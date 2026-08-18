# -*- coding: utf-8 -*-
"""短剧 JSON：角色/场景绑图 + 分镜循环 + 成片拼接。"""

from __future__ import annotations

import copy
import glob
import json
import os
import re
import uuid
import urllib.request
from typing import Any

import folder_paths
import torch
from comfy.cli_args import args
from comfy_execution.graph import ExecutionBlocker
from server import PromptServer

CAT = "short-drama/json"
ROLE_KEYS = ("角色图片", "角色档案")
SCENE_KEYS = ("场景图片", "场景档案")
_NUM = re.compile(r"(\d+(?:\.\d+)?)")
_TAG = re.compile(r"<([^<>]+)>")
_PIC = re.compile(r"^Picture_(\d+)$")
_PIC_REF = re.compile(r"[<［\[]\s*Picture\s*(\d+)\s*[>］\]]", re.I)


def used_picture_indices(shot_prompt: str) -> set[int]:
    """从 shot_prompt 里收集 <Picture N>；空集合表示未解析到任何参考图编号。"""
    return {int(m) for m in _PIC_REF.findall(shot_prompt or "")}


def used_picture_indices_by_name(shot_prompt: str, slots: list[dict[str, Any]]) -> set[int]:
    """按 <角色/场景名> 回退解析 Picture 槽位。"""
    pics = {s["name"]: int(s["picture"]) for s in slots if s.get("name") and s.get("picture")}
    if not pics:
        return set()
    hit = set()
    for tag in _TAG.findall(shot_prompt or ""):
        name = str(tag).strip()
        if name in pics:
            hit.add(pics[name])
    return hit


def resolve_picture_gate(shot_prompt: str | None, data: dict[str, Any]) -> set[int] | None:
    """
    返回本镜应放行的 Picture 编号集合。
    None = 不过滤（全部放行）；set = 只放行这些槽。
    无尖括号标签时不过滤，避免接错 shot_name 时把参考图全清掉。
    """
    if not isinstance(shot_prompt, str) or not shot_prompt.strip():
        return None
    gate = used_picture_indices(shot_prompt)
    if not gate:
        gate = used_picture_indices_by_name(shot_prompt, discover_slots(data))
    if not gate and not _TAG.findall(shot_prompt):
        return None
    return gate


class _FlexReturns(tuple):
    """可 JSON 序列化；下标越界时仍返回 *，供 ComfyUI 校验动态槽。"""

    def __new__(cls):
        return tuple.__new__(cls, ("STRING", "*"))

    def __getitem__(self, i):
        if isinstance(i, slice):
            start, stop, step = i.indices(256)
            return tuple(self[j] for j in range(start, stop, step))
        i = int(i)
        if i < 0:
            return tuple.__getitem__(self, i)
        return "STRING" if i == 0 else "*"


_FLEX_RETURNS = _FlexReturns()


def _slot_count_from_names(names) -> int:
    n = 0
    for key in names or []:
        m = _PIC.match(str(key))
        if m:
            n = max(n, int(m.group(1)))
    return n


def _apply_binder_schema(n: int) -> int:
    return max(0, int(n))


def _parse_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"短剧 JSON 解析失败: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("短剧提示词必须是 JSON 对象")
    return data


def _first_dict(data: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    for key in keys:
        block = data.get(key)
        if isinstance(block, dict) and block:
            return block
    return {}


def discover_slots(data: dict[str, Any]) -> list[dict[str, Any]]:
    slots, seen = [], set()
    for kind, keys in (("角色", ROLE_KEYS), ("场景", SCENE_KEYS)):
        for name in _first_dict(data, keys):
            key = str(name).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            pic = len(slots) + 1
            slots.append({"name": key, "kind": kind, "picture": pic})
    return slots


def _empty_image():
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def _is_real_image(img) -> bool:
    """未接线 / 空值不算参考图，允许不绑定角色。"""
    if img is None:
        return False
    t = img[0] if isinstance(img, (list, tuple)) and img else img
    return isinstance(t, torch.Tensor) and t.ndim >= 3 and t.numel() > 0


def apply_bound_pictures(text: str, bound_name_to_pic: dict[str, int]) -> str:
    """只给已接线的角色/场景加 <Picture N>；未绑的名称保留尖括号，不占参考图序号。"""
    if not text:
        return ""
    text = re.sub(r"/<\s*Picture\s+\d+\s*>", "", text, flags=re.I)
    text = _PIC_REF.sub("", text)

    def repl(m: re.Match) -> str:
        name = m.group(1).strip()
        low = name.lower().replace(" ", "")
        if low.startswith("picture") or low.startswith("image"):
            return m.group(0)
        pic = bound_name_to_pic.get(name)
        return f"<{name}>/<Picture {pic}>" if pic else m.group(0)

    return _TAG.sub(repl, text)


def bind_slot_images(slots: list[dict[str, Any]], kwargs: dict, gate: set[int] | None) -> tuple[dict[str, int], list]:
    """按槽位收集已接参考图；gate 为本镜放行的旧 Picture 号。返回 名称→紧凑序号 与按槽输出（未接为 None）。"""
    bound: dict[str, int] = {}
    pics: list = []
    compact = 0
    n = len(slots) or _slot_count_from_names(kwargs)
    if slots:
        for slot in slots:
            i = int(slot["picture"])
            img = kwargs.get(f"Picture_{i}")
            if not _is_real_image(img) or (gate is not None and i not in gate):
                img = None
            if img is not None:
                compact += 1
                bound[str(slot["name"])] = compact
            pics.append(img)
        return bound, pics
    for i in range(1, n + 1):
        img = kwargs.get(f"Picture_{i}")
        pics.append(img if _is_real_image(img) else None)
    return bound, pics


def list_shots(data: dict[str, Any]) -> list[tuple[str, list]]:
    seq = data.get("分镜序列")
    shots = []
    if isinstance(seq, dict):
        keys = sorted(seq, key=lambda n: (int(m.group()) if (m := re.search(r"\d+", str(n))) else 999, str(n)))
        for key in keys:
            clips = seq[key]
            if clips is None or clips == []:
                continue
            if isinstance(clips, dict):
                clips = [clips]
            if isinstance(clips, list):
                shots.append((str(key), clips))
    elif isinstance(seq, list):
        for i, item in enumerate(seq, 1):
            clips = item if isinstance(item, list) else [item] if isinstance(item, dict) else None
            if clips:
                shots.append((f"分镜{i}", clips))
    return shots


def _seconds(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    m = _NUM.search(str(raw or ""))
    return float(m.group(1)) if m else 0.0


def _clip_duration(clip: dict) -> float:
    if "时长" in clip:
        return _seconds(clip["时长"])
    text = str(clip.get("时间") or "")
    if m := re.search(r"时长\s*[:：]\s*(\d+(?:\.\d+)?)", text):
        return float(m.group(1))
    if m := re.search(r"(\d+(?:\.\d+)?)\s*[-~～到至]\s*(\d+(?:\.\d+)?)", text):
        return float(m.group(2))
    ends = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:s|S|秒)", text)]
    return max(ends) if ends else 0.0


def shot_duration(clips: list) -> float:
    total = sum(_clip_duration(c) for c in clips if isinstance(c, dict))
    return total if total > 0 else 10.0


def total_duration(data: dict[str, Any]) -> float:
    """JSON 全部分镜时长之和（秒）。"""
    return float(sum(shot_duration(clips) for _, clips in list_shots(data)))


def override_shot_duration(csv: str, index: int, fallback: float) -> float:
    parts = [p.strip() for p in str(csv or "").replace("，", ",").split(",") if p.strip() != ""]
    if not parts:
        return float(fallback)
    i = max(1, int(index)) - 1
    if i >= len(parts):
        return float(fallback)
    v = _seconds(parts[i])
    return v if v > 0 else float(fallback)


_CLIP_TEXT_KEYS = ("场景", "镜头", "画面", "声音", "构图", "光影", "衔接", "禁止", "对白", "H3", "h3")
_H3_KEYS = ("H3", "h3")


def _clip_blob(clips: list) -> str:
    parts: list[str] = []
    for c in clips:
        if not isinstance(c, dict):
            continue
        for key in _CLIP_TEXT_KEYS:
            val = c.get(key)
            if isinstance(val, dict):
                parts.extend(str(x) for x in (*val.keys(), *val.values()))
            elif val is not None:
                parts.append(str(val))
    return "\n".join(parts)


def _names_in_text(names: list[str], text: str) -> list[str]:
    name_set = {n for n in names if n}
    hit = [t for t in _TAG.findall(text or "") if t in name_set]
    seen, ordered = set(), []
    for n in hit:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    if ordered:
        return ordered
    for name in sorted(name_set, key=len, reverse=True):
        if name in text and name not in ordered:
            ordered.append(name)
    return ordered


def _expand_ref_tags(text: str, pics: dict[str, int]) -> str:
    if not text or not pics:
        return text

    def repl(m: re.Match) -> str:
        name = m.group(1).strip()
        if name.lower().startswith(("picture", "image")):
            return m.group(0)
        pic = pics.get(name)
        return f"<{name}>/<Picture {pic}>" if pic else m.group(0)

    return _TAG.sub(repl, str(text))


def _expand_clips_refs(clips: list, pics: dict[str, int]) -> list:
    out = []
    for c in clips:
        if not isinstance(c, dict):
            out.append(c)
            continue
        nc = dict(c)
        for key in ("场景", "镜头", "画面", "声音", "构图", "光影", "衔接", "禁止", "H3", "h3"):
            if isinstance(nc.get(key), str):
                nc[key] = _expand_ref_tags(nc[key], pics)
        out.append(nc)
    return out


def _h3_from_clips(clips: list) -> str:
    """分镜若写了 H3 字段，直接把英文 H3 提示词送给 MiniMax，不再 dumping 中文 JSON。"""
    parts: list[str] = []
    for c in clips:
        if not isinstance(c, dict):
            continue
        for key in _H3_KEYS:
            val = c.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
                break
    return "\n\n".join(parts)


def filter_shot_context(data: dict[str, Any], clips: list) -> dict[str, Any]:
    out = {k: v for k, v in data.items() if k not in {"分镜序列", "参考绑定"}}
    blob = _clip_blob(clips)
    roles = _first_dict(data, ROLE_KEYS)
    scenes = _first_dict(data, SCENE_KEYS)
    for k in ROLE_KEYS + SCENE_KEYS:
        out.pop(k, None)

    used_roles = _names_in_text(list(roles), blob)
    used_scenes = _names_in_text(list(scenes), blob)
    for c in clips:
        if isinstance(c, dict) and isinstance(c.get("对白"), dict):
            for spk in c["对白"]:
                if spk in roles and spk not in used_roles:
                    used_roles.append(spk)

    # 画面/对白未点名的角色不要回退成「全员」
    if roles:
        filtered_roles = {k: roles[k] for k in used_roles if k in roles}
        if filtered_roles:
            out["角色图片"] = filtered_roles
    if scenes:
        filtered_scenes = {k: scenes[k] for k in used_scenes if k in scenes}
        out["场景图片"] = filtered_scenes or {k: scenes[k] for k in list(scenes)[:1]}

    # 不改写「镜头规则」：工具只裁剪本镜用到的角色/场景，文案由用户 JSON 决定
    out["当前场景"] = "、".join((out.get("场景图片") or {})) or "（无）"
    return out


def build_shot_prompt(data: dict[str, Any], index: int) -> tuple[str, str, int, float, str]:
    shots = list_shots(data)
    if not shots:
        raise ValueError("JSON 中没有可用的分镜序列")
    n = len(shots)
    idx = max(1, min(int(index), n))
    name, clips = shots[idx - 1]
    out = filter_shot_context(data, clips)
    pics = {s["name"]: s["picture"] for s in discover_slots(data)}
    clips = _expand_clips_refs(clips, pics)
    h3 = _h3_from_clips(clips)
    if h3:
        return h3, name, n, shot_duration(clips), f"ShortDrama_{name}"
    out.update({"当前分镜": name, "分镜序号": idx, "分镜总数": n, "分镜序列": clips})
    return json.dumps(out, ensure_ascii=False, indent=2), name, n, shot_duration(clips), f"ShortDrama_{name}"


def _latest_mp4(output_dir: str, prefix: str) -> str | None:
    """按文件名前缀取最新 mp4；避免 ShortDrama_分镜1 误匹配到 分镜10。"""
    pref = str(prefix or "")
    base_pref = os.path.basename(pref)
    files = []
    for pat in (os.path.join(output_dir, f"{pref}*.mp4"), os.path.join(output_dir, "**", f"{pref}*.mp4")):
        for f in glob.glob(pat, recursive=True):
            if not os.path.isfile(f):
                continue
            name = os.path.basename(f)
            if "merged" in name.lower():
                continue
            if not name.startswith(base_pref):
                continue
            rest = name[len(base_pref) :]
            # 前缀后须结束或接 _ / - / .，不能再跟数字（防止 分镜1 吃到 分镜10）
            if rest and rest[0].isdigit():
                continue
            files.append(f)
    return max(files, key=os.path.getmtime) if files else None


def _shot_names_for_batch(prompt_json: str, start: int, end: int) -> list[str]:
    """本批序号 start..end（含）→ JSON 分镜名列表。分镜名≠序号。"""
    shots = list_shots(_parse_json(prompt_json))
    if not shots:
        raise ValueError("JSON 中没有可用的分镜序列")
    names = []
    for i in range(int(start), int(end) + 1):
        if i < 1 or i > len(shots):
            raise ValueError(f"分镜序号越界: {i}（共 {len(shots)} 镜）")
        names.append(shots[i - 1][0])
    return names


def _prompt_get_node(prompt: dict, node_id) -> dict | None:
    if not isinstance(prompt, dict):
        return None
    if node_id in prompt and isinstance(prompt[node_id], dict):
        return prompt[node_id]
    key = str(node_id)
    node = prompt.get(key)
    return node if isinstance(node, dict) else None


def _resolve_graph_string(prompt: dict, value, _depth: int = 0) -> str:
    """解析 API prompt 里的字符串；连线多为 [node_id, slot]。"""
    if _depth > 8:
        return ""
    if isinstance(value, str):
        return value
    if not (isinstance(value, (list, tuple)) and value):
        return ""
    src = _prompt_get_node(prompt, value[0])
    if not src:
        return ""
    inputs = src.get("inputs") or {}
    for key in ("value", "string", "text", "prompt_json", "STRING"):
        if key not in inputs:
            continue
        got = _resolve_graph_string(prompt, inputs[key], _depth + 1) if not isinstance(inputs[key], str) else inputs[key]
        if isinstance(got, str) and got.strip():
            return got
    for v in inputs.values():
        if isinstance(v, str) and len(v.strip()) > 2:
            return v
        if isinstance(v, (list, tuple)):
            got = _resolve_graph_string(prompt, v, _depth + 1)
            if got.strip():
                return got
    return ""


def _prompt_json_from_workflow(prompt: dict | None) -> str:
    """从同图「分镜选择」自动取 prompt_json，拼接节点无需再接线。"""
    if not isinstance(prompt, dict):
        return ""
    for node in prompt.values():
        if not isinstance(node, dict) or node.get("class_type") != "ShortDramaJSONShotSelector":
            continue
        text = _resolve_graph_string(prompt, (node.get("inputs") or {}).get("prompt_json"))
        if text.strip():
            return text
    return ""


def clamp_run(total: int, start: int, end: int) -> tuple[int, int]:
    """开始分镜:结束分镜 → (当前索引, 剩余镜数)。开始不能大于结束。"""
    total = max(0, int(total))
    if total <= 0:
        return 1, 1
    last = max(1, min(int(end), total))
    cur = max(1, min(int(start), last))
    return cur, last - cur + 1


class ShortDramaJSONSlotParser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "总时长": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 1.0,
                        "max": 300.0,
                        "step": 1.0,
                        "tooltip": "点「刷新槽位标签」从 JSON 分镜时长求和填入；可用左右箭头微调。",
                    },
                ),
                "prompt_json": ("STRING", {"forceInput": True, "multiline": True, "default": ""}),
            },
            "optional": {
                "shot_prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "default": "",
                        "tooltip": "接「分镜选择/循环」的 shot_prompt；只放行本镜引用的参考图。角色槽可不接图。",
                    },
                ),
            },
        }

    RETURN_TYPES = _FLEX_RETURNS
    RETURN_NAMES = ("prompt_json", "shot_prompt", "总时长")
    FUNCTION = "run"
    CATEGORY = CAT

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, prompt_json=None, **kwargs):
        n = 0
        if isinstance(prompt_json, str) and prompt_json.strip():
            try:
                n = max(n, len(discover_slots(_parse_json(prompt_json))))
            except Exception:
                pass
        n = max(n, _slot_count_from_names(input_types), _slot_count_from_names(kwargs))
        _apply_binder_schema(n)
        return True

    def run(self, prompt_json, 总时长=10.0, shot_prompt=None, **kwargs):
        if not isinstance(prompt_json, str):
            prompt_json = "" if prompt_json is None else str(prompt_json)
        if not isinstance(shot_prompt, str):
            shot_prompt = None
        data = _parse_json(prompt_json)
        slots = discover_slots(data)
        n = len(slots) or _slot_count_from_names(kwargs)
        _apply_binder_schema(n)
        gate = resolve_picture_gate(shot_prompt, data)
        bound, pics = bind_slot_images(slots, kwargs, gate)
        if n > len(pics):
            pics.extend([None] * (n - len(pics)))
        rewritten = apply_bound_pictures(shot_prompt or "", bound)
        if not rewritten.strip():
            rewritten = prompt_json or ""
        dur = 总时长 if 总时长 is not None else kwargs.get("时长微调", 10.0)
        # 顺序：prompt_json, Picture_1..N, shot_prompt, 总时长 —— Picture 槽位从 1 起，避免挤乱 MiniMax 参考图连线
        return (prompt_json or "",) + tuple(pics[:n]) + (rewritten, float(dur))


class ShortDramaJSONShotSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_json": ("STRING", {"forceInput": True, "multiline": True, "default": ""}),
                "开始分镜": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1, "tooltip": "按 JSON 分镜排列顺序从第几条开始（含）。键名如「分镜4」是分镜名，不等于序号。"}),
                "结束分镜": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1, "tooltip": "跑到顺序第几条（含）。2:2 只跑第2条。"}),
                "分镜时长": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("shot_prompt", "shot_name", "shot_index", "shot_count", "duration", "filename_prefix")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, prompt_json, 开始分镜=None, 结束分镜=None, 分镜时长="", **kw):
        data = _parse_json(prompt_json)
        start = 开始分镜 if 开始分镜 is not None else kw.get("分镜索引", 1)
        end = 结束分镜 if 结束分镜 is not None else kw.get("分镜结束", kw.get("分镜数量", start))
        cur, remain = clamp_run(len(list_shots(data)), start, end)
        prompt, name, _n, duration, prefix = build_shot_prompt(data, cur)
        return prompt, name, cur, remain, float(override_shot_duration(分镜时长, cur, duration)), prefix


class ShortDramaJSONAutoNextShot:
    """兼容旧工作流：逻辑已并入 ShortDramaJSONConcatBatch，本节点不再续跑。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_index": ("INT", {"default": 1, "min": 1, "max": 64}),
                "shot_count": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {"trigger": ("VIDEO",)},
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = CAT
    OUTPUT_NODE = True
    DEPRECATED = True

    def run(self, shot_index, shot_count, trigger=None):
        _ = (shot_index, shot_count, trigger)
        return {"ui": {"text": ["已并入「续跑并拼接」，可删除本节点"]}}


def _queue_next_shot(prompt_dict: dict, next_index: int, remain: int, batch_start: int) -> str:
    new_prompt = copy.deepcopy(prompt_dict)
    found = False
    for node in new_prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.setdefault("inputs", {})
        ctype = node.get("class_type")
        if ctype == "ShortDramaJSONShotSelector":
            inputs["开始分镜"] = next_index
            inputs["结束分镜"] = next_index + remain - 1
            for old in ("分镜索引", "分镜结束", "分镜数量"):
                inputs.pop(old, None)
            found = True
        elif ctype == "ShortDramaJSONConcatBatch":
            inputs["merge_batch_start"] = batch_start
    if not found:
        return "未找到分镜选择节点"

    payload = {
        "prompt": new_prompt,
        "client_id": getattr(PromptServer.instance, "client_id", None) or str(uuid.uuid4()),
    }
    if not payload["client_id"]:
        payload.pop("client_id", None)
    port = int(getattr(args, "port", 8188) or 8188)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/prompt",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    return f"已排队分镜{next_index}（剩余{remain}）｜{body[:100]}"


def _load_video_av(path: str) -> tuple[torch.Tensor, dict | None, float]:
    import av
    import numpy as np

    container = av.open(path)
    try:
        if not any(s.type == "video" for s in container.streams):
            raise ValueError(f"无视频轨: {path}")
        v_stream = next(s for s in container.streams if s.type == "video")
        fps = float(v_stream.average_rate) if v_stream.average_rate else 24.0
        frames = [torch.from_numpy(f.to_ndarray(format="rgb24").astype(np.float32) / 255.0) for f in container.decode(video=0)]
        if not frames:
            raise ValueError(f"视频无帧: {path}")
        images = torch.stack(frames, dim=0)

        audio = None
        try:
            container.seek(0)
            if any(s.type == "audio" for s in container.streams):
                a_stream = next(s for s in container.streams if s.type == "audio")
                chunks = []
                for frame in container.decode(audio=0):
                    arr = frame.to_ndarray()
                    if arr.ndim == 1:
                        arr = arr[None, :]
                    chunks.append(torch.from_numpy(arr.astype("float32")))
                if chunks:
                    wave = torch.cat(chunks, dim=-1)[:2]
                    audio = {"waveform": wave.unsqueeze(0), "sample_rate": int(a_stream.rate or 44100)}
        except Exception:
            audio = None
        return images, audio, fps
    finally:
        container.close()


def _concat_audio(parts: list[dict]) -> dict:
    sr = next((int(p["sample_rate"]) for p in parts if p.get("sample_rate")), 44100)
    waves = []
    for p in parts:
        w = p["waveform"]
        if w.dim() == 2:
            w = w.unsqueeze(0)
        waves.append(w.to(dtype=torch.float32))
        sr = int(p.get("sample_rate") or sr)
    ch = max(w.shape[1] for w in waves)
    aligned = []
    for w in waves:
        if w.shape[1] < ch:
            w = torch.cat([w, torch.zeros((w.shape[0], ch - w.shape[1], w.shape[2]), dtype=w.dtype)], dim=1)
        elif w.shape[1] > ch:
            w = w[:, :ch]
        aligned.append(w)
    return {"waveform": torch.cat(aligned, dim=-1), "sample_rate": sr}


def _blocked3(msg: str | None = None):
    return (ExecutionBlocker(msg), ExecutionBlocker(None), ExecutionBlocker(None))


class ShortDramaJSONConcatBatch:
    """未完成本批则自动 Queue 下一镜；最后一镜按分镜名拼接成片。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_index": ("INT", {"default": 1, "min": 1, "max": 64}),
                "shot_count": ("INT", {"default": 1, "min": 1, "max": 64}),
                "prefix_root": ("STRING", {"default": "ShortDrama_"}),
            },
            "optional": {
                "trigger": ("VIDEO",),
                "merge_batch_start": ("INT", {"default": 0, "min": 0, "max": 64}),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    OUTPUT_NODE = True

    def run(self, shot_index, shot_count, prefix_root="ShortDrama_", trigger=None, merge_batch_start=0, prompt=None):
        _ = trigger
        idx, remain = int(shot_index), int(shot_count)
        batch_start = int(merge_batch_start or 0)

        # 本批未结束：续跑下一镜，下游成片先挡住
        if remain > 1:
            if not isinstance(prompt, dict) or not prompt:
                return {"ui": {"text": ["无法获取 prompt，续跑失败"]}, "result": _blocked3()}
            start = batch_start if batch_start > 0 else idx
            try:
                msg = _queue_next_shot(prompt, idx + 1, remain - 1, start)
            except Exception as exc:
                msg = f"续跑失败: {exc}"
            return {"ui": {"text": [msg]}, "result": _blocked3()}

        # 单镜批次：不拼
        if batch_start <= 0:
            return {"ui": {"text": [f"本批完成（单镜{idx}，无需拼接）"]}, "result": _blocked3()}

        # 最后一镜：按分镜名拼接
        out_dir = folder_paths.get_output_directory()
        root = str(prefix_root or "ShortDrama_")
        end = idx
        start = batch_start
        try:
            prompt_json = _prompt_json_from_workflow(prompt)
            if not prompt_json.strip():
                raise ValueError("无法从「分镜选择/循环」读取 prompt_json，请确认同图已接 JSON")
            names = _shot_names_for_batch(prompt_json, start, end)
            paths = []
            for name in names:
                path = _latest_mp4(out_dir, f"{root}{name}")
                if not path:
                    raise FileNotFoundError(f"缺少分镜视频: {root}{name}*.mp4")
                paths.append(path)

            packs = [_load_video_av(p) for p in paths]
            fps = packs[-1][2]
            h0, w0 = packs[0][0].shape[1], packs[0][0].shape[2]
            resized, aud_parts = [], []
            for imgs, aud, _fps in packs:
                if imgs.shape[1] != h0 or imgs.shape[2] != w0:
                    x = torch.nn.functional.interpolate(imgs.permute(0, 3, 1, 2), size=(h0, w0), mode="bilinear", align_corners=False)
                    imgs = x.permute(0, 2, 3, 1)
                resized.append(imgs)
                if aud is not None:
                    aud_parts.append(aud)
                else:
                    samples = max(1, int(imgs.shape[0] / max(fps, 1e-3) * 44100))
                    aud_parts.append({"waveform": torch.zeros((1, 1, samples), dtype=torch.float32), "sample_rate": 44100})
            label = " + ".join(names)
            info = f"已拼接 {label} 共{len(paths)}段｜" + " + ".join(os.path.basename(p) for p in paths)
            return {"ui": {"text": [info]}, "result": (torch.cat(resized, dim=0), _concat_audio(aud_parts), info)}
        except Exception as exc:
            return {"ui": {"text": [f"拼接失败: {exc}"]}, "result": _blocked3(f"拼接失败: {exc}")}


NODE_CLASS_MAPPINGS = {
    "ShortDramaJSONSlotParser": ShortDramaJSONSlotParser,
    "ShortDramaJSONShotSelector": ShortDramaJSONShotSelector,
    "ShortDramaJSONAutoNextShot": ShortDramaJSONAutoNextShot,
    "ShortDramaJSONConcatBatch": ShortDramaJSONConcatBatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ShortDramaJSONSlotParser": "短剧JSON · 角色场景图绑定",
    "ShortDramaJSONShotSelector": "短剧JSON · 分镜选择/循环",
    "ShortDramaJSONAutoNextShot": "短剧JSON · 续跑下一镜（已并入，可删）",
    "ShortDramaJSONConcatBatch": "短剧JSON · 续跑并拼接",
}
