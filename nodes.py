# -*- coding: utf-8 -*-
"""短剧 JSON：角色/场景绑图 + 分镜循环 + 成片拼接。"""

from __future__ import annotations

import copy
import glob
import json
import logging
import os
import re
import threading
import uuid
import urllib.request
from typing import Any

import folder_paths
import torch
from comfy.cli_args import args
from comfy_execution.graph import ExecutionBlocker
from server import PromptServer

_log = logging.getLogger("ComfyUI-ShortDramaJSON")

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


def _apply_binder_schema(n: int, cls=None) -> int:
    """按槽位数登记 RETURN；左右同序：prompt_json, Picture_*, 上一镜末帧, shot_prompt, 总时长。"""
    n = max(0, int(n))
    target = cls
    if target is None:
        return n
    names = ("prompt_json",) + tuple(f"Picture_{i}" for i in range(1, n + 1)) + ("上一镜末帧", "shot_prompt", "总时长")
    types = ("STRING",) + tuple("IMAGE" for _ in range(n)) + ("IMAGE", "STRING", "FLOAT")
    target.RETURN_NAMES = names
    target.RETURN_TYPES = types
    return n


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


def _is_usable_ref_image(img) -> bool:
    """真实参考图（排除空值）。"""
    return _is_real_image(img)


PREV_FRAME_SLOT = "上一镜末帧"
_DIALOGUE_TAG = re.compile(r"<d>\s*\[[^\]]*\]\s*.+?</d>", re.I | re.S)


def shot_has_dialogue(text: str) -> bool:
    """本镜提示词是否含 <d>…</d> 对白。无对白镜不应再喂上一镜人声音频。"""
    return bool(_DIALOGUE_TAG.search(text or ""))


def _inject_prev_frame_prompt(text: str, as_first_frame: bool) -> str:
    """确保提示词引用 <上一镜末帧>；可选写成 I2VA 首帧对齐。是否接 ref_video/audio/latent 由工作流自行接线。"""
    tag = f"<{PREV_FRAME_SLOT}>"
    t = (text or "").strip()
    if as_first_frame:
        line = (
            f"For the target video, at 0.00 seconds into the target video, "
            f"{tag} (from [Shot 1]) is fully referenced."
        )
        if "is fully referenced" in t[:400] or "How the reference pictures align" in t[:400]:
            if tag not in t:
                t = f"{t}\n\nContinuity anchor: {tag} is the last frame of the previous shot."
        elif not t:
            t = line
        elif tag in t and line.split(tag)[0] in t:
            pass
        else:
            t = re.sub(
                r"^For the target video, at 0\.00 seconds into the target video, .+? is fully referenced\.\s*",
                "",
                t,
                count=1,
                flags=re.I | re.S,
            ).strip()
            t = f"{line}\n\n{t}" if t else line
    elif tag not in t:
        t = f"{t}\n\nReference still: {tag} is the last frame of the previous shot for continuity.".strip()
    return t


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


def override_shot_tail_frames(csv: str | int | float | None, index: int, fallback: int = 0) -> int:
    """按镜解析续接帧数 CSV（与分镜时长同格式）。0=该镜不续接；空串用 fallback。兼容旧版单整数。"""
    if isinstance(csv, bool):
        return max(0, int(fallback))
    if isinstance(csv, (int, float)):
        return max(0, int(csv))
    parts = [p.strip() for p in str(csv or "").replace("，", ",").split(",") if p.strip() != ""]
    if not parts:
        return max(0, int(fallback))
    i = max(1, int(index)) - 1
    if i >= len(parts):
        return max(0, int(fallback))
    try:
        return max(0, int(float(parts[i])))
    except ValueError:
        return max(0, int(fallback))


_CLIP_TEXT_KEYS = ("场景", "镜头", "画面", "声音", "构图", "光影", "衔接", "禁止", "对白", "内容")
# 直通英文提示词：有「内容」字段则不再 dumping 中文 JSON
_H3_KEYS = ("内容",)


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
        for key in ("场景", "镜头", "画面", "声音", "构图", "光影", "衔接", "禁止", "内容"):
            if isinstance(nc.get(key), str):
                nc[key] = _expand_ref_tags(nc[key], pics)
        out.append(nc)
    return out


def _h3_from_clips(clips: list) -> str:
    """分镜若写了「内容」字段，直接把英文提示词送给 MiniMax，不再 dumping 中文 JSON。"""
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
                "上一镜末帧": (
                    "IMAGE",
                    {
                        "tooltip": "接「上一镜末帧」节点。有有效图时自动并入末位 Picture，并写入提示词。",
                    },
                ),
                "末帧续接": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "开启后预留末位 Picture 槽，并把上一镜末帧并入参考图。",
                    },
                ),
                "末帧作首帧": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "把上一镜末帧写成 I2VA 首帧对齐（0.00s fully referenced）。",
                    },
                ),
            },
        }

    RETURN_TYPES = _FLEX_RETURNS
    RETURN_NAMES = ("prompt_json", "上一镜末帧", "shot_prompt", "总时长")
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
        # 末帧续接多预留一槽
        cont = kwargs.get("末帧续接", True)
        if cont is None and isinstance(input_types, dict):
            cont = True
        if cont:
            n += 1
        _apply_binder_schema(n, cls)
        return True

    def run(
        self,
        prompt_json,
        总时长=10.0,
        shot_prompt=None,
        上一镜末帧=None,
        末帧续接=True,
        末帧作首帧=True,
        **kwargs,
    ):
        if not isinstance(prompt_json, str):
            prompt_json = "" if prompt_json is None else str(prompt_json)
        if not isinstance(shot_prompt, str):
            shot_prompt = None
        data = _parse_json(prompt_json)
        slots = discover_slots(data)
        n = len(slots) or _slot_count_from_names(kwargs)
        # 末帧续接：有有效上一镜末帧才并入；没有就空槽（None），不传假图
        use_prev = bool(末帧续接) and _is_usable_ref_image(上一镜末帧)
        out_n = n + (1 if bool(末帧续接) else 0)
        _apply_binder_schema(out_n, self.__class__)
        gate = resolve_picture_gate(shot_prompt, data)
        bound, pics = bind_slot_images(slots, kwargs, gate)
        if n > len(pics):
            pics.extend([None] * (n - len(pics)))
        pics = list(pics[:n])

        text = shot_prompt or ""
        prev_out = None
        if use_prev:
            text = _inject_prev_frame_prompt(text, bool(末帧作首帧))
            compact = max(bound.values(), default=0) + 1
            bound[PREV_FRAME_SLOT] = compact
            pics.append(上一镜末帧)
            prev_out = 上一镜末帧
        elif bool(末帧续接):
            pics.append(None)  # 预留槽位但空，MiniMax 跳过

        rewritten = apply_bound_pictures(text, bound)
        if not rewritten.strip():
            rewritten = prompt_json or ""
        dur = 总时长 if 总时长 is not None else kwargs.get("时长微调", 10.0)
        # 与输入同序：prompt_json, Picture_1..out_n, 上一镜末帧, shot_prompt, 总时长
        return (prompt_json or "",) + tuple(pics[:out_n]) + (prev_out, rewritten, float(dur))


def _silent_audio(num_samples: int = 1024, sample_rate: int = 44100) -> dict:
    n = max(1, int(num_samples))
    return {"waveform": torch.zeros((1, 1, n), dtype=torch.float32), "sample_rate": int(sample_rate)}


# MiniMax H3：参考视频至少 5 帧，且裁切后须满足 n % 17 == 5（5 / 22 / 39… 合法）
_MINIMAX_REF_VIDEO_MIN_FRAMES = 5
# 续跑默认最多 22 帧（≈0.9s@24fps）。传满 5 秒会在第 2 镜 VAE 编码参考视频时长时间假死。
_MINIMAX_REF_VIDEO_MAX_FRAMES = 22
_REF_VIDEO_MAX_SHORT_EDGE = 768


def _largest_valid_ref_frames(n: int, cap: int) -> int:
    """最大合法帧数：<=min(n,cap) 且 n%17==5（不足则退到 5）。"""
    n = min(int(n), int(cap))
    if n < _MINIMAX_REF_VIDEO_MIN_FRAMES:
        return _MINIMAX_REF_VIDEO_MIN_FRAMES
    while n > _MINIMAX_REF_VIDEO_MIN_FRAMES and n % 17 != 5:
        n -= 1
    if n % 17 != 5:
        return _MINIMAX_REF_VIDEO_MIN_FRAMES
    return n


def _downscale_ref_video(images: torch.Tensor, max_short: int = _REF_VIDEO_MAX_SHORT_EDGE) -> torch.Tensor:
    """缩小参考视频短边，降低第 2 镜起 VAE 编码压力。"""
    if images is None or int(images.shape[0]) <= 0:
        return images
    h, w = int(images.shape[1]), int(images.shape[2])
    short = min(h, w)
    if short <= int(max_short):
        return images.contiguous()
    scale = float(max_short) / float(short)
    nh = max(16, int(round(h * scale / 16.0) * 16))
    nw = max(16, int(round(w * scale / 16.0) * 16))
    x = torch.nn.functional.interpolate(
        images.permute(0, 3, 1, 2).float(),
        size=(nh, nw),
        mode="bilinear",
        align_corners=False,
    )
    return x.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)


def _ensure_minimax_ref_video(
    images: torch.Tensor,
    *,
    max_frames: int = _MINIMAX_REF_VIDEO_MAX_FRAMES,
) -> torch.Tensor:
    """补齐/裁到 MiniMax 参考视频合法帧数；从末尾截取（保留衔接关键帧）。"""
    video = images.contiguous()
    n = int(video.shape[0])
    if n < _MINIMAX_REF_VIDEO_MIN_FRAMES:
        last = video[-1:]
        pad = last.expand(_MINIMAX_REF_VIDEO_MIN_FRAMES - n, *video.shape[1:]).clone()
        video = torch.cat([video, pad], dim=0)
        n = int(video.shape[0])
    keep = _largest_valid_ref_frames(n, max_frames)
    # 必须取尾部：video[:keep] 会丢掉真正的上一镜末帧
    return video[-keep:].contiguous().clone()


def _trim_tail_av(images: torch.Tensor, aud: dict | None, fps: float, frames: int):
    """只保留成片末尾 N 帧的视频与对应时长音频；frames<=0 表示整段。"""
    fps = max(float(fps or 0), 1e-3)
    total = int(images.shape[0])
    n_req = int(frames)
    if n_req <= 0 or total <= 0:
        video = images.contiguous().clone()
        n_keep = total
    else:
        n_keep = max(1, min(total, n_req))
        video = images[-n_keep:].contiguous().clone()

    if aud is not None and isinstance(aud, dict) and aud.get("waveform") is not None:
        w = aud["waveform"]
        sr = int(aud.get("sample_rate") or 44100)
        if w.dim() == 2:
            w = w.unsqueeze(0)
        t = int(w.shape[-1])
        if n_req <= 0:
            samples_keep = t
        else:
            samples_keep = max(1, min(t, int(round(n_keep / fps * sr))))
        aud_out = {"waveform": w[..., -samples_keep:].contiguous().clone(), "sample_rate": sr}
    else:
        samples = max(1024, int(round(max(n_keep, 1) / fps * 44100)))
        aud_out = _silent_audio(samples, 44100)
    return video, aud_out, n_keep


def _load_prev_shot_bundle(
    prompt_json,
    shot_index: int,
    *,
    enabled: bool = True,
    batch_first: int = 1,
    prefix_root: str = "ShortDrama_",
    tail_frames: int = 22,
):
    """取上一镜末帧/视频/音频。无上一镜时三者皆为 None（与未引用参考图一样，下游跳过不传）。"""
    def _none(msg: str):
        return None, None, None, msg

    if not enabled:
        return _none("续接帧数为0或不启用")
    idx = int(shot_index)
    first = int(batch_first) if int(batch_first or 0) > 0 else 1
    if idx <= first:
        return _none(f"分镜{idx}为批首，无上一镜成片")

    try:
        data = _parse_json(prompt_json)
        shots = list_shots(data)
    except Exception as exc:
        return _none(f"JSON 无效: {exc}")

    prev_i = idx - 1
    if prev_i < 1 or prev_i > len(shots):
        return _none(f"上一镜序号无效: {prev_i}")

    prev_name = shots[prev_i - 1][0]
    root = str(prefix_root or "ShortDrama_").strip() or "ShortDrama_"
    path = _latest_mp4(folder_paths.get_output_directory(), f"{root}{prev_name}")
    if not path:
        return _none(f"未找到上一镜视频: {root}{prev_name}*.mp4（请先跑完上一镜并 SaveVideo）")

    try:
        images, aud, fps = _load_video_av(path)
        n_tail = int(tail_frames if tail_frames is not None else 22)
        video, aud, n_keep = _trim_tail_av(images, aud, fps, n_tail)
        # 先缩小再截合法帧数，避免第 2 镜 VAE 编码参考视频卡死
        video = _downscale_ref_video(video)
        video = _ensure_minimax_ref_video(video)
        sr = int(aud.get("sample_rate") or 44100)
        need = max(1024, int(round(video.shape[0] / max(float(fps), 1e-3) * sr)))
        w = aud["waveform"]
        if w.dim() == 2:
            w = w.unsqueeze(0)
        if w.shape[-1] < need:
            pad = torch.zeros((*w.shape[:-1], need - w.shape[-1]), dtype=w.dtype, device=w.device)
            w = torch.cat([w, pad], dim=-1)
        elif w.shape[-1] > need:
            w = w[..., -need:]
        aud = {"waveform": w.contiguous().clone(), "sample_rate": sr}
        frame = video[-1:].contiguous().clone()
    except Exception as exc:
        return _none(f"读上一镜成片失败: {exc}")

    if int(tail_frames or 0) <= 0:
        clip_desc = f"整段→{int(video.shape[0])}帧"
    else:
        clip_desc = f"末{n_keep}帧源→参考{int(video.shape[0])}帧@{int(video.shape[2])}x{int(video.shape[1])}"
    info = f"已取 {os.path.basename(path)} → 分镜{idx}｜末帧 + {clip_desc} + 音频"
    _log.info("[ShortDramaJSON] %s", info)
    return frame, video, aud, info


class ShortDramaJSONPrevLastFrame:
    """已并入「分镜选择/循环」。保留兼容旧工作流。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_json": ("STRING", {"forceInput": True, "multiline": True, "default": ""}),
                "shot_index": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "成片查找前缀": ("STRING", {"default": "ShortDrama_", "tooltip": "只填根前缀，如 ShortDrama_。用于按「前缀+分镜名」找成片；不是 SaveVideo 的保存文件名。"}),
            },
            "optional": {
                "merge_batch_start": ("INT", {"default": 0, "min": 0, "max": 64}),
                "启用": ("BOOLEAN", {"default": True}),
                "续接帧数": ("INT", {"default": 22, "min": 0, "max": 240, "step": 1, "tooltip": "截取上一镜最后 N 帧；0=不续接（本兼容节点）。主节点请用 CSV。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("上一镜末帧", "上一镜视频", "上一镜音频", "has_prev", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    DEPRECATED = True

    def run(
        self,
        prompt_json,
        shot_index=1,
        成片查找前缀="ShortDrama_",
        merge_batch_start=0,
        启用=True,
        续接帧数=22,
    ):
        n_tail = int(续接帧数 if 续接帧数 is not None else 22)
        frame, video, aud, info = _load_prev_shot_bundle(
            prompt_json,
            shot_index,
            enabled=bool(启用) and n_tail > 0,
            batch_first=int(merge_batch_start or 0),
            prefix_root=成片查找前缀,
            tail_frames=n_tail,
        )
        return frame, video, aud, (0 if video is None else 1), info


class ShortDramaJSONShotSelector:
    """选当前镜；可选输出上一镜末帧 / video / audio（批首镜不传）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_json": ("STRING", {"forceInput": True, "multiline": True, "default": ""}),
                "开始分镜": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1, "tooltip": "按 JSON 分镜排列顺序从第几条开始（含）。键名如「分镜4」是分镜名，不等于序号。"}),
                "结束分镜": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1, "tooltip": "跑到顺序第几条（含）。2:2 只跑第2条。"}),
                "分镜时长": ("STRING", {"default": "", "multiline": False}),
                "续接帧数": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "与分镜时长同格式，按镜填续接帧数。例 0,5,0,5,5,5：第3镜为0=不续接；非0=取上一镜末 N 帧。第1镜通常为0。",
                    },
                ),
                "成片查找前缀": (
                    "STRING",
                    {
                        "default": "ShortDrama_",
                        "tooltip": "只填根前缀 ShortDrama_。查找上一镜用「成片查找前缀+分镜名」。不要填/不要接「保存文件名」（那是 ShortDrama_分镜N）。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "FLOAT", "STRING", "IMAGE", "IMAGE", "AUDIO", "INT")
    RETURN_NAMES = (
        "shot_prompt",
        "shot_name",
        "shot_index",
        "shot_count",
        "duration",
        "保存文件名",
        "上一镜末帧",
        "上一镜视频",
        "上一镜音频",
        "续接帧数",
    )
    FUNCTION = "run"
    CATEGORY = CAT

    def run(
        self,
        prompt_json,
        开始分镜=None,
        结束分镜=None,
        分镜时长="",
        续接帧数="",
        成片查找前缀="ShortDrama_",
        **kw,
    ):
        data = _parse_json(prompt_json)
        start = 开始分镜 if 开始分镜 is not None else kw.get("分镜索引", 1)
        end = 结束分镜 if 结束分镜 is not None else kw.get("分镜结束", kw.get("分镜数量", start))
        cur, remain = clamp_run(len(list_shots(data)), start, end)
        prompt, name, _n, duration, save_name = build_shot_prompt(data, cur)
        # 兼容旧工作流 widgets：末帧续接(bool)+续接帧数(int) 错位进「续接帧数」「成片查找前缀」
        raw_tail = 续接帧数 if 续接帧数 is not None else kw.get("续接帧数", "")
        prefix = 成片查找前缀 if isinstance(成片查找前缀, str) else "ShortDrama_"
        if isinstance(raw_tail, bool) or (
            isinstance(raw_tail, str) and raw_tail.strip().lower() in ("true", "false")
        ):
            legacy_on = bool(raw_tail) if isinstance(raw_tail, bool) else raw_tail.strip().lower() == "true"
            if isinstance(成片查找前缀, (int, float)) and not isinstance(成片查找前缀, bool):
                n_tail = int(成片查找前缀) if legacy_on else 0
                prefix = "ShortDrama_"
            else:
                n_tail = 22 if legacy_on else 0
                prefix = str(成片查找前缀 or "ShortDrama_")
        else:
            n_tail = override_shot_tail_frames(raw_tail, cur, 0)
            prefix = str(prefix or "ShortDrama_")
        # 始终按本镜续接帧数输出；0=不续接（None）。是否接到 MiniMax / Motion Context 由工作流自行拉线
        frame, video, aud, _info = _load_prev_shot_bundle(
            prompt_json,
            cur,
            enabled=n_tail > 0,
            batch_first=1,
            prefix_root=prefix,
            tail_frames=n_tail,
        )
        # 无 <d> 对白的镜头：不传上一镜人声音频，避免参考轨把台词续进静默镜
        if aud is not None and not shot_has_dialogue(prompt):
            _log.info("[ShortDramaJSON] shot %s silent → drop prev dialogue audio", cur)
            aud = None
        return (
            prompt,
            name,
            cur,
            remain,
            float(override_shot_duration(分镜时长, cur, duration)),
            save_name,
            frame,
            video,
            aud,
            int(n_tail),
        )


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

    client_id = getattr(PromptServer.instance, "client_id", None) or str(uuid.uuid4())
    payload = {"prompt": new_prompt, "client_id": client_id}
    port = int(getattr(args, "port", 8188) or 8188)
    url = f"http://127.0.0.1:{port}/prompt"

    def _post():
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
            _log.info("[ShortDramaJSON] queued shot %s: %s", next_index, body[:200])
        except Exception as exc:
            _log.error("[ShortDramaJSON] queue shot %s failed: %s", next_index, exc)

    # 异步入队，避免在当前执行线程里同步 HTTP 自调用导致阻塞/超时
    threading.Thread(target=_post, name=f"ShortDramaJSON-queue-{next_index}", daemon=True).start()
    return f"已异步排队分镜{next_index}（剩余{remain}）"


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


# 与 H3 Motion Context Save/Load 同格式（video+audio safetensors），按分镜序号自动续接。
_H3_LATENT_PREFIX = "h3_context/clip"
try:
    from safetensors.torch import load_file as _st_load, save_file as _st_save
except ImportError:
    _st_load = _st_save = None


def _h3_streams_from_latent(latent):
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        return list(samples.unbind())
    if isinstance(samples, (list, tuple)):
        return list(samples)
    return [samples]


def _h3_latent_slot_path(filename_prefix: str, clip_index: int) -> str:
    root = folder_paths.get_output_directory()
    p = (filename_prefix or _H3_LATENT_PREFIX).strip().replace("\\", "/").strip("/")
    if "/" in p:
        sub, name = p.rsplit("/", 1)
        folder = os.path.join(root, *sub.split("/"))
    else:
        folder, name = root, p or "clip"
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "%s_%05d.safetensors" % (name, int(clip_index)))


class ShortDramaJSONH3SaveLatent:
    """保存本镜 Sampler 输出的 H3 AV latent，供下一镜 Motion Context 续接。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "shot_index": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "filename_prefix": ("STRING", {"default": _H3_LATENT_PREFIX}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("latent_path",)
    FUNCTION = "save"
    CATEGORY = CAT
    OUTPUT_NODE = True

    def save(self, latent, shot_index=1, filename_prefix=_H3_LATENT_PREFIX):
        if _st_save is None:
            raise RuntimeError("ShortDramaJSON: safetensors 不可用，无法保存 latent")
        parts = _h3_streams_from_latent(latent)
        if len(parts) < 2:
            raise ValueError("ShortDramaJSON: latent 无音频流，请接 H3 AV Sampler 输出")
        video = parts[0].detach().cpu().contiguous()
        audio = parts[1].detach().cpu().contiguous()
        idx = max(1, int(shot_index))
        path = _h3_latent_slot_path(filename_prefix, idx)
        _st_save({"video": video, "audio": audio}, path, metadata={"format": "h3_motion_context_av_v1"})
        _log.info("[ShortDramaJSON] saved H3 latent shot %s → %s", idx, path)
        return {"ui": {"text": [f"已保存分镜{idx} latent"]}, "result": (path,)}


class ShortDramaJSONH3LoadPrevLatent:
    """按 shot_index 加载上一镜 latent；第 1 镜或缺失时返回 None（Motion Context 通传）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_index": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "filename_prefix": ("STRING", {"default": _H3_LATENT_PREFIX}),
            },
            "optional": {
                "续接帧数": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 240,
                        "step": 1,
                        "tooltip": "接「分镜选择/循环」的续接帧数。0=本镜不续接（返回 None，Motion Context 通传）；-1/不接=只按 shot_index。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("context_latent",)
    FUNCTION = "load"
    CATEGORY = CAT

    @classmethod
    def IS_CHANGED(cls, shot_index, filename_prefix=_H3_LATENT_PREFIX, 续接帧数=-1):
        if 续接帧数 is not None and int(续接帧数) == 0:
            return "skip0"
        prev = int(shot_index) - 1
        if prev < 1:
            return float("NaN")
        path = _h3_latent_slot_path(filename_prefix, prev)
        if not os.path.isfile(path):
            return float("NaN")
        return "%s:%d" % (path, os.stat(path).st_mtime_ns)

    def load(self, shot_index, filename_prefix=_H3_LATENT_PREFIX, 续接帧数=-1):
        if _st_load is None:
            raise RuntimeError("ShortDramaJSON: safetensors 不可用，无法加载 latent")
        if 续接帧数 is not None and int(续接帧数) == 0:
            _log.info("[ShortDramaJSON] 续接帧数=0 → 不加载上一镜 latent（通传）")
            return (None,)
        prev = int(shot_index) - 1
        if prev < 1:
            _log.info("[ShortDramaJSON] shot %s 无上一镜 latent（通传）", shot_index)
            return (None,)
        path = _h3_latent_slot_path(filename_prefix, prev)
        if not os.path.isfile(path):
            _log.info("[ShortDramaJSON] 缺少上一镜 latent: %s（通传）", path)
            return (None,)
        data = _st_load(path)
        if "video" not in data or "audio" not in data:
            raise ValueError(f"ShortDramaJSON: 不是 H3 Motion Context latent: {path}")
        _log.info("[ShortDramaJSON] loaded prev latent shot %s ← %s", prev, path)
        return ({"samples": [data["video"], data["audio"]]},)


# Motion Context 原版 context_length 为 COMBO ["22","5","39","56"]，不能直接接 INT
_MOTION_CONTEXT_LENGTH_OPTS = ["22", "5", "39", "56"]


class ShortDramaJSONMotionContextLength:
    """续接帧数 INT → Motion Context context_length（COMBO），不改动 H3-Motion-Context 节点。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "续接帧数": (
                    "INT",
                    {
                        "default": 22,
                        "min": 0,
                        "max": 240,
                        "step": 1,
                        "tooltip": "接「分镜选择/循环」的续接帧数。对齐到 5/22/39/56；0 时仍输出 22（请同时把续接帧数接到「加载上一镜 Latent」以通传）。",
                    },
                ),
            }
        }

    RETURN_TYPES = (_MOTION_CONTEXT_LENGTH_OPTS,)
    RETURN_NAMES = ("context_length",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, 续接帧数=22):
        n = int(续接帧数 if 续接帧数 is not None else 0)
        if n <= 0:
            return ("22",)
        for g in (56, 39, 22, 5):
            if g <= n:
                return (str(g),)
        return ("5",)


class ShortDramaJSONConcatBatch:
    """未完成本批则自动 Queue 下一镜；最后一镜按分镜名拼接成片。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_index": ("INT", {"default": 1, "min": 1, "max": 64}),
                "shot_count": ("INT", {"default": 1, "min": 1, "max": 64}),
                "成片查找前缀": (
                    "STRING",
                    {
                        "default": "ShortDrama_",
                        "tooltip": "只填根前缀 ShortDrama_。拼接时按「成片查找前缀+分镜名」找各镜文件。不要接「保存文件名」。",
                    },
                ),
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

    def run(self, shot_index, shot_count, 成片查找前缀="ShortDrama_", trigger=None, merge_batch_start=0, prompt=None):
        _ = trigger
        idx, remain = int(shot_index), int(shot_count)
        batch_start = int(merge_batch_start or 0)
        root = str(成片查找前缀 or "ShortDrama_").strip() or "ShortDrama_"

        # 本批未结束：续跑下一镜，下游成片先挡住
        if remain > 1:
            if not isinstance(prompt, dict):
                return {"ui": {"text": ["无法获取 prompt，续跑失败"]}, "result": _blocked3()}
            if not prompt:
                return {"ui": {"text": ["prompt 为空，续跑失败"]}, "result": _blocked3()}
            start = batch_start if batch_start > 0 else idx
            try:
                msg = _queue_next_shot(prompt, idx + 1, remain - 1, start)
            except Exception as exc:
                msg = f"续跑失败: {exc}"
            _log.info("[ShortDramaJSON] concat batch remain=%s → %s", remain, msg)
            return {"ui": {"text": [msg]}, "result": _blocked3()}

        # 单镜批次：不拼
        if batch_start <= 0:
            return {"ui": {"text": [f"本批完成（单镜{idx}，无需拼接）"]}, "result": _blocked3()}

        # 最后一镜：按分镜名拼接
        out_dir = folder_paths.get_output_directory()
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
    "ShortDramaJSONPrevLastFrame": ShortDramaJSONPrevLastFrame,
    "ShortDramaJSONAutoNextShot": ShortDramaJSONAutoNextShot,
    "ShortDramaJSONConcatBatch": ShortDramaJSONConcatBatch,
    "ShortDramaJSONH3SaveLatent": ShortDramaJSONH3SaveLatent,
    "ShortDramaJSONH3LoadPrevLatent": ShortDramaJSONH3LoadPrevLatent,
    "ShortDramaJSONMotionContextLength": ShortDramaJSONMotionContextLength,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ShortDramaJSONSlotParser": "短剧JSON · 角色场景图绑定",
    "ShortDramaJSONPrevLastFrame": "短剧JSON · 上一镜成片（已并入，可删）",
    "ShortDramaJSONShotSelector": "短剧JSON · 分镜选择/循环",
    "ShortDramaJSONAutoNextShot": "短剧JSON · 续跑下一镜（已并入，可删）",
    "ShortDramaJSONConcatBatch": "短剧JSON · 续跑并拼接",
    "ShortDramaJSONH3SaveLatent": "短剧JSON · 保存本镜 Latent",
    "ShortDramaJSONH3LoadPrevLatent": "短剧JSON · 加载上一镜 Latent",
    "ShortDramaJSONMotionContextLength": "短剧JSON · 续接帧数→Motion Context",
}
