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
    """从 shot_prompt 里收集 <Picture N>；空集合表示本镜不需要任何参考图。"""
    return {int(m) for m in _PIC_REF.findall(shot_prompt or "")}


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


def _clip_blob(clips: list) -> str:
    parts: list[str] = []
    for c in clips:
        if not isinstance(c, dict):
            continue
        for key in ("场景", "镜头", "画面", "声音", "对白"):
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
        for key in ("场景", "镜头", "画面", "声音"):
            if isinstance(nc.get(key), str):
                nc[key] = _expand_ref_tags(nc[key], pics)
        out.append(nc)
    return out


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
    out.update({"当前分镜": name, "分镜序号": idx, "分镜总数": n, "分镜序列": clips})
    return json.dumps(out, ensure_ascii=False, indent=2), name, n, shot_duration(clips), f"ShortDrama_{name}"


def _latest_mp4(output_dir: str, prefix: str) -> str | None:
    files = [
        f
        for pat in (os.path.join(output_dir, f"{prefix}*.mp4"), os.path.join(output_dir, "**", f"{prefix}*.mp4"))
        for f in glob.glob(pat, recursive=True)
        if os.path.isfile(f) and "merged" not in os.path.basename(f).lower()
    ]
    return max(files, key=os.path.getmtime) if files else None


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
                        "tooltip": "接「分镜选择/循环」的 shot_prompt；只放行本镜引用的 <Picture N>，其余参考图不送入下游。",
                    },
                ),
            },
        }

    RETURN_TYPES = _FLEX_RETURNS
    RETURN_NAMES = ("prompt_json", "总时长")
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
        data = _parse_json(prompt_json)
        n = len(discover_slots(data)) or _slot_count_from_names(kwargs)
        _apply_binder_schema(n)
        gate = None
        if isinstance(shot_prompt, str) and shot_prompt.strip():
            gate = used_picture_indices(shot_prompt)
        pics = []
        for i in range(1, n + 1):
            img = kwargs.get(f"Picture_{i}")
            if img is None:
                img = _empty_image()
            # 接了 shot_prompt 时：未引用槽位送空图，避免角色参考图泄漏进 MiniMax
            if gate is not None and i not in gate:
                img = _empty_image()
            pics.append(img)
        dur = 总时长 if 总时长 is not None else kwargs.get("时长微调", 10.0)
        return (prompt_json or "",) + tuple(pics) + (float(dur),)


class ShortDramaJSONShotSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_json": ("STRING", {"forceInput": True, "multiline": True, "default": ""}),
                "开始分镜": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1, "tooltip": "从第几镜开始（含）。不能大于结束分镜。"}),
                "结束分镜": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1, "tooltip": "跑到第几镜（含）。2:2 只跑第2镜。"}),
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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_index": ("INT", {"default": 1, "min": 1, "max": 64}),
                "shot_count": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {"trigger": ("VIDEO",)},
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = CAT
    OUTPUT_NODE = True

    def run(self, shot_index, shot_count, trigger=None, prompt=None, unique_id=None):
        _ = (trigger, unique_id)
        idx, remain = int(shot_index), int(shot_count)
        if remain <= 1:
            return {"ui": {"text": [f"本批完成（索引{idx}）"]}}
        if not isinstance(prompt, dict) or not prompt:
            return {"ui": {"text": ["无法获取 prompt，续跑失败"]}}

        start = idx
        for node in prompt.values():
            if isinstance(node, dict) and node.get("class_type") == "ShortDramaJSONConcatBatch":
                prev = int(node.get("inputs", {}).get("merge_batch_start") or 0)
                if prev > 0:
                    start = prev
                break
        try:
            msg = self._queue(prompt, idx + 1, remain - 1, start)
        except Exception as exc:
            msg = f"续跑失败: {exc}"
        return {"ui": {"text": [msg]}}

    @staticmethod
    def _queue(prompt_dict: dict, next_index: int, remain: int, batch_start: int) -> str:
        new_prompt = copy.deepcopy(prompt_dict)
        found = False
        for node in new_prompt.values():
            if not isinstance(node, dict):
                continue
            inputs = node.setdefault("inputs", {})
            if node.get("class_type") == "ShortDramaJSONShotSelector":
                inputs["开始分镜"] = next_index
                inputs["结束分镜"] = next_index + remain - 1
                for old in ("分镜索引", "分镜结束", "分镜数量"):
                    inputs.pop(old, None)
                found = True
            elif node.get("class_type") == "ShortDramaJSONConcatBatch":
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


class ShortDramaJSONConcatBatch:
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
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "info")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, shot_index, shot_count, prefix_root="ShortDrama_", trigger=None, merge_batch_start=0):
        _ = trigger
        remain, end, start = int(shot_count), int(shot_index), int(merge_batch_start or 0)
        if start <= 0 or remain > 1:
            return (ExecutionBlocker(None), ExecutionBlocker(None), ExecutionBlocker(None))

        out_dir = folder_paths.get_output_directory()
        root = str(prefix_root or "ShortDrama_")
        try:
            paths = []
            for i in range(start, end + 1):
                path = _latest_mp4(out_dir, f"{root}分镜{i}")
                if not path:
                    raise FileNotFoundError(f"缺少分镜视频: {root}分镜{i}*.mp4")
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
            info = f"已拼接分镜{start}-{end} 共{len(paths)}段｜" + " + ".join(os.path.basename(p) for p in paths)
            return torch.cat(resized, dim=0), _concat_audio(aud_parts), info
        except Exception as exc:
            return (ExecutionBlocker(f"拼接失败: {exc}"), ExecutionBlocker(None), ExecutionBlocker(str(exc)))


NODE_CLASS_MAPPINGS = {
    "ShortDramaJSONSlotParser": ShortDramaJSONSlotParser,
    "ShortDramaJSONShotSelector": ShortDramaJSONShotSelector,
    "ShortDramaJSONAutoNextShot": ShortDramaJSONAutoNextShot,
    "ShortDramaJSONConcatBatch": ShortDramaJSONConcatBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShortDramaJSONSlotParser": "短剧JSON · 角色场景图绑定",
    "ShortDramaJSONShotSelector": "短剧JSON · 分镜选择/循环",
    "ShortDramaJSONAutoNextShot": "短剧JSON · 续跑下一镜",
    "ShortDramaJSONConcatBatch": "短剧JSON · 本批全部分镜拼接",
}
