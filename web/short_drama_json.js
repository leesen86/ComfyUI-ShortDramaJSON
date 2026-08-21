import { app } from "../../../scripts/app.js";

const BINDER = "ShortDramaJSONSlotParser";
const SELECTOR = "ShortDramaJSONShotSelector";
const ROLE_KEYS = ["角色图片", "角色档案"];
const SCENE_KEYS = ["场景图片", "场景档案"];

function parsePrompt(raw) {
  try {
    const data = JSON.parse(String(raw || "").trim());
    return data && typeof data === "object" ? data : null;
  } catch {
    return null;
  }
}

function widgetText(node) {
  if (!node) return "";
  for (const name of ["value", "string", "text", "prompt", "prompt_json"]) {
    const w = node.widgets?.find((x) => x.name === name);
    if (w?.value != null && String(w.value).trim()) return String(w.value);
  }
  for (const w of node.widgets || []) {
    if (typeof w?.value === "string" && w.value.trim().startsWith("{")) return w.value;
  }
  for (const v of node.widgets_values || []) {
    if (typeof v === "string" && v.trim().startsWith("{")) return v;
  }
  return "";
}

function readUpstreamPrompt(node) {
  const input = node.inputs?.find((i) => i.name === "prompt_json");
  if (!input || input.link == null) return "";
  const link = app.graph.links?.[input.link];
  const upstream = link && app.graph.getNodeById(link.origin_id);
  if (!upstream) return "";
  let text = widgetText(upstream);
  if (text) return text;
  const upIn = upstream.inputs?.find((i) => i.type === "STRING" && i.link != null);
  if (!upIn) return "";
  const src = app.graph.getNodeById(app.graph.links?.[upIn.link]?.origin_id);
  return widgetText(src) || "";
}

function firstDict(data, keys) {
  for (const key of keys) {
    const block = data?.[key];
    if (block && typeof block === "object" && !Array.isArray(block) && Object.keys(block).length) return block;
  }
  return {};
}

function discoverSlots(data) {
  const slots = [];
  const seen = new Set();
  const push = (name, kind) => {
    const key = String(name || "").trim();
    if (!key || seen.has(key)) return;
    seen.add(key);
    slots.push({ name: key, kind, picture: slots.length + 1 });
  };
  for (const name of Object.keys(firstDict(data, ROLE_KEYS))) push(name, "角色");
  for (const name of Object.keys(firstDict(data, SCENE_KEYS))) push(name, "场景");
  return slots;
}

function neededSlotCount(slots) {
  if (!slots.length) return 0;
  return Math.max(slots.length, ...slots.map((s) => Number(s.picture) || 0));
}

function syncBinderSockets(node, count, reservePrev = false) {
  const n = Math.max(0, count | 0) + (reservePrev ? 1 : 0);
  const keepIn = new Map();
  const keepOut = new Map();
  // 保留所有已接线输入（含末帧续接等），避免加载时重排槽位后丢线/崩前端
  for (const inp of node.inputs || []) {
    if (inp?.name && inp.link != null) keepIn.set(inp.name, inp.link);
  }
  for (const out of node.outputs || []) {
    if (out?.name && out.links?.length) keepOut.set(out.name, [...out.links]);
  }

  const retarget = (linkIds, originSlot) => {
    if (!linkIds?.length) return;
    for (const lid of linkIds) {
      const link = app.graph?.links?.[lid];
      if (link) link.origin_slot = originSlot;
    }
  };
  const retargetIn = (linkId, targetSlot) => {
    if (linkId == null) return;
    const link = app.graph?.links?.[linkId];
    if (link) link.target_slot = targetSlot;
  };

  if (!node.inputs?.some((x) => x.name === "prompt_json")) node.addInput("prompt_json", "STRING");
  if (!node.inputs?.some((x) => x.name === "shot_prompt")) node.addInput("shot_prompt", "STRING");
  if (!node.inputs?.some((x) => x.name === "上一镜末帧")) node.addInput("上一镜末帧", "IMAGE", { shape: 7 });
  if (!node.outputs?.some((x) => x.name === "prompt_json")) node.addOutput("prompt_json", "STRING");
  if (!node.outputs?.some((x) => x.name === "shot_prompt")) node.addOutput("shot_prompt", "STRING");
  for (let i = 1; i <= n; i++) {
    const name = `Picture_${i}`;
    if (!node.inputs?.some((x) => x.name === name)) node.addInput(name, "IMAGE", { shape: 7 });
    if (!node.outputs?.some((x) => x.name === name)) node.addOutput(name, "IMAGE");
  }
  if (!node.outputs?.some((x) => x.name === "总时长")) node.addOutput("总时长", "FLOAT");
  if (!node.outputs?.some((x) => x.name === "上一镜末帧")) node.addOutput("上一镜末帧", "IMAGE");

  // 去掉 JSON 里没有的空槽（有连线的先留着，避免拆 MiniMax 线）
  if (node.inputs) {
    for (let i = node.inputs.length - 1; i >= 0; i--) {
      const m = String(node.inputs[i].name || "").match(/^Picture_(\d+)$/);
      if (m && Number(m[1]) > n && node.inputs[i].link == null) node.removeInput(i);
    }
  }
  if (node.outputs) {
    for (let i = node.outputs.length - 1; i >= 0; i--) {
      const m = String(node.outputs[i].name || "").match(/^Picture_(\d+)$/);
      if (m && Number(m[1]) > n && !node.outputs[i].links?.length) node.removeOutput(i);
    }
  }

  const byNameIn = Object.fromEntries((node.inputs || []).map((s) => [s.name, s]));
  const byNameOut = Object.fromEntries((node.outputs || []).map((s) => [s.name, s]));
  const keepExtraIn = (node.inputs || []).filter(
    (s) =>
      s.name !== "prompt_json" &&
      s.name !== "shot_prompt" &&
      s.name !== "上一镜末帧" &&
      !/^Picture_\d+$/.test(s.name)
  );
  // 左右同序：Picture_* → 上一镜末帧 → shot_prompt（总时长仅右侧）
  node.inputs = [
    "prompt_json",
    ...Array.from({ length: n }, (_, i) => `Picture_${i + 1}`),
    "上一镜末帧",
    "shot_prompt",
  ]
    .map((name) => byNameIn[name])
    .filter(Boolean)
    .concat(keepExtraIn);
  node.outputs = [
    "prompt_json",
    ...Array.from({ length: n }, (_, i) => `Picture_${i + 1}`),
    "上一镜末帧",
    "shot_prompt",
    "总时长",
  ]
    .map((name) => byNameOut[name])
    .filter(Boolean);

  // 末帧续接：末位 Picture 左右都藏起来，改走专用「上一镜末帧」；旧 MiniMax 线迁过去
  const prevPicName = reservePrev && n > 0 ? `Picture_${n}` : null;
  if (prevPicName && keepOut.has(prevPicName)) {
    const moved = keepOut.get(prevPicName) || [];
    const dest = keepOut.get("上一镜末帧") || [];
    keepOut.set("上一镜末帧", [...new Set([...dest, ...moved])]);
    keepOut.delete(prevPicName);
  }

  for (let i = 0; i < node.inputs.length; i++) {
    const inp = node.inputs[i];
    if (!inp) continue;
    const isPrevPic = prevPicName && inp.name === prevPicName;
    inp.hidden = Boolean(isPrevPic);
    if (/^Picture_\d+$/.test(inp.name || "") || inp.name === "上一镜末帧") inp.shape = 7;
    if (inp.name === "上一镜末帧") {
      inp.label = "上一镜末帧";
      inp.localized_name = "上一镜末帧";
    }
    const linkId = keepIn.has(inp.name) ? keepIn.get(inp.name) : inp.link;
    if (linkId != null) {
      inp.link = linkId;
      retargetIn(linkId, i);
    }
  }
  for (let i = 0; i < node.outputs.length; i++) {
    const out = node.outputs[i];
    if (!out) continue;
    const isPrevPic = prevPicName && out.name === prevPicName;
    out.hidden = Boolean(isPrevPic);
    if (out.name === "prompt_json") {
      out.label = "prompt_json";
      out.localized_name = "prompt_json";
    } else if (out.name === "shot_prompt") {
      out.label = "shot_prompt";
      out.localized_name = "shot_prompt";
    } else if (out.name === "总时长") {
      out.label = "总时长";
      out.localized_name = "总时长";
      out.widget = null;
    } else if (out.name === "上一镜末帧") {
      out.label = "上一镜末帧";
      out.localized_name = "上一镜末帧";
    }
    if (keepOut.has(out.name)) {
      out.links = keepOut.get(out.name);
      retarget(out.links, i);
    } else if (isPrevPic) {
      out.links = null;
    }
  }

  node.size[1] = Math.max(140, 120 + (n + 3) * 28);
}

function clipDuration(clip) {
  if (!clip || typeof clip !== "object") return 0;
  if ("时长" in clip) {
    const m = String(clip["时长"]).match(/(\d+(?:\.\d+)?)/);
    return m ? Number(m[1]) : typeof clip["时长"] === "number" ? clip["时长"] : 0;
  }
  const text = String(clip["时间"] || "");
  const tagged = text.match(/时长\s*[:：]\s*(\d+(?:\.\d+)?)/);
  if (tagged) return Number(tagged[1]);
  const range = text.match(/(\d+(?:\.\d+)?)\s*[-~～到至]\s*(\d+(?:\.\d+)?)/);
  if (range) return Number(range[2]);
  const ends = [...text.matchAll(/(\d+(?:\.\d+)?)\s*(?:s|S|秒)/g)].map((m) => Number(m[1]));
  return ends.length ? Math.max(...ends) : 0;
}

function listShotClips(data) {
  const seq = data?.["分镜序列"];
  if (!seq) return [];
  if (Array.isArray(seq)) {
    return seq.map((item, i) => ({
      name: `分镜${i + 1}`,
      clips: Array.isArray(item) ? item : item && typeof item === "object" ? [item] : [],
    }));
  }
  return Object.keys(seq)
    .filter((k) => seq[k] != null && !(Array.isArray(seq[k]) && !seq[k].length))
    .sort((a, b) => Number((a.match(/\d+/) || [999])[0]) - Number((b.match(/\d+/) || [999])[0]))
    .map((name) => {
      const raw = seq[name];
      return { name, clips: Array.isArray(raw) ? raw : raw && typeof raw === "object" ? [raw] : [] };
    });
}

function formatShotDurations(data) {
  return listShotClips(data)
    .map(({ clips }) => String(clips.reduce((s, c) => s + clipDuration(c), 0) || 10))
    .join(",");
}

/** 默认续接帧数 CSV：第 1 镜 0，其后 22（与分镜时长同格式） */
function formatShotTailFrames(data) {
  const n = listShotClips(data).length;
  if (!n) return "";
  return Array.from({ length: n }, (_, i) => (i === 0 ? "0" : "22")).join(",");
}

function totalDurationSeconds(data) {
  return listShotClips(data).reduce((sum, { clips }) => {
    const v = clips.reduce((s, c) => s + clipDuration(c), 0);
    return sum + (v > 0 ? v : 10);
  }, 0);
}

let _suppressWidgetRefresh = false;
function applyWidgetValue(node, widget, value) {
  if (!widget) return;
  try {
    widget.setValue?.(value);
  } catch {
    widget.value = value;
  }
  if (widget.value !== value) widget.value = value;
  if (widget.inputEl && "value" in widget.inputEl) widget.inputEl.value = value;
  if (widget.element && "value" in widget.element) widget.element.value = value;
  const idx = node.widgets?.indexOf(widget);
  if (idx >= 0 && Array.isArray(node.widgets_values) && idx < node.widgets_values.length) {
    node.widgets_values[idx] = value;
  }
  if (!_suppressWidgetRefresh) {
    try {
      widget.callback?.(value, app.canvas, node, null, null);
    } catch {
      /* ignore */
    }
  }
  node.setDirtyCanvas?.(true, true);
}

function findWidget(node, ...names) {
  return node.widgets?.find((x) => names.includes(x.name)) || null;
}

function refreshBinderLabels(node) {
  try {
    const raw = readUpstreamPrompt(node);
    const parsed = raw ? parsePrompt(raw) : null;
    const slots = parsed ? discoverSlots(parsed) : [];
    const contW = findWidget(node, "末帧续接");
    const reservePrev = contW ? Boolean(contW.value) : true;
    syncBinderSockets(node, neededSlotCount(slots), reservePrev);

    const totalSec = parsed ? totalDurationSeconds(parsed) : 0;
    const durWidget = findWidget(node, "总时长", "时长微调");
    if (durWidget && parsed) {
      _suppressWidgetRefresh = true;
      try {
        applyWidgetValue(node, durWidget, totalSec);
      } finally {
        _suppressWidgetRefresh = false;
      }
    }

    const status = findWidget(node, "_slot_status");
    if (status) {
      const base = !raw
        ? "请从提示词节点改 JSON"
        : !parsed
          ? "上游 JSON 无效"
          : slots.length
            ? `共${slots.length}槽（角色可不接图）｜` + slots.map((s) => `${s.kind}·${s.name}`).join(" | ")
            : "角色图片/场景图片为空（可不绑图）";
      status.value = reservePrev ? `${base}｜+末帧续接` : base;
    }
    const durOut = node.outputs?.find((o) => o.name === "总时长");
    if (durOut) {
      durOut.label = "总时长";
      durOut.localized_name = "总时长";
      durOut.widget = null;
    }
    const byPic = new Map(slots.map((s) => [s.picture, s]));
    const nSlots = neededSlotCount(slots);
    const prevPic = reservePrev ? nSlots + 1 : -1;
    for (const sock of [...(node.inputs || []), ...(node.outputs || [])]) {
      if (sock.name === "总时长" || sock.name === "prompt_json" || sock.name === "shot_prompt") continue;
      if (sock.name === "上一镜末帧") {
        sock.label = "上一镜末帧";
        sock.localized_name = "上一镜末帧";
        sock.hidden = false;
        continue;
      }
      const m = String(sock.name || "").match(/^Picture_(\d+)$/);
      if (!m) continue;
      const picNum = Number(m[1]);
      const meta = byPic.get(picNum);
      if (picNum === prevPic) {
        // 末位 Picture 左右都隐藏，与专用「上一镜末帧」口对齐
        sock.hidden = true;
        sock.label = sock.name;
        sock.localized_name = sock.name;
      } else {
        sock.hidden = false;
        sock.label = meta ? `${meta.kind}·${meta.name}` : sock.name;
        sock.localized_name = sock.label;
      }
    }
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
  } catch (err) {
    console.error("[ShortDramaJSON] refreshBinderLabels failed", err);
  }
}

function refreshSelectorStatus(node, { syncCount = false, syncDurations = false } = {}) {
  try {
    const raw = readUpstreamPrompt(node);
    const parsed = raw ? parsePrompt(raw) : null;
    const shots = parsed ? listShotClips(parsed).map((s) => s.name) : [];
    const durWidget = findWidget(node, "分镜时长");
    const tailWidget = findWidget(node, "续接帧数");
    const status = findWidget(node, "_shot_status");
    const idxWidget = findWidget(node, "开始分镜", "分镜索引", "shot_index");
    const endWidget = findWidget(node, "结束分镜", "分镜结束", "分镜数量");
    const csv = parsed ? formatShotDurations(parsed) : "";
    const tailCsv = parsed ? formatShotTailFrames(parsed) : "";

    _suppressWidgetRefresh = true;
    try {
      if (syncCount && endWidget && shots.length) applyWidgetValue(node, endWidget, shots.length);
      if (durWidget && ((syncDurations && csv) || (!String(durWidget.value || "").trim() && csv))) {
        applyWidgetValue(node, durWidget, csv);
      }
      if (
        tailWidget &&
        typeof tailWidget.value !== "boolean" &&
        ((syncDurations && tailCsv) || (!String(tailWidget.value ?? "").trim() && tailCsv))
      ) {
        applyWidgetValue(node, tailWidget, tailCsv);
      }
      const start = Number(idxWidget?.value ?? 1);
      const last = Number(endWidget?.value ?? start);
      if (idxWidget && start > last) applyWidgetValue(node, idxWidget, last);
      if (endWidget && Number(endWidget.value) < Number(idxWidget?.value ?? 1)) {
        applyWidgetValue(node, endWidget, Number(idxWidget.value));
      }
    } finally {
      _suppressWidgetRefresh = false;
    }

    if (status) {
      if (!shots.length) status.value = "未识别到分镜序列";
      else {
        const total = shots.length;
        const cur = Math.min(Math.max(1, Number(idxWidget?.value ?? 1)), total);
        const last = Math.min(Math.max(cur, Number(endWidget?.value ?? cur)), total);
        const batch = cur === last ? `只跑第 ${cur} 镜` : `本批 ${cur}–${last} 镜`;
        const tails = String(tailWidget?.value || tailCsv || "");
        const parts = tails.split(/[,，]/).map((x) => x.trim()).filter(Boolean);
        const curTail = parts[cur - 1];
        const curHint =
          curTail == null || curTail === ""
            ? ""
            : Number(curTail) <= 0
              ? "本镜不续接"
              : `本镜末${curTail}帧`;
        status.value = `共${total}镜｜${batch}｜时长 ${String(durWidget?.value || csv)}｜续接 ${tails || "—"}${curHint ? "｜" + curHint : ""}`;
      }
    }
    app.graph?.setDirtyCanvas?.(true, true);
  } catch (err) {
    console.error("[ShortDramaJSON] refreshSelectorStatus failed", err);
  }
}

function hookRefresh(nodeType, refresh) {
  for (const key of ["onNodeCreated", "onConnectionsChange", "onConfigure"]) {
    const orig = nodeType.prototype[key];
    nodeType.prototype[key] = function () {
      const r = orig?.apply(this, arguments);
      if (key === "onNodeCreated") refresh.setup?.(this);
      // 接线后稍晚再刷标签，避免抢在 link 写入前拆插座
      const delay = key === "onConnectionsChange" ? 30 : 50;
      setTimeout(() => refresh(this), delay);
      return r;
    };
  }
}

app.registerExtension({
  name: "short-drama.ShortDramaJSON",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === BINDER) {
      hookRefresh(nodeType, Object.assign(refreshBinderLabels, {
        setup(node) {
          if (!node.widgets?.find((w) => w.name === "_slot_status")) {
            const w = node.addWidget("text", "_slot_status", "等待提示词…", () => {});
            w.serialize = false;
          }
          if (!node.widgets?.find((w) => w.name === "刷新槽位标签")) {
            node.addWidget("button", "刷新槽位标签", null, () => refreshBinderLabels(node));
          }
          const cont = findWidget(node, "末帧续接");
          if (cont) {
            const prev = cont.callback;
            cont.callback = (...args) => {
              const out = prev?.apply(cont, args);
              refreshBinderLabels(node);
              return out;
            };
          }
          const dur = node.widgets?.find((w) => w.name === "总时长" || w.name === "时长微调");
          if (dur) {
            dur.name = "总时长";
            dur.label = "总时长";
            const idx = node.widgets.indexOf(dur);
            if (idx > 0) {
              node.widgets.splice(idx, 1);
              node.widgets.unshift(dur);
            }
          }
        },
      }));
    }

    if (nodeData.name === SELECTOR) {
      hookRefresh(nodeType, Object.assign(refreshSelectorStatus, {
        setup(node) {
          if (!node.widgets?.find((w) => w.name === "_shot_status")) {
            const w = node.addWidget("text", "_shot_status", "等待分镜…", () => {});
            w.serialize = false;
          }
          if (!node.widgets?.find((w) => w.name === "刷新分镜信息")) {
            node.addWidget("button", "刷新分镜信息", "刷新", () => {
              refreshSelectorStatus(node, { syncCount: true, syncDurations: true });
            });
          }
          for (const name of ["开始分镜", "结束分镜", "分镜索引", "分镜结束", "分镜时长", "续接帧数", "成片查找前缀"]) {
            const w = node.widgets?.find((x) => x.name === name);
            if (!w) continue;
            const prev = w.callback;
            w.callback = (...args) => {
              const out = prev?.apply(w, args);
              const startW = findWidget(node, "开始分镜", "分镜索引");
              const endW = findWidget(node, "结束分镜", "分镜结束", "分镜数量");
              if (startW && endW) {
                const start = Number(startW.value ?? 1);
                const last = Number(endW.value ?? 1);
                _suppressWidgetRefresh = true;
                try {
                  if ((name === "开始分镜" || name === "分镜索引") && start > last) {
                    applyWidgetValue(node, startW, last);
                  } else if ((name === "结束分镜" || name === "分镜结束") && last < start) {
                    applyWidgetValue(node, endW, start);
                  }
                } finally {
                  _suppressWidgetRefresh = false;
                }
              }
              refreshSelectorStatus(node);
              return out;
            };
          }
        },
      }));
    }

    if (nodeData.name === "ShortDramaJSONConcatBatch") {
      const orig = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = orig?.apply(this, arguments);
        const hide = this.widgets?.find((w) => w.name === "merge_batch_start");
        if (hide) {
          hide.hidden = true;
          hide.computeSize = () => [0, -4];
        }
        return r;
      };
    }
  },
});
