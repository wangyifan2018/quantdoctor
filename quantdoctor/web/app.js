const byId = (id) => document.getElementById(id);
const labels = {
  missing: "缺少结果", failed: "运行失败", transcribed: "有转写", cleanup_empty: "清理后为空",
  first_step_eos: "首步 EOS", eos_unlocated: "EOS 位置待确认", other_stop: "其他停止条件",
  empty_unknown: "空转写 · 证据不足", baseline_unusable: "基线不可用", unpaired: "配对条件不一致",
};
const sources = { fixture: "模拟数据", torch: "PyTorch 结果", target: "设备结果" };
const transitions = {
  recovered_text: "恢复文本", new_empty: "新增空转写", still_empty: "仍为空",
  stable_text: "两次均有文本", repair_missing: "复验缺结果", repair_failed: "复验运行失败",
  original_unavailable: "原始结果不可用",
};
const percentage = (value) => (value * 100).toFixed(1) + "%";
const tensorType = row => row ? row.dtype + " [" + row.shape.join(", ") + "]" : "缺失";
const alignmentLabels = {
  aligned: "已对齐",
  missing_step: "缺少另一侧的生成步",
  run_mismatch: "缺少匹配的运行条件或请求摘要",
  context_mismatch: "输入或生成前缀不同",
  score_stage_mismatch: "分数采集阶段不同",
  stop_set_mismatch: "有效停止 ID 不同",
  missing_scores: "未采集参考或全部停止 token 的分数，或基线已停止",
};

function text(id, value) {
  byId(id).textContent = value;
}

function result(side, output, run) {
  text(side + "-precision", ["backbone", "audio", "head", "kv"].map((key) =>
    key + " " + run.precision[key]).join(" · "));
  if (!output) {
    text(side + "-text", "未采集");
    text(side + "-stop", "");
    return;
  }
  const finalText = output.text.trim() ? output.text : "∅";
  text(side + "-text", output.raw_text == null ? finalText : "最终  " + finalText + "\n清理前  " + output.raw_text);
  text(side + "-stop", [
    "停止 " + (output.finish_reason ?? "未采集"),
    "生成数 " + (output.generated_count ?? "未采集"),
    "首 ID " + (output.first_token_id ?? "未采集"),
  ].join(" · "));
}

function showReplayAudit(audit) {
  if (audit.status === "not_supplied") {
    text("artifact-status", "未提供运行关联的 checkpoint，组件范围依据工单与运行声明。");
    return;
  }
  byId("replay-artifacts").open = true;
  const groups = audit.changed_components.join("、") || "无";
  text("artifact-status", (audit.matches_expectation ? "实际张量范围符合工单" : "实际张量范围与工单不一致，本次样本结果仅供观察")
    + " · 变更组件 " + groups + " · " + audit.changed_tensors.length + " 个张量");
  text("artifact-config", "两份 checkpoint 摘要匹配运行记录。"
    + (audit.changed_configs.length ? "需复核配置差异：" : "配置变化：")
    + (audit.changed_configs.join("、") || "无")
    + (audit.tie_declaration_changed ? " · 绑定声明变化" : "")
    + "。部署验收仍需 engine 和板端证据。");
  const list = byId("artifact-tensors");
  for (const row of audit.changed_tensors.slice(0, 12)) {
    const line = document.createElement("div");
    line.className = "tensor-row";
    const name = document.createElement("code");
    name.textContent = row.name;
    const value = document.createElement("span");
    value.textContent = tensorType(row.original) + " → " + tensorType(row.candidate);
    line.append(name, value);
    list.append(line);
  }
  if (audit.changed_tensors.length > 12) {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = "此处显示前 12 条。完整差异见 CLI 复验 JSON。";
    list.append(note);
  }
}

function showReplay(replay, original) {
  const [field, value] = Object.entries(replay.change)[0];
  const component = field.split(".")[1];
  text("replay-change", component + " " + original.precision[component] + " → " + value
    + " · 工单样本 " + replay.work_order.sample_id);
  for (const [code, count] of Object.entries(replay.counts)) {
    if (!count && !["recovered_text", "new_empty"].includes(code)) continue;
    const badge = document.createElement("span");
    badge.textContent = transitions[code] + " " + count;
    byId("replay-counts").append(badge);
  }
  const metrics = Object.entries(replay.metrics).filter(([, values]) => values.paired);
  text("replay-metrics", metrics.length ? metrics.map(([name, values]) =>
    name.toUpperCase() + " " + [values.baseline, values.original, values.repaired].map(percentage).join(" → ")
      + "（三组共 " + values.paired + " 条）").join(" · ") : "无可评分的三组共同样本");

  const costs = replay.costs ? Object.entries(replay.costs.measurements) : [];
  const costLabels = { rtf_p95: "RTF P95", peak_memory_bytes: "峰值内存", artifact_bytes: "产物大小" };
  text("replay-costs", costs.length ? costs.map(([key, value]) => {
    const format = key.endsWith("_bytes")
      ? (number) => (number / 1048576).toFixed(1) + " MiB"
      : (number) => number.toFixed(3);
    return costLabels[key] + " " + format(value.original) + " → " + format(value.repaired);
  }).join(" · ") : "未导入可配对的性能数据");
  showReplayAudit(replay.checkpoint_audit);
  byId("replay-overview").hidden = false;
  byId("repaired-column").hidden = false;
  byId("replay-filter").hidden = false;
  byId("transcripts").classList.add("with-repair");
}

function showDecodeEvidence(original, repaired = []) {
  const picker = byId("decode-step");
  const chart = byId("decode-chart");
  const source = new Map(original.map(row => [row.step, row]));
  const repair = new Map(repaired.map(row => [row.step, row]));
  const steps = [...new Set([...source.keys(), ...repair.keys()])].sort((a, b) => a - b);
  picker.replaceChildren();
  chart.replaceChildren();
  byId("step-control").hidden = steps.length < 2;
  if (!steps.length) {
    text("decode-note", "未采集 token 分数。可以导入与运行及请求匹配的解码证据。");
    picker.onchange = null;
    return;
  }
  for (const step of steps) picker.add(new Option(String(step), String(step)));
  picker.value = String(original.find(row => row.selected_changed)?.step ?? steps[0]);

  function render() {
    chart.replaceChildren();
    const step = Number(picker.value);
    const pair = source.get(step);
    const fixed = repair.get(step);
    const anchor = pair?.status === "aligned" ? pair : fixed;
    const sameStage = pair?.status !== "aligned" || pair.score_stage === fixed?.score_stage;
    const repairedMargin = sameStage ? fixed?.candidate_margin : null;
    const stage = anchor?.score_stage === "raw_logits" ? "原始 logits" : "规则处理后 logits";
    text("decode-note", anchor?.status === "aligned"
      ? "生成步 " + step + " · " + stage + " · 分差 = 基线选中的 token #" + anchor.reference_id
        + " 分数 − 最高有效停止分数。中线为零。"
      : "生成步 " + step + " · 分差不可比。下方保留已观察到的 token 选择。");
    const rows = [
      ["基线", anchor?.baseline_margin, pair?.baseline_selected ?? fixed?.baseline_selected,
        pair?.baseline_top1 ?? fixed?.baseline_top1, anchor?.status],
      ["原候选", pair?.candidate_margin, pair?.candidate_selected, pair?.candidate_top1, pair?.status],
    ];
    if (repaired.length) rows.push(["修复后", repairedMargin, fixed?.candidate_selected, fixed?.candidate_top1,
      sameStage ? fixed?.status : "score_stage_mismatch"]);
    const extent = Math.max(1, ...rows.map(row => Math.abs(row[1] ?? 0)));
    for (const [label, margin, selected, top1, status] of rows) {
      const container = document.createElement("div");
      container.className = "decode-row";
      const name = document.createElement("strong");
      name.textContent = label;
      const track = document.createElement("div");
      track.className = "margin-track";
      track.setAttribute("aria-hidden", "true");
      const value = document.createElement("span");
      value.className = "margin-value";
      value.textContent = margin == null ? "—" : (margin > 0 ? "+" : "") + margin.toFixed(3);
      if (margin != null) {
        const bar = document.createElement("span");
        const width = Math.abs(margin) / extent * 50;
        bar.style.width = width + "%";
        bar.style.left = (margin < 0 ? 50 - width : 50) + "%";
        if (margin < 0) bar.className = "negative";
        track.append(bar);
      }
      const detail = document.createElement("p");
      detail.className = "muted";
      detail.textContent = (selected == null ? "选择未采集" : "选择 #" + selected)
        + (top1 == null ? "" : " · Top-1 #" + top1)
        + (status && status !== "aligned" ? " · " + alignmentLabels[status] : "");
      container.append(name, track, value, detail);
      chart.append(container);
    }
    if (pair?.status === "aligned" && pair.baseline_margin > 0 && pair.candidate_margin < 0
        && pair.candidate_stopped) {
      const detail = document.createElement("p");
      detail.className = "transition";
      detail.textContent = "结束符超过基线参考 token，且原候选选中了停止 ID。可结合单变量复验判断这项变化是否与空转写有关。";
      chart.append(detail);
    }
  }

  picker.onchange = render;
  render();
}

function showCheckpoint(data) {
  const audit = data.kind === "checkpoint_audit";
  text("source", data.payload_hashed ? "本地张量内容已读取" : "本地 checkpoint 头部");
  document.querySelector("h1").textContent = audit ? "模型产物差异审计" : "模型权重检查";
  text("model", (data.model_type ?? "未声明 model_type") + " · 声明量化 "
    + (data.declared_quantization ?? "未声明") + " · 权重绑定 "
    + (data.tied_word_embeddings == null ? "未声明" : data.tied_word_embeddings ? "开启" : "关闭"));
  const changed = (data.changed_components ?? []).join("、") || "无";
  text("checkpoint-status", audit
    ? "有张量变化的组件：" + changed + (data.tie_declaration_changed ? " · 权重绑定声明变化" : "")
      + (data.expected_component ? " · 仅 " + data.expected_component + " 发生变化：" + (data.matches_expectation ? "符合" : "不符合") : "")
    : "下方为 checkpoint 中实际存储的 dtype 与字节数。运行时精度需核对部署证据。");
  const names = [...new Set([...Object.keys(data.original_components ?? {}), ...Object.keys(data.components)])];
  const maxBytes = Math.max(1, ...names.flatMap(name => [
    data.original_components?.[name]?.bytes ?? 0, data.components[name]?.bytes ?? 0,
  ]));
  const formatSize = value => value >= 1048576 ? (value / 1048576).toFixed(1) + " MiB" : value + " B";
  for (const name of names) {
    const group = document.createElement("section");
    group.className = "checkpoint-group";
    const heading = document.createElement("h3");
    heading.textContent = name;
    group.append(heading);
    if (audit) {
      const original = document.createElement("p");
      original.className = "muted";
      original.textContent = "原始 " + formatSize(data.original_components[name]?.bytes ?? 0);
      group.append(original);
    }
    const component = data.components[name];
    const size = document.createElement("p");
    size.textContent = formatSize(component?.bytes ?? 0) + " · " + (component?.scale_count ?? 0) + " 个 scale";
    const meter = document.createElement("meter");
    meter.min = 0;
    meter.max = maxBytes;
    meter.value = component?.bytes ?? 0;
    meter.setAttribute("aria-label", name + " 的存储字节数");
    const breakdown = document.createElement("p");
    breakdown.className = "muted";
    breakdown.textContent = Object.entries(component?.dtype_bytes ?? {}).map(([dtype, bytes]) =>
      dtype + " " + formatSize(bytes)).join(" · ");
    group.append(size, meter, breakdown);
    byId("component-plot").append(group);
  }
  const picker = byId("component-picker");
  picker.add(new Option("全部组件", ""));
  for (const name of names) picker.add(new Option(name, name));
  text("tensor-heading", audit ? "实际变更张量" : "张量清单");
  const records = audit ? data.changed_tensors : data.tensors;
  function render() {
    const rows = records.filter(row => !picker.value || row.component === picker.value);
    const items = byId("tensors");
    items.replaceChildren();
    text("tensor-count", rows.length > 100 ? "共 " + rows.length + " 条，显示前 100 条。完整结果可通过 CLI 导出。"
      : rows.length + " 条记录");
    for (const row of rows.slice(0, 100)) {
      const line = document.createElement("div");
      line.className = "tensor-row";
      const name = document.createElement("code");
      name.textContent = row.name;
      const info = document.createElement("span");
      info.textContent = audit ? tensorType(row.original) + " → " + tensorType(row.candidate) : tensorType(row);
      line.append(name, info);
      items.append(line);
    }
  }
  picker.addEventListener("change", render);
  render();
  byId("checkpoint").hidden = false;
}

function showCandidatePerformance(data) {
  const performance = data.performance;
  const fields = {
    cer: ["CER", percentage],
    wer: ["WER", percentage],
    unavailable: ["失败或缺失", String],
    empty: ["空转写", String],
    rtf_p95: ["RTF P95", value => value.toFixed(3)],
    peak_memory_bytes: ["峰值内存", value => (value / 1048576).toFixed(1) + " MiB"],
    artifact_bytes: ["产物大小", value => (value / 1048576).toFixed(1) + " MiB"],
  };
  const columns = [...performance.objectives, "artifact_bytes"];
  const notes = {
    compared: "仅在 " + performance.complete_candidates + " 个记录完整的候选间比较点估计。另一个方案各项目标均不差且至少一项更好时，标为被支配。",
    insufficient_complete_costs: "至少需要两份同时包含 RTF P95 和峰值内存的记录才判断前沿。",
    insufficient_common_quality: "共同质量数据不足，暂不判断前沿。",
  };
  text("performance-note", "协议 " + performance.protocol_id + " · " + notes[performance.status]
    + " 产物大小单列。");
  const table = byId("performance-table");
  const headings = table.createTHead().insertRow();
  for (const label of ["候选", ...columns.map(key => fields[key][0]), "前沿状态"]) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    headings.append(cell);
  }
  const tbody = table.createTBody();
  for (const candidate of data.candidates) {
    const entry = performance.runs[candidate.id];
    const row = tbody.insertRow();
    const heading = document.createElement("th");
    heading.scope = "row";
    heading.textContent = candidate.id;
    row.append(heading);
    for (const key of columns) {
      const value = entry.values[key];
      row.insertCell().textContent = value == null ? "未测或不可比" : fields[key][1](value);
    }
    const state = row.insertCell();
    if (entry.frontier === "non_dominated") {
      state.textContent = "非支配";
      state.className = "performance-frontier";
    } else if (entry.frontier === "dominated") {
      state.textContent = "被 " + entry.dominated_by.join("、") + " 支配";
    } else {
      state.textContent = entry.missing_costs.length ? "缺性能记录" : "待评估";
    }
  }
  byId("candidate-performance").hidden = false;
}

function showCandidates(data) {
  const create = (tag, value, className) => {
    const element = document.createElement(tag);
    if (value != null) element.textContent = value;
    if (className) element.className = className;
    return element;
  };
  const pp = value => (value > 0 ? "+" : "") + (value * 100).toFixed(2);
  text("source", sources[data.baseline.source]);
  document.querySelector("h1").textContent = "候选方案对照";
  text("model", data.model + " · 基线 " + data.baseline.id + " · " + data.samples.length + " 条评测记录");
  for (const run of data.candidates) {
    const counts = run.outcomes;
    const precision = Object.entries(run.precision).map(([component, value]) => component + " " + value).join(" · ");
    const group = create("section", null, "checkpoint-group");
    group.append(create("h3", run.id), create("p", precision, "muted"),
      create("p", "有文本 " + counts.text + " · 空 " + counts.empty + " · 失败 " + counts.failed + " · 缺失 " + counts.missing));
    byId("candidate-outcomes").append(group);
  }
  for (const [metric, summary] of Object.entries(data.metrics)) {
    const card = create("section", null, "replay-overview");
    card.append(create("h2", metric.toUpperCase() + " · 共同样本"),
      create("p", summary.common_samples + " / " + summary.eligible_samples + " 条有标注记录 · "
        + summary.reference_units + " 个参考单位" + (summary.baseline_error_rate == null ? ""
          : " · 基线 " + percentage(summary.baseline_error_rate)), "muted"));
    if (!summary.common_samples) {
      card.append(create("p", "没有全部候选与基线均可评分的样本。"));
      byId("candidate-metrics").append(card);
      continue;
    }
    const extent = Math.max(0.001, ...summary.candidates.flatMap(row =>
      [row.delta_vs_reference, ...(row.interval_95 ?? [])].map(Math.abs)));
    const x = value => (50 + value / extent * 48) + "%";
    card.append(create("p", "参照 " + data.reference_run_id + " · 误差率差值（百分点），中线为零，左侧表示误差更低。", "muted"));
    for (const row of summary.candidates) {
      const line = create("div", null, "interval-row");
      const track = create("div", null, "margin-track interval-track");
      track.setAttribute("aria-hidden", "true");
      if (row.interval_95) {
        const band = create("span", null, "interval-band");
        band.style.left = x(row.interval_95[0]);
        band.style.width = (row.interval_95[1] - row.interval_95[0]) / extent * 48 + "%";
        track.append(band);
      }
      const point = create("span", null, "interval-point");
      point.style.left = x(row.delta_vs_reference);
      track.append(point);
      const interval = row.interval_95 ? " · 95% [" + row.interval_95.map(pp).join(", ") + "]"
        : " · 样本不足，未计算区间";
      line.append(create("strong", row.run_id), track,
        create("div", percentage(row.error_rate) + " · Δ " + pp(row.delta_vs_reference) + interval));
      card.append(line);
    }
    byId("candidate-metrics").append(card);
  }
  if (data.performance) showCandidatePerformance(data);
  const picker = byId("candidate-sample");
  for (const sample of data.samples) picker.add(new Option(sample.id));
  const rows = new Map(data.samples.map(sample => [sample.id, sample]));
  function renderSample() {
    const sample = rows.get(picker.value);
    text("candidate-reference", "参考转写 " + (sample.reference ?? "未提供"));
    text("candidate-cohort", sample.common_scoring ? "计入共同评分集合" : "未计入共同评分集合，保留原始结果");
    const list = byId("candidate-sample-results");
    list.replaceChildren();
    const outputs = [
      [data.baseline.id, sample.baseline_status, sample.baseline_text],
      ...data.candidates.map(run => [run.id, sample.candidates[run.id].status, sample.candidates[run.id].text]),
    ];
    for (const [id, status, transcript] of outputs) {
      const row = create("div", null, "tensor-row");
      row.append(create("strong", id), create("span", (labels[status] ?? status) + " · " + (transcript?.trim() || "∅")));
      list.append(row);
    }
  }
  picker.addEventListener("change", renderSample);
  renderSample();
  byId("candidate-board").hidden = false;
}

function showPreflight(data) {
  document.querySelector("h1").textContent = "目标设备预检";
  text("source", sources[data.source]);
  text("model", data.model + " · " + data.candidate_id + " · " + data.profile.id
    + " (" + data.profile.platform + ", " + data.profile.edge_commit.slice(0, 8) + ")");
  for (const [key, result] of Object.entries(data.checks)) {
    const card = document.createElement("section");
    card.className = "checkpoint-group";
    const title = document.createElement("h2");
    title.textContent = key === "candidate" ? "原候选" : "工单方案";
    const state = document.createElement("p");
    state.textContent = result.state === "blocked" ? "存在已知约束" : "待设备验证";
    const precision = document.createElement("p");
    precision.className = "muted";
    precision.textContent = Object.entries(result.precision).map(([part, dtype]) => part + " " + dtype).join(" · ");
    card.append(title, state, precision);
    const issues = document.createElement("ul");
    for (const issue of result.issues) {
      const row = document.createElement("li");
      row.textContent = (issue.severity === "block" ? "阻止部署预检 · " : "需复核 · ") + issue.detail
        + (issue.components ? " " + issue.components.join("、") : "");
      if (issue.source) {
        const link = document.createElement("a");
        link.href = issue.source;
        link.textContent = "查看依据";
        row.append(" ", link);
      }
      issues.append(row);
    }
    if (issues.children.length) card.append(issues);
    byId("target-checks").append(card);
  }
  byId("target-preflight").hidden = false;
}

function setup(data) {
  if (data.kind === "target_preflight") {
    showPreflight(data);
    return;
  }
  if (data.kind === "candidate_comparison") {
    showCandidates(data);
    return;
  }
  if (data.kind === "checkpoint" || data.kind === "checkpoint_audit") {
    showCheckpoint(data);
    return;
  }
  const { baseline, candidate, replay } = data;
  const replayRows = new Map(Object.entries(replay?.samples ?? {}));
  text("source", baseline.source === candidate.source
    ? sources[baseline.source]
    : sources[baseline.source] + " / " + sources[candidate.source]);
  text("model", data.model + " · " + baseline.id + " → " + candidate.id);
  const metrics = byId("metrics");
  if (!data.comparable) {
    metrics.textContent = "比较条件不一致，配对指标和自动修复建议已停用。";
  } else {
    const empty = document.createElement("p");
    const counts = (side) => data.outcomes[side].empty + "/" + data.outcomes[side].completed;
    empty.textContent = "空转写 / 已完成  " + counts("baseline") + " → " + counts("candidate");
    metrics.append(empty);
    for (const [key, metric] of Object.entries(data.metrics)) {
      const row = document.createElement("p");
      row.textContent = metric.paired
        ? key.toUpperCase() + "  " + percentage(metric.baseline) + " → " + percentage(metric.candidate) + " · " + metric.paired + " 个配对"
        : key.toUpperCase() + "  无可评分配对";
      metrics.append(row);
    }
  }
  if (replay) showReplay(replay, candidate);

  const list = byId("samples");
  const filter = byId("sample-filter");
  let selected = replay?.work_order.sample_id;
  function renderSample(sample, button) {
    selected = sample.id;
    byId("detail").hidden = false;
    for (const entry of list.querySelectorAll("button")) entry.setAttribute("aria-pressed", entry === button);
    text("sample-id", sample.id);
    text("reference", sample.reference == null ? "参考文本未提供" : "参考 " + sample.reference);
    result("baseline", sample.baseline, baseline);
    result("candidate", sample.candidate, candidate);
    showDecodeEvidence(sample.decode_evidence, replayRows.get(sample.id)?.decode_evidence);
    if (replay) {
      const row = replayRows.get(sample.id);
      result("repaired", row.output, replay.repair_run);
      const edits = row.edits ? " · 编辑数 " + row.edits.original + " → " + row.edits.repaired : "";
      text("sample-transition", transitions[row.transition] + edits);
      byId("sample-transition").hidden = false;
    }
    const finding = sample.finding;
    text("finding-title", labels[finding.code]);
    text("evidence", finding.evidence);
    text("attribution", finding.attribution);
    const panel = byId("intervention");
    panel.hidden = !finding.action;
    panel.open = false;
    if (finding.action) {
      const reviewed = replay?.work_order.sample_id === sample.id;
      text("action", finding.action.title + (reviewed ? " · 已有复验结果" : " · 待复验"));
      text("change", JSON.stringify(finding.action.change, null, 2));
      text("controls", finding.action.controls);
      text("verify", finding.action.verify);
    }
  }

  function populate() {
    list.replaceChildren();
    const rows = data.samples.filter((sample) => {
      if (filter.value === "all") return true;
      if (filter.value === "issues") return sample.finding.code !== "transcribed";
      const row = replayRows.get(sample.id);
      return row && (row.transition !== "stable_text" || row.edits?.original !== row.edits?.repaired);
    });
    let activeButton;
    for (const sample of rows) {
      const button = document.createElement("button");
      button.type = "button";
      const transition = replayRows.get(sample.id)?.transition;
      button.textContent = sample.id + " · " + (transition ? transitions[transition] : labels[sample.finding.code]);
      button.addEventListener("click", () => renderSample(sample, button));
      list.append(button);
      if (sample.id === selected) activeButton = button;
    }
    activeButton ??= list.querySelector("button");
    if (activeButton) {
      activeButton.click();
    } else {
      list.textContent = "没有匹配的样本";
      byId("detail").hidden = true;
    }
  }

  filter.addEventListener("change", populate);
  populate();
  byId("workbench").hidden = false;
}

fetch("/report.json").then((response) => {
  if (!response.ok) throw new Error("HTTP " + response.status);
  return response.json();
}).then(setup).catch((error) => {
  text("error", "无法加载报告 · " + error.message);
  byId("error").hidden = false;
});
