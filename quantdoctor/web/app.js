const byId = (id) => document.getElementById(id);
const labels = {
  missing: "缺少结果", failed: "运行失败", transcribed: "有转写", cleanup_empty: "清理后为空",
  first_step_eos: "首步 EOS", eos_unlocated: "EOS 位置待确认", other_stop: "其他停止条件",
  empty_unknown: "空转写 · 证据不足", baseline_unusable: "基线不可用", unpaired: "配对条件不一致",
};
const sources = { fixture: "模拟数据", torch: "PyTorch 结果", target: "设备结果" };

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

function setup(data) {
  const { baseline, candidate } = data;
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
      const percentage = (value) => (value * 100).toFixed(1) + "%";
      row.textContent = metric.paired
        ? key.toUpperCase() + "  " + percentage(metric.baseline) + " → " + percentage(metric.candidate) + " · " + metric.paired + " 个配对"
        : key.toUpperCase() + "  无可评分配对";
      metrics.append(row);
    }
  }

  const list = byId("samples");
  const filter = byId("sample-filter");
  let selected;
  function renderSample(sample, button) {
    selected = sample.id;
    byId("detail").hidden = false;
    for (const entry of list.querySelectorAll("button")) entry.setAttribute("aria-pressed", entry === button);
    text("sample-id", sample.id);
    text("reference", sample.reference == null ? "参考文本未提供" : "参考 " + sample.reference);
    result("baseline", sample.baseline, baseline);
    result("candidate", sample.candidate, candidate);
    const finding = sample.finding;
    text("finding-title", labels[finding.code]);
    text("evidence", finding.evidence);
    text("attribution", finding.attribution);
    const panel = byId("intervention");
    panel.hidden = !finding.action;
    panel.open = false;
    if (finding.action) {
      text("action", finding.action.title + " · 待复验");
      text("change", JSON.stringify(finding.action.change, null, 2));
      text("controls", finding.action.controls);
      text("verify", finding.action.verify);
    }
  }

  function populate() {
    list.replaceChildren();
    const rows = data.samples.filter((sample) =>
      filter.value === "all" || sample.finding.code !== "transcribed");
    let activeButton;
    for (const sample of rows) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = sample.id + " · " + labels[sample.finding.code];
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
