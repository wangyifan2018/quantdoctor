"""Evidence rules and single-variable intervention planning."""

from dataclasses import asdict, dataclass
import unicodedata

from .model import Comparison, Output, Sample
from .audio_evidence import audio_inputs_agree, sample_evidence

ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
ERRORS = {"error", "cancelled", "not-finished"}


@dataclass(frozen=True)
class Action:
    title: str
    change: dict[str, str]
    verify: str
    controls: str


@dataclass(frozen=True)
class Finding:
    code: str
    evidence: str
    attribution: str
    action: Action | None = None


def classify(output: Output | None):
    if output is None:
        return "missing"
    if output.finish_reason in ERRORS:
        return "failed"
    if output.text.strip():
        return "transcribed"
    if output.raw_text and output.raw_text.strip():
        return "cleanup_empty"
    if (output.generated_count == 1 and output.finish_reason == "end-of-sequence"
            and output.first_token_id is not None and output.first_token_id in (output.stop_token_ids or [])):
        return "first_step_eos"
    if output.finish_reason == "end-of-sequence":
        return "eos_unlocated"
    if output.finish_reason in {"max-length", "stop-words"}:
        return "other_stop"
    return "empty_unknown"


def nvfp4_intervention(comparison: Comparison):
    precision = comparison.candidate.precision
    if comparison.model != ASR_MODEL or precision["backbone"] != "nvfp4":
        return None
    verify = "在同一设备复跑失败样本和留出集，核对有效精度、空转写率、CER/WER 和 RTF。"
    controls = "固定输入、设备、解码和其他组件精度。"
    if precision["audio"] == "fp8":
        return Action("仅恢复 Audio 到 FP16", {"precision.audio": "fp16"}, verify, controls)
    if precision["audio"] == "fp16" and precision["head"] in {"nvfp4", "fp8"}:
        return Action("仅恢复 LM Head 到 FP16", {"precision.head": "fp16"}, verify, controls)
    if precision["audio"] == precision["head"] == "fp16":
        return Action("仅将 Backbone 切换到 FP8，建立质量对照", {"precision.backbone": "fp8"}, verify, controls)
    return None


def diagnose(comparison: Comparison, sample: Sample):
    differences = comparison.baseline.differences(comparison.candidate)
    state = classify(sample.candidate)
    if differences:
        return Finding("unpaired", "比较条件不一致：" + "、".join(differences), "待补齐配对条件")
    if not audio_inputs_agree(sample.baseline, sample.candidate, expected=sample.audio_sha256):
        return Finding("input_mismatch", "捕获的音频摘要与另一次运行或评测清单冲突。", "先复核实际输入音频")
    if state in {"failed", "missing"}:
        return Finding(state, "候选运行失败或缺少样本记录。", "核对失败阶段、请求索引和日志")
    if state == "transcribed":
        return Finding(state, "候选含有正文，可在配对指标中查看退化。", "无空转写")
    if state == "cleanup_empty":
        return Finding(state, "清理前有文本，清理后为空。", "差异位于文本清理",
                       Action("回放文本清理规则", {"postprocessing": "replay"},
                              "核对正文、合法前缀和 stop 规则。", "固定原始输出和评分规则。"))
    if classify(sample.baseline) != "transcribed":
        return Finding("baseline_unusable", "基线没有可用正文。", "先建立浮点基线")
    if state in {"first_step_eos", "eos_unlocated"}:
        evidence = ("首个生成 token 命中有效结束 ID。" if state == "first_step_eos"
                    else "运行以 EOS 结束，缺少首 token 或有效停止 ID。")
        action = nvfp4_intervention(comparison)
        return Finding(state, evidence, "量化归因待受控复验" if action else "待补齐组件精度或阶段证据", action)
    if state == "other_stop":
        return Finding(state, f"运行报告 {sample.candidate.finish_reason}。",
                       "先核对停止规则与生成长度")
    return Finding(state, "仅能确认候选空转写。", "补采结束原因、生成长度和清理前输出")


def work_order(comparison: Comparison, sample_id: str):
    sample = next((row for row in comparison.samples if row.id == sample_id), None)
    if sample is None:
        raise ValueError(f"sample {sample_id!r} not found")
    finding = diagnose(comparison, sample)
    if finding.action is None:
        raise ValueError(f"no proposed intervention for {sample_id}: {finding.attribution}")
    return {
        "schema_version": 1,
        "status": "pending_verification",
        "model": comparison.model,
        "sample_id": sample.id,
        "evaluation_dataset_sha256": comparison.evaluation_dataset_sha256,
        "request_document_sha256": comparison.request_document_sha256,
        "request_protocol_sha256": comparison.request_protocol.sha256 if comparison.request_protocol else None,
        "baseline": asdict(comparison.baseline),
        "candidate": asdict(comparison.candidate),
        "finding": asdict(finding),
    }


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for row, item in enumerate(left, start=1):
        current = [row]
        for col, other in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[col] + 1,
                               previous[col - 1] + (item != other)))
        previous = current
    return previous[-1]


def tokens(text, language):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    cleaned = "".join(" " if unicodedata.category(char).startswith("P") else char
                      for char in normalized)
    return cleaned.split() if language == "en" else list("".join(cleaned.split()))


def evaluate(comparison: Comparison):
    metrics = {}
    for language, name in (("en", "wer"), ("zh", "cer")):
        total = paired = 0
        errors = {"baseline": 0, "candidate": 0}
        for row in comparison.samples:
            if row.language != language or not row.reference or any(
                    classify(output) in {"missing", "failed"} for output in (row.baseline, row.candidate)):
                continue
            if not audio_inputs_agree(row.baseline, row.candidate, expected=row.audio_sha256):
                continue
            reference = tokens(row.reference, language)
            if not reference:
                continue
            paired += 1
            total += len(reference)
            for side in errors:
                errors[side] += edit_distance(reference, tokens(getattr(row, side).text, language))
        metrics[name] = {"paired": paired, "reference_units": total, **{
            side: errors[side] / total if total else None for side in errors
        }}
    return metrics


def report(comparison: Comparison):
    comparable = not comparison.baseline.differences(comparison.candidate)
    outcomes = {}
    for side in ("baseline", "candidate"):
        outputs = [getattr(row, side) for row in comparison.samples]
        states = [classify(output) for output in outputs]
        missing, failed = states.count("missing"), states.count("failed")
        completed = len(outputs) - missing - failed
        empty = completed - states.count("transcribed")
        outcomes[side] = {"empty": empty, "completed": completed, "missing": missing, "failed": failed}
    return {
        "model": comparison.model,
        "evaluation_dataset_sha256": comparison.evaluation_dataset_sha256,
        "request_document_sha256": comparison.request_document_sha256,
        "request_protocol": asdict(comparison.request_protocol) if comparison.request_protocol else None,
        "baseline": asdict(comparison.baseline),
        "candidate": asdict(comparison.candidate),
        "samples": [
            {**asdict(row), "finding": asdict(diagnose(comparison, row)), **sample_evidence(
                row.baseline, row.candidate,
                comparable and bool(comparison.request_document_sha256),
                expected=row.audio_sha256,
            )}
            for row in comparison.samples
        ],
        "metrics": evaluate(comparison) if comparable else {},
        "outcomes": outcomes,
        "comparable": comparable,
    }
