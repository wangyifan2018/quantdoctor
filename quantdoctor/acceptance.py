"""Evaluate an explicit deployment policy against imported, paired evidence."""

from hashlib import sha256
import json

from .model import SHA256, nonempty, require

UNAVAILABLE = {"missing", "failed"}


def _limit(data, key, minimum, maximum):
    value = data.get(key)
    require(type(value) in (int, float) and minimum <= value <= maximum,
            f"policy {key} must be between {minimum} and {maximum}")
    return value


def _rule(code, status, observed=None, limit=None, *, reason=None):
    return {"code": code, "status": status, "observed": observed, "limit": limit, "reason": reason}


def _upper_rule(code, value, maximum):
    if value is None:
        return _rule(code, "insufficient", limit=maximum, reason="缺少可比较记录")
    return _rule(code, "meets" if value <= maximum + 1e-9 else "fails", value, maximum)


def _quality_rule(metric, name, candidate, maximum, minimum, coverage):
    observed = next(row for row in metric["candidates"] if row["run_id"] == candidate)
    code = f"{name}_increase_pp"
    value = observed["delta_vs_baseline"]
    if (metric["common_samples"] < minimum or not metric["eligible_samples"]
            or metric["common_samples"] / metric["eligible_samples"] < coverage):
        return _rule(code, "insufficient", limit=maximum, reason="共同样本数或覆盖率未达到门槛")
    if metric["bootstrap_unit"] != "speaker" or observed["interval_95"] is None:
        return _rule(code, "insufficient", value * 100, maximum, reason="缺少按说话人计算的配对区间")
    low, high = (bound * 100 for bound in observed["interval_95"])
    status = "meets" if high <= maximum + 1e-9 else "fails" if low > maximum + 1e-9 else "insufficient"
    return {**_rule(code, status, value * 100, maximum,
                    reason="配对区间跨过门槛" if status == "insufficient" else None), "interval_95": [low, high]}


def _empty_delta(samples, run_id, coverage):
    paired = [
        (row["baseline_status"], row["candidates"][run_id]["status"]) for row in samples
        if not row["input_mismatch"] and row["baseline_status"] not in UNAVAILABLE
        and row["candidates"][run_id]["status"] not in UNAVAILABLE
    ]
    if not paired or len(paired) / len(samples) < coverage:
        return None
    return 100 * sum((candidate != "transcribed") - (baseline != "transcribed")
                     for baseline, candidate in paired) / len(paired)


def _percent_change(baseline, candidate, key):
    first, second = baseline.get(key), candidate.get(key)
    return 100 * (second - first) / first if first and second is not None else None


def _savings_rule(baseline, candidate, minimum):
    savings = {}
    for key in ("peak_memory_bytes", "artifact_bytes"):
        change = _percent_change(baseline, candidate, key)
        if change is not None:
            savings[key] = -change
    best = max(savings.values(), default=None)
    if best is not None and best + 1e-9 >= minimum:
        status = "meets"
    elif len(savings) == 2:
        status = "fails"
    else:
        status = "insufficient"
    return {**_rule("resource_savings_pct", status, best, minimum,
                    reason="缺少另一项可比较的资源记录" if status == "insufficient" else None),
            "measurements": savings}


def evaluate_acceptance(report, policy):
    require(isinstance(policy, dict) and type(policy.get("schema_version")) is int
            and policy["schema_version"] == 1 and nonempty(policy.get("policy_id")),
            "acceptance requires schema_version 1 and a policy ID")
    require(set(policy) == {
        "schema_version", "policy_id", "environment_id", "evaluation_dataset_sha256", "request_document_sha256",
        "protocol_id", "quality", "min_common_samples", "min_coverage", "max_empty_increase_pp",
        "max_unavailable_rate", "max_rtf_increase_pct", "min_resource_savings_pct",
    }, "unexpected or missing acceptance policy fields")
    baseline = report["baseline"]
    for field, expected in (
        ("environment_id", baseline["environment_id"]),
        ("evaluation_dataset_sha256", report["evaluation_dataset_sha256"]),
        ("request_document_sha256", report["request_document_sha256"]),
    ):
        require(policy.get(field) == expected, f"acceptance {field} does not match the comparison")
    require(all(value == "fp16" for value in baseline["precision"].values()),
            "acceptance requires a same-environment FP16 baseline")
    require(report["reference_run_id"] == baseline["id"], "acceptance intervals must use the FP16 baseline as reference")
    require(nonempty(policy.get("protocol_id")), "acceptance requires a measurement protocol ID")
    performance = report.get("performance")
    require(not performance or performance["protocol_id"] == policy["protocol_id"],
            "performance protocol does not match acceptance policy")
    quality = policy.get("quality")
    require(isinstance(quality, dict) and quality and quality.keys() <= {"cer", "wer"},
            "acceptance quality requires CER and/or WER limits")
    require({name for name, metric in report["metrics"].items() if metric["eligible_samples"]} <= quality.keys(),
            "acceptance policy must cover every language present in the scored data")
    maxima = {name: _limit(quality, name, 0, 100) for name in quality}
    minimum = policy.get("min_common_samples")
    require(type(minimum) is int and minimum > 0, "acceptance requires a positive min_common_samples")
    coverage = _limit(policy, "min_coverage", 0, 1)
    require(coverage > 0, "acceptance min_coverage must be positive")
    empty_limit = _limit(policy, "max_empty_increase_pp", 0, 100)
    unavailable_limit = _limit(policy, "max_unavailable_rate", 0, 1)
    rtf_limit = _limit(policy, "max_rtf_increase_pct", 0, 1000)
    savings_limit = _limit(policy, "min_resource_savings_pct", 0, 100)
    samples = report["samples"]
    outcomes = []
    for run in report["candidates"]:
        run_id = run["id"]
        checks = [_quality_rule(report["metrics"][name], name, run_id, maximum, minimum, coverage)
                  for name, maximum in maxima.items()]
        counts = run["outcomes"]
        baseline_costs = performance["baseline_measurements"] if performance else {}
        candidate_costs = performance["runs"][run_id]["values"] if performance else {}
        checks.extend([
            _upper_rule("unavailable_rate", (counts["missing"] + counts["failed"]) / len(samples), unavailable_limit),
            _upper_rule("empty_increase_pp", _empty_delta(samples, run_id, coverage), empty_limit),
            _upper_rule("rtf_increase_pct", _percent_change(baseline_costs, candidate_costs, "rtf_p95"), rtf_limit),
            _savings_rule(baseline_costs, candidate_costs, savings_limit),
        ])
        if report["input_mismatch_samples"]:
            checks.append(_rule("input_integrity", "insufficient", report["input_mismatch_samples"], 0,
                                reason="存在实际音频与评测清单或运行记录冲突"))
        if "unknown" in run["precision"].values():
            checks.append(_rule("precision_evidence", "insufficient", reason="候选包含未知精度组件"))
        statuses = {row["status"] for row in checks}
        state = "outside_policy" if "fails" in statuses else "insufficient_evidence" if "insufficient" in statuses else "meets_policy"
        outcomes.append({"run_id": run_id, "status": state, "checks": checks})
    canonical = json.dumps(policy, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return {"policy_id": policy["policy_id"], "policy_sha256": sha256(canonical.encode("ascii")).hexdigest(),
            "source": baseline["source"], "deployment_verified": False, "candidates": outcomes}
