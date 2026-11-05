"""Controlled comparison of an original run and one precision intervention."""

from dataclasses import asdict, replace

from .checkpoint import audit_checkpoints
from .diagnostics import classify, edit_distance, evaluate, tokens, work_order
from .model import Comparison, Sample, nonempty, require
from .performance import MEASUREMENTS, read_measurements
from .trace import compare_steps


def _controls(before: Comparison, after: Comparison, order):
    require(before.model == after.model and before.baseline == after.baseline,
            "repair must reuse the same model and baseline run")
    require(after.candidate.id != before.candidate.id, "repair requires a new candidate run ID")
    differences = before.candidate.differences(after.candidate)
    require(not differences, "repair changed comparison controls: " + ", ".join(differences))
    for field in ("evaluation_dataset_sha256", "request_document_sha256"):
        original, repaired = getattr(before, field), getattr(after, field)
        require(original and original == repaired, f"repair requires matching {field}")
    require(all("unknown" not in run.precision.values()
                for run in (before.baseline, before.candidate, after.candidate)),
            "effective precision must be known for all components")
    changed = {
        f"precision.{component}": value
        for component, value in after.candidate.precision.items()
        if value != before.candidate.precision[component]
    }
    expected = order["finding"]["action"]["change"]
    require(changed == expected, f"repair changed {changed}; expected only {expected}")
    return changed


def _transition(original, repaired):
    first, second = classify(original), classify(repaired)
    if first in {"missing", "failed"}:
        return "original_unavailable"
    if second == "missing":
        return "repair_missing"
    if second == "failed":
        return "repair_failed"
    if first == "transcribed":
        return "stable_text" if second == "transcribed" else "new_empty"
    return "recovered_text" if second == "transcribed" else "still_empty"


def _sample_edit_delta(original: Sample, repaired: Sample, transition):
    if transition in {"original_unavailable", "repair_missing", "repair_failed"} or not original.reference:
        return None
    if original.language not in {"zh", "en"}:
        return None
    reference = tokens(original.reference, original.language)
    if not reference:
        return None
    before = edit_distance(reference, tokens(original.candidate.text, original.language))
    after = edit_distance(reference, tokens(repaired.candidate.text, original.language))
    return {"original": before, "repaired": after, "reference_units": len(reference)}


def _cohort_metrics(before, after, shared):
    original = evaluate(replace(before, samples=[row for row in before.samples if row.id in shared]))
    repaired = evaluate(replace(after, samples=[row for row in after.samples if row.id in shared]))
    return {
        metric: {"paired": values["paired"], "reference_units": values["reference_units"],
                 "baseline": values["baseline"], "original": values["candidate"],
                 "repaired": repaired[metric]["candidate"]}
        for metric, values in original.items()
    }


def _compare_costs(data, before, after):
    if data is None:
        return None
    require(isinstance(data, dict) and type(data.get("schema_version")) is int
            and data["schema_version"] == 1 and nonempty(data.get("protocol_id")),
            "costs require schema_version 1 and a measurement protocol ID")
    runs = {}
    for side, run in (("original", before.candidate), ("repaired", after.candidate)):
        entry = data.get(side)
        require(isinstance(entry, dict) and entry.get("run_id") == run.id,
                f"costs {side} run ID does not match")
        runs[side] = read_measurements(entry)
    measurements = {}
    for key in MEASUREMENTS:
        original, repaired = (runs[side].get(key) for side in ("original", "repaired"))
        if original is not None and repaired is not None:
            measurements[key] = {"original": original, "repaired": repaired, "delta": repaired - original}
    return {"protocol_id": data["protocol_id"], "measurements": measurements}


def _checkpoint_evidence(before, after, changed, checkpoints):
    if checkpoints is None:
        return {"status": "not_supplied"}
    original_hash, repaired_hash = before.candidate.checkpoint_sha256, after.candidate.checkpoint_sha256
    require(original_hash and repaired_hash, "checkpoint audit requires checkpoint_sha256 on both candidate runs")
    component = next(iter(changed)).split(".", 1)[1]
    require(component in ("audio", "backbone", "head"), "checkpoint audit does not map this precision component")
    audit = audit_checkpoints(*checkpoints, expected_component=component)
    require(audit["model_type"] == "qwen3_asr", "ASR repair requires Qwen3-ASR checkpoint artifacts")
    require(audit["original_checkpoint_sha256"] == original_hash
            and audit["candidate_checkpoint_sha256"] == repaired_hash,
            "checkpoint content does not match the candidate run fingerprints")
    return {"status": "matched_run_fingerprints", **audit}


def verify_repair(before: Comparison, after: Comparison, sample_id: str, costs=None, *, checkpoints=None):
    """Check declared controls and show paired evidence. Does not certify deployment."""
    order = work_order(before, sample_id)
    expected = order["finding"]["action"]["change"]
    require(len(expected) == 1 and all(key.startswith("precision.") for key in expected),
            "repair replay currently requires a single precision change")
    changed = _controls(before, after, order)
    checkpoint = _checkpoint_evidence(before, after, changed, checkpoints)
    status = "compared"
    if checkpoint.get("matches_expectation") is False:
        status = "checkpoint_scope_mismatch"
    elif checkpoint.get("changed_configs"):
        status = "checkpoint_config_review"
    repaired_rows = {row.id: row for row in after.samples}
    require(repaired_rows.keys() == {row.id for row in before.samples},
            "repair must preserve sample IDs, including missing results")
    samples = {}
    shared = set()
    counts = {code: 0 for code in ("recovered_text", "new_empty", "still_empty", "stable_text",
                                  "repair_missing", "repair_failed", "original_unavailable")}
    for original in before.samples:
        repaired = repaired_rows[original.id]
        require((original.reference, original.language, original.baseline) ==
                (repaired.reference, repaired.language, repaired.baseline),
                f"{original.id}: reference, language or baseline output changed")
        transition = _transition(original.candidate, repaired.candidate)
        counts[transition] += 1
        edits = _sample_edit_delta(original, repaired, transition)
        if edits is not None and classify(original.baseline) not in {"failed", "missing"}:
            shared.add(original.id)
        samples[original.id] = {
            "transition": transition,
            "output": asdict(repaired.candidate) if repaired.candidate is not None else None,
            "edits": edits,
            "decode_evidence": compare_steps(
                original.baseline.trace if original.baseline else (),
                repaired.candidate.trace if repaired.candidate else (),
            ),
        }

    return {
        "schema_version": 1,
        "model": before.model,
        "source": before.candidate.source,
        "status": status,
        "deployment_acceptance": "not_assessed",
        "work_order": order,
        "repair_run": asdict(after.candidate),
        "change": changed,
        "counts": counts,
        "metrics": _cohort_metrics(before, after, shared),
        "samples": samples,
        "costs": _compare_costs(costs, before, after),
        "checkpoint_audit": checkpoint,
    }
