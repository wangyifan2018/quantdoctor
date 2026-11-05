"""Controlled comparison of an original run and one precision intervention."""

from dataclasses import asdict, replace
import math

from .diagnostics import classify, edit_distance, evaluate, tokens, work_order
from .model import Comparison, Sample, nonempty, require


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
        runs[side] = entry
    measurements = {}
    for key in ("rtf_p95", "peak_memory_bytes", "artifact_bytes"):
        original, repaired = (runs[side].get(key) for side in ("original", "repaired"))
        for value in (original, repaired):
            if value is None:
                continue
            if key.endswith("_bytes"):
                require(type(value) is int and 0 <= value <= 2**63 - 1, f"invalid costs {key}")
            else:
                require(type(value) in (int, float) and 0 <= value <= 1e9 and math.isfinite(value),
                        f"invalid costs {key}")
        if original is not None and repaired is not None:
            measurements[key] = {"original": original, "repaired": repaired, "delta": repaired - original}
    return {"protocol_id": data["protocol_id"], "measurements": measurements}


def verify_repair(before: Comparison, after: Comparison, sample_id: str, costs=None):
    """Check declared controls and show paired evidence. Does not certify deployment."""
    order = work_order(before, sample_id)
    expected = order["finding"]["action"]["change"]
    require(len(expected) == 1 and all(key.startswith("precision.") for key in expected),
            "repair replay currently requires a single precision change")
    changed = _controls(before, after, order)
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
        }

    return {
        "schema_version": 1,
        "model": before.model,
        "source": before.candidate.source,
        "status": "compared",
        "deployment_acceptance": "not_assessed",
        "work_order": order,
        "repair_run": asdict(after.candidate),
        "change": changed,
        "counts": counts,
        "metrics": _cohort_metrics(before, after, shared),
        "samples": samples,
        "costs": _compare_costs(costs, before, after),
    }
