"""Controlled comparison of an original run and one precision intervention."""

from collections import defaultdict
from dataclasses import asdict, replace

from .checkpoint import audit_checkpoints
from .audio_evidence import audio_inputs_agree, sample_evidence
from .diagnostics import classify, edit_distance, evaluate, tokens, work_order
from .model import Comparison, Sample, nonempty, require
from .performance import MEASUREMENTS, read_measurements
from .quality import MIN_INTERVAL_UNITS, paired_intervals

COHORT_FINGERPRINTS = ("evaluation_dataset_sha256", "request_document_sha256")


def _controls(before: Comparison, after: Comparison, order):
    require(before.model == after.model and before.baseline == after.baseline,
            "repair must reuse the same model and baseline run")
    require(after.candidate.id != before.candidate.id, "repair requires a new candidate run ID")
    differences = before.candidate.differences(after.candidate)
    require(not differences, "repair changed comparison controls: " + ", ".join(differences))
    for field in COHORT_FINGERPRINTS:
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


def _holdout_identity(diagnosis, before, after):
    require(diagnosis.model == before.model, "holdout must reuse the diagnosis model")
    for field in COHORT_FINGERPRINTS:
        first, second = getattr(diagnosis, field), getattr(before, field)
        require(first and second and first != second, f"holdout requires a distinct {field} from diagnosis")
    identities = {}
    for field in ("id", "audio_sha256", "speaker_id"):
        first = {getattr(row, field) for row in diagnosis.samples}
        second = {getattr(row, field) for row in before.samples}
        require(None not in first and None not in second, f"holdout isolation requires complete {field}")
        require(first.isdisjoint(second), f"diagnosis and holdout overlap by {field}")
        identities[field] = len(first), len(second)
    require(all(audio_inputs_agree(row.baseline, row.candidate, expected=row.audio_sha256)
                for run in (diagnosis, before, after) for row in run.samples),
            "holdout isolation requires captured audio to agree with its manifest")
    controls = {}
    for role in ("baseline", "candidate"):
        first, second = getattr(diagnosis, role), getattr(before, role)
        differences = first.differences(second)
        if first.precision != second.precision:
            differences.append("precision")
        require(not differences, f"holdout {role} differs from diagnosis: {', '.join(differences)}")
        hashes = {}
        for field, left, right in (
            ("checkpoint_sha256", first.checkpoint_sha256, second.checkpoint_sha256),
            ("engine_sha256", first.worker_evidence.engine_sha256 if first.worker_evidence else None,
             second.worker_evidence.engine_sha256 if second.worker_evidence else None),
        ):
            if left and right:
                require(left == right, f"holdout {role} {field} differs from diagnosis")
                hashes[field] = left
        controls[role] = {"diagnosis_run_id": first.id, "evaluation_run_id": second.id, "matched_artifacts": hashes}
    return {
        "status": "disjoint_declared_identities",
        "diagnosis_samples": len(diagnosis.samples), "evaluation_samples": len(before.samples),
        "diagnosis_speakers": identities["speaker_id"][0], "evaluation_speakers": identities["speaker_id"][1],
        **{field: getattr(before, field) for field in COHORT_FINGERPRINTS},
        "controls": controls,
    }


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
    if transition in {"original_unavailable", "repair_missing", "repair_failed", "input_mismatch"} or not original.reference:
        return None
    if original.language not in {"zh", "en"}:
        return None
    reference = tokens(original.reference, original.language)
    if not reference:
        return None
    before = edit_distance(reference, tokens(original.candidate.text, original.language))
    after = edit_distance(reference, tokens(repaired.candidate.text, original.language))
    return {"original": before, "repaired": after, "reference_units": len(reference)}


def _edit_effect(edits):
    if edits is None:
        return "unscored"
    if edits["repaired"] < edits["original"]:
        return "fewer_edits"
    return "more_edits" if edits["repaired"] > edits["original"] else "same_edits"


def _cohort_metrics(before, after, shared):
    original = evaluate(replace(before, samples=[row for row in before.samples if row.id in shared]))
    repaired = evaluate(replace(after, samples=[row for row in after.samples if row.id in shared]))
    return {
        metric: {"paired": values["paired"], "reference_units": values["reference_units"],
                 "baseline": values["baseline"], "original": values["candidate"],
                 "repaired": repaired[metric]["candidate"]}
        for metric, values in original.items()
    }


def _holdout_intervals(metrics, before, rows, shared):
    for language, metric in (("zh", "cer"), ("en", "wer")):
        speakers = defaultdict(lambda: [0, 0])
        for row in before.samples:
            if row.id in shared and row.language == language:
                edits = rows[row.id]["edits"]
                pair = speakers[row.speaker_id]
                pair[0] += edits["reference_units"]
                pair[1] += edits["repaired"] - edits["original"]
        paired = [(units, [delta]) for _, (units, delta) in sorted(speakers.items())]
        metric_values = metrics[metric]
        units = metric_values["reference_units"]
        metric_values.update({
            "delta_vs_original": sum(delta for _, [delta] in paired) / units if units else None,
            "interval_95": paired_intervals(paired, 1)[0],
            "speaker_count": len(paired), "min_speakers": MIN_INTERVAL_UNITS,
        })


def _compare_costs(data, before, after, *, holdout):
    if data is None:
        return None
    require(isinstance(data, dict) and type(data.get("schema_version")) is int
            and data["schema_version"] == 1 and nonempty(data.get("protocol_id")),
            "costs require schema_version 1 and a measurement protocol ID")
    if holdout:
        for field in COHORT_FINGERPRINTS:
            require(data.get(field) == getattr(before, field), f"holdout costs require matching {field}")
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


def verify_repair(before: Comparison, after: Comparison, sample_id: str, costs=None, *, checkpoints=None,
                  diagnosis: Comparison | None = None):
    """Check declared controls and show paired evidence. Does not certify deployment."""
    order = work_order(diagnosis if diagnosis is not None else before, sample_id)
    expected = order["finding"]["action"]["change"]
    require(len(expected) == 1 and all(key.startswith("precision.") for key in expected),
            "repair replay currently requires a single precision change")
    changed = _controls(before, after, order)
    holdout = _holdout_identity(diagnosis, before, after) if diagnosis is not None else None
    performance = _compare_costs(costs, before, after, holdout=holdout is not None)
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
                                  "repair_missing", "repair_failed", "original_unavailable", "input_mismatch")}
    edit_counts = dict.fromkeys(("fewer_edits", "more_edits", "same_edits", "unscored"), 0)
    for original in before.samples:
        repaired = repaired_rows[original.id]
        require((original.reference, original.language, original.speaker_id, original.baseline, original.audio_sha256) ==
                (repaired.reference, repaired.language, repaired.speaker_id, repaired.baseline, repaired.audio_sha256),
                f"{original.id}: reference, language, speaker, audio fingerprint or baseline output changed")
        transition = (_transition(original.candidate, repaired.candidate)
                      if audio_inputs_agree(original.baseline, original.candidate, repaired.candidate,
                                            expected=original.audio_sha256)
                      else "input_mismatch")
        counts[transition] += 1
        edits = _sample_edit_delta(original, repaired, transition)
        edit_effect = _edit_effect(edits)
        edit_counts[edit_effect] += 1
        if edits is not None and classify(original.baseline) not in {"failed", "missing"}:
            shared.add(original.id)
        samples[original.id] = {
            "transition": transition,
            "output": asdict(repaired.candidate) if repaired.candidate is not None else None,
            "edits": edits,
            "edit_effect": edit_effect,
            **sample_evidence(original.baseline, repaired.candidate, expected=original.audio_sha256),
        }

    metrics = _cohort_metrics(before, after, shared)
    if holdout is not None:
        _holdout_intervals(metrics, before, samples, shared)
    return {
        "schema_version": 1,
        "model": before.model,
        "source": before.candidate.source,
        "status": status,
        "deployment_acceptance": "not_assessed",
        "work_order": order,
        "baseline_run": asdict(before.baseline),
        "original_run": asdict(before.candidate),
        "repair_run": asdict(after.candidate),
        "holdout": holdout,
        "change": changed,
        "counts": counts,
        "edit_counts": edit_counts,
        "metrics": metrics,
        "samples": samples,
        "costs": performance,
        "checkpoint_audit": checkpoint,
    }
