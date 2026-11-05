"""Measurement validation and point-estimate tradeoffs for ASR comparisons."""

import math

from .model import nonempty, require


MEASUREMENTS = ("rtf_p95", "peak_memory_bytes", "artifact_bytes")
FRONTIER_COSTS = ("rtf_p95", "peak_memory_bytes")


def read_measurements(entry):
    require(isinstance(entry, dict), "performance measurements must be an object")
    measurements = {}
    for key in MEASUREMENTS:
        value = entry.get(key)
        if value is None:
            continue
        if key.endswith("_bytes"):
            require(type(value) is int and 0 <= value <= 2**63 - 1, f"invalid costs {key}")
        else:
            require(type(value) in (int, float) and 0 <= value <= 1e9 and math.isfinite(value),
                    f"invalid costs {key}")
        measurements[key] = value
    return measurements


def _dominates(left, right):
    return all(a <= b for a, b in zip(left, right, strict=True)) and any(
        a < b for a, b in zip(left, right, strict=True))


def compare_performance(report, data):
    require(isinstance(data, dict) and type(data.get("schema_version")) is int
            and data["schema_version"] == 1 and nonempty(data.get("protocol_id")),
            "performance requires schema_version 1 and a protocol ID")
    baseline = report["baseline"]
    for field, expected in (
        ("source", baseline["source"]),
        ("environment_id", baseline["environment_id"]),
        ("evaluation_dataset_sha256", report["evaluation_dataset_sha256"]),
        ("request_document_sha256", report["request_document_sha256"]),
    ):
        require(data.get(field) == expected, f"performance {field} does not match the comparison")

    measurements = data.get("runs")
    require(isinstance(measurements, dict), "performance requires a runs object")
    require(measurements.keys() <= {run["id"] for run in report["candidates"]} | {baseline["id"]},
            "performance contains run IDs outside the comparison")
    baseline_values = read_measurements(measurements.get(baseline["id"], {}))
    scores = report["metrics"]
    languages = [name for name, metric in scores.items() if metric["eligible_samples"]]
    missing_quality = [name for name in languages if not scores[name]["common_samples"]]
    by_metric = {name: {row["run_id"]: row["error_rate"] for row in scores[name]["candidates"]}
                 for name in languages}
    objectives = [*languages, "unavailable", "empty", *FRONTIER_COSTS]
    rows = {}
    vectors = {}
    for run in report["candidates"]:
        run_id = run["id"]
        entry = read_measurements(measurements[run_id]) if run_id in measurements else {}
        missing = [name for name in FRONTIER_COSTS if name not in entry]
        outcomes = run["outcomes"]
        values = {name: by_metric[name][run_id] for name in languages}
        values.update({"unavailable": outcomes["missing"] + outcomes["failed"], "empty": outcomes["empty"]})
        values.update({name: entry.get(name) for name in MEASUREMENTS})
        rows[run_id] = {"values": values, "missing_costs": missing, "frontier": "not_assessed", "dominated_by": []}
        if languages and not missing_quality and not missing:
            vectors[run_id] = tuple(values[name] for name in objectives)

    status = "compared"
    if report["input_mismatch_samples"]:
        status = "input_mismatch"
    elif not languages or missing_quality:
        status = "insufficient_common_quality"
    elif len(vectors) < 2:
        status = "insufficient_complete_costs"
    else:
        for run_id, vector in vectors.items():
            dominators = [other_id for other_id, other in vectors.items()
                          if other_id != run_id and _dominates(other, vector)]
            rows[run_id]["frontier"] = "dominated" if dominators else "non_dominated"
            rows[run_id]["dominated_by"] = dominators
    return {"protocol_id": data["protocol_id"], "status": status, "complete_candidates": len(vectors),
            "baseline_measurements": baseline_values,
            "objectives": objectives,
            "missing_quality": missing_quality, "runs": rows}
