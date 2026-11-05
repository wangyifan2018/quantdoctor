"""Compare several ASR candidates on the same scored samples."""

from collections import Counter
from dataclasses import asdict
from random import Random

from .audio_evidence import audio_inputs_agree
from .diagnostics import classify, edit_distance, tokens
from .model import require
from .performance import compare_performance

UNAVAILABLE = {"missing", "failed"}
MIN_INTERVAL_SAMPLES = 10
RESAMPLES = 1000
SEED = 17


def _validate(comparisons):
    require(2 <= len(comparisons) <= 12, "comparison requires 2 to 12 candidate runs")
    first = comparisons[0]
    require(first.evaluation_dataset_sha256 and first.request_document_sha256,
            "candidate comparison requires dataset and request fingerprints")
    require(len({run.candidate.id for run in comparisons}) == len(comparisons),
            "candidate run IDs must be distinct")
    expected = {row.id: (row.reference, row.language, row.baseline, row.audio_sha256) for row in first.samples}
    for comparison in comparisons:
        require(comparison.model == first.model and comparison.baseline == first.baseline,
                "all candidates must share the same model and baseline run")
        require(not comparison.baseline.differences(comparison.candidate),
                f"{comparison.candidate.id}: run controls differ from the baseline")
        require(comparison.evaluation_dataset_sha256 == first.evaluation_dataset_sha256
                and comparison.request_document_sha256 == first.request_document_sha256,
                f"{comparison.candidate.id}: dataset or request fingerprint differs")
        observed = {row.id: (row.reference, row.language, row.baseline, row.audio_sha256) for row in comparison.samples}
        require(expected == observed,
                f"{comparison.candidate.id}: sample IDs, references, audio fingerprints or baseline outputs differ")


def _percentile(values, fraction):
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def _paired_intervals(rows, candidates):
    if len(rows) < MIN_INTERVAL_SAMPLES:
        return [None] * candidates
    rng = Random(SEED)
    results = [[] for _ in range(candidates)]
    for _ in range(RESAMPLES):
        totals = [0] * candidates
        units = 0
        for _ in rows:
            length, _, deltas = rows[rng.randrange(len(rows))]
            units += length
            for index, delta in enumerate(deltas):
                totals[index] += delta
        for index, delta in enumerate(totals):
            results[index].append(delta / units)
    return [[_percentile(sorted_values, 0.025), _percentile(sorted_values, 0.975)]
            for sorted_values in map(sorted, results)]


def _measure(rows, candidate_ids, eligible_count, reference_index):
    units = sum(row[0] for row in rows)
    baseline = sum(row[1] for row in rows)
    paired = rows if reference_index is None else [
        (length, errors, [delta - deltas[reference_index] for delta in deltas])
        for length, errors, deltas in rows
    ]
    intervals = _paired_intervals(paired, len(candidate_ids))
    scored = []
    for index, run_id in enumerate(candidate_ids):
        delta = sum(row[2][index] for row in rows)
        relative = sum(row[2][index] for row in paired)
        scored.append({
            "run_id": run_id,
            "error_rate": (baseline + delta) / units if units else None,
            "delta_vs_baseline": delta / units if units else None,
            "delta_vs_reference": relative / units if units else None,
            "interval_95": intervals[index],
        })
    if units:
        scored.sort(key=lambda entry: entry["error_rate"])
    return {
        "eligible_samples": eligible_count,
        "common_samples": len(rows),
        "reference_units": units,
        "baseline_error_rate": baseline / units if units else None,
        "candidates": scored,
    }


def compare_candidates(comparisons, reference_run_id=None, *, costs=None):
    _validate(comparisons)
    first = comparisons[0]
    indices = [{row.id: row for row in comparison.samples} for comparison in comparisons]
    candidate_ids = [comparison.candidate.id for comparison in comparisons]
    reference_run_id = reference_run_id or first.baseline.id
    require(reference_run_id == first.baseline.id or reference_run_id in candidate_ids,
            "reference run must be the baseline or one of the compared candidates")
    reference_index = None if reference_run_id == first.baseline.id else candidate_ids.index(reference_run_id)
    contributions = {"cer": [], "wer": []}
    eligible = Counter()
    outcomes = [Counter() for _ in comparisons]
    samples = []
    for row in sorted(first.samples, key=lambda item: item.id):
        candidates = [index[row.id].candidate for index in indices]
        states = [classify(output) for output in candidates]
        for counts, state in zip(outcomes, states, strict=True):
            counts["missing" if state == "missing" else "failed" if state == "failed"
                   else "text" if state == "transcribed" else "empty"] += 1
        metric = {"zh": "cer", "en": "wer"}.get(row.language)
        reference = tokens(row.reference, row.language) if metric and row.reference else []
        input_match = audio_inputs_agree(row.baseline, *candidates, expected=row.audio_sha256)
        common = (bool(reference) and input_match and classify(row.baseline) not in UNAVAILABLE
                  and UNAVAILABLE.isdisjoint(states))
        if reference:
            eligible[metric] += 1
        if common:
            baseline = edit_distance(reference, tokens(row.baseline.text, row.language))
            deltas = [edit_distance(reference, tokens(output.text, row.language)) - baseline for output in candidates]
            contributions[metric].append((len(reference), baseline, deltas))
        samples.append({
            "id": row.id, "reference": row.reference, "common_scoring": common,
            "input_mismatch": not input_match,
            "baseline_text": row.baseline.text if row.baseline else None,
            "baseline_status": classify(row.baseline),
            "candidates": {run_id: {"status": state, "text": output.text if output else None}
                           for run_id, state, output in zip(candidate_ids, states, candidates, strict=True)},
        })
    report = {
        "schema_version": 1, "kind": "candidate_comparison", "model": first.model,
        "baseline": asdict(first.baseline),
        "reference_run_id": reference_run_id,
        "evaluation_dataset_sha256": first.evaluation_dataset_sha256,
        "request_document_sha256": first.request_document_sha256,
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "minimum_samples": MIN_INTERVAL_SAMPLES},
        "candidates": [{**asdict(comparison.candidate), "outcomes": {
            key: counts[key] for key in ("text", "empty", "failed", "missing")
        }} for comparison, counts in zip(comparisons, outcomes, strict=True)],
        "metrics": {key: _measure(rows, candidate_ids, eligible[key], reference_index)
                    for key, rows in contributions.items()},
        "samples": samples,
        "input_mismatch_samples": sum(sample["input_mismatch"] for sample in samples),
    }
    if costs is not None:
        report["performance"] = compare_performance(report, costs)
    return report
