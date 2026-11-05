"""Compare several ASR candidates on the same scored samples."""

from collections import Counter
from dataclasses import asdict, dataclass
from random import Random

from .audio_evidence import audio_inputs_agree, sample_evidence
from .diagnostics import classify, diagnose, edit_distance, tokens
from .model import require
from .performance import compare_performance

UNAVAILABLE = {"missing", "failed"}
MIN_INTERVAL_UNITS = 10
RESAMPLES = 1000
SEED = 17


@dataclass(frozen=True)
class ScoredSample:
    units: int
    baseline_errors: int
    deltas: list[int]
    speaker_id: str | None


def _validate(comparisons):
    require(2 <= len(comparisons) <= 12, "comparison requires 2 to 12 candidate runs")
    first = comparisons[0]
    require(first.evaluation_dataset_sha256 and first.request_document_sha256,
            "candidate comparison requires dataset and request fingerprints")
    require(len({run.candidate.id for run in comparisons}) == len(comparisons),
            "candidate run IDs must be distinct")
    expected = {row.id: (row.reference, row.language, row.baseline, row.audio_sha256, row.speaker_id)
                for row in first.samples}
    for comparison in comparisons:
        require(comparison.model == first.model and comparison.baseline == first.baseline,
                "all candidates must share the same model and baseline run")
        require(not comparison.baseline.differences(comparison.candidate),
                f"{comparison.candidate.id}: run controls differ from the baseline")
        require(comparison.evaluation_dataset_sha256 == first.evaluation_dataset_sha256
                and comparison.request_document_sha256 == first.request_document_sha256,
                f"{comparison.candidate.id}: dataset or request fingerprint differs")
        observed = {row.id: (row.reference, row.language, row.baseline, row.audio_sha256, row.speaker_id)
                    for row in comparison.samples}
        require(expected == observed,
                f"{comparison.candidate.id}: sample IDs, references, speaker IDs, audio fingerprints or baseline outputs differ")


def _percentile(values, fraction):
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def paired_intervals(rows, candidates):
    if len(rows) < MIN_INTERVAL_UNITS:
        return [None] * candidates
    rng = Random(SEED)
    results = [[] for _ in range(candidates)]
    for _ in range(RESAMPLES):
        totals = [0] * candidates
        units = 0
        for _ in rows:
            length, deltas = rows[rng.randrange(len(rows))]
            units += length
            for index, delta in enumerate(deltas):
                totals[index] += delta
        for index, delta in enumerate(totals):
            results[index].append(delta / units)
    return [[_percentile(sorted_values, 0.025), _percentile(sorted_values, 0.975)]
            for sorted_values in map(sorted, results)]


def _relative(deltas, reference_index):
    reference = deltas[reference_index] if reference_index is not None else 0
    return [value - reference for value in deltas]


def _measure(rows, candidate_ids, eligible_count, reference_index, has_speakers):
    units = sum(row.units for row in rows)
    baseline = sum(row.baseline_errors for row in rows)
    groups = {}
    for row in rows:
        if row.speaker_id is not None:
            groups.setdefault(row.speaker_id, []).append(row)
    speaker_groups = []
    speaker_units = []
    for speaker, records in sorted(groups.items()):
        length = sum(row.units for row in records)
        base_errors = sum(row.baseline_errors for row in records)
        deltas = [sum(row.deltas[index] for row in records) for index in range(len(candidate_ids))]
        relative = _relative(deltas, reference_index)
        speaker_units.append((length, relative))
        speaker_groups.append({
            "speaker_id": speaker, "samples": len(records), "reference_units": length,
            "baseline_error_rate": base_errors / length,
            "candidates": {run_id: {"error_rate": (base_errors + deltas[index]) / length,
                                    "delta_vs_reference": relative[index] / length}
                           for index, run_id in enumerate(candidate_ids)},
        })
    common_with_speaker = sum(len(records) for records in groups.values())
    if has_speakers and common_with_speaker < len(rows):
        sampling_unit, paired, interval_status = "speaker", [], "missing_speaker_ids"
    elif has_speakers:
        sampling_unit = "speaker"
        paired = speaker_units
        interval_status = "available" if len(paired) >= MIN_INTERVAL_UNITS else "insufficient_units"
    else:
        sampling_unit = "utterance"
        paired = [(row.units, _relative(row.deltas, reference_index)) for row in rows]
        interval_status = "available" if len(paired) >= MIN_INTERVAL_UNITS else "insufficient_units"
    intervals = paired_intervals(paired, len(candidate_ids))
    deltas = [sum(row.deltas[index] for row in rows) for index in range(len(candidate_ids))]
    relative = _relative(deltas, reference_index)
    scored = []
    for index, run_id in enumerate(candidate_ids):
        delta = deltas[index]
        scored.append({
            "run_id": run_id,
            "error_rate": (baseline + delta) / units if units else None,
            "delta_vs_baseline": delta / units if units else None,
            "delta_vs_reference": relative[index] / units if units else None,
            "interval_95": intervals[index],
        })
    if units:
        scored.sort(key=lambda entry: entry["error_rate"])
    return {
        "eligible_samples": eligible_count,
        "common_samples": len(rows),
        "reference_units": units,
        "baseline_error_rate": baseline / units if units else None,
        "common_with_speaker": common_with_speaker,
        "bootstrap_unit": sampling_unit,
        "bootstrap_units": len(paired),
        "interval_status": interval_status,
        "speaker_groups": speaker_groups,
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
    speaker_metrics = set()
    outcomes = [Counter() for _ in comparisons]
    samples = []
    for row in sorted(first.samples, key=lambda item: item.id):
        pairs = [index[row.id] for index in indices]
        candidates = [pair.candidate for pair in pairs]
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
            if row.speaker_id is not None:
                speaker_metrics.add(metric)
        baseline_edits = edit_distance(reference, tokens(row.baseline.text, row.language)) if common else None
        scores = {}
        for comparison, pair, state in zip(comparisons, pairs, states, strict=True):
            output = pair.candidate
            scores[comparison.candidate.id] = {
                "status": state, "text": output.text if output else None,
                "finding": asdict(diagnose(comparison, pair)),
                "edits": edit_distance(reference, tokens(output.text, row.language)) if common else None,
                "finish_reason": output.finish_reason if output else None,
                "generated_count": output.generated_count if output else None,
                "first_token_id": output.first_token_id if output else None,
                **sample_evidence(pair.baseline, output, expected=pair.audio_sha256),
            }
        if common:
            deltas = [score["edits"] - baseline_edits for score in scores.values()]
            contributions[metric].append(ScoredSample(len(reference), baseline_edits, deltas, row.speaker_id))
        samples.append({
            "id": row.id, "reference": row.reference, "common_scoring": common,
            "speaker_id": row.speaker_id,
            "input_mismatch": not input_match,
            "baseline_text": row.baseline.text if row.baseline else None,
            "baseline_status": classify(row.baseline),
            "baseline_edits": baseline_edits, "reference_units": len(reference) if common else None,
            "metric": metric,
            "candidates": scores,
        })
    report = {
        "schema_version": 1, "kind": "candidate_comparison", "model": first.model,
        "baseline": asdict(first.baseline),
        "reference_run_id": reference_run_id,
        "evaluation_dataset_sha256": first.evaluation_dataset_sha256,
        "request_document_sha256": first.request_document_sha256,
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "minimum_units": MIN_INTERVAL_UNITS},
        "candidates": [{**asdict(comparison.candidate), "outcomes": {
            key: counts[key] for key in ("text", "empty", "failed", "missing")
        }} for comparison, counts in zip(comparisons, outcomes, strict=True)],
        "metrics": {key: _measure(rows, candidate_ids, eligible[key], reference_index, key in speaker_metrics)
                    for key, rows in contributions.items()},
        "samples": samples,
        "input_mismatch_samples": sum(sample["input_mismatch"] for sample in samples),
    }
    if costs is not None:
        report["performance"] = compare_performance(report, costs)
    return report
