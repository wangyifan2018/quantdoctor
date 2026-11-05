"""Import the ordinary Edge-LLM llm_inference JSON result format."""

from .model import SHA256, nonempty, require


def _responses(document):
    require(isinstance(document, dict) and isinstance(document.get("responses"), list),
            "Edge output requires a responses array")
    result = {}
    for entry in document["responses"]:
        require(isinstance(entry, dict), "invalid Edge response")
        index = entry.get("request_idx")
        require(type(index) is int and index >= 0 and index not in result,
                "request_idx must be a unique nonnegative integer")
        require(isinstance(entry.get("output_text"), str), f"missing output_text for {index}")
        result[index] = {
            "text": entry["output_text"],
            "finish_reason": entry.get("finish_reason"),
            "generated_count": entry.get("generated_token_count"),
        }
    return result


def _attach_trace(rows, side, manifest, trace):
    require(isinstance(trace, dict) and type(trace.get("schema_version")) is int
            and trace["schema_version"] == 1 and isinstance(trace.get("samples"), dict),
            "decode trace requires schema_version 1 and a samples object")
    run = manifest.get(side)
    require(isinstance(run, dict) and run.get("id") and trace.get("run_id") == run["id"],
            f"{side} trace does not match the run ID")
    fingerprint = manifest.get("request_document_sha256")
    require(isinstance(fingerprint, str) and SHA256.fullmatch(fingerprint)
            and trace.get("request_document_sha256") == fingerprint,
            f"{side} trace requires the matching request fingerprint")
    by_id = {row["id"]: row[side] for row in rows}
    for sample_id, steps in trace["samples"].items():
        output = by_id.get(sample_id)
        require(output is not None, f"{side} trace has no response for {sample_id}")
        output["trace"] = steps


def import_edge(manifest, baseline, candidate, *, baseline_trace=None, candidate_trace=None):
    require(isinstance(manifest, dict) and isinstance(manifest.get("samples"), list),
            "manifest requires a samples array")
    baseline_by_index, candidate_by_index = _responses(baseline), _responses(candidate)
    rows = []
    indices = set()
    for row in manifest["samples"]:
        require(isinstance(row, dict) and nonempty(row.get("id")), "manifest sample ID is required")
        index = row.get("request_idx")
        require(type(index) is int and index >= 0 and index not in indices,
                "manifest request_idx must be a unique nonnegative integer")
        indices.add(index)
        rows.append({
            "id": row.get("id"), "reference": row.get("reference"), "language": row.get("language"),
            "baseline": baseline_by_index.get(index), "candidate": candidate_by_index.get(index),
        })
    require(not (baseline_by_index.keys() | candidate_by_index.keys()) - indices,
            "Edge outputs include request indices absent from the manifest")
    for side, trace in (("baseline", baseline_trace), ("candidate", candidate_trace)):
        if trace is not None:
            _attach_trace(rows, side, manifest, trace)
    return {**manifest, "samples": rows}
