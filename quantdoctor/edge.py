"""Import the ordinary Edge-LLM llm_inference JSON result format."""

from .model import require


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


def import_edge(manifest, baseline, candidate):
    require(isinstance(manifest, dict) and isinstance(manifest.get("samples"), list),
            "manifest requires a samples array")
    baseline_by_index, candidate_by_index = _responses(baseline), _responses(candidate)
    rows = []
    indices = set()
    for row in manifest["samples"]:
        require(isinstance(row, dict), "invalid sample manifest entry")
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
    return {**manifest, "samples": rows}
