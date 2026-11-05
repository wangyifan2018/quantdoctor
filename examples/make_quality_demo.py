"""Generate synthetic multi-candidate ASR comparisons for the local dashboard."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from quantdoctor.model import load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    original = load_json(root / "demo-comparison.json")
    repaired = load_json(root / "demo-retest.json")
    fp8 = deepcopy(repaired)
    fp8["candidate"]["id"] = "fp8-fp16-audio-demo"
    fp8["candidate"]["precision"]["backbone"] = "fp8"

    for index in range(24):
        language = "zh" if index < 12 else "en"
        reference = f"切换到频道{index}。" if language == "zh" else f"Switch to channel {index}."
        truncated = "切换频道。" if language == "zh" else "Switch channels."
        for run, prediction in (
            (original, "" if index % 4 == 0 else truncated if index % 5 == 0 else reference),
            (repaired, truncated if index % 4 == 0 else reference),
            (fp8, "" if index % 9 == 0 else reference),
        ):
            run["samples"].append({"id": f"sim-{index:02d}", "reference": reference, "language": language,
                                   "baseline": {"text": reference}, "candidate": {"text": prediction}})

    args.output.mkdir()
    for name, run in (("original", original), ("repaired", repaired), ("fp8", fp8)):
        for field in ("evaluation_dataset_sha256", "request_document_sha256"):
            run[field] = sha256(("synthetic-quality-v1:" + field).encode()).hexdigest()
        (args.output / f"{name}.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    costs = {
        "schema_version": 1, "protocol_id": "synthetic-warmup-then-repeat",
        "source": "fixture", "environment_id": original["baseline"]["environment_id"],
        "evaluation_dataset_sha256": original["evaluation_dataset_sha256"],
        "request_document_sha256": original["request_document_sha256"],
        "runs": {
            run["candidate"]["id"]: {"rtf_p95": rtf, "peak_memory_bytes": memory * 1048576,
                                      "artifact_bytes": size * 1048576}
            for run, rtf, memory, size in (
                (original, 0.19, 1440, 550), (repaired, 0.22, 1520, 630), (fp8, 0.21, 1680, 760)
            )
        },
    }
    (args.output / "costs.json").write_text(json.dumps(costs, indent=2) + "\n")


if __name__ == "__main__":
    main()
