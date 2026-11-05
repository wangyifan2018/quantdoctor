"""Generate synthetic multi-candidate ASR comparisons for the local dashboard."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from quantdoctor.model import load_json


def _fixture_hash(value):
    return sha256(value.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--holdout", action="store_true", help="Separate the work-order sample from the evaluation cohort")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    original = load_json(root / "demo-comparison.json")
    repaired = load_json(root / "demo-retest.json")
    fp8 = deepcopy(repaired)
    fp8["candidate"]["id"] = "fp8-fp16-audio-demo"
    fp8["candidate"]["precision"]["backbone"] = "fp8"

    for run in (original, repaired, fp8):
        for sample in run["samples"]:
            sample["speaker_id"] = "demo-" + sample["id"]
    for index in range(48):
        language = "zh" if index < 24 else "en"
        reference = f"切换到频道{index}。" if language == "zh" else f"Switch to channel {index}."
        truncated = "切换频道。" if language == "zh" else "Switch channels."
        restored = truncated if index % 4 == 0 else reference
        if args.holdout and index in (7, 31):
            restored = ""
        for run, prediction in (
            (original, "" if index % 4 == 0 else truncated if index % 5 == 0 else reference),
            (repaired, restored),
            (fp8, "" if index % 9 == 0 else reference),
        ):
            run["samples"].append({"id": f"sim-{index:02d}", "speaker_id": f"speaker-{index // 2:02d}",
                                   "reference": reference, "language": language,
                                   "baseline": {"text": reference}, "candidate": {"text": prediction}})

    args.output.mkdir()
    fields = ("evaluation_dataset_sha256", "request_document_sha256")
    tag = "synthetic-holdout" if args.holdout else "synthetic-quality-speakers-v2"
    if args.holdout:
        diagnosis = {**original, "samples": [row for row in original["samples"] if row["id"].startswith("ASR-")]}
        for row in diagnosis["samples"]:
            audio = (row["baseline"] or {}).get("audio") or {}
            row["audio_sha256"] = audio.get("input_sha256") or _fixture_hash("diagnostic-audio:" + row["id"])
        for field in fields:
            diagnosis[field] = _fixture_hash("synthetic-diagnosis:" + field)
        (args.output / "diagnosis.json").write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n")
    for name, run in (("original", original), ("repaired", repaired), ("fp8", fp8)):
        if args.holdout:
            run["samples"] = [row for row in run["samples"] if row["id"].startswith("sim-")]
            for row in run["samples"]:
                row["audio_sha256"] = _fixture_hash("holdout-audio:" + row["id"])
            for role in ("baseline", "candidate"):
                run[role]["id"] = "holdout-" + run[role]["id"]
        for field in fields:
            run[field] = _fixture_hash(tag + ":" + field)
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
    costs["runs"][original["baseline"]["id"]] = {
        "rtf_p95": 0.20, "peak_memory_bytes": 1850 * 1048576, "artifact_bytes": 950 * 1048576,
    }
    (args.output / "costs.json").write_text(json.dumps(costs, indent=2) + "\n")
    policy = {key: costs[key] for key in ("schema_version", "protocol_id", "environment_id",
                                          "evaluation_dataset_sha256", "request_document_sha256")}
    policy.update(policy_id="fixture-asr-gate", quality={"cer": 1.0, "wer": 1.0},
                  min_common_samples=20, min_coverage=0.95, max_empty_increase_pp=0.5,
                  max_unavailable_rate=0.0, max_rtf_increase_pct=5.0, min_resource_savings_pct=10.0)
    (args.output / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")


if __name__ == "__main__":
    main()
