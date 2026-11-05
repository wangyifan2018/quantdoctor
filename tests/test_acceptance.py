from pathlib import Path
import unittest

from quantdoctor.acceptance import evaluate_acceptance
from quantdoctor.model import Comparison, load_json
from quantdoctor.quality import compare_candidates


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        self.runs = [load_json(examples / filename) for filename in ("demo-comparison.json", "demo-retest.json")]
        for run in self.runs:
            run["samples"] = [
                {"id": f"sample-{index}", "reference": "open door", "language": "en", "speaker_id": f"s{index}",
                 "baseline": {"text": "open door"}, "candidate": {"text": "open door"}}
                for index in range(12)
            ]
        first = self.runs[0]
        self.costs = {
            "schema_version": 1, "protocol_id": "fixture", "source": "fixture",
            "environment_id": first["baseline"]["environment_id"],
            "evaluation_dataset_sha256": first["evaluation_dataset_sha256"],
            "request_document_sha256": first["request_document_sha256"],
            "runs": {
                first["baseline"]["id"]: {"rtf_p95": 1, "peak_memory_bytes": 1000, "artifact_bytes": 1000},
                first["candidate"]["id"]: {"rtf_p95": 1.05, "peak_memory_bytes": 900, "artifact_bytes": 1000},
                self.runs[1]["candidate"]["id"]: {"rtf_p95": 1.10, "peak_memory_bytes": 900},
            },
        }
        self.policy = {key: self.costs[key] for key in ("schema_version", "protocol_id", "environment_id",
                                                       "evaluation_dataset_sha256", "request_document_sha256")}
        self.policy.update(policy_id="fixture", quality={"wer": 1}, min_common_samples=10,
                           min_coverage=0.9, max_empty_increase_pp=0.5, max_unavailable_rate=0,
                           max_rtf_increase_pct=5, min_resource_savings_pct=10)

    def evaluate(self, use_costs=True, reference=None):
        report = compare_candidates([Comparison.parse(run) for run in self.runs], reference,
                                    costs=self.costs if use_costs else None)
        return evaluate_acceptance(report, self.policy)

    def test_fixture_can_meet_policy_without_verifying_deployment(self):
        outcome = self.evaluate()
        self.assertFalse(outcome["deployment_verified"])
        self.assertEqual(outcome["source"], "fixture")
        self.assertEqual(outcome["candidates"][0]["status"], "meets_policy")
        self.assertEqual(outcome["candidates"][1]["status"], "outside_policy")
        checks = {check["code"]: check for check in outcome["candidates"][0]["checks"]}
        self.assertEqual(checks["wer_increase_pp"]["interval_95"], [0, 0])
        self.assertEqual(checks["resource_savings_pct"]["observed"], 10)

    def test_unavailable_costs_or_utterance_only_intervals_are_insufficient(self):
        self.assertEqual(self.evaluate(use_costs=False)["candidates"][0]["status"], "insufficient_evidence")
        for run in self.runs:
            for row in run["samples"]:
                row.pop("speaker_id")
        outcome = self.evaluate()["candidates"][0]
        self.assertEqual(outcome["status"], "insufficient_evidence")
        self.assertIn("说话人", outcome["checks"][0]["reason"])

    def test_quality_interval_crossing_limit_remains_indeterminate(self):
        self.runs[0]["samples"][0]["candidate"]["text"] = "open"
        result = self.evaluate()["candidates"][0]
        check = result["checks"][0]
        self.assertEqual(check["status"], "insufficient")
        self.assertEqual(result["status"], "insufficient_evidence")

    def test_rejects_different_measurement_protocol_or_reference(self):
        with self.assertRaisesRegex(ValueError, "FP16 baseline as reference"):
            self.evaluate(reference=self.runs[0]["candidate"]["id"])
        self.policy["protocol_id"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "performance protocol"):
            self.evaluate()
