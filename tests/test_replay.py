from pathlib import Path
import unittest

from quantdoctor.model import Comparison, load_json
from quantdoctor.replay import verify_repair


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.original = load_json(EXAMPLES / "demo-comparison.json")
        self.repaired = load_json(EXAMPLES / "demo-retest.json")
        self.costs = load_json(EXAMPLES / "demo-costs.json")

    def replay(self, costs=None):
        return verify_repair(Comparison.parse(self.original), Comparison.parse(self.repaired), "ASR-014", costs)

    def test_replay_reports_recovered_and_regressed_on_common_cohort(self):
        report = self.replay(self.costs)
        self.assertEqual(report["source"], "fixture")
        self.assertEqual(report["deployment_acceptance"], "not_assessed")
        self.assertEqual(report["change"], {"precision.audio": "fp16"})
        self.assertEqual(report["counts"]["recovered_text"], 2)
        self.assertEqual(report["counts"]["new_empty"], 1)
        self.assertEqual(report["counts"]["original_unavailable"], 1)
        self.assertEqual(report["samples"]["ASR-039"]["edits"]["repaired"], 2)
        self.assertEqual(report["metrics"]["wer"]["paired"], 1)
        self.assertEqual(report["metrics"]["wer"]["repaired"], 1.0)
        self.assertEqual(report["metrics"]["cer"]["paired"], 3)
        self.assertLess(report["metrics"]["cer"]["repaired"], report["metrics"]["cer"]["original"])
        self.assertAlmostEqual(report["costs"]["measurements"]["rtf_p95"]["delta"], 0.03)

    def test_repair_fails_if_another_component_changed(self):
        self.repaired["candidate"]["precision"]["head"] = "fp8"
        with self.assertRaisesRegex(ValueError, "expected only"):
            self.replay()

    def test_repair_requires_same_request_and_anchor(self):
        self.repaired["request_document_sha256"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "matching request_document"):
            self.replay()
        self.repaired["request_document_sha256"] = self.original["request_document_sha256"]
        self.repaired["samples"][0]["baseline"]["text"] = "different"
        with self.assertRaisesRegex(ValueError, "baseline output changed"):
            self.replay()

    def test_failed_retest_is_not_recovery_or_scored(self):
        self.repaired["samples"][0]["candidate"]["finish_reason"] = "error"
        report = self.replay()
        self.assertEqual(report["samples"]["ASR-014"]["transition"], "repair_failed")
        self.assertEqual(report["counts"]["repair_failed"], 1)
        self.assertEqual(report["metrics"]["cer"]["paired"], 2)

    def test_costs_require_matching_run_and_valid_measurement(self):
        self.costs["repaired"]["run_id"] = "another"
        with self.assertRaisesRegex(ValueError, "run ID"):
            self.replay(self.costs)
        self.costs["repaired"]["run_id"] = self.repaired["candidate"]["id"]
        self.costs["original"]["peak_memory_bytes"] = -1
        with self.assertRaisesRegex(ValueError, "invalid costs"):
            self.replay(self.costs)
