from copy import deepcopy
from pathlib import Path
import unittest

from quantdoctor.model import Comparison, load_json
from quantdoctor.quality import compare_candidates


class QualityTests(unittest.TestCase):
    def setUp(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        self.original = load_json(examples / "demo-comparison.json")
        self.repaired = load_json(examples / "demo-retest.json")

    def compare(self, *runs):
        return compare_candidates([Comparison.parse(run) for run in runs])

    def test_common_cohort_and_coverage_do_not_hide_missing_results(self):
        report = self.compare(self.original, self.repaired)
        self.assertEqual(report["metrics"]["wer"]["eligible_samples"], 2)
        self.assertEqual(report["metrics"]["wer"]["common_samples"], 1)
        self.assertEqual(report["metrics"]["wer"]["candidates"][1]["error_rate"], 1.0)
        self.assertEqual(report["candidates"][1]["outcomes"]["missing"], 1)
        self.assertIsNone(report["metrics"]["wer"]["candidates"][0]["interval_95"])

    def test_paired_interval_is_deterministic_and_uses_shared_reference_lengths(self):
        for index in range(20):
            reference = "open door" if index % 2 == 0 else "door"
            row = {"id": f"test-{index}", "language": "en", "reference": reference,
                   "baseline": {"text": reference}}
            self.original["samples"].append({**row, "candidate": {"text": ""}})
            self.repaired["samples"].append({**row, "candidate": {"text": reference}})
        first = self.compare(self.original, self.repaired)["metrics"]["wer"]
        reordered = deepcopy(self.repaired)
        reordered["samples"].reverse()
        self.assertEqual(first, self.compare(self.original, reordered)["metrics"]["wer"])
        self.assertEqual(first["common_samples"], 21)
        for row in first["candidates"]:
            lo, hi = row["interval_95"]
            self.assertLessEqual(lo, row["delta_vs_baseline"])
            self.assertGreaterEqual(hi, row["delta_vs_baseline"])

    def test_rejects_control_mismatch_and_duplicate_run(self):
        with self.assertRaisesRegex(ValueError, "run IDs must be distinct"):
            self.compare(self.original, self.original)
        self.repaired["request_document_sha256"] = "2" * 64
        with self.assertRaisesRegex(ValueError, "dataset or request fingerprint differs"):
            self.compare(self.original, self.repaired)

    def test_can_measure_the_repair_against_the_original_candidate(self):
        runs = [Comparison.parse(run) for run in (self.original, self.repaired)]
        report = compare_candidates(runs, reference_run_id=runs[0].candidate.id)
        original, repaired = report["metrics"]["wer"]["candidates"]
        self.assertEqual(original["delta_vs_reference"], 0)
        self.assertEqual(repaired["delta_vs_baseline"], 1)
        self.assertEqual(repaired["delta_vs_reference"], 0.5)
        with self.assertRaisesRegex(ValueError, "reference run"):
            compare_candidates(runs, "unknown")
