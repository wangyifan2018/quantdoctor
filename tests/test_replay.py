from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import unittest

from quantdoctor.model import Comparison, load_json, scoring_sha256
from quantdoctor.replay import verify_repair
from quantdoctor.request_protocol import RequestProtocol


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
        self.assertEqual(report["edit_counts"], {"fewer_edits": 2, "more_edits": 1, "same_edits": 1, "unscored": 1})
        self.assertEqual(report["samples"]["ASR-039"]["edits"]["repaired"], 2)
        self.assertEqual(report["metrics"]["wer"]["paired"], 1)
        self.assertEqual(report["metrics"]["wer"]["repaired"], 1.0)
        self.assertEqual(report["metrics"]["cer"]["paired"], 3)
        self.assertLess(report["metrics"]["cer"]["repaired"], report["metrics"]["cer"]["original"])
        self.assertAlmostEqual(report["costs"]["measurements"]["rtf_p95"]["delta"], 0.03)

    def test_recovered_text_can_have_more_reference_edits(self):
        for run in (self.original, self.repaired):
            run["samples"][0]["reference"] = "开"
        self.repaired["samples"][0]["candidate"]["text"] = "完全无关"
        row = self.replay()["samples"]["ASR-014"]
        self.assertEqual(row["transition"], "recovered_text")
        self.assertEqual(row["edit_effect"], "more_edits")
        self.assertEqual(row["edits"], {"original": 1, "repaired": 4, "reference_units": 1})

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
        self.assertEqual(report["samples"]["ASR-014"]["edit_effect"], "unscored")
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


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.diagnosis = load_json(EXAMPLES / "demo-comparison.json")
        self.original = deepcopy(self.diagnosis)
        self.repaired = load_json(EXAMPLES / "demo-retest.json")
        for row in self.diagnosis["samples"]:
            captured = (row["baseline"] or {}).get("audio", {}).get("input_sha256")
            row["audio_sha256"] = captured or sha256(row["id"].encode()).hexdigest()
            row["speaker_id"] = "diagnosis-speaker"
        for run, text in ((self.original, "open"), (self.repaired, "open door")):
            run["evaluation_dataset_sha256"] = "a" * 64
            run["request_document_sha256"] = "b" * 64
            for side in ("baseline", "candidate"):
                run[side]["id"] = "holdout-" + run[side]["id"]
            run["samples"] = [
                {"id": f"clip-{index}", "speaker_id": f"speaker-{index % 10}", "reference": "open door", "language": "en",
                 "audio_sha256": sha256(f"clip-{index}".encode()).hexdigest(),
                 "baseline": {"text": "open door"}, "candidate": {"text": text}}
                for index in range(12)
            ]

    def replay(self, costs=None):
        return verify_repair(Comparison.parse(self.original), Comparison.parse(self.repaired), "ASR-014",
                             costs, diagnosis=Comparison.parse(self.diagnosis))

    def test_holdout_separates_anchor_and_counts_speakers_in_interval(self):
        result = self.replay()
        self.assertEqual(result["work_order"]["sample_id"], "ASR-014")
        self.assertNotIn("ASR-014", result["samples"])
        self.assertEqual(result["work_order"]["candidate"]["id"], "nvfp4-demo")
        self.assertEqual(result["original_run"]["id"], "holdout-nvfp4-demo")
        self.assertEqual(result["holdout"]["evaluation_samples"], 12)
        self.assertEqual(result["holdout"]["evaluation_speakers"], 10)
        self.assertEqual(result["holdout"]["controls"]["candidate"]["matched_artifacts"], {})
        self.assertEqual(result["metrics"]["wer"]["interval_95"], [-0.5, -0.5])
        self.assertEqual(result["metrics"]["wer"]["speaker_count"], 10)
        self.assertEqual(result["metrics"]["wer"]["delta_vs_original"], -0.5)
        self.assertIsNone(result["metrics"]["cer"]["delta_vs_original"])
        self.assertEqual(result["deployment_acceptance"], "not_assessed")
        for run in (self.original, self.repaired):
            run["samples"] = [row for row in run["samples"] if row["speaker_id"] != "speaker-9"]
        low_coverage = self.replay()["metrics"]["wer"]
        self.assertEqual(low_coverage["speaker_count"], 9)
        self.assertIsNone(low_coverage["interval_95"])

    def test_holdout_rejects_leakage_even_with_distinct_dataset_fingerprints(self):
        for field in ("id", "audio_sha256", "speaker_id"):
            with self.subTest(field=field):
                original = self.original["samples"][0][field]
                for run in (self.original, self.repaired):
                    run["samples"][0][field] = self.diagnosis["samples"][0][field]
                with self.assertRaisesRegex(ValueError, f"overlap by {field}"):
                    self.replay()
                for run in (self.original, self.repaired):
                    run["samples"][0][field] = original
        self.diagnosis["samples"][0].pop("speaker_id")
        with self.assertRaisesRegex(ValueError, "complete speaker_id"):
            self.replay()

    def test_holdout_rejects_captured_audio_conflict(self):
        self.repaired["samples"][0]["candidate"]["audio"] = {"input_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "captured audio"):
            self.replay()

    def test_holdout_detects_decode_drift_even_when_declared_settings_ids_match(self):
        request = {"requests": [{"messages": [{"role": "user", "content": [{"type": "audio", "audio": "clip.wav"}]}]}],
                   "top_k": 1}
        for comparison in (self.diagnosis, self.original, self.repaired):
            comparison["request_protocol"] = asdict(RequestProtocol.from_document(request))
        self.assertEqual(self.replay()["holdout"]["request_protocol"]["status"], "matching_files")
        self.diagnosis["request_protocol"] = asdict(RequestProtocol.from_document({**request, "max_generate_length": 1}))
        with self.assertRaisesRegex(ValueError, "request file settings differ from diagnosis: max_generate_length"):
            self.replay()

    def test_holdout_rejects_different_diagnosis_settings_or_artifacts(self):
        for role in ("baseline", "candidate"):
            self.diagnosis[role]["request_settings_id"] = "different-decode"
        with self.assertRaisesRegex(ValueError, "baseline differs from diagnosis: request_settings_id"):
            self.replay()
        for role in ("baseline", "candidate"):
            self.diagnosis[role]["request_settings_id"] = self.original[role]["request_settings_id"]
        for run in (self.diagnosis, self.original):
            run["candidate"]["worker_evidence"] = {
                "profile_id": run["candidate"]["environment_id"], "source": "fixture", "engine_sha256": "e" * 64,
                "job_id": "1" * 64, "request_artifact_sha256": "2" * 64, "response_artifact_sha256": "3" * 64,
                "request_document_sha256": run["request_document_sha256"],
                "evaluation_dataset_sha256": run["evaluation_dataset_sha256"],
                "scoring_sha256": scoring_sha256(Comparison.parse(run).samples, "candidate"),
            }
        controls = self.replay()["holdout"]["controls"]
        self.assertEqual(controls["candidate"]["matched_artifacts"], {"engine_sha256": "e" * 64})
        self.diagnosis["candidate"]["worker_evidence"]["engine_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "engine_sha256 differs from diagnosis"):
            self.replay()

    def test_holdout_costs_require_cohort_fingerprints(self):
        costs = load_json(EXAMPLES / "demo-costs.json")
        with self.assertRaisesRegex(ValueError, "costs require matching evaluation_dataset_sha256"):
            self.replay(costs)
        for field in ("evaluation_dataset_sha256", "request_document_sha256"):
            costs[field] = self.original[field]
        for role in ("original", "repaired"):
            costs[role]["run_id"] = "holdout-" + costs[role]["run_id"]
        self.assertAlmostEqual(self.replay(costs)["costs"]["measurements"]["rtf_p95"]["delta"], 0.03)
