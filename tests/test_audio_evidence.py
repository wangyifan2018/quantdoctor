from pathlib import Path
import unittest

from quantdoctor.diagnostics import report
from quantdoctor.edge import import_edge
from quantdoctor.model import Comparison, load_json
from quantdoctor.replay import verify_repair


class AudioEvidenceTests(unittest.TestCase):
    def setUp(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        self.original = load_json(examples / "demo-comparison.json")
        self.repaired = load_json(examples / "demo-retest.json")

    def sample(self):
        return report(Comparison.parse(self.original))["samples"][0]

    def test_encoder_difference_is_localized_with_paired_sketch(self):
        row = self.sample()
        self.assertEqual([entry["status"] for entry in row["audio_evidence"]], ["same", "same", "different"])
        self.assertEqual(row["boundary"], {"stage": "encoder", "earlier_evidence_complete": True})
        self.assertAlmostEqual(row["audio_evidence"][2]["sketch"]["max_abs_error"], 0.11)
        result = verify_repair(Comparison.parse(self.original), Comparison.parse(self.repaired), "ASR-014")
        self.assertIsNone(result["samples"]["ASR-014"]["boundary"])
        self.assertEqual(result["samples"]["ASR-014"]["audio_evidence"][2]["status"], "same")

    def test_input_mismatch_stops_attribution_and_excludes_score(self):
        self.original["samples"][0]["candidate"]["audio"]["input_sha256"] = "1" * 64
        row = self.sample()
        self.assertEqual(row["finding"]["code"], "input_mismatch")
        self.assertIsNone(row["finding"]["action"])
        self.assertEqual(row["boundary"]["stage"], "audio_input")
        self.assertEqual(row["decode_evidence"][0]["status"], "input_mismatch")
        self.assertEqual(report(Comparison.parse(self.original))["metrics"]["cer"]["paired"], 2)

    def test_missing_or_incompatible_preprocessing_keeps_gap_visible(self):
        self.original["samples"][0]["candidate"]["audio"]["mel"]["representation"] = "other-layout"
        row = self.sample()
        self.assertEqual(row["audio_evidence"][1]["status"], "representation_mismatch")
        self.assertFalse(row["boundary"]["earlier_evidence_complete"])
        self.original["samples"][0]["candidate"]["audio"]["encoder"]["sketch"] = [[99999999, 1.0]]
        with self.assertRaisesRegex(ValueError, "indices"):
            self.sample()

    def test_repair_conflicting_audio_is_not_counted_as_recovery(self):
        self.repaired["samples"][0]["candidate"]["audio"]["input_sha256"] = "1" * 64
        result = verify_repair(Comparison.parse(self.original), Comparison.parse(self.repaired), "ASR-014")
        self.assertEqual(result["samples"]["ASR-014"]["transition"], "input_mismatch")
        self.assertEqual(result["samples"]["ASR-014"]["edit_effect"], "unscored")
        self.assertEqual(result["counts"]["input_mismatch"], 1)

    def test_both_runs_can_conflict_with_expected_audio(self):
        row = self.original["samples"][0]
        row["candidate"]["audio"]["input_sha256"] = row["baseline"]["audio"]["input_sha256"]
        row["audio_sha256"] = "2" * 64
        observed = self.sample()
        self.assertEqual(observed["finding"]["code"], "input_mismatch")
        self.assertEqual(observed["boundary"]["stage"], "audio_input")

    def test_sidecar_joins_using_run_and_request(self):
        manifest = {**self.original, "samples": [{"id": "ASR-014", "request_idx": 0}]}
        response = {"responses": [{"request_idx": 0, "output_text": ""}]}
        audio = {"schema_version": 1, "run_id": manifest["candidate"]["id"],
                 "request_document_sha256": manifest["request_document_sha256"],
                 "samples": {"ASR-014": self.original["samples"][0]["candidate"]["audio"]}}
        joined = Comparison.parse(import_edge(manifest, response, response, candidate_audio=audio))
        self.assertEqual(joined.samples[0].candidate.audio.encoder.shape, (13, 1024))
        audio["run_id"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "run ID"):
            import_edge(manifest, response, response, candidate_audio=audio)
