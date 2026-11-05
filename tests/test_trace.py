from pathlib import Path
import unittest

from quantdoctor.diagnostics import report
from quantdoctor.edge import import_edge
from quantdoctor.model import Comparison, load_json
from quantdoctor.replay import verify_repair


class TraceTests(unittest.TestCase):
    def setUp(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        self.original = load_json(examples / "demo-comparison.json")
        self.repaired = load_json(examples / "demo-retest.json")

    def first(self):
        return report(Comparison.parse(self.original))["samples"][0]["decode_evidence"][0]

    def test_margin_uses_the_same_reference_token_in_both_runs(self):
        evidence = self.first()
        self.assertEqual(evidence["status"], "aligned")
        self.assertEqual(evidence["reference_id"], 101)
        self.assertTrue(evidence["selected_changed"])
        self.assertTrue(evidence["candidate_stopped"])
        self.assertAlmostEqual(evidence["baseline_margin"], 1.4)
        self.assertAlmostEqual(evidence["candidate_margin"], -0.8)
        replay = verify_repair(Comparison.parse(self.original), Comparison.parse(self.repaired), "ASR-014")
        self.assertAlmostEqual(replay["samples"]["ASR-014"]["decode_evidence"][0]["candidate_margin"], 1.1)

    def test_mismatched_context_never_generates_a_margin(self):
        self.original["samples"][0]["candidate"]["trace"][0]["context_sha256"] = "2" * 64
        evidence = self.first()
        self.assertEqual(evidence["status"], "context_mismatch")
        self.assertNotIn("candidate_margin", evidence)
        self.assertEqual(evidence["candidate_selected"], 23)

    def test_missing_scores_and_score_stage(self):
        entry = self.original["samples"][0]["candidate"]["trace"][0]
        entry["scores"].pop()
        self.assertEqual(self.first()["status"], "missing_scores")
        entry["score_stage"] = "raw_logits"
        self.assertEqual(self.first()["status"], "score_stage_mismatch")

    def test_trace_cannot_contradict_first_token(self):
        self.original["samples"][0]["candidate"]["first_token_id"] = 105
        with self.assertRaisesRegex(ValueError, "first token contradicts"):
            self.first()

    def test_edge_sidecar_requires_request_identity(self):
        manifest = {**self.original, "samples": [{"id": "ASR-014", "request_idx": 0}]}
        output = {"responses": [{"request_idx": 0, "output_text": "", "finish_reason": "end-of-sequence",
                                 "generated_token_count": 1}]}
        trace = {"schema_version": 1, "run_id": self.original["candidate"]["id"],
                 "request_document_sha256": self.original["request_document_sha256"],
                 "samples": {"ASR-014": self.original["samples"][0]["candidate"]["trace"]}}
        normalized = Comparison.parse(import_edge(manifest, output, output, candidate_trace=trace))
        self.assertEqual(normalized.samples[0].candidate.first_token_id, 23)
        trace["request_document_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "request fingerprint"):
            import_edge(manifest, output, output, candidate_trace=trace)
