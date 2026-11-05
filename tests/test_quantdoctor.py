import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from quantdoctor.diagnostics import report, work_order
from quantdoctor.edge import import_edge
from quantdoctor.model import Comparison, file_sha256, load_json
from quantdoctor.qat import QATRecipe, read_examples, require_disjoint, template

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo-comparison.json"


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.bundle = load_json(DEMO)

    def report(self):
        return report(Comparison.parse(self.bundle))

    def test_empty_output_paths(self):
        outcome = self.report()
        cases = [row["finding"] for row in outcome["samples"]]
        self.assertEqual([case["code"] for case in cases],
                         ["first_step_eos", "cleanup_empty", "empty_unknown", "transcribed", "missing"])
        self.assertEqual(cases[0]["action"]["change"], {"precision.audio": "fp16"})
        self.assertIsNone(cases[2]["action"])
        self.assertEqual(outcome["outcomes"]["candidate"],
                         {"empty": 3, "completed": 4, "missing": 1, "failed": 0})
        self.assertEqual(outcome["metrics"]["wer"]["paired"], 1)
        self.assertEqual(outcome["metrics"]["wer"]["candidate"], 0.5)
        self.assertEqual(outcome["metrics"]["cer"]["candidate"], 1.0)

    def test_mismatched_runs_are_not_compared(self):
        self.bundle["candidate"]["environment_id"] = "another-board"
        outcome = self.report()
        self.assertFalse(outcome["comparable"])
        self.assertEqual(outcome["metrics"], {})
        self.assertTrue(all(row["finding"]["code"] == "unpaired" for row in outcome["samples"]))

    def test_work_order_has_provenance_and_requires_evidence(self):
        comparison = Comparison.parse(self.bundle)
        order = work_order(comparison, "ASR-014")
        self.assertEqual(order["status"], "pending_verification")
        self.assertEqual(order["candidate"]["id"], "nvfp4-demo")
        self.assertEqual(order["finding"]["action"]["change"], {"precision.audio": "fp16"})
        with self.assertRaisesRegex(ValueError, "no proposed intervention"):
            work_order(comparison, "ASR-039")

    def test_eos_without_token_data_has_no_location(self):
        row = self.bundle["samples"][0]["candidate"]
        row.pop("first_token_id")
        row.pop("stop_token_ids")
        row.pop("trace")
        self.bundle["candidate"]["precision"].update(audio="fp16", head="nvfp4")
        finding = self.report()["samples"][0]["finding"]
        self.assertEqual(finding["code"], "eos_unlocated")
        self.assertEqual(finding["action"]["change"], {"precision.head": "fp16"})

    def test_runtime_failure_and_empty_baseline_take_precedence(self):
        self.bundle["samples"][0]["candidate"]["finish_reason"] = "error"
        self.bundle["samples"][2]["baseline"]["text"] = ""
        outcome = self.report()
        self.assertEqual(outcome["samples"][0]["finding"]["code"], "failed")
        self.assertEqual(outcome["samples"][2]["finding"]["code"], "baseline_unusable")

    def test_import_edge_preserves_missing_and_refuses_unmapped_indices(self):
        self.bundle["samples"] = [{"id": "A", "request_idx": 2}, {"id": "B", "request_idx": 1}]
        baseline = {"responses": [{"request_idx": 2, "output_text": "a"}]}
        candidate = {"responses": [{"request_idx": 1, "output_text": "", "finish_reason": "end-of-sequence"}]}
        result = Comparison.parse(import_edge(self.bundle, baseline, candidate))
        self.assertIsNone(result.samples[0].candidate)
        self.assertEqual(result.samples[1].candidate.finish_reason, "end-of-sequence")
        candidate["responses"][0]["request_idx"] = 9
        with self.assertRaisesRegex(ValueError, "absent from the manifest"):
            import_edge(self.bundle, baseline, candidate)


class QATContractTests(unittest.TestCase):
    def test_selective_projection_scope(self):
        data = template()
        self.assertEqual(QATRecipe.parse(data).train_modules, ("model.layers.27.mlp.down_proj",))
        for invalid in ("lm_head", "model.layers.28.self_attn.k_proj", "model.layers.*"):
            data["train_modules"] = [invalid]
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                QATRecipe.parse(data)

    def test_teacher_forcing_and_split_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "test.wav"
            audio.write_bytes(b"synthetic fixture")
            row = {"id": "a", "audio": audio.name, "audio_sha256": file_sha256(audio),
                   "prompt_ids": [1, 3, 4], "target_ids": [5, 6, 7]}
            path = root / "dataset.json"
            path.write_text(json.dumps([row]))
            train = read_examples(path, 3, 3, 9)
            self.assertEqual(train[0].teacher_forcing_ids(), [1, 3, 4, 5, 6])
            row["id"] = "different-id"
            path.write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, "overlap by audio_sha256"):
                require_disjoint(read_examples(path, 3, 3, 9), train)
            audio.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                read_examples(path, 3, 3, 9)


if __name__ == "__main__":
    unittest.main()
