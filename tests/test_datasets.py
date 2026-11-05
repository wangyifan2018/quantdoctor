from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
import unittest
import wave

from quantdoctor.datasets import calibration_source, load_asr_dataset, prepare_evaluation
from quantdoctor.diagnostics import report
from quantdoctor.edge import import_edge
from quantdoctor.model import Comparison, load_json


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.write_wav("cal.wav", 16_000, b"\x00\x00")
        self.write_wav("eval.wav", 16_000, b"\x01\x00")
        self.manifest = self.root / "dataset.json"
        self.rows = [
            {"id": "c1", "audio": "cal.wav", "text": "calibration text", "split": "calibration", "language": "en"},
            {"id": "e1", "audio": "eval.wav", "text": "开门", "split": "evaluation", "language": "zh"},
        ]
        self.write_manifest()

    def write_wav(self, name, rate, sample):
        with wave.open(str(self.root / name), "wb") as clip:
            clip.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            clip.writeframes(sample * 160)

    def write_manifest(self):
        self.manifest.write_text(json.dumps({"schema_version": 1, "samples": self.rows}))

    def test_calibration_source_only_returns_calibration_and_detects_changes(self):
        dataset = load_asr_dataset(self.manifest)
        generate = calibration_source(self.manifest)
        self.assertEqual(generate.calib_name, "quantdoctor:" + dataset.digest("calibration"))
        self.assertEqual(list(generate()), [((self.root / "cal.wav").read_bytes(), "calibration text")])
        self.write_wav("cal.wav", 16_000, b"\x02\x00")
        with self.assertRaisesRegex(ValueError, "audio changed"):
            list(generate())

    def test_evaluation_bundle_pairs_edge_outputs_without_leaking_references(self):
        metadata = load_json(Path(__file__).resolve().parents[1] / "examples" / "demo-comparison.json")
        request, manifest = prepare_evaluation(load_asr_dataset(self.manifest), metadata, PurePosixPath("/board/asr"))
        self.assertEqual(request["requests"][0]["messages"][1]["content"][0]["audio"], "/board/asr/eval.wav")
        self.assertEqual(request["top_k"], 1)
        self.assertNotIn("开门", json.dumps(request, ensure_ascii=False))
        self.assertEqual(manifest["samples"], [{
            "id": "e1", "request_idx": 0, "reference": "开门", "language": "zh",
            "audio_sha256": sha256((self.root / "eval.wav").read_bytes()).hexdigest(),
        }])
        fp16 = {"responses": [{"request_idx": 0, "output_text": "开门"}]}
        nvfp4 = {"responses": [{"request_idx": 0, "output_text": ""}]}
        result = report(Comparison.parse(import_edge(manifest, fp16, nvfp4)))
        self.assertEqual(result["metrics"]["cer"]["candidate"], 1.0)
        self.assertEqual(result["evaluation_dataset_sha256"], manifest["evaluation_dataset_sha256"])
        expected = sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(result["request_document_sha256"], expected)

    def test_calibration_and_evaluation_reject_identical_audio(self):
        (self.root / "eval.wav").write_bytes((self.root / "cal.wav").read_bytes())
        with self.assertRaisesRegex(ValueError, "both calibration and evaluation"):
            load_asr_dataset(self.manifest)

    def test_invalid_calibration_sample_rate(self):
        self.write_wav("cal.wav", 8000, b"\x00\x00")
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            load_asr_dataset(self.manifest)

    def test_paths_cannot_escape_manifest_directory(self):
        self.rows[0]["audio"] = "../outside.wav"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "stay inside"):
            load_asr_dataset(self.manifest)

    def test_speaker_isolation_and_propagation(self):
        self.assertEqual(load_asr_dataset(self.manifest).summary()["speaker_isolation"], "incomplete")
        self.rows[0]["speaker_id"] = self.rows[1]["speaker_id"] = "speaker-a"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "speaker 'speaker-a' is present"):
            load_asr_dataset(self.manifest)
        self.rows[1]["speaker_id"] = "speaker-b"
        self.write_manifest()
        dataset = load_asr_dataset(self.manifest)
        self.assertEqual(dataset.summary()["speaker_isolation"], "disjoint_declared_ids")
        metadata = load_json(Path(__file__).resolve().parents[1] / "examples" / "demo-comparison.json")
        _, prepared = prepare_evaluation(dataset, metadata)
        self.assertEqual(prepared["samples"][0]["speaker_id"], "speaker-b")
        self.rows[1]["speaker_id"] = "speaker-c"
        self.write_manifest()
        self.assertNotEqual(dataset.digest("evaluation"), load_asr_dataset(self.manifest).digest("evaluation"))


if __name__ == "__main__":
    unittest.main()
