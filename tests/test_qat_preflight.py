from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
import wave

from quantdoctor.model import file_sha256
from quantdoctor.qat import QATRecipe, load_qat_inputs, template


def make_qat_fixture(root):
    snapshot = root / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({
        "model_type": "qwen3_asr",
        "thinker_config": {
            "audio_token_id": 151676, "audio_config": {"encoder_layers": 18},
            "text_config": {"hidden_size": 1024, "num_hidden_layers": 28, "vocab_size": 151936,
                            "intermediate_size": 3072, "num_attention_heads": 16,
                            "num_key_value_heads": 8, "head_dim": 128},
        },
    }))
    size = 1024 * 3072 * 2
    header = json.dumps({"thinker.model.layers.27.mlp.down_proj.weight": {
        "shape": [1024, 3072], "dtype": "F16", "data_offsets": [0, size]
    }}).encode()
    with (snapshot / "model.safetensors").open("wb") as stream:
        stream.write(len(header).to_bytes(8, "little") + header)
        stream.truncate(8 + len(header) + size)
    for index, split in enumerate(("calibration", "train", "evaluation")):
        path = root / f"{split}.wav"
        with wave.open(str(path), "wb") as clip:
            clip.setparams((1, 2, 16000, 0, "NONE", ""))
            clip.writeframes(bytes((index, 0)) * 160)
        row = {"id": split, "audio": path.name, "speaker_id": f"spk-{index}"}
        if split == "evaluation":
            document = {"schema_version": 1, "samples": [
                {**row, "split": "evaluation", "text": "开门", "language": "zh"}
            ]}
        else:
            document = [{**row, "audio_sha256": file_sha256(path),
                         "prompt_ids": [1, 151676, 2], "target_ids": [3, 4]}]
        (root / f"{split}.json").write_text(json.dumps(document))
    recipe = root / "qat.json"
    recipe.write_text(json.dumps({**template(), "evaluation_manifest": "evaluation.json",
                                  "model_revision": "fixture-revision"}))
    return recipe, snapshot


class QatPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.recipe, self.snapshot = make_qat_fixture(self.root)

    def check(self):
        recipe = QATRecipe.parse(json.loads(self.recipe.read_text()))
        return load_qat_inputs(recipe, self.recipe, self.snapshot)[2]

    def test_local_input_reports_no_device_verification(self):
        report = self.check()
        self.assertFalse(report["training_verified"])
        self.assertEqual(report["modules"][0]["shape"], [1024, 3072])
        self.assertEqual(report["splits"]["evaluation"]["speaker_count"], 1)
        self.assertEqual(report["splits"]["train"]["audio_header_checked"], 1)
        self.assertIn("evaluation", report["manifest_sha256"])

    def test_holdout_rejects_same_speaker_or_audio(self):
        path = self.root / "evaluation.json"
        document = json.loads(path.read_text())
        document["samples"][0]["speaker_id"] = "spk-1"
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "holdout overlap by speaker_id"):
            self.check()
        document["samples"][0].update(speaker_id="spk-2", audio="calibration.wav")
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "holdout overlap by audio_sha256"):
            self.check()

    def test_incompatible_projection_and_tokens_fail_before_training(self):
        original = json.loads(self.recipe.read_text())
        self.recipe.write_text(json.dumps({**original, "train_modules": ["model.layers.26.mlp.down_proj"]}))
        with self.assertRaisesRegex(ValueError, "checkpoint weight is missing"):
            self.check()
        self.recipe.write_text(json.dumps(original))
        path = self.root / "train.json"
        data = json.loads(path.read_text())
        data[0]["target_ids"] = [151936]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "outside vocabulary"):
            self.check()

    def test_readable_wav_with_wrong_rate_fails(self):
        path = self.root / "train.wav"
        with wave.open(str(path), "wb") as clip:
            clip.setparams((1, 2, 8000, 0, "NONE", ""))
            clip.writeframes(b"\x00\x00" * 160)
        manifest = self.root / "train.json"
        rows = json.loads(manifest.read_text())
        rows[0]["audio_sha256"] = file_sha256(path)
        manifest.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, "requires 16 kHz"):
            self.check()
