from pathlib import Path
import json
from tempfile import TemporaryDirectory
import unittest

from quantdoctor.checkpoint import audit_checkpoints, inspect_checkpoint


def write_shard(path, tensors):
    header = {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.original, self.repaired = self.root / "original", self.root / "repaired"
        for directory in (self.original, self.repaired):
            directory.mkdir()
            (directory / "config.json").write_text(json.dumps({
                "model_type": "qwen3_asr",
                "thinker_config": {"text_config": {"tie_word_embeddings": True}},
            }))
        self.backbone = ("thinker.model.layers.0.mlp.up_proj.weight", "U8", [1, 1], b"A")
        self.audio = ("thinker.audio_tower.layers.0.fc1.weight", "F8_E4M3", [1, 1], b"B")
        self.audio_scale = ("thinker.audio_tower.layers.0.fc1.weight_scale", "F32", [], b"1234")
        self.float_audio = (self.audio[0], "F16", [1, 1], b"CD")
        write_shard(self.original / "model.safetensors", [self.backbone, self.audio, self.audio_scale])
        write_shard(self.repaired / "model.safetensors", [self.backbone, self.float_audio])

    def test_headers_classify_payload_and_declared_quantization_separately(self):
        (self.original / "hf_quant_config.json").write_text('{"quantization":{"quant_algo":"MIXED_PRECISION"}}')
        inventory = inspect_checkpoint(self.original)
        self.assertTrue(inventory["tied_word_embeddings"])
        self.assertEqual(inventory["declared_quantization"], "MIXED_PRECISION")
        self.assertFalse(inventory["payload_hashed"])
        self.assertEqual(inventory["components"]["audio"]["dtype_bytes"], {"F8_E4M3": 1, "F32": 4})
        self.assertEqual(inventory["components"]["audio"]["scale_count"], 1)
        self.assertNotIn("sha256", inventory["tensors"][0])

    def test_audit_matches_audio_only_and_detects_same_shape_weight_changes(self):
        audit = audit_checkpoints(self.original, self.repaired, "audio")
        self.assertTrue(audit["matches_expectation"])
        self.assertEqual(audit["changed_components"], ["audio"])
        self.assertEqual(len(audit["changed_tensors"]), 2)
        changed_backbone = (*self.backbone[:3], b"Z")
        write_shard(self.repaired / "model.safetensors", [changed_backbone, self.float_audio])
        audit = audit_checkpoints(self.original, self.repaired, "audio")
        self.assertFalse(audit["matches_expectation"])
        self.assertEqual(audit["changed_components"], ["audio", "backbone"])

    def test_shard_index_and_data_validation(self):
        (self.repaired / "model.safetensors").unlink()
        write_shard(self.repaired / "part1.safetensors", [self.float_audio])
        write_shard(self.repaired / "part2.safetensors", [self.backbone])
        index = self.repaired / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {self.audio[0]: "part1.safetensors", self.backbone[0]: "part2.safetensors"}}))
        audit = audit_checkpoints(self.original, self.repaired, "audio")
        self.assertTrue(audit["matches_expectation"])
        write_shard(self.repaired / "part1.safetensors", [(*self.float_audio[:3], b"C")])
        with self.assertRaisesRegex(ValueError, "byte size mismatch"):
            inspect_checkpoint(self.repaired)
