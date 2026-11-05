"""Create tiny synthetic checkpoints and a fingerprint-bound replay fixture."""

import argparse
import json
from pathlib import Path

from quantdoctor.checkpoint import inspect_checkpoint

ROOT = Path(__file__).resolve().parent
BACKBONE = "thinker.model.layers.0.mlp.up_proj.weight"
AUDIO = "thinker.audio_tower.layers.0.fc1.weight"


def write_checkpoint(directory, audio_dtype, audio_payload):
    directory.mkdir()
    (directory / "config.json").write_text('{"model_type":"qwen3_asr"}\n', encoding="utf-8")
    header = {
        BACKBONE: {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
        AUDIO: {"dtype": audio_dtype, "shape": [1], "data_offsets": [1, 1 + len(audio_payload)]},
    }
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    (directory / "model.safetensors").write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"A" + audio_payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    for run, template, dtype, payload in (
        ("original", "demo-comparison.json", "F8_E4M3", b"B"),
        ("repaired", "demo-retest.json", "F16", b"CD"),
    ):
        checkpoint = args.output / run
        write_checkpoint(checkpoint, dtype, payload)
        data = json.loads((ROOT / template).read_text(encoding="utf-8"))
        data["candidate"]["checkpoint_sha256"] = inspect_checkpoint(checkpoint, hash_weights=True)["checkpoint_sha256"]
        (args.output / f"{run}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
