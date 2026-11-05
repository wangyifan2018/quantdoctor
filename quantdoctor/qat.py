"""Platform-independent contract for a selective Qwen3-ASR QAT job."""

from dataclasses import dataclass
import math
from pathlib import Path
import re

from .diagnostics import ASR_MODEL
from .model import SHA256, file_sha256, load_json, nonempty, parse_speaker_id, require, valid_ids

EDGE_COMMIT = "1231d6415ce38d0b2f20b3308ac24e58ec4100a8"
MODULE = re.compile(r"model\.layers\.([0-9]+)\.(?:mlp\.(?:gate|up|down)_proj|self_attn\.(?:q|k|v|o)_proj)")


@dataclass(frozen=True)
class QATRecipe:
    model_revision: str
    backbone: str
    train_modules: tuple[str, ...]
    calibration_manifest: str
    train_manifest: str
    steps: int
    learning_rate: float
    seed: int

    @classmethod
    def parse(cls, data):
        require(isinstance(data, dict) and type(data.get("schema_version")) is int
                and data["schema_version"] == 1, "expected QAT schema_version 1")
        require(data.get("model") == ASR_MODEL and data.get("edge_commit") == EDGE_COMMIT,
                "this worker requires Qwen3-ASR-0.6B and its pinned Edge-LLM commit")
        require(data.get("backbone") in ("fp8", "nvfp4"), "QAT backbone must be fp8 or nvfp4")
        for key in ("model_revision", "calibration_manifest", "train_manifest"):
            require(nonempty(data.get(key)), f"QAT requires {key}")
        modules = data.get("train_modules")
        require(isinstance(modules, list) and 1 <= len(modules) <= 14,
                "select 1 to 14 decoder projections")
        for module in modules:
            match = MODULE.fullmatch(module) if isinstance(module, str) else None
            require(match and str(int(match[1])) == match[1] and int(match[1]) < 28,
                    f"invalid Qwen3-ASR-0.6B projection {module!r}")
        require(len(set(modules)) == len(modules), "duplicate train_modules")
        steps, lr, seed = (data.get(key) for key in ("steps", "learning_rate", "seed"))
        require(type(steps) is int and 1 <= steps <= 10_000, "steps must be in 1..10000")
        require(type(lr) in (int, float) and 0 < lr <= 0.01 and math.isfinite(lr),
                "learning_rate must be positive and <= 0.01")
        require(type(seed) is int and 0 <= seed < 2**32, "invalid seed")
        return cls(data["model_revision"], data["backbone"], tuple(modules),
                   data["calibration_manifest"], data["train_manifest"], steps, float(lr), seed)


@dataclass(frozen=True)
class TrainingExample:
    id: str
    audio: Path
    audio_sha256: str
    prompt_ids: list[int]
    target_ids: list[int]
    speaker_id: str | None = None

    def teacher_forcing_ids(self):
        return self.prompt_ids + self.target_ids[:-1]


def read_examples(path, limit, audio_token_id, vocab_size):
    rows = load_json(path)
    require(isinstance(rows, list) and 1 <= len(rows) <= limit,
            f"{path.name} must have 1 to {limit} samples")
    samples = []
    seen = set()
    audio_hashes = {}
    for row in rows:
        require(isinstance(row, dict) and nonempty(row.get("id")), "sample id is required")
        sample_id = row["id"]
        require(sample_id not in seen, f"duplicate training sample ID {sample_id}")
        seen.add(sample_id)
        prompt, target = row.get("prompt_ids"), row.get("target_ids")
        require(valid_ids(prompt) and 2 <= len(prompt) <= 512 and prompt.count(audio_token_id) == 1
                and prompt[-1] != audio_token_id, f"{sample_id}: invalid prompt_ids/audio placeholder")
        require(valid_ids(target) and 1 <= len(target) <= 256 and audio_token_id not in target,
                f"{sample_id}: invalid target_ids")
        require(max(prompt + target) < vocab_size, f"{sample_id}: token outside vocabulary")
        audio = row.get("audio")
        digest = row.get("audio_sha256")
        require(nonempty(audio) and isinstance(digest, str) and SHA256.fullmatch(digest),
                f"{sample_id}: local audio path and SHA-256 are required")
        audio_path = (path.parent / audio).resolve()
        if audio_path not in audio_hashes:
            audio_hashes[audio_path] = file_sha256(audio_path)
        require(audio_hashes[audio_path] == digest, f"{sample_id}: audio SHA-256 mismatch")
        samples.append(TrainingExample(sample_id, audio_path, digest, prompt, target,
                                       parse_speaker_id(row.get("speaker_id"))))
    return samples


def require_disjoint(calibration, training):
    for key in ("id", "audio_sha256", "speaker_id"):
        overlap = ({getattr(row, key) for row in calibration} & {getattr(row, key) for row in training}) - {None}
        require(not overlap, f"calibration/training overlap by {key}")


def template():
    return {
        "schema_version": 1, "model": ASR_MODEL, "model_revision": "<HF revision>",
        "edge_commit": EDGE_COMMIT, "backbone": "nvfp4",
        "train_modules": ["model.layers.27.mlp.down_proj"],
        "calibration_manifest": "calibration.json", "train_manifest": "train.json",
        "steps": 30, "learning_rate": 0.00001, "seed": 17,
    }
