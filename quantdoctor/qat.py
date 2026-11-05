"""Platform-independent contract for a selective Qwen3-ASR QAT job."""

from dataclasses import dataclass
import math
from pathlib import Path
import re
import wave

from .checkpoint import inspect_checkpoint
from .datasets import load_asr_dataset
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
    evaluation_manifest: str | None = None

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
        evaluation = data.get("evaluation_manifest")
        require(evaluation is None or nonempty(evaluation), "evaluation_manifest must be a local manifest path")
        return cls(data["model_revision"], data["backbone"], tuple(modules),
                   data["calibration_manifest"], data["train_manifest"], steps, float(lr), seed, evaluation)


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


def _snapshot_config(snapshot):
    require(not (snapshot / "hf_quant_config.json").exists(), "QAT starts from a float HF snapshot")
    config = load_json(snapshot / "config.json")
    require(isinstance(config, dict) and config.get("model_type") == "qwen3_asr"
            and not config.get("quantization_config"), "expected float Qwen3-ASR snapshot")
    thinker = config.get("thinker_config")
    require(isinstance(thinker, dict), "missing Qwen3-ASR thinker config")
    text, audio = thinker.get("text_config"), thinker.get("audio_config")
    require(isinstance(text, dict) and isinstance(audio, dict)
            and (text.get("num_hidden_layers"), text.get("hidden_size"), audio.get("encoder_layers"))
            == (28, 1024, 18), "expected Qwen3-ASR-0.6B architecture")
    dimensions = ("intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim")
    require(tuple(text.get(key) for key in dimensions) == (3072, 16, 8, 128),
            "expected Qwen3-ASR-0.6B projection dimensions")
    token_id, vocab = thinker.get("audio_token_id"), text.get("vocab_size")
    require(type(token_id) is int and type(vocab) is int and 0 <= token_id < vocab,
            "missing valid audio_token_id or vocab_size")
    return token_id, vocab


def _readable_wav_count(samples):
    checked = 0
    for sample in samples:
        if sample.audio.suffix.lower() != ".wav":
            continue
        try:
            with wave.open(str(sample.audio), "rb") as clip:
                require(clip.getframerate() == 16000 and 0 < clip.getnframes() <= 320000,
                        f"{sample.id}: QAT requires 16 kHz audio of at most 20 seconds")
                checked += 1
        except (wave.Error, EOFError):
            # SoundFile on the worker supports encodings beyond the stdlib PCM reader.
            continue
    return checked


def _split_summary(samples):
    wav_count = _readable_wav_count(samples)
    return {
        "count": len(samples),
        "speaker_count": len({row.speaker_id for row in samples if row.speaker_id is not None}),
        "missing_speaker_ids": sum(row.speaker_id is None for row in samples),
        "audio_header_checked": wav_count,
        "audio_decode_pending": len(samples),
    }


def load_qat_inputs(recipe, recipe_path, snapshot):
    """Validate local model/data contracts without importing a training runtime."""
    token_id, vocab = _snapshot_config(snapshot)
    inventory = inspect_checkpoint(snapshot)
    tensors = {row["name"]: row for row in inventory["tensors"]}
    shapes = {"gate_proj": [3072, 1024], "up_proj": [3072, 1024], "down_proj": [1024, 3072],
              "q_proj": [2048, 1024], "k_proj": [1024, 1024], "v_proj": [1024, 1024], "o_proj": [1024, 2048]}
    audited_modules = []
    for module in recipe.train_modules:
        tensor = tensors.get(f"thinker.{module}.weight")
        require(tensor and tensor["dtype"] in ("F16", "BF16", "F32")
                and tensor["shape"] == shapes[module.rsplit(".", 1)[-1]],
                f"{module}: floating-point checkpoint weight is missing or shape is incompatible")
        audited_modules.append({"name": module, "dtype": tensor["dtype"], "shape": tensor["shape"]})
    manifests = {split: (recipe_path.parent / filename).resolve()
                 for split, filename in (("calibration", recipe.calibration_manifest), ("train", recipe.train_manifest))}
    calibration = read_examples(manifests["calibration"], 128, token_id, vocab)
    training = read_examples(manifests["train"], 4096, token_id, vocab)
    require_disjoint(calibration, training)
    fingerprints = {name: file_sha256(path) for name, path in manifests.items()}
    warnings = []
    splits = {"calibration": _split_summary(calibration), "train": _split_summary(training)}
    if recipe.evaluation_manifest is not None:
        evaluation = load_asr_dataset((recipe_path.parent / recipe.evaluation_manifest).resolve())
        holdout = evaluation.select("evaluation")
        require(holdout, "evaluation_manifest needs an evaluation split")
        examples = calibration + training
        for field, left, right in (
            ("id", {row.id for row in examples}, {row.id for row in holdout}),
            ("audio_sha256", {row.audio_sha256 for row in examples}, {row.sha256 for row in holdout}),
            ("speaker_id", {row.speaker_id for row in examples}, {row.speaker_id for row in holdout}),
        ):
            require(not (left & right) - {None}, f"evaluation holdout overlap by {field}")
        splits["evaluation"] = {"count": len(holdout),
                                "speaker_count": len({row.speaker_id for row in holdout if row.speaker_id is not None}),
                                "missing_speaker_ids": sum(row.speaker_id is None for row in holdout)}
        fingerprints["evaluation"] = evaluation.digest("evaluation")
    else:
        warnings.append("未提供评测清单，留出集隔离待核对。")
    if any(split["missing_speaker_ids"] for split in splits.values()):
        warnings.append("部分样本缺少说话人 ID，无法完整核对说话人隔离。")
    if any(split["audio_header_checked"] < split["count"] for name, split in splits.items() if name != "evaluation"):
        warnings.append("部分训练或校准音频未完成本机 WAV 头部检查。")
    report = {
        "schema_version": 1, "kind": "qat_preflight", "model": ASR_MODEL,
        "model_revision": recipe.model_revision, "status": "local_checks_passed",
        "backbone": recipe.backbone, "steps": recipe.steps, "audio_token_id": token_id, "vocab_size": vocab,
        "splits": splits, "manifest_sha256": fingerprints, "modules": audited_modules, "warnings": warnings,
        "training_verified": False, "deployment_verified": False,
    }
    return calibration, training, report


def template():
    return {
        "schema_version": 1, "model": ASR_MODEL, "model_revision": "<HF revision>",
        "edge_commit": EDGE_COMMIT, "backbone": "nvfp4",
        "train_modules": ["model.layers.27.mlp.down_proj"],
        "calibration_manifest": "calibration.json", "train_manifest": "train.json",
        "steps": 30, "learning_rate": 0.00001, "seed": 17,
    }
