"""Opt-in, version-bound Linux/CUDA adapter. The local application never imports ML dependencies."""

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from .model import file_sha256, load_json, require
from .qat import EDGE_COMMIT, QATRecipe, read_examples, require_disjoint


def _check_worker(edge_repo):
    require(sys.platform == "linux", "QAT requires a Linux NVIDIA worker")
    commit = subprocess.check_output(["git", "-C", str(edge_repo), "rev-parse", "HEAD"], text=True).strip()
    require(commit == EDGE_COMMIT, f"QAT requires Edge-LLM {EDGE_COMMIT}")
    changed = subprocess.check_output(
        ["git", "-C", str(edge_repo), "status", "--porcelain", "--", "tensorrt_edgellm"], text=True)
    require(not changed, "use a clean Edge-LLM Python source tree for QAT")
    versions = {name: version(name) for name in ("nvidia-modelopt", "transformers", "torch", "soundfile", "safetensors")}
    for name, expected in (("nvidia-modelopt", "0.45.0"), ("transformers", "5.14.1"), ("torch", "2.13.0")):
        require(versions[name].split("+")[0] == expected, f"QAT requires {name}=={expected}")
    return versions


def _load_data(recipe, recipe_path, snapshot):
    require(not (snapshot / "hf_quant_config.json").exists(), "QAT starts from a float HF snapshot")
    config = load_json(snapshot / "config.json")
    require(config.get("model_type") == "qwen3_asr" and not config.get("quantization_config"),
            "expected float Qwen3-ASR snapshot")
    thinker = config.get("thinker_config") or {}
    text, audio = thinker.get("text_config", {}), thinker.get("audio_config", {})
    require((text.get("num_hidden_layers"), text.get("hidden_size"), audio.get("encoder_layers"))
            == (28, 1024, 18), "expected Qwen3-ASR-0.6B architecture")
    token_id, vocab = thinker.get("audio_token_id"), text.get("vocab_size")
    require(type(token_id) is int and type(vocab) is int and 0 <= token_id < vocab,
            "missing valid audio_token_id or vocab_size")
    manifests = {
        "calibration": (recipe_path.parent / recipe.calibration_manifest).resolve(),
        "train": (recipe_path.parent / recipe.train_manifest).resolve(),
    }
    calibration = read_examples(manifests["calibration"], 128, token_id, vocab)
    training = read_examples(manifests["train"], 4096, token_id, vocab)
    require_disjoint(calibration, training)
    return calibration, training, {name: file_sha256(path) for name, path in manifests.items()}


def _weight_files(snapshot):
    single = snapshot / "model.safetensors"
    if single.is_file():
        return [single]
    index = load_json(snapshot / "model.safetensors.index.json")
    mapping = index.get("weight_map") if isinstance(index, dict) else None
    require(isinstance(mapping, dict) and mapping and all(
        isinstance(name, str) and Path(name).name == name and name.endswith(".safetensors")
        for name in mapping.values()
    ), "invalid checkpoint shard names")
    shards = sorted(set(mapping.values()))
    return [snapshot / name for name in shards]


def _snapshot_fingerprint(snapshot, weights):
    files = set(weights)
    for pattern in ("*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken", "*.py"):
        files.update(snapshot.glob(pattern))
    checksums = {path.name: file_sha256(path) for path in sorted(files) if path.is_file()}
    return sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest()


def _audit_float_weights(model, weights, safe_open):
    shapes = {}
    for shard in weights:
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():
                require(name not in shapes, f"duplicate checkpoint tensor {name}")
                shapes[name] = tuple(handle.get_slice(name).get_shape())
    for name, parameter in model.named_parameters():
        require(shapes.get(f"thinker.{name}") == tuple(parameter.shape),
                f"missing or mismatched float checkpoint parameter thinker.{name}")


def _audit_export(checkpoint, recipe, safe_open, torch):
    with safe_open(checkpoint / "model.safetensors", framework="pt") as handle:
        keys = set(handle.keys())
        for module in recipe.train_modules:
            prefix = f"thinker.{module}"
            require({prefix + ".weight", prefix + ".weight_scale"} <= keys,
                    f"{module}: packed weight or scale is missing")
            weight = handle.get_tensor(prefix + ".weight")
            expected = (torch.float8_e4m3fn,) if recipe.backbone == "fp8" else (torch.uint8, torch.int8)
            require(weight.dtype in expected, f"{module}: unexpected packed dtype {weight.dtype}")
        require(not any(
            key.endswith(".weight_scale")
            and key.startswith(("thinker.audio_tower.", "thinker.lm_head."))
            for key in keys
        ), "exported Audio and LM Head must remain unquantized")
    return sorted(recipe.train_modules)


def _batches(model, torch, soundfile, extractor, prepare):
    def batch(example, training):
        with soundfile.SoundFile(example.audio) as clip:
            require(clip.samplerate == 16000 and 0 < clip.frames <= 320000,
                    f"{example.id}: expected 16 kHz audio up to 20 seconds")
            pcm = clip.read(dtype="float32", always_2d=True).mean(axis=1)
        features = extractor(pcm, sampling_rate=16000, return_tensors="pt",
                             return_attention_mask=True, padding=False, truncation=False)
        lengths = features.attention_mask.sum(dim=-1).to(torch.int64)
        ragged = prepare(features.input_features, lengths, n_window=model.audio_tower.config["n_window"],
                         dtype=torch.float32)
        ids = example.teacher_forcing_ids() if training else example.prompt_ids + example.target_ids
        return {
            "input_ids": torch.tensor([ids], device="cuda"),
            "audio_padded_feature": ragged["padded_feature"].to("cuda", dtype=torch.float16),
            "audio_indices": ragged["padded_mask_after_cnn_indices"].to("cuda"),
            "audio_attn_mask": ragged["attention_mask"].to("cuda", dtype=torch.float16),
            "logits_to_keep": len(example.target_ids) if training else 1,
        }
    return batch


def _train(model, recipe, examples, batch, torch):
    model.eval()
    model.requires_grad_(False)
    pairs = []
    for name in recipe.train_modules:
        module = model.get_submodule(name)
        quantizer = getattr(module, "weight_quantizer", None)
        require(quantizer is not None and quantizer.is_enabled and quantizer.fake_quant,
                f"selected projection must have an enabled fake weight quantizer: {name}")
        weight = module.weight
        require(weight.is_floating_point(), f"{name}: expected trainable float weight")
        weight.requires_grad_(True)
        pairs.append((weight, torch.nn.Parameter(weight.detach().float().clone())))

    masters = [master for _, master in pairs]
    optimizer = torch.optim.AdamW(masters, lr=recipe.learning_rate, weight_decay=0)
    losses = []
    order = list(range(len(examples)))
    rng = random.Random(recipe.seed)
    for step in range(recipe.steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        example = examples[order[step % len(order)]]
        optimizer.zero_grad(set_to_none=True)
        for weight, _ in pairs:
            weight.grad = None
        logits = model(**batch(example, training=True)).logits
        targets = torch.tensor(example.target_ids, device="cuda")
        require(logits.shape[:2] == (1, len(example.target_ids)), "teacher-forcing alignment failed")
        loss = torch.nn.functional.cross_entropy(logits[0].float(), targets)
        require(torch.isfinite(loss).item(), f"nonfinite QAT loss at step {step + 1}")
        loss.backward()
        for weight, master in pairs:
            require(weight.grad is not None, "selected projection has no gradient")
            master.grad = weight.grad.float()
        torch.nn.utils.clip_grad_norm_(masters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for weight, master in pairs:
                weight.copy_(master)
        losses.append({"step": step + 1, "sample_id": example.id, "loss": float(loss.detach())})
    return losses


def run(recipe_path, snapshot, edge_repo, destination):
    recipe_doc = load_json(recipe_path)
    recipe = QATRecipe.parse(recipe_doc)
    versions = _check_worker(edge_repo)
    require(not destination.exists(), "QAT destination already exists")
    require(destination.parent.is_dir(), "QAT destination parent must exist")

    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from safetensors import safe_open
    import soundfile
    import torch
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    require(torch.cuda.is_available(), "CUDA is unavailable")
    sys.path.insert(0, str(edge_repo))
    import tensorrt_edgellm
    require(Path(tensorrt_edgellm.__file__).resolve().is_relative_to(edge_repo),
            "Edge-LLM import did not resolve to the pinned checkout")
    from tensorrt_edgellm.quantization.models.qwen3_asr.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )
    from tensorrt_edgellm.quantization.models.qwen3_asr.modeling_qwen3_asr_audio import prepare_audio_inputs
    from tensorrt_edgellm.quantization.quantization_configs import build_quant_config
    from tensorrt_edgellm.quantization.quantize import (
        _fix_generation_config_for_strict_validate, _normalize_tied_weights_keys,
    )
    from tensorrt_edgellm.quantization.qwen3_asr_loader import postprocess_qwen3_asr_checkpoint

    calibration, training, manifest_hashes = _load_data(recipe, recipe_path, snapshot)
    weights = _weight_files(snapshot)
    snapshot_id = _snapshot_fingerprint(snapshot, weights)
    torch.manual_seed(recipe.seed)
    torch.cuda.manual_seed_all(recipe.seed)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(str(snapshot), torch.float16, "cuda")
    _audit_float_weights(model, weights, safe_open)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160,
                                        n_fft=400, return_attention_mask=True, padding_value=0.0)
    batch = _batches(model, torch, soundfile, extractor, prepare_audio_inputs)
    config = build_quant_config(recipe.backbone, None, None)

    @torch.no_grad()
    def calibrate(current):
        for example in calibration:
            current(**batch(example, training=False))

    model = mtq.quantize(model, config, forward_loop=calibrate)
    losses = _train(model, recipe, training, batch, torch)
    _fix_generation_config_for_strict_validate(model)
    _normalize_tied_weights_keys(model)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary) / "result"
        checkpoint = stage / "checkpoint"
        checkpoint.mkdir(parents=True)
        with torch.inference_mode():
            export_hf_checkpoint(model, dtype=torch.float16, export_dir=str(checkpoint))
        tokenizer.save_pretrained(checkpoint)
        for name in ("generation_config.json", "processor_config.json", "preprocessor_config.json", "chat_template.jinja"):
            if (snapshot / name).is_file() and not (checkpoint / name).exists():
                shutil.copy2(snapshot / name, checkpoint / name)
        postprocess_qwen3_asr_checkpoint(str(snapshot), str(checkpoint))
        require((checkpoint / "hf_quant_config.json").is_file(), "ModelOpt quantization metadata is missing")
        audited = _audit_export(checkpoint, recipe, safe_open, torch)
        report = {
            "schema_version": 1, "status": "checkpoint_exported", "target_validation": "pending",
            "recipe": recipe_doc, "snapshot_sha256": snapshot_id, "dataset_sha256": manifest_hashes,
            "versions": versions, "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
            "platform": platform.platform(), "training_steps": losses, "audited_modules": audited,
        }
        (stage / "qat-result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        stage.rename(destination)
