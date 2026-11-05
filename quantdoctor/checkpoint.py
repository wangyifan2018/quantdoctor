"""Read local Hugging Face safetensors metadata without loading a model."""

from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path

from .model import load_json, nonempty, require

MAX_HEADER = 100_000_000
BITS = {
    "BOOL": 8, "I8": 8, "U8": 8, "I16": 16, "U16": 16, "I32": 32, "U32": 32,
    "I64": 64, "U64": 64, "F16": 16, "BF16": 16, "F32": 32, "F64": 64,
    "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8,
}
SCALE_SUFFIXES = {
    "weight_scale", "weight_scale_2", "input_scale", "input_scale_2", "pre_quant_scale", "scales",
    "q_scale", "k_scale", "v_scale",
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate safetensors JSON key {key}")
        result[key] = value
    return result


def _read_shard(path, hash_weights):
    with path.open("rb") as stream:
        file_size = os.fstat(stream.fileno()).st_size
        prefix = stream.read(8)
        require(len(prefix) == 8, f"{path.name}: missing safetensors header size")
        size = int.from_bytes(prefix, "little")
        require(0 < size <= MAX_HEADER and size <= file_size - 8, f"{path.name}: invalid safetensors header size")
        encoded = stream.read(size)
        require(encoded.startswith(b"{") and len(encoded) == size, f"{path.name}: invalid safetensors header")
        data = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
        payload_start = 8 + size
        payload_size = file_size - payload_start
        metadata = data.pop("__metadata__", {})
        require(isinstance(metadata, dict) and all(isinstance(value, str) for value in metadata.values()),
                f"{path.name}: invalid safetensors metadata")
        entries = []
        for name, value in data.items():
            require(nonempty(name) and isinstance(value, dict), f"{path.name}: invalid tensor entry")
            dtype, shape, offsets = (value.get(key) for key in ("dtype", "shape", "data_offsets"))
            require(nonempty(dtype) and isinstance(shape, list)
                    and all(type(dim) is int and dim >= 0 for dim in shape), f"{name}: invalid dtype or shape")
            require(isinstance(offsets, list) and len(offsets) == 2 and all(type(n) is int for n in offsets)
                    and 0 <= offsets[0] <= offsets[1] <= payload_size, f"{name}: invalid offsets")
            start, end = offsets
            if dtype in BITS:
                require(prod(shape) * BITS[dtype] == (end - start) * 8, f"{name}: dtype/shape byte size mismatch")
            entries.append((start, end, {"name": name, "dtype": dtype, "shape": shape, "bytes": end - start}))
        position = 0
        for start, end, row in sorted(entries, key=lambda entry: entry[:2]):
            require(start == position, f"{path.name}: overlapping or unindexed payload")
            if hash_weights:
                stream.seek(payload_start + start)
                digest = sha256()
                remaining = end - start
                while remaining:
                    chunk = stream.read(min(remaining, 1024 * 1024))
                    require(chunk, f"{path.name}: truncated tensor payload")
                    digest.update(chunk)
                    remaining -= len(chunk)
                row["sha256"] = digest.hexdigest()
            position = end
        require(position == payload_size, f"{path.name}: trailing unindexed payload")
    return [row for _, _, row in entries]


def _shards(snapshot):
    single = snapshot / "model.safetensors"
    if single.is_file():
        return [single], None
    index = load_json(snapshot / "model.safetensors.index.json")
    mapping = index.get("weight_map") if isinstance(index, dict) else None
    require(isinstance(mapping, dict) and mapping and all(
        isinstance(name, str) and Path(name).name == name and name.endswith(".safetensors")
        for name in mapping.values()
    ), "invalid safetensors weight_map")
    return [snapshot / name for name in sorted(set(mapping.values()))], mapping


def _component(name):
    short = name.removeprefix("thinker.")
    for prefix, group in (("audio_tower.", "audio"), ("lm_head.", "head"), ("model.", "backbone")):
        if short.startswith(prefix):
            return group
    return "other"


def _object(path):
    data = load_json(path)
    require(isinstance(data, dict), f"{path.name}: expected a JSON object")
    return data


def _digest(items):
    return sha256(json.dumps(items, ensure_ascii=True, separators=(",", ":")).encode("ascii")).hexdigest()


def inspect_checkpoint(snapshot: Path, *, hash_weights=False):
    config = _object(snapshot / "config.json")
    quant_file = snapshot / "hf_quant_config.json"
    quant = _object(quant_file).get("quantization") if quant_file.is_file() else config.get("quantization_config")
    algo = quant.get("quant_algo") or quant.get("quant_method") if isinstance(quant, dict) else None
    thinker = config.get("thinker_config")
    text = thinker.get("text_config") if isinstance(thinker, dict) else config
    tied = text.get("tie_word_embeddings") if isinstance(text, dict) else None
    files, weight_map = _shards(snapshot)
    observed = {}
    tensors = []
    for shard in files:
        for row in _read_shard(shard, hash_weights):
            name = row["name"]
            require(name not in observed, f"duplicate checkpoint tensor {name}")
            observed[name] = shard.name
            tensors.append({**row, "component": _component(name), "shard": shard.name,
                            "scale": name.rpartition(".")[2] in SCALE_SUFFIXES})
    require(tensors, "checkpoint has no tensors")
    require(weight_map is None or weight_map == observed, "weight_map and shard tensors disagree")
    tensors.sort(key=lambda row: row["name"])
    components = {}
    for row in tensors:
        group = components.setdefault(row["component"], {"tensor_count": 0, "scale_count": 0, "bytes": 0, "dtype_bytes": {}})
        group["tensor_count"] += 1
        group["scale_count"] += int(row["scale"])
        group["bytes"] += row["bytes"]
        group["dtype_bytes"][row["dtype"]] = group["dtype_bytes"].get(row["dtype"], 0) + row["bytes"]
    if hash_weights:
        for name, group in components.items():
            group["tensor_content_sha256"] = _digest([
                [row["name"], row["dtype"], row["shape"], row["sha256"]]
                for row in tensors if row["component"] == name
            ])
    return {"schema_version": 1, "kind": "checkpoint", "model_type": config.get("model_type"),
            "declared_quantization": algo, "tied_word_embeddings": tied,
            "payload_hashed": hash_weights, "components": components, "tensors": tensors}


def audit_checkpoints(before: Path, after: Path, expected_component=None):
    original = inspect_checkpoint(before, hash_weights=True)
    candidate = inspect_checkpoint(after, hash_weights=True)
    require(nonempty(original["model_type"]) and original["model_type"] == candidate["model_type"],
            "checkpoint audit requires the same model_type")
    left = {row["name"]: row for row in original["tensors"]}
    right = {row["name"]: row for row in candidate["tensors"]}
    changed = []
    for name in sorted(left.keys() | right.keys()):
        first, second = left.get(name), right.get(name)
        fields = ("dtype", "shape", "sha256")
        if first is None or second is None or any(first[field] != second[field] for field in fields):
            changed.append({"name": name, "component": (first or second)["component"],
                            "original": first, "candidate": second})
    components = sorted({row["component"] for row in changed})
    tie_changed = original["tied_word_embeddings"] != candidate["tied_word_embeddings"]
    return {
        "schema_version": 1, "kind": "checkpoint_audit", "model_type": original["model_type"],
        "changed_components": components, "changed_tensors": changed,
        "original_components": original["components"], "components": candidate["components"],
        "tied_word_embeddings": candidate["tied_word_embeddings"], "tie_declaration_changed": tie_changed,
        "declared_quantization": candidate["declared_quantization"],
        "expected_component": expected_component, "payload_hashed": True,
        "matches_expectation": (
            components == [expected_component] and not tie_changed if expected_component is not None else None
        ),
    }
