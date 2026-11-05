"""Versioned evidence exchanged between the CLI, Web workbench and workers."""

from dataclasses import dataclass
import json
from pathlib import Path


def load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def valid_ids(value):
    return isinstance(value, list) and all(type(token) is int and token >= 0 for token in value)


@dataclass(frozen=True)
class Run:
    id: str
    source: str
    model_revision: str
    environment_id: str
    request_settings_id: str
    precision: dict

    @classmethod
    def parse(cls, data):
        require(isinstance(data, dict), "run must be an object")
        fields = ("id", "source", "model_revision", "environment_id", "request_settings_id")
        require(all(nonempty(data.get(field)) for field in fields), "run provenance is incomplete")
        require(data["source"] in {"fixture", "torch", "target"}, "invalid evidence source")
        precision = data.get("precision")
        require(isinstance(precision, dict) and set(precision) == {"backbone", "audio", "head", "kv"},
                "precision requires backbone, audio, head and kv")
        require(all(v in ("fp16", "bf16", "fp8", "nvfp4", "unknown") for v in precision.values()),
                "invalid effective precision")
        return cls(*(data[field] for field in fields), dict(precision))

    def differences(self, other):
        return [key for key in ("source", "model_revision", "environment_id", "request_settings_id")
                if getattr(self, key) != getattr(other, key)]


@dataclass(frozen=True)
class Output:
    text: str
    raw_text: str | None
    finish_reason: str | None
    generated_count: int | None
    first_token_id: int | None
    stop_token_ids: list[int] | None

    @classmethod
    def parse(cls, data):
        if data is None:
            return None
        require(isinstance(data, dict) and isinstance(data.get("text"), str), "output text is required")
        for key in ("raw_text", "finish_reason"):
            require(data.get(key) is None or isinstance(data[key], str), f"invalid {key}")
        for key in ("generated_count", "first_token_id"):
            value = data.get(key)
            require(value is None or type(value) is int and value >= 0, f"invalid {key}")
        require(data.get("stop_token_ids") is None or valid_ids(data["stop_token_ids"]),
                "invalid stop_token_ids")
        return cls(*(data.get(key) for key in cls.__dataclass_fields__))


@dataclass(frozen=True)
class Sample:
    id: str
    baseline: Output | None
    candidate: Output | None
    reference: str | None
    language: str | None


@dataclass(frozen=True)
class Comparison:
    model: str
    baseline: Run
    candidate: Run
    samples: list[Sample]

    @classmethod
    def parse(cls, data):
        require(isinstance(data, dict) and type(data.get("schema_version")) is int
                and data["schema_version"] == 1, "expected comparison schema_version 1")
        require(nonempty(data.get("model")), "model is required")
        baseline = Run.parse(data.get("baseline"))
        candidate = Run.parse(data.get("candidate"))
        require(baseline.id != candidate.id, "run IDs must differ")
        rows = data.get("samples")
        require(isinstance(rows, list) and bool(rows), "at least one sample is required")
        seen = set()
        samples = []
        for row in rows:
            require(isinstance(row, dict) and nonempty(row.get("id")), "sample id is required")
            require(row["id"] not in seen, f"duplicate sample id {row['id']}")
            seen.add(row["id"])
            reference, language = row.get("reference"), row.get("language")
            require(reference is None or isinstance(reference, str), "reference must be text")
            require(language in (None, "zh", "en"), "language must be en or zh")
            samples.append(Sample(row["id"], Output.parse(row.get("baseline")),
                                  Output.parse(row.get("candidate")), reference, language))
        return cls(data["model"], baseline, candidate, samples)
