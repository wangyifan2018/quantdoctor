"""Versioned evidence exchanged between the CLI, Web workbench and workers."""

from dataclasses import dataclass
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .audio_evidence import AudioEvidence
    from .request_protocol import RequestProtocol
    from .trace import DecodeStep


SHA256 = re.compile(r"[a-f0-9]{64}")


def load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def parse_speaker_id(value):
    require(value is None or nonempty(value) and value == value.strip(),
            "speaker_id must be a nonempty string without surrounding whitespace")
    return value


def valid_ids(value):
    return isinstance(value, list) and all(type(token) is int and token >= 0 for token in value)


def file_sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def document_sha256(document):
    canonical = json.dumps(document, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return sha256(canonical.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class InvocationEvidence:
    artifact_sha256: str
    binary_sha256: str
    exit_code: int
    audio_checked: int
    audio_manifest_checked: int

    @classmethod
    def parse(cls, value):
        if value is None:
            return None
        require(isinstance(value, dict) and value.keys() == cls.__dataclass_fields__.keys(),
                "invalid Infer invocation evidence fields")
        require(all(isinstance(value[key], str) and SHA256.fullmatch(value[key])
                    for key in ("artifact_sha256", "binary_sha256")), "invalid Infer invocation evidence digest")
        require(type(value["exit_code"]) is int and value["exit_code"] in (0, 1)
                and type(value["audio_checked"]) is int and value["audio_checked"] > 0
                and type(value["audio_manifest_checked"]) is int
                and 0 <= value["audio_manifest_checked"] <= value["audio_checked"],
                "invalid Infer invocation status or audio counts")
        return cls(**value)


@dataclass(frozen=True)
class WorkerEvidence:
    job_id: str
    profile_id: str
    source: str
    engine_sha256: str
    request_artifact_sha256: str
    response_artifact_sha256: str
    request_document_sha256: str
    evaluation_dataset_sha256: str
    scoring_sha256: str
    request_protocol_sha256: str | None = None
    invocation: InvocationEvidence | None = None

    @classmethod
    def parse(cls, value):
        if value is None:
            return None
        required = cls.__dataclass_fields__.keys() - {"request_protocol_sha256", "invocation"}
        require(isinstance(value, dict) and required <= value.keys() <= cls.__dataclass_fields__.keys(),
                "invalid worker evidence fields")
        require(nonempty(value["profile_id"]) and value["source"] in ("fixture", "worker"),
                "invalid worker evidence profile or source")
        for key in required - {"profile_id", "source"}:
            require(isinstance(value[key], str) and SHA256.fullmatch(value[key]), f"invalid worker evidence {key}")
        protocol = value.get("request_protocol_sha256")
        require(protocol is None or isinstance(protocol, str) and SHA256.fullmatch(protocol),
                "invalid worker evidence request_protocol_sha256")
        return cls(**{**value, "invocation": InvocationEvidence.parse(value.get("invocation"))})


@dataclass(frozen=True)
class Run:
    id: str
    source: str
    model_revision: str
    environment_id: str
    request_settings_id: str
    precision: dict
    checkpoint_sha256: str | None = None
    worker_evidence: WorkerEvidence | None = None

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
        checkpoint = data.get("checkpoint_sha256")
        require(checkpoint is None or isinstance(checkpoint, str) and SHA256.fullmatch(checkpoint),
                "checkpoint_sha256 must be a lowercase SHA-256 digest")
        evidence = WorkerEvidence.parse(data.get("worker_evidence"))
        if evidence is not None:
            source = "target" if evidence.source == "worker" else "fixture"
            require(source == data["source"] and evidence.profile_id == data["environment_id"],
                    "worker evidence source or profile does not match the run")
        return cls(*(data[field] for field in fields), dict(precision), checkpoint, evidence)

    def differences(self, other):
        differences = [key for key in ("source", "model_revision", "environment_id", "request_settings_id")
                       if getattr(self, key) != getattr(other, key)]
        first = self.worker_evidence.invocation if self.worker_evidence else None
        second = other.worker_evidence.invocation if other.worker_evidence else None
        if first and second and first.binary_sha256 != second.binary_sha256:
            differences.append("infer_binary_sha256")
        return differences


@dataclass(frozen=True)
class Output:
    text: str
    raw_text: str | None
    finish_reason: str | None
    generated_count: int | None
    first_token_id: int | None
    stop_token_ids: list[int] | None
    trace: tuple["DecodeStep", ...] = ()
    audio: "AudioEvidence | None" = None

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
        from .trace import parse_steps
        from .audio_evidence import AudioEvidence

        trace = parse_steps(data.get("trace", ()))
        count, first_id, stop_ids = (data.get(key) for key in ("generated_count", "first_token_id", "stop_token_ids"))
        require(count is None or not trace or trace[-1].step < count, "decode step exceeds generated token count")
        if trace and trace[0].step == 0:
            first = trace[0]
            require(first_id is None or first_id == first.selected_id, "first token contradicts decode trace")
            require(stop_ids is None or set(stop_ids) == set(first.stop_ids), "stop IDs contradict decode trace")
            first_id, stop_ids = first.selected_id, list(first.stop_ids)
        audio = AudioEvidence.parse(data["audio"]) if data.get("audio") is not None else None
        return cls(data["text"], data.get("raw_text"), data.get("finish_reason"), count, first_id, stop_ids, trace, audio)


@dataclass(frozen=True)
class Sample:
    id: str
    baseline: Output | None
    candidate: Output | None
    reference: str | None
    language: str | None
    audio_sha256: str | None = None
    speaker_id: str | None = None


def scoring_sha256(samples, side):
    rows = []
    for sample in sorted(samples, key=lambda row: row.id):
        output = getattr(sample, side)
        score_fields = (None if output is None else
                        [output.text, output.finish_reason, output.generated_count])
        rows.append([sample.id, sample.reference, sample.language, sample.audio_sha256, sample.speaker_id, score_fields])
    return document_sha256(rows)


@dataclass(frozen=True)
class Comparison:
    model: str
    baseline: Run
    candidate: Run
    samples: list[Sample]
    evaluation_dataset_sha256: str | None = None
    request_document_sha256: str | None = None
    request_protocol: "RequestProtocol | None" = None

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
            digest = row.get("audio_sha256")
            require(digest is None or isinstance(digest, str) and SHA256.fullmatch(digest),
                    "sample audio_sha256 must be a lowercase SHA-256 digest")
            samples.append(Sample(row["id"], Output.parse(row.get("baseline")),
                                  Output.parse(row.get("candidate")), reference, language, digest,
                                  parse_speaker_id(row.get("speaker_id"))))
        fingerprints = tuple(data.get(key) for key in ("evaluation_dataset_sha256", "request_document_sha256"))
        require(all(value is None or isinstance(value, str) and SHA256.fullmatch(value) for value in fingerprints),
                "dataset and request fingerprints must be lowercase SHA-256 digests")
        from .request_protocol import RequestProtocol

        protocol = RequestProtocol.parse(data.get("request_protocol"))
        require(protocol is None or fingerprints[1], "request protocol requires a request document fingerprint")
        for side, run in (("baseline", baseline), ("candidate", candidate)):
            evidence = run.worker_evidence
            if evidence is not None:
                require((evidence.evaluation_dataset_sha256, evidence.request_document_sha256) == fingerprints,
                        f"{side}: worker evidence dataset or request fingerprint differs")
                require(evidence.scoring_sha256 == scoring_sha256(samples, side),
                        f"{side}: worker evidence does not match the normalized scoring rows")
                require(evidence.request_protocol_sha256 is None or
                        protocol is not None and protocol.sha256 == evidence.request_protocol_sha256,
                        f"{side}: worker evidence does not match the request protocol")
                if evidence.invocation:
                    require(evidence.invocation.audio_checked == len(samples)
                            and evidence.invocation.audio_manifest_checked == sum(row.audio_sha256 is not None for row in samples),
                            f"{side}: Infer invocation audio counts differ from the comparison")
        return cls(data["model"], baseline, candidate, samples, *fingerprints, protocol)
