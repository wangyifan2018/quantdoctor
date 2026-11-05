"""Offline stage handoff. This module neither imports ML libraries nor runs jobs."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from .model import SHA256, nonempty, require


class Stage(str, Enum):
    PTQ = "ptq"
    QAT = "qat"
    PROBE = "probe"
    EXPORT = "export"
    BUILD = "build"
    INFER = "infer"
    PROFILE = "profile"


@dataclass(frozen=True)
class StageArtifacts:
    inputs: frozenset[str]
    outputs: frozenset[str]


ARTIFACTS = {
    Stage.PTQ: StageArtifacts(frozenset({"snapshot", "recipe", "calibration"}), frozenset({"checkpoint"})),
    Stage.QAT: StageArtifacts(
        frozenset({"snapshot", "recipe", "calibration", "training"}), frozenset({"checkpoint", "training_report"})),
    Stage.PROBE: StageArtifacts(
        frozenset({"subject", "probe_plan", "requests"}), frozenset({"observations"})),
    Stage.EXPORT: StageArtifacts(frozenset({"checkpoint", "export_config"}), frozenset({"onnx_bundle"})),
    Stage.BUILD: StageArtifacts(frozenset({"onnx_bundle", "build_config"}), frozenset({"engine_bundle"})),
    Stage.INFER: StageArtifacts(frozenset({"engine_bundle", "requests"}), frozenset({"responses"})),
    Stage.PROFILE: StageArtifacts(frozenset({"engine_bundle", "requests", "profile_config"}), frozenset({"profile"})),
}


def _document(data, keys):
    require(isinstance(data, dict) and set(data) == keys, f"expected fields: {', '.join(sorted(keys))}")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "expected worker contract schema_version 1")


def _strings(data, keys):
    require(all(nonempty(data[key]) for key in keys), f"required nonempty strings: {', '.join(keys)}")


def _digests(data, roles):
    require(isinstance(data, dict) and data.keys() == roles, f"expected artifact roles: {', '.join(sorted(roles))}")
    require(all(isinstance(value, str) and SHA256.fullmatch(value) for value in data.values()),
            "artifact references must be lowercase SHA-256 digests")
    return MappingProxyType(dict(data))


@dataclass(frozen=True)
class WorkerJob:
    stage: Stage
    backend: str
    profile_id: str
    inputs: Mapping[str, str]

    @classmethod
    def parse(cls, data):
        _document(data, {"schema_version", "stage", "backend", "profile_id", "inputs"})
        _strings(data, ("backend", "profile_id"))
        stage = Stage(data["stage"])
        return cls(stage, data["backend"], data["profile_id"], _digests(data["inputs"], ARTIFACTS[stage].inputs))

    @property
    def document(self):
        return {"schema_version": 1, "stage": self.stage.value, "backend": self.backend,
                "profile_id": self.profile_id, "inputs": dict(self.inputs)}

    @property
    def id(self):
        canonical = json.dumps(self.document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return sha256(canonical.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class WorkerResult:
    job_id: str
    profile_id: str
    source: str
    status: str
    outputs: Mapping[str, str]
    message: str

    @classmethod
    def parse(cls, data, job: WorkerJob):
        _document(data, {"schema_version", "job_id", "profile_id", "source", "status", "outputs", "message"})
        require(data["job_id"] == job.id and data["profile_id"] == job.profile_id,
                "worker result does not match the job/profile")
        require(data["source"] in ("fixture", "worker"), "source must be fixture or worker")
        status = data["status"]
        require(status in ("succeeded", "failed", "unsupported"), "invalid worker result status")
        require(isinstance(data["message"], str), "message must be a string")
        require(status == "succeeded" or nonempty(data["message"]), "failed/unsupported result requires a reason")
        roles = ARTIFACTS[job.stage].outputs if status == "succeeded" else frozenset()
        return cls(job.id, job.profile_id, data["source"], status, _digests(data["outputs"], roles), data["message"])


class CapabilityState(str, Enum):
    UNVERIFIED = "unverified"
    UNSUPPORTED = "unsupported"
    VERIFIED = "verified"


@dataclass(frozen=True)
class Capability:
    state: CapabilityState
    reason: str
    evidence_id: str | None = None

    def __post_init__(self):
        require(nonempty(self.reason), "capability requires a reason")
        require(self.state != CapabilityState.VERIFIED or nonempty(self.evidence_id),
                "verified capability requires evidence")


class WorkerAdapter(Protocol):
    """Device implementations live outside the local diagnostics workbench."""

    def capability(self, job: WorkerJob) -> Capability:
        """Report support for the exact stage, adapter version and execution profile."""
        ...

    def execute(self, job: WorkerJob, inputs: Mapping[str, Path], staging: Path) -> WorkerResult:
        """Consume verified local inputs and write outputs under their artifact role names."""
        ...
