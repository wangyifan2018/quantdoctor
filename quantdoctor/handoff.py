"""Construct offline jobs and verify the byte identity of staged handoffs."""

from pathlib import Path

from .artifacts import artifact_digest, verify_artifacts
from .contracts import ARTIFACTS, Stage, WorkerJob, WorkerResult
from .model import require


HANDOFFS = {
    (Stage.PTQ, Stage.EXPORT): "checkpoint",
    (Stage.QAT, Stage.EXPORT): "checkpoint",
    (Stage.EXPORT, Stage.BUILD): "onnx_bundle",
    (Stage.BUILD, Stage.INFER): "engine_bundle",
    (Stage.BUILD, Stage.PROFILE): "engine_bundle",
}


def create_job(stage: Stage, backend: str, profile_id: str, root: Path):
    require(root.is_dir() and not root.is_symlink(), "artifact root must be a local directory")
    return WorkerJob.parse({
        "schema_version": 1, "stage": stage.value, "backend": backend, "profile_id": profile_id,
        "inputs": {role: artifact_digest(root / role) for role in sorted(ARTIFACTS[stage].inputs)},
    })


def next_job(upstream: WorkerJob, result: WorkerResult, outputs: Path, stage: Stage,
             backend: str, profile_id: str, inputs: Path, *, allow_fixture=False):
    require(result.job_id == upstream.id and result.profile_id == upstream.profile_id,
            "upstream result does not match job/profile")
    require(result.status == "succeeded", "upstream job must succeed before handoff")
    require(result.source == "worker" or allow_fixture, "fixture handoff requires --allow-fixture")
    role = HANDOFFS.get((upstream.stage, stage))
    require(role is not None, f"unsupported handoff {upstream.stage.value} → {stage.value}")
    if role == "engine_bundle":
        require(profile_id == upstream.profile_id, "engine handoff requires the same build and execution profile")
    verify_artifacts(outputs, result.outputs)
    downstream = create_job(stage, backend, profile_id, inputs)
    require(downstream.inputs[role] == result.outputs[role], f"{role}: staged input differs from upstream output")
    return downstream
