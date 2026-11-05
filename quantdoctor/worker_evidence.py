"""Import Edge responses directly from locally checked Infer job bundles."""

from hashlib import sha256
import json

from .artifacts import file_artifact_sha256, verify_artifacts
from .contracts import Stage, WorkerJob, WorkerResult
from .edge import import_edge
from .model import Comparison, SHA256, document_sha256, load_json, require, scoring_sha256


def _bound_json(path, expected):
    require(path.is_file() and not path.is_symlink(), f"{path.name}: expected a regular JSON artifact")
    payload = path.read_bytes()
    require(file_artifact_sha256(sha256(payload).hexdigest()) == expected, f"{path.name}: artifact digest mismatch")
    return json.loads(payload)


def _infer_evidence(directory, manifest, side):
    job = WorkerJob.parse(load_json(directory / "job.json"))
    require(job.stage == Stage.INFER, "worker evidence requires an infer job")
    result = WorkerResult.parse(load_json(directory / "result.json"), job)
    require(result.status == "succeeded", "worker evidence requires a successful infer result")
    run = manifest.get(side)
    require(isinstance(run, dict) and run.get("environment_id") == job.profile_id
            and run.get("source") == ("target" if result.source == "worker" else "fixture"),
            f"{side}: worker source or profile differs from the run")
    inputs, outputs = directory / "inputs", directory / "outputs"
    verify_artifacts(inputs, {"engine_bundle": job.inputs["engine_bundle"]})
    require(outputs.is_dir() and not outputs.is_symlink(), "worker outputs must be a local directory")
    requests = _bound_json(inputs / "requests", job.inputs["requests"])
    require(document_sha256(requests) == manifest.get("request_document_sha256"),
            f"{side}: worker requests do not match the evaluation request fingerprint")
    responses = _bound_json(outputs / "responses", result.outputs["responses"])
    evidence = {
        "job_id": job.id, "profile_id": job.profile_id, "source": result.source,
        "engine_sha256": job.inputs["engine_bundle"], "request_artifact_sha256": job.inputs["requests"],
        "response_artifact_sha256": result.outputs["responses"],
        "request_document_sha256": manifest["request_document_sha256"],
        "evaluation_dataset_sha256": manifest["evaluation_dataset_sha256"],
    }
    return responses, evidence, requests


def import_worker_edge(manifest, baseline, candidate, **observations):
    require(isinstance(manifest, dict) and all(isinstance(manifest.get(key), str)
            and SHA256.fullmatch(manifest[key])
            for key in ("evaluation_dataset_sha256", "request_document_sha256")),
            "worker import requires evaluation dataset and request fingerprints")
    baseline_response, baseline_evidence, requests = _infer_evidence(baseline, manifest, "baseline")
    candidate_response, candidate_evidence, _ = _infer_evidence(candidate, manifest, "candidate")
    require(baseline_evidence["job_id"] != candidate_evidence["job_id"],
            "baseline and candidate require distinct infer jobs")
    bundle = import_edge(manifest, baseline_response, candidate_response, request_document=requests, **observations)
    comparison = Comparison.parse(bundle)
    for side, evidence in (("baseline", baseline_evidence), ("candidate", candidate_evidence)):
        if comparison.request_protocol is not None:
            evidence["request_protocol_sha256"] = comparison.request_protocol.sha256
        bundle[side]["worker_evidence"] = {**evidence, "scoring_sha256": scoring_sha256(comparison.samples, side)}
    Comparison.parse(bundle)
    return bundle
