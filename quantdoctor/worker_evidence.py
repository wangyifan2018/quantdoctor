"""Import Edge responses directly from locally checked Infer job bundles."""

from hashlib import sha256
import json

from .artifacts import file_artifact_sha256, verify_artifacts
from .contracts import Stage, WorkerJob, WorkerResult
from .edge import import_edge
from .infer import validate_invocation
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
    invocation = None
    if "invocation" in result.outputs:
        invocation = _bound_json(outputs / "invocation", result.outputs["invocation"])
        validate_invocation(invocation, job, result.source, requests, responses)
        evidence["invocation"] = {"artifact_sha256": result.outputs["invocation"],
                                  "binary_sha256": invocation["binary_sha256"], "exit_code": invocation["exit_code"],
                                  "audio_checked": len(invocation["audio_sha256"]), "audio_manifest_checked": 0}
    return responses, evidence, requests, invocation


def import_worker_edge(manifest, baseline, candidate, **observations):
    require(isinstance(manifest, dict) and all(isinstance(manifest.get(key), str)
            and SHA256.fullmatch(manifest[key])
            for key in ("evaluation_dataset_sha256", "request_document_sha256")),
            "worker import requires evaluation dataset and request fingerprints")
    baseline_response, baseline_evidence, requests, baseline_invocation = _infer_evidence(baseline, manifest, "baseline")
    candidate_response, candidate_evidence, _, candidate_invocation = _infer_evidence(candidate, manifest, "candidate")
    require(baseline_evidence["job_id"] != candidate_evidence["job_id"],
            "baseline and candidate require distinct infer jobs")
    bundle = import_edge(manifest, baseline_response, candidate_response, request_document=requests, **observations)
    comparison = Comparison.parse(bundle)
    if baseline_invocation and candidate_invocation:
        require(baseline_invocation["audio_sha256"] == candidate_invocation["audio_sha256"],
                "baseline and candidate Infer invocations read different audio bytes")
    for side, evidence, invocation in (("baseline", baseline_evidence, baseline_invocation),
                                       ("candidate", candidate_evidence, candidate_invocation)):
        if invocation is not None:
            require(comparison.request_protocol is not None, "Infer invocation requires a uniform ASR request protocol")
            for row, sample in zip(manifest["samples"], comparison.samples, strict=True):
                actual = invocation["audio_sha256"][row["request_idx"]]
                output = getattr(sample, side)
                captured = output.audio.input_sha256 if output and output.audio else None
                require(all(value is None or value == actual for value in (sample.audio_sha256, captured)),
                        f"{side} {sample.id}: Infer audio fingerprint differs from manifest or capture")
                evidence["invocation"]["audio_manifest_checked"] += sample.audio_sha256 is not None
        if comparison.request_protocol is not None:
            evidence["request_protocol_sha256"] = comparison.request_protocol.sha256
        bundle[side]["worker_evidence"] = {**evidence, "scoring_sha256": scoring_sha256(comparison.samples, side)}
    Comparison.parse(bundle)
    return bundle
