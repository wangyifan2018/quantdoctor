"""Plan and collect single-device ASR runs using the pinned Edge CLI layout."""

import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from tempfile import TemporaryDirectory

from .artifacts import artifact_digest, verify_artifacts
from .contracts import EDGE_ASR_INFER_BACKEND, Stage, WorkerJob, WorkerResult
from .edge import parse_edge_responses
from .model import SHA256, file_sha256, load_json, require
from .qat import EDGE_COMMIT
from .request_protocol import RequestProtocol


ENGINE_ROOT = "inputs/engine_bundle"
REQUESTS = "inputs/requests"
RESPONSES = "outputs/responses"
INFER_FLAGS = (f"--engineDir={ENGINE_ROOT}/llm", f"--multimodalEngineDir={ENGINE_ROOT}",
               f"--inputFile={REQUESTS}", f"--outputFile={RESPONSES}")


def _binary_path(value):
    if not isinstance(value, str) or "\0" in value:
        return False
    path = PurePosixPath(value)
    return path.is_absolute() and ".." not in path.parts and bool(path.name)


def _audio_files(bundle, requests):
    for request in requests["requests"]:
        for message in request["messages"]:
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if item["type"] == "audio":
                        yield bundle / item["audio"]


def _load_plan(bundle, profile, binary):
    require(bundle.is_dir() and not bundle.is_symlink(), "expected an Infer bundle directory")
    job = WorkerJob.parse(load_json(bundle / "job.json"))
    require(job.stage == Stage.INFER and job.backend == EDGE_ASR_INFER_BACKEND,
            f"expected an infer job with backend {EDGE_ASR_INFER_BACKEND}")
    require(job.profile_id == profile.id and profile.edge_commit == EDGE_COMMIT,
            "Infer requires a matching target profile for the pinned Edge-LLM commit")
    require(profile.platform in {"jetson_orin", "jetson_thor", "drive_thor"}, "Infer requires an NVIDIA target profile")
    require(_binary_path(str(binary)), "llm_inference executable must be an absolute path on the worker")
    verify_artifacts(bundle / "inputs", job.inputs)
    engine = bundle / ENGINE_ROOT
    require(not any((engine / name).exists() for name in ("visual", "action")),
            "Infer adapter requires a standalone ASR engine bundle")
    for name in ("llm/llm.engine", "llm/config.json", "llm/tokenizer.json", "llm/embedding.safetensors",
                 "audio/audio_encoder.engine", "audio/config.json"):
        path = engine / name
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty Infer artifact: engine_bundle/{name}")
    template_paths = (engine / "llm" / name for name in ("chat_template.jinja", "chat_template.model"))
    templates = [path for path in template_paths if path.exists()]
    require(len(templates) == 1 and templates[0].is_file() and templates[0].stat().st_size > 0,
            "engine_bundle/llm requires exactly one chat_template.jinja or chat_template.model")
    audio_config = load_json(engine / "audio" / "config.json")
    require(isinstance(audio_config, dict) and audio_config.get("model_type") == "qwen3_asr_thinker",
            "engine_bundle/audio must declare Qwen3-ASR")
    llm_config = load_json(engine / "llm" / "config.json")
    build = llm_config.get("builder_config") if isinstance(llm_config, dict) else None
    require(isinstance(build, dict) and type(build.get("tp_size", 1)) is int and build.get("tp_size", 1) == 1
            and type(build.get("max_batch_size")) is int,
            "engine_bundle/llm requires a single-device builder_config with max_batch_size")
    requests = load_json(bundle / REQUESTS)
    protocol = RequestProtocol.from_document(requests)
    require(protocol is not None, "Infer adapter requires a uniform single-raw-audio ASR request template")
    require(not requests.get("available_lora_weights") and not any(row.get("lora_name") for row in requests["requests"]),
            "Infer adapter requires all model artifacts inside engine_bundle; LoRA is unavailable")
    require(1 <= protocol.settings["batch_size"] <= min(build["max_batch_size"], 16),
            "request batch size exceeds the engine or Edge CLI limit")
    require(0 < protocol.settings["max_generate_length"] <= 2**63 - 1,
            "request max_generate_length must be a positive signed 64-bit integer")
    default_logprobs = protocol.settings["num_logprobs"]
    logprobs = [default_logprobs] + [row.get("num_logprobs", default_logprobs) for row in requests["requests"]]
    require(all(type(value) is int and 0 <= value <= 50 for value in logprobs),
            "request num_logprobs must be an integer from 0 to 50")
    plan = {"schema_version": 1, "status": "prepared", "job_id": job.id, "profile_id": profile.id,
            "edge_commit": EDGE_COMMIT, "argv": [str(binary), *INFER_FLAGS], "working_directory": "bundle_root",
            "request_count": len(requests["requests"]), "request_protocol_sha256": protocol.sha256,
            "execution_verified": False}
    return job, requests, plan


def plan_infer(bundle, profile, binary):
    return _load_plan(bundle, profile, binary)[2]


def validate_invocation(record, job, source, requests, responses):
    keys = {"schema_version", "job_id", "profile_id", "source", "edge_commit", "argv", "binary_sha256",
            "exit_code", "audio_sha256"}
    require(isinstance(record, dict) and record.keys() == keys and type(record["schema_version"]) is int
            and record["schema_version"] == 1, "invalid Infer invocation schema")
    require(record["job_id"] == job.id and record["profile_id"] == job.profile_id
            and record["source"] == source and record["edge_commit"] == EDGE_COMMIT,
            "Infer invocation does not match the job, source or pinned profile")
    argv = record["argv"]
    require(isinstance(argv, list) and len(argv) == 5 and _binary_path(argv[0]) and argv[1:] == list(INFER_FLAGS),
            "Infer invocation differs from the fixed CLI arguments; file-setting overrides are unsupported")
    binary = record["binary_sha256"]
    require(isinstance(binary, str) and SHA256.fullmatch(binary), "invalid Infer executable fingerprint")
    exit_code = record["exit_code"]
    require(type(exit_code) is int and exit_code in (0, 1), "Infer process ended without a collectable result")
    require(isinstance(requests, dict) and isinstance(requests.get("requests"), list) and requests["requests"],
            "Infer invocation requires a nonempty request list")
    audio = record["audio_sha256"]
    require(isinstance(audio, list) and len(audio) == len(requests["requests"])
            and all(isinstance(value, str) and SHA256.fullmatch(value) for value in audio),
            "Infer invocation requires one audio fingerprint per request")
    require(isinstance(responses, dict) and responses.get("input_file") == REQUESTS,
            "Infer response input_file does not match the staged request path")
    rows = parse_edge_responses(responses)
    require(rows.keys() == set(range(len(audio))), "Infer responses must cover every staged request index")
    failed = sum(row["finish_reason"] == "error" for row in rows.values())
    require(exit_code == 0 or failed, "nonzero Infer exit without recorded request errors")
    return failed


def _audio_hashes(bundle, requests):
    hashes = {}
    result = []
    for path in _audio_files(bundle, requests):
        if path not in hashes:
            require(path.is_file(), f"request audio file is unavailable: {path}")
            hashes[path] = file_sha256(path)
        result.append(hashes[path])
    return result


def run_infer(bundle, profile, binary, *, fixture=False):
    require(fixture or sys.platform == "linux", "device inference requires Linux; use --fixture with a synthetic executable")
    job, requests, plan = _load_plan(bundle, profile, binary)
    require(binary.is_file() and os.access(binary, os.X_OK), "llm_inference executable is missing or not executable")
    audio_hashes = _audio_hashes(bundle, requests)
    binary_hash = file_sha256(binary)
    result_path, outputs = bundle / "result.json", bundle / "outputs"
    require(not result_path.exists() and not result_path.is_symlink(), "Infer bundle already has a result")
    outputs.mkdir()
    source = "fixture" if fixture else "worker"
    result = {"schema_version": 1, "job_id": job.id, "profile_id": job.profile_id, "source": source,
              "status": "failed", "outputs": {}, "message": ""}
    try:
        with (
            (outputs / "stdout.log").open("xb") as stdout,
            (outputs / "stderr.log").open("xb") as stderr,
        ):
            completed = subprocess.run(plan["argv"], cwd=bundle, stdout=stdout, stderr=stderr, check=False)
        verify_artifacts(bundle / "inputs", job.inputs)
        require(audio_hashes == _audio_hashes(bundle, requests), "request audio changed during Infer execution")
        require(binary_hash == file_sha256(binary), "llm_inference executable changed during execution")
        invocation = {"schema_version": 1, "job_id": job.id, "profile_id": job.profile_id, "source": source,
                      "edge_commit": EDGE_COMMIT, "argv": plan["argv"], "binary_sha256": binary_hash,
                      "exit_code": completed.returncode, "audio_sha256": audio_hashes}
        path = outputs / "invocation"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(invocation, stream, indent=2)
        response_path = bundle / RESPONSES
        response_hash = artifact_digest(response_path)
        failed = validate_invocation(invocation, job, source, requests, load_json(response_path))
        result.update(status="succeeded", outputs={"responses": response_hash, "invocation": artifact_digest(path)},
                      message=f"Collected complete response records ({failed} request errors; exit {completed.returncode}).")
    except (ValueError, OSError) as error:
        result["message"] = str(error)
    WorkerResult.parse(result, job)
    with TemporaryDirectory(dir=bundle) as temporary:
        pending = Path(temporary) / "result.json"
        pending.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        os.link(pending, result_path)
    return result
