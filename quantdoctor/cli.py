"""Local entry points. Device jobs are always explicit."""

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from tempfile import TemporaryDirectory

from .acceptance import evaluate_acceptance
from .artifacts import artifact_digest, verify_artifacts
from .checkpoint import audit_checkpoints, inspect_checkpoint
from .contracts import Stage, WorkerJob, WorkerResult
from .datasets import load_asr_dataset, prepare_evaluation
from .diagnostics import report, work_order
from .edge import import_edge
from .handoff import create_job, next_job
from .infer import plan_infer, run_infer
from .model import Comparison, load_json, require
from .qat import QATRecipe, load_qat_inputs, template
from .preflight import TargetProfile, preflight
from .quality import compare_candidates
from .replay import COHORT_FINGERPRINTS, verify_repair
from .request_protocol import inspect_request_protocol
from .worker_evidence import import_worker_edge


def emit(document, output=None):
    encoded = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output is None:
        print(encoded, end="")
    else:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)


def _replay_evidence_options(command):
    command.add_argument("--original-checkpoint", type=Path)
    command.add_argument("--repaired-checkpoint", type=Path)
    command.add_argument("--diagnosis", type=Path, help="Comparison that supplied the work-order sample, for holdout replay")


def _checkpoint_paths(args):
    checkpoints = args.original_checkpoint, args.repaired_checkpoint
    require(all(checkpoints) or not any(checkpoints), "both checkpoint directories are required for audit")
    return checkpoints if all(checkpoints) else None


def _replay_options(command):
    command.add_argument("--costs", type=Path)
    _replay_evidence_options(command)


def _diagnosis(args):
    return Comparison.parse(load_json(args.diagnosis)) if args.diagnosis else None


def _replay(before, args):
    after = Comparison.parse(load_json(args.retest))
    return verify_repair(
        before, after, args.sample_id, load_json(args.costs) if args.costs else None,
        checkpoints=_checkpoint_paths(args), diagnosis=_diagnosis(args),
    )


def _compare_replay(comparisons, selection, performance, checkpoints, diagnosis):
    original, repaired, sample_id = selection
    runs = {comparison.candidate.id: comparison for comparison in comparisons}
    require(original in runs and repaired in runs, "replay run IDs must name candidates in the comparison")
    costs = None
    if performance is not None:
        costs = {"schema_version": 1, "protocol_id": performance["protocol_id"],
                 **{field: getattr(runs[original], field) for field in COHORT_FINGERPRINTS}}
        for side, run_id in (("original", original), ("repaired", repaired)):
            costs[side] = {"run_id": run_id, **performance["runs"][run_id]["values"]}
    return verify_repair(runs[original], runs[repaired], sample_id, costs, checkpoints=checkpoints, diagnosis=diagnosis)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="quantdoctor")
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare", help="Compare candidates on a shared ASR cohort")
    compare.add_argument("comparisons", type=Path, nargs="+")
    compare.add_argument("--reference-run", help="Use this run for paired error-rate differences")
    compare.add_argument("--costs", type=Path, help="Attach same-protocol candidate performance records")
    compare.add_argument("--policy", type=Path, help="Evaluate a deployment policy against imported evidence")
    compare.add_argument("--replay", nargs=3, metavar=("ORIGINAL", "REPAIRED", "SAMPLE_ID"),
                         help="Show a controlled precision repair between two candidate run IDs")
    _replay_evidence_options(compare)
    inspect = commands.add_parser("inspect", help="Inspect a local HF safetensors checkpoint")
    inspect.add_argument("snapshot", type=Path)
    inspect.add_argument("--hash-weights", action="store_true")
    audit = commands.add_parser("audit-checkpoints", help="Compare the actual tensor payload of two checkpoints")
    audit.add_argument("original", type=Path)
    audit.add_argument("candidate", type=Path)
    audit.add_argument("--expect-component", choices=("audio", "backbone", "head", "other"))
    target = commands.add_parser("preflight", help="Check a candidate and optional repair against a target profile")
    target.add_argument("comparison", type=Path)
    target.add_argument("profile", type=Path)
    target.add_argument("--sample-id")
    for command in (compare, inspect, audit, target):
        destination = command.add_mutually_exclusive_group()
        destination.add_argument("-o", "--output", type=Path)
        destination.add_argument("--port", type=int, help="Open a loopback inspection workbench")
    dataset_check = commands.add_parser("dataset-validate", help="Check a local ASR dataset and fingerprint each split")
    dataset_check.add_argument("dataset", type=Path)
    dataset_check.add_argument("--require-speakers", action="store_true", help="Require a speaker ID for every sample")
    evaluation = commands.add_parser("prepare-eval", help="Export Edge ASR requests and scoring manifest")
    evaluation.add_argument("dataset", type=Path)
    evaluation.add_argument("comparison_metadata", type=Path)
    evaluation.add_argument("--audio-root", type=PurePosixPath, help="Dataset root as seen on the target")
    evaluation.add_argument("--output", type=Path, required=True)
    request_audit = commands.add_parser("request-audit", help="Inspect and compare file-level Edge ASR request settings")
    request_audit.add_argument("request", type=Path)
    request_audit.add_argument("--against", type=Path)
    request_audit.add_argument("-o", "--output", type=Path)
    edge_import = commands.add_parser("import-edge", help="Pair two llm_inference result files")
    edge_import.add_argument("--requests", type=Path, help="Attach the matching request JSON for protocol auditing")
    worker_import = commands.add_parser("import-worker-edge", help="Pair two locally verified infer bundles")
    for importer in (edge_import, worker_import):
        for name in ("manifest", "baseline", "candidate"):
            importer.add_argument(name, type=Path)
        for flag in ("baseline-trace", "candidate-trace", "baseline-audio", "candidate-audio"):
            importer.add_argument(f"--{flag}", type=Path)
        importer.add_argument("-o", "--output", type=Path)
    planner = commands.add_parser("plan", help="Export a single-variable work order for a sample")
    planner.add_argument("comparison", type=Path)
    planner.add_argument("sample_id")
    planner.add_argument("-o", "--output", type=Path)
    replay = commands.add_parser("replay", help="Compare a single-variable repair with its original run")
    replay.add_argument("comparison", type=Path)
    replay.add_argument("retest", type=Path)
    replay.add_argument("sample_id")
    _replay_options(replay)
    replay.add_argument("-o", "--output", type=Path)
    for name in ("report", "serve"):
        command = commands.add_parser(name)
        command.add_argument("comparison", type=Path)
        if name == "serve":
            command.add_argument("--port", type=int, default=8765)
            command.add_argument("--retest", type=Path)
            command.add_argument("--sample-id")
            _replay_options(command)
        else:
            command.add_argument("-o", "--output", type=Path)

    commands.add_parser("qat-template").add_argument("-o", "--output", type=Path)
    qat_check = commands.add_parser("qat-validate", help="Validate a QAT recipe and optionally inspect local inputs")
    qat_check.add_argument("recipe", type=Path)
    qat_check.add_argument("--snapshot", type=Path)
    qat_check.add_argument("--port", type=int, help="Open a loopback preflight workbench")
    commands.add_parser("job-validate", help="Validate an offline worker job and print its digest").add_argument("job", type=Path)
    commands.add_parser("artifact-digest", help="Print the digest of a local artifact file or directory").add_argument(
        "artifact", type=Path)
    inputs = commands.add_parser("job-input-verify", help="Verify a worker job's local input artifacts")
    inputs.add_argument("job", type=Path)
    inputs.add_argument("artifact_root", type=Path)
    result = commands.add_parser("job-result-validate", help="Validate a worker result against its job")
    result.add_argument("job", type=Path)
    result.add_argument("result", type=Path)
    result.add_argument("--artifact-root", type=Path, help="Verify the successful result's output artifacts")
    build_job = commands.add_parser("job-create", help="Create an offline job from staged input roles")
    chain_job = commands.add_parser("job-next", help="Verify the previous result and build the next offline job")
    chain_job.add_argument("upstream_job", type=Path)
    chain_job.add_argument("upstream_result", type=Path)
    chain_job.add_argument("upstream_outputs", type=Path)
    chain_job.add_argument("--allow-fixture", action="store_true")
    for command in (build_job, chain_job):
        command.add_argument("stage", choices=[stage.value for stage in Stage])
        command.add_argument("backend")
        command.add_argument("profile_id")
        command.add_argument("inputs", type=Path)
        command.add_argument("-o", "--output", type=Path)
    infer_plan = commands.add_parser("infer-plan", help="Check an ASR Infer bundle and print the pinned CLI arguments")
    infer_run = commands.add_parser("infer-run", help="Run a pinned ASR Infer bundle on a Linux worker")
    for command in (infer_plan, infer_run):
        command.add_argument("bundle", type=Path)
        command.add_argument("profile", type=Path)
        command.add_argument("--binary", required=True, type=Path, help="Absolute path to the worker executable")
    infer_plan.add_argument("-o", "--output", type=Path)
    infer_run.add_argument("--fixture", action="store_true", help="Run a synthetic executable and label all evidence as fixture")
    worker = commands.add_parser("qat-run", help="Run selective QAT on a Linux/CUDA worker")
    worker.add_argument("recipe", type=Path)
    for flag in ("snapshot", "edge-repo", "output"):
        worker.add_argument(f"--{flag}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in {"compare", "inspect", "audit-checkpoints", "preflight"}:
            if args.command == "compare":
                checkpoints = _checkpoint_paths(args)
                require(checkpoints is None or args.replay, "checkpoint audit requires --replay")
                require(not args.diagnosis or args.replay, "--diagnosis requires --replay")
                comparisons = [Comparison.parse(load_json(path)) for path in args.comparisons]
                data = compare_candidates(comparisons, args.reference_run,
                                          costs=load_json(args.costs) if args.costs else None)
                if args.policy:
                    data["acceptance"] = evaluate_acceptance(data, load_json(args.policy))
                if args.replay:
                    data["replay"] = _compare_replay(comparisons, args.replay, data.get("performance"),
                                                     checkpoints, _diagnosis(args))
            elif args.command == "inspect":
                data = inspect_checkpoint(args.snapshot, hash_weights=args.hash_weights)
            elif args.command == "audit-checkpoints":
                data = audit_checkpoints(args.original, args.candidate, args.expect_component)
            else:
                data = preflight(Comparison.parse(load_json(args.comparison)),
                                 TargetProfile.parse(load_json(args.profile)), args.sample_id)
            if args.port is None:
                emit(data, args.output)
            else:
                from .server import serve
                serve(data, args.port)
        elif args.command == "dataset-validate":
            dataset = load_asr_dataset(args.dataset)
            require(not args.require_speakers or all(row.speaker_id is not None for row in dataset.samples),
                    "speaker_id is required for every dataset sample")
            emit(dataset.summary())
        elif args.command == "prepare-eval":
            dataset = load_asr_dataset(args.dataset)
            request, scoring = prepare_evaluation(dataset, load_json(args.comparison_metadata), args.audio_root)
            output = args.output.absolute()
            require(not output.exists(), "evaluation destination already exists")
            with TemporaryDirectory(dir=output.parent) as temporary:
                stage = Path(temporary) / "evaluation"
                stage.mkdir()
                emit(request, stage / "input.json")
                emit(scoring, stage / "manifest.json")
                stage.rename(output)
            print(output)
        elif args.command == "request-audit":
            documents = [load_json(args.request)]
            if args.against:
                documents.append(load_json(args.against))
            emit(inspect_request_protocol(*documents), args.output)
        elif args.command in {"import-edge", "import-worker-edge"}:
            observations = dict(
                baseline_trace=load_json(args.baseline_trace) if args.baseline_trace else None,
                candidate_trace=load_json(args.candidate_trace) if args.candidate_trace else None,
                baseline_audio=load_json(args.baseline_audio) if args.baseline_audio else None,
                candidate_audio=load_json(args.candidate_audio) if args.candidate_audio else None,
            )
            manifest = load_json(args.manifest)
            if args.command == "import-worker-edge":
                bundle = import_worker_edge(manifest, args.baseline, args.candidate, **observations)
            else:
                if args.requests:
                    observations["request_document"] = load_json(args.requests)
                bundle = import_edge(manifest, load_json(args.baseline), load_json(args.candidate), **observations)
            Comparison.parse(bundle)
            emit(bundle, args.output)
        elif args.command == "plan":
            comparison = Comparison.parse(load_json(args.comparison))
            emit(work_order(comparison, args.sample_id), args.output)
        elif args.command == "replay":
            before = Comparison.parse(load_json(args.comparison))
            emit(_replay(before, args), args.output)
        elif args.command in {"report", "serve"}:
            before = Comparison.parse(load_json(args.comparison))
            data = report(before)
            if args.command == "serve":
                from .server import serve
                if args.retest:
                    require(args.sample_id, "--retest requires --sample-id")
                    data["replay"] = _replay(before, args)
                else:
                    require(not any((args.sample_id, args.costs, args.diagnosis,
                                     args.original_checkpoint, args.repaired_checkpoint)),
                            "repair options require --retest")
                serve(data, args.port)
            else:
                emit(data, args.output)
        elif args.command == "qat-template":
            emit(template(), args.output)
        elif args.command == "qat-validate":
            recipe = QATRecipe.parse(load_json(args.recipe))
            if args.snapshot is None:
                require(args.port is None, "--port requires --snapshot")
                print("QAT recipe schema validated. Pass --snapshot to check local inputs.")
            else:
                _, _, data = load_qat_inputs(recipe, args.recipe, args.snapshot)
                if args.port is None:
                    emit(data)
                else:
                    from .server import serve
                    serve(data, args.port)
        elif args.command == "artifact-digest":
            print(artifact_digest(args.artifact))
        elif args.command in {"job-create", "job-next"}:
            stage = Stage(args.stage)
            if args.command == "job-create":
                job = create_job(stage, args.backend, args.profile_id, args.inputs)
            else:
                upstream = WorkerJob.parse(load_json(args.upstream_job))
                result = WorkerResult.parse(load_json(args.upstream_result), upstream)
                job = next_job(upstream, result, args.upstream_outputs, stage, args.backend, args.profile_id,
                               args.inputs, allow_fixture=args.allow_fixture)
            emit(job.document, args.output)
        elif args.command in {"job-validate", "job-input-verify", "job-result-validate"}:
            job = WorkerJob.parse(load_json(args.job))
            if args.command == "job-validate":
                print(job.id)
            elif args.command == "job-input-verify":
                emit({"job_id": job.id, "verified_inputs": verify_artifacts(args.artifact_root, job.inputs)})
            else:
                result = WorkerResult.parse(load_json(args.result), job)
                if args.artifact_root:
                    require(result.status == "succeeded", "only successful worker results have output artifacts")
                    emit({"job_id": job.id, "source": result.source,
                          "verified_outputs": verify_artifacts(args.artifact_root, result.outputs),
                          "execution_verified": False})
                else:
                    print(f"Result schema validated: {result.source}/{result.status}. Artifacts and execution are unverified.")
        elif args.command in {"infer-plan", "infer-run"}:
            profile = TargetProfile.parse(load_json(args.profile))
            if args.command == "infer-plan":
                emit(plan_infer(args.bundle, profile, args.binary), args.output)
            else:
                result = run_infer(args.bundle, profile, args.binary, fixture=args.fixture)
                emit(result)
                return int(result["status"] != "succeeded")
        elif args.command == "qat-run":
            from .qat_worker import run
            run(*(path.resolve() for path in (args.recipe, args.snapshot, args.edge_repo, args.output)))
            print(f"QAT checkpoint exported to {args.output}. Target validation pending.")
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, ImportError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"quantdoctor: {error}", file=sys.stderr)
        return 1
    return 0
