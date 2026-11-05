"""Local entry points. Device jobs are always explicit."""

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from tempfile import TemporaryDirectory

from .artifacts import artifact_digest, verify_artifacts
from .checkpoint import audit_checkpoints, inspect_checkpoint
from .contracts import WorkerJob, WorkerResult
from .datasets import load_asr_dataset, prepare_evaluation
from .diagnostics import report, work_order
from .edge import import_edge
from .model import Comparison, load_json, require
from .qat import QATRecipe, template
from .quality import compare_candidates
from .replay import verify_repair


def emit(document, output=None):
    encoded = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output is None:
        print(encoded, end="")
    else:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)


def _replay_options(command):
    command.add_argument("--costs", type=Path)
    command.add_argument("--original-checkpoint", type=Path)
    command.add_argument("--repaired-checkpoint", type=Path)


def _replay(before, args):
    after = Comparison.parse(load_json(args.retest))
    checkpoints = args.original_checkpoint, args.repaired_checkpoint
    require(all(checkpoints) or not any(checkpoints), "both checkpoint directories are required for audit")
    return verify_repair(
        before, after, args.sample_id, load_json(args.costs) if args.costs else None,
        checkpoints=checkpoints if all(checkpoints) else None,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="quantdoctor")
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare", help="Compare candidates on a shared ASR cohort")
    compare.add_argument("comparisons", type=Path, nargs="+")
    compare.add_argument("--reference-run", help="Use this run for paired error-rate differences")
    inspect = commands.add_parser("inspect", help="Inspect a local HF safetensors checkpoint")
    inspect.add_argument("snapshot", type=Path)
    inspect.add_argument("--hash-weights", action="store_true")
    audit = commands.add_parser("audit-checkpoints", help="Compare the actual tensor payload of two checkpoints")
    audit.add_argument("original", type=Path)
    audit.add_argument("candidate", type=Path)
    audit.add_argument("--expect-component", choices=("audio", "backbone", "head", "other"))
    for command in (compare, inspect, audit):
        destination = command.add_mutually_exclusive_group()
        destination.add_argument("-o", "--output", type=Path)
        destination.add_argument("--port", type=int, help="Open a loopback inspection workbench")
    commands.add_parser("dataset-validate", help="Check a local ASR dataset and fingerprint each split").add_argument(
        "dataset", type=Path)
    evaluation = commands.add_parser("prepare-eval", help="Export Edge ASR requests and scoring manifest")
    evaluation.add_argument("dataset", type=Path)
    evaluation.add_argument("comparison_metadata", type=Path)
    evaluation.add_argument("--audio-root", type=PurePosixPath, help="Dataset root as seen on the target")
    evaluation.add_argument("--output", type=Path, required=True)
    importer = commands.add_parser("import-edge", help="Pair two llm_inference result files")
    for name in ("manifest", "baseline", "candidate"):
        importer.add_argument(name, type=Path)
    importer.add_argument("--baseline-trace", type=Path)
    importer.add_argument("--candidate-trace", type=Path)
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
    commands.add_parser("qat-validate").add_argument("recipe", type=Path)
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
    worker = commands.add_parser("qat-run", help="Run selective QAT on a Linux/CUDA worker")
    worker.add_argument("recipe", type=Path)
    for flag in ("snapshot", "edge-repo", "output"):
        worker.add_argument(f"--{flag}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in {"compare", "inspect", "audit-checkpoints"}:
            if args.command == "compare":
                data = compare_candidates([Comparison.parse(load_json(path)) for path in args.comparisons],
                                          args.reference_run)
            elif args.command == "inspect":
                data = inspect_checkpoint(args.snapshot, hash_weights=args.hash_weights)
            else:
                data = audit_checkpoints(args.original, args.candidate, args.expect_component)
            if args.port is None:
                emit(data, args.output)
            else:
                from .server import serve
                serve(data, args.port)
        elif args.command == "dataset-validate":
            emit(load_asr_dataset(args.dataset).summary())
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
        elif args.command == "import-edge":
            bundle = import_edge(
                *(load_json(path) for path in (args.manifest, args.baseline, args.candidate)),
                baseline_trace=load_json(args.baseline_trace) if args.baseline_trace else None,
                candidate_trace=load_json(args.candidate_trace) if args.candidate_trace else None,
            )
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
                    require(not any((args.sample_id, args.costs, args.original_checkpoint, args.repaired_checkpoint)),
                            "repair options require --retest")
                serve(data, args.port)
            else:
                emit(data, args.output)
        elif args.command == "qat-template":
            emit(template(), args.output)
        elif args.command == "qat-validate":
            QATRecipe.parse(load_json(args.recipe))
            print("QAT recipe schema validated. Model, datasets and device are checked by the worker.")
        elif args.command == "artifact-digest":
            print(artifact_digest(args.artifact))
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
