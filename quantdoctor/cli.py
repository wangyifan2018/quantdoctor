"""Local entry points. Device jobs are always explicit."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .contracts import WorkerJob, WorkerResult
from .diagnostics import report, work_order
from .edge import import_edge
from .model import Comparison, load_json
from .qat import QATRecipe, template


def emit(document, output=None):
    encoded = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output is None:
        print(encoded, end="")
    else:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="quantdoctor")
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import-edge", help="Pair two llm_inference result files")
    for name in ("manifest", "baseline", "candidate"):
        importer.add_argument(name, type=Path)
    importer.add_argument("-o", "--output", type=Path)
    planner = commands.add_parser("plan", help="Export a single-variable work order for a sample")
    planner.add_argument("comparison", type=Path)
    planner.add_argument("sample_id")
    planner.add_argument("-o", "--output", type=Path)
    for name in ("report", "serve"):
        command = commands.add_parser(name)
        command.add_argument("comparison", type=Path)
        if name == "serve":
            command.add_argument("--port", type=int, default=8765)
        else:
            command.add_argument("-o", "--output", type=Path)

    commands.add_parser("qat-template").add_argument("-o", "--output", type=Path)
    commands.add_parser("qat-validate").add_argument("recipe", type=Path)
    commands.add_parser("job-validate", help="Validate an offline worker job and print its digest").add_argument("job", type=Path)
    result = commands.add_parser("job-result-validate", help="Validate a worker result against its job")
    result.add_argument("job", type=Path)
    result.add_argument("result", type=Path)
    worker = commands.add_parser("qat-run", help="Run selective QAT on a Linux/CUDA worker")
    worker.add_argument("recipe", type=Path)
    for flag in ("snapshot", "edge-repo", "output"):
        worker.add_argument(f"--{flag}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "import-edge":
            bundle = import_edge(*(load_json(path) for path in (args.manifest, args.baseline, args.candidate)))
            Comparison.parse(bundle)
            emit(bundle, args.output)
        elif args.command == "plan":
            comparison = Comparison.parse(load_json(args.comparison))
            emit(work_order(comparison, args.sample_id), args.output)
        elif args.command in {"report", "serve"}:
            data = report(Comparison.parse(load_json(args.comparison)))
            if args.command == "serve":
                from .server import serve
                serve(data, args.port)
            else:
                emit(data, args.output)
        elif args.command == "qat-template":
            emit(template(), args.output)
        elif args.command == "qat-validate":
            QATRecipe.parse(load_json(args.recipe))
            print("QAT recipe schema validated. Model, datasets and device are checked by the worker.")
        elif args.command in {"job-validate", "job-result-validate"}:
            job = WorkerJob.parse(load_json(args.job))
            if args.command == "job-validate":
                print(job.id)
            else:
                result = WorkerResult.parse(load_json(args.result), job)
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
