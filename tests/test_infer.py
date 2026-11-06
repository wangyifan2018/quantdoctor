from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import wave

from quantdoctor.artifacts import artifact_digest, verify_artifacts
from quantdoctor.contracts import EDGE_ASR_INFER_BACKEND, Stage, WorkerResult
from quantdoctor.diagnostics import classify
from quantdoctor.handoff import create_job
from quantdoctor.infer import plan_infer, run_infer
from quantdoctor.model import Comparison, document_sha256, file_sha256, load_json
from quantdoctor.preflight import TargetProfile
from quantdoctor.qat import EDGE_COMMIT
from quantdoctor.worker_evidence import import_worker_edge


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


STUB = '''#!/usr/bin/env python3
import json
from pathlib import Path
import sys

args = dict(flag.split("=", 1) for flag in sys.argv[1:])
root = Path(args["--multimodalEngineDir"])
assert (root / "audio" / "audio_encoder.engine").is_file()
mode = (Path(args["--engineDir"]) / "llm.engine").read_text()
print("Synthetic Edge process: " + mode)
if mode == "no-output":
    sys.exit(1)
requests = json.loads(Path(args["--inputFile"]).read_text())["requests"]
error = mode == "error"
responses = []
for index, _ in enumerate(requests):
    failed = error and index == 0
    responses.append({"request_idx": index, "output_text": "failed" if failed else "open door",
                      "finish_reason": "error" if failed else "end-of-sequence"})
Path(args["--outputFile"]).write_text(json.dumps({"input_file": args["--inputFile"], "responses": responses}))
sys.exit(int(error))
'''


class InferTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binary = self.root / "fake llm_inference"
        self.binary.write_text(STUB, encoding="utf-8")
        self.binary.chmod(0o700)
        self.audio = self.root / "clip.wav"
        with wave.open(str(self.audio), "wb") as clip:
            clip.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            clip.writeframes(b"\x01\x00" * 160)
        self.requests = {"top_k": 1, "requests": [{"messages": [{"role": "user", "content": [
            {"type": "audio", "audio": str(self.audio)},
        ]}]}]}
        examples = Path(__file__).resolve().parents[1] / "examples"
        self.manifest = load_json(examples / "demo-comparison.json")
        self.profile = TargetProfile("demo-thor", "jetson_thor", EDGE_COMMIT)
        self.manifest["request_document_sha256"] = document_sha256(self.requests)
        self.manifest["samples"] = [{"id": "clip", "request_idx": 0, "reference": "open door", "language": "en",
                                     "audio_sha256": file_sha256(self.audio)}]

    def stage(self, name, mode):
        bundle = self.root / name
        engines = bundle / "inputs" / "engine_bundle"
        for folder, files in (
            ("llm", {"llm.engine": mode, "config.json": '{"builder_config":{"max_batch_size":1}}',
                     "tokenizer.json": "{}", "embedding.safetensors": "fixture", "chat_template.model": "qwen3"}),
            ("audio", {"audio_encoder.engine": "audio fixture", "config.json": '{"model_type":"qwen3_asr_thinker"}'}),
        ):
            (engines / folder).mkdir(parents=True)
            for filename, content in files.items():
                (engines / folder / filename).write_text(content, encoding="utf-8")
        write_json(bundle / "inputs" / "requests", self.requests)
        return bundle, self.write_job(bundle)

    def write_job(self, bundle):
        job = create_job(Stage.INFER, EDGE_ASR_INFER_BACKEND, self.profile.id, bundle / "inputs")
        write_json(bundle / "job.json", job.document)
        return job

    def test_plan_uses_bundle_root_for_multimodal_and_checks_layout(self):
        bundle, job = self.stage("baseline", "baseline")
        plan = plan_infer(bundle, self.profile, Path("/worker/bin/llm_inference"))
        self.assertEqual(plan["job_id"], job.id)
        self.assertIn("--multimodalEngineDir=inputs/engine_bundle", plan["argv"])
        self.assertNotIn("--multimodalEngineDir=inputs/engine_bundle/audio", plan["argv"])
        self.assertFalse(plan["execution_verified"])
        self.assertFalse((bundle / "outputs").exists())
        path = bundle / "inputs" / "engine_bundle" / "audio" / "audio_encoder.engine"
        path.rename(path.with_name("wrong.engine"))
        self.write_job(bundle)
        with self.assertRaisesRegex(ValueError, "audio/audio_encoder.engine"):
            plan_infer(bundle, self.profile, self.binary)

    def test_fixture_partial_failures_and_audio_checks_survive_import(self):
        self.requests["requests"].append(deepcopy(self.requests["requests"][0]))
        self.manifest["request_document_sha256"] = document_sha256(self.requests)
        self.manifest["samples"].append({**self.manifest["samples"][0], "id": "clip-2", "request_idx": 1})
        baseline, _ = self.stage("baseline", "baseline")
        candidate, job = self.stage("candidate", "error")
        run_infer(baseline, self.profile, self.binary, fixture=True)
        result = run_infer(candidate, self.profile, self.binary, fixture=True)
        worker = WorkerResult.parse(result, job)
        self.assertEqual(worker.status, "succeeded")
        self.assertEqual(worker.source, "fixture")
        verify_artifacts(candidate / "outputs", worker.outputs)
        parsed = Comparison.parse(import_worker_edge(self.manifest, baseline, candidate))
        self.assertEqual(classify(parsed.samples[0].candidate), "failed")
        self.assertEqual(classify(parsed.samples[1].candidate), "transcribed")
        evidence = parsed.candidate.worker_evidence.invocation
        self.assertEqual((evidence.exit_code, evidence.audio_checked, evidence.audio_manifest_checked), (1, 2, 2))
        self.assertEqual(evidence.binary_sha256, file_sha256(self.binary))
        changed = deepcopy(self.manifest)
        changed["samples"][0]["audio_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "audio fingerprint differs"):
            import_worker_edge(changed, baseline, candidate)

    def test_collect_rejects_overrides_even_if_sidecar_digest_is_updated(self):
        baseline, _ = self.stage("baseline", "baseline")
        candidate, _ = self.stage("candidate", "error")
        for path in (baseline, candidate):
            run_infer(path, self.profile, self.binary, fixture=True)
        invocation = candidate / "outputs" / "invocation"
        edited = load_json(invocation)
        edited["argv"].append("--maxGenerateLength=1")
        write_json(invocation, edited)
        result = load_json(candidate / "result.json")
        result["outputs"]["invocation"] = artifact_digest(invocation)
        write_json(candidate / "result.json", result)
        with self.assertRaisesRegex(ValueError, "fixed CLI arguments"):
            import_worker_edge(self.manifest, baseline, candidate)

    def test_missing_output_retains_logs_and_preserves_existing_result(self):
        bundle, job = self.stage("broken", "no-output")
        result = run_infer(bundle, self.profile, self.binary, fixture=True)
        self.assertEqual(WorkerResult.parse(result, job).status, "failed")
        self.assertIn("no-output", (bundle / "outputs" / "stdout.log").read_text())
        with self.assertRaisesRegex(ValueError, "already has a result"):
            run_infer(bundle, self.profile, self.binary, fixture=True)
        self.assertEqual(load_json(bundle / "result.json"), result)

    def test_different_infer_executables_are_noncomparable(self):
        baseline, _ = self.stage("baseline", "baseline")
        candidate, _ = self.stage("candidate", "error")
        run_infer(baseline, self.profile, self.binary, fixture=True)
        self.binary.write_text(STUB + "\n# different binary bytes\n", encoding="utf-8")
        run_infer(candidate, self.profile, self.binary, fixture=True)
        comparison = Comparison.parse(import_worker_edge(self.manifest, baseline, candidate))
        self.assertEqual(comparison.baseline.differences(comparison.candidate), ["infer_binary_sha256"])
