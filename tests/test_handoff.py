from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from quantdoctor.artifacts import artifact_digest
from quantdoctor.contracts import Stage, WorkerResult
from quantdoctor.handoff import create_job, next_job


class HandoffTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.build_inputs, self.outputs, self.infer_inputs = (root / name for name in ("build", "out", "infer"))
        for directory, artifacts in (
            (self.build_inputs, {"onnx_bundle": "fixture ONNX", "build_config": "{}"}),
            (self.outputs, {"engine_bundle": "fixture engine"}),
            (self.infer_inputs, {"engine_bundle": "fixture engine", "requests": "{}"}),
        ):
            directory.mkdir()
            for role, data in artifacts.items():
                (directory / role).write_text(data)
        self.upstream = create_job(Stage.BUILD, "fixture", "thor-a", self.build_inputs)
        self.result = {
            "schema_version": 1, "job_id": self.upstream.id, "profile_id": "thor-a", "source": "fixture",
            "status": "succeeded", "message": "", "outputs": {
                "engine_bundle": artifact_digest(self.outputs / "engine_bundle"),
            },
        }

    def next(self, *, profile="thor-a", allow_fixture=True, stage=Stage.INFER):
        result = WorkerResult.parse(self.result, self.upstream)
        return next_job(self.upstream, result, self.outputs, stage, "fixture", profile,
                        self.infer_inputs, allow_fixture=allow_fixture)

    def test_engine_handoff_checks_profile_and_transferred_bytes(self):
        downstream = self.next()
        self.assertEqual(downstream.document["inputs"]["engine_bundle"], self.result["outputs"]["engine_bundle"])
        with self.assertRaisesRegex(ValueError, "same build and execution profile"):
            self.next(profile="thor-b")
        (self.infer_inputs / "engine_bundle").write_text("different")
        with self.assertRaisesRegex(ValueError, "staged input differs"):
            self.next()

    def test_failed_or_fixture_result_cannot_silently_chain(self):
        with self.assertRaisesRegex(ValueError, "allow-fixture"):
            self.next(allow_fixture=False)
        with self.assertRaisesRegex(ValueError, "unsupported handoff"):
            self.next(stage=Stage.PTQ)
        self.result.update(status="failed", message="build failed", outputs={})
        with self.assertRaisesRegex(ValueError, "must succeed"):
            self.next()

    def test_result_payload_is_verified_on_original_side(self):
        (self.outputs / "engine_bundle").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "artifact digest mismatch"):
            self.next()
