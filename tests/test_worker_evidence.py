from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from quantdoctor.artifacts import artifact_digest
from quantdoctor.contracts import Stage
from quantdoctor.handoff import create_job
from quantdoctor.model import Comparison, document_sha256, load_json
from quantdoctor.worker_evidence import import_worker_edge


def write_json(path, document):
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


class WorkerEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        fixture = Path(__file__).resolve().parents[1] / "examples" / "demo-comparison.json"
        self.manifest = load_json(fixture)
        self.requests = {"requests": [{"fixture": True}]}
        self.manifest["request_document_sha256"] = document_sha256(self.requests)
        self.manifest["samples"] = [{"id": "clip", "request_idx": 0, "reference": "开窗", "language": "zh"}]
        self.baseline = self.bundle("baseline", "开窗")
        self.candidate = self.bundle("candidate", "")

    def bundle(self, side, transcript):
        path = self.root / side
        engine = path / "inputs" / "engine_bundle"
        engine.mkdir(parents=True)
        (engine / "model.engine").write_text(side, encoding="utf-8")
        write_json(path / "inputs" / "requests", self.requests)
        job = create_job(Stage.INFER, "test", self.manifest[side]["environment_id"], path / "inputs")
        write_json(path / "job.json", job.document)
        responses = path / "outputs" / "responses"
        responses.parent.mkdir()
        write_json(responses, {"responses": [{"request_idx": 0, "output_text": transcript}]})
        write_json(path / "result.json", {
            "schema_version": 1, "job_id": job.id, "profile_id": job.profile_id, "source": "fixture",
            "status": "succeeded", "outputs": {"responses": artifact_digest(responses)}, "message": "fixture",
        })
        return path

    def test_receipt_links_checked_engine_response_and_scoring_rows(self):
        bundle = import_worker_edge(self.manifest, self.baseline, self.candidate)
        comparison = Comparison.parse(bundle)
        evidence = comparison.candidate.worker_evidence
        self.assertEqual(evidence.job_id, load_json(self.candidate / "result.json")["job_id"])
        self.assertEqual(evidence.engine_sha256, artifact_digest(self.candidate / "inputs" / "engine_bundle"))
        self.assertEqual(evidence.response_artifact_sha256, artifact_digest(self.candidate / "outputs" / "responses"))
        self.assertEqual(comparison.samples[0].candidate.text, "")
        for field, value in (("candidate", {"text": "edited"}), ("reference", "替换")):
            changed = deepcopy(bundle)
            changed["samples"][0][field] = value
            with self.assertRaisesRegex(ValueError, "normalized scoring rows"):
                Comparison.parse(changed)

    def test_rejects_tampered_artifacts(self):
        for role in ("inputs/engine_bundle/model.engine", "inputs/requests", "outputs/responses"):
            with self.subTest(role=role):
                path = self.candidate / role
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "artifact digest mismatch"):
                    import_worker_edge(self.manifest, self.baseline, self.candidate)
                path.write_bytes(original)

    def test_rejects_duplicate_job_and_mismatched_manifest_identity(self):
        with self.assertRaisesRegex(ValueError, "distinct infer jobs"):
            import_worker_edge(self.manifest, self.baseline, self.baseline)
        self.manifest["candidate"]["environment_id"] = "wrong-profile"
        with self.assertRaisesRegex(ValueError, "source or profile"):
            import_worker_edge(self.manifest, self.baseline, self.candidate)
        self.manifest["request_document_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "evaluation request fingerprint"):
            import_worker_edge(self.manifest, self.baseline, self.candidate)
