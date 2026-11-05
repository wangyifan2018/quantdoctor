import unittest

from quantdoctor.contracts import Capability, CapabilityState, WorkerJob, WorkerResult


class WorkerContractTests(unittest.TestCase):
    def setUp(self):
        self.document = {
            "schema_version": 1,
            "stage": "infer",
            "backend": "edge-cli-v1",
            "profile_id": "thor-v1",
            "inputs": {"engine_bundle": "a" * 64, "requests": "b" * 64},
        }
        self.job = WorkerJob.parse(self.document)

    def test_identity_is_order_independent_and_binds_inputs(self):
        self.document["inputs"] = dict(reversed(list(self.document["inputs"].items())))
        self.assertEqual(WorkerJob.parse(self.document).id, self.job.id)
        self.document["inputs"]["requests"] = "c" * 64
        self.assertNotEqual(WorkerJob.parse(self.document).id, self.job.id)

    def test_job_rejects_missing_roles_and_paths(self):
        self.document["inputs"]["requests"] = "/path/to/requests.json"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            WorkerJob.parse(self.document)
        del self.document["inputs"]["requests"]
        with self.assertRaisesRegex(ValueError, "artifact roles"):
            WorkerJob.parse(self.document)

    def test_result_must_bind_to_job_and_output_roles(self):
        result = {
            "schema_version": 1, "job_id": self.job.id, "profile_id": self.job.profile_id,
            "source": "fixture", "status": "succeeded", "outputs": {"responses": "c" * 64},
            "message": "",
        }
        self.assertEqual(WorkerResult.parse(result, self.job).source, "fixture")
        with self.assertRaisesRegex(ValueError, "does not match"):
            WorkerResult.parse({**result, "profile_id": "orin-v1"}, self.job)
        with self.assertRaisesRegex(ValueError, "artifact roles"):
            WorkerResult.parse({**result, "outputs": {}}, self.job)
        with self.assertRaisesRegex(ValueError, "requires a reason"):
            WorkerResult.parse({**result, "status": "unsupported", "outputs": {}}, self.job)

    def test_verified_requires_evidence(self):
        with self.assertRaisesRegex(ValueError, "requires evidence"):
            Capability(CapabilityState.VERIFIED, "test")
        pending = Capability(CapabilityState.UNVERIFIED, "no device run")
        self.assertIsNone(pending.evidence_id)
