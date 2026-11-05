from pathlib import Path
import unittest

from quantdoctor.model import Comparison, load_json
from quantdoctor.preflight import TargetProfile, preflight
from quantdoctor.qat import EDGE_COMMIT


class PreflightTests(unittest.TestCase):
    def setUp(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "demo-comparison.json"
        self.data = load_json(example)

    def preflight(self, platform, commit=EDGE_COMMIT):
        profile = TargetProfile.parse({"schema_version": 1, "id": "fixture-target",
                                       "platform": platform, "edge_commit": commit})
        return preflight(Comparison.parse(self.data), profile, "ASR-014")

    def test_audio_repair_removes_combo_warning_but_orin_stays_blocked(self):
        result = self.preflight("jetson_orin")
        self.assertFalse(result["deployment_verified"])
        self.assertEqual(result["checks"]["candidate"]["state"], "blocked")
        self.assertEqual(result["checks"]["proposed"]["state"], "blocked")
        issues = result["checks"]["proposed"]["issues"]
        self.assertEqual([issue["code"] for issue in issues], ["orin_precision"])
        self.assertEqual(issues[0]["components"], ["backbone"])

    def test_thor_and_unrecognized_version_do_not_promote_to_verified(self):
        result = self.preflight("drive_thor")
        self.assertEqual(result["checks"]["proposed"]["state"], "unverified")
        self.assertEqual(result["checks"]["proposed"]["issues"], [])
        self.assertEqual(result["checks"]["candidate"]["issues"][0]["code"], "asr_empty_output_risk")
        result = self.preflight("jetson_orin", "1" * 40)
        self.assertEqual(result["checks"]["candidate"]["issues"][0]["code"], "edge_rules_unavailable")
        self.assertEqual(result["checks"]["candidate"]["state"], "unverified")

    def test_mac_blocks_cuda_path(self):
        result = self.preflight("macos")
        self.assertEqual(result["checks"]["proposed"]["issues"][0]["code"], "macos_cuda_runtime")
        self.assertEqual(result["checks"]["proposed"]["state"], "blocked")
