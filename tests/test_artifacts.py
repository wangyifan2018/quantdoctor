from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from quantdoctor.artifacts import artifact_digest, verify_artifacts


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_directory_digest_stable_across_relocation_and_sensitive_to_paths(self):
        first = self.root / "first"
        second = self.root / "second"
        for directory, filenames in ((first, ("a", "b")), (second, ("b", "a"))):
            directory.mkdir()
            for filename in filenames:
                (directory / filename).write_text(filename)
        self.assertEqual(artifact_digest(first), artifact_digest(second))
        (second / "a").rename(second / "c")
        self.assertNotEqual(artifact_digest(first), artifact_digest(second))

    def test_verify_role_detects_payload_changes_and_missing_files(self):
        path = self.root / "responses"
        path.write_text('{"responses":[]}')
        expected = {"responses": artifact_digest(path)}
        self.assertEqual(verify_artifacts(self.root, expected), expected)
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            verify_artifacts(self.root, expected)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "regular artifact"):
            verify_artifacts(self.root, expected)

    def test_symlinks_are_rejected(self):
        target = self.root / "target"
        target.write_text("payload")
        tree = self.root / "tree"
        tree.mkdir()
        (tree / "link").symlink_to(target)
        with self.assertRaisesRegex(ValueError, "regular artifact"):
            artifact_digest(tree)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            artifact_digest(tree / "link")
