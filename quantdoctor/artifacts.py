"""Content verification for offline worker handoffs."""

from collections.abc import Mapping
from hashlib import sha256
import json
import os
from pathlib import Path

from .model import file_sha256, require


def _raise_walk_error(error):
    raise error


def artifact_digest(path: Path):
    require(not path.is_symlink(), f"artifact symlinks are not supported: {path}")
    if path.is_file():
        return sha256(b"quantdoctor:file:v1\0" + bytes.fromhex(file_sha256(path))).hexdigest()
    require(path.is_dir(), f"expected a regular artifact file or directory: {path}")
    files = []
    for root, dirs, names in os.walk(path, onerror=_raise_walk_error):
        directory = Path(root)
        for name in dirs:
            require(not (directory / name).is_symlink(), f"artifact symlinks are not supported: {directory / name}")
        for name in names:
            child = directory / name
            require(not child.is_symlink() and child.is_file(), f"expected a regular artifact file: {child}")
            files.append((child.relative_to(path).as_posix(), file_sha256(child)))
    require(files, f"artifact directory is empty: {path}")
    manifest = {"schema_version": 1, "files": sorted(files)}
    encoded = json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256(b"quantdoctor:directory:v1\0" + encoded.encode("ascii")).hexdigest()


def verify_artifacts(root: Path, expected: Mapping[str, str]):
    require(root.is_dir() and not root.is_symlink(), "artifact root must be a local directory")
    verified = {}
    for role, digest in expected.items():
        require(role and Path(role).name == role and role not in {".", ".."}, "invalid artifact role")
        actual = artifact_digest(root / role)
        require(actual == digest, f"{role}: artifact digest mismatch (expected {digest}, got {actual})")
        verified[role] = actual
    return verified
