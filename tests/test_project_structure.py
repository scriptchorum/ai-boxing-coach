from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boxing_coach.model_assets import (  # noqa: E402
    AssetError,
    find_project_root,
    load_manifest,
    select_asset,
    verify_asset,
)


class ProjectStructureTests(unittest.TestCase):
    def test_project_root_is_discoverable(self) -> None:
        self.assertEqual(find_project_root(ROOT / "tests"), ROOT)

    def test_model_manifest_uses_pinned_generation_and_hash(self) -> None:
        manifest = load_manifest(ROOT)
        asset = select_asset(manifest)
        self.assertIn("generation=", asset["url"])
        self.assertEqual(len(asset["sha256"]), 64)
        self.assertFalse(asset["tracked_in_git"])

    def test_asset_verifier_accepts_matching_file(self) -> None:
        payload = b"synthetic-model-placeholder"
        asset = {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.task"
            path.write_bytes(payload)
            verify_asset(path, asset)

    def test_asset_verifier_rejects_bad_hash(self) -> None:
        payload = b"synthetic-model-placeholder"
        asset = {"size_bytes": len(payload), "sha256": "0" * 64}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.task"
            path.write_bytes(payload)
            with self.assertRaises(AssetError):
                verify_asset(path, asset)

    def test_lock_contains_verified_runtime(self) -> None:
        lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
        for requirement in (
            "av==19.0.1",
            "mediapipe==1.1.1",
            "numpy==2.5.4",
            "opencv-contrib-python==5.0.0.93",
            "streamlit==1.66.0",
        ):
            self.assertIn(requirement, lock)

    def test_manifest_is_valid_json_without_private_paths(self) -> None:
        text = (ROOT / "configs" / "model-assets.json").read_text(encoding="utf-8")
        parsed = json.loads(text)
        self.assertEqual(parsed["schema_version"], "1.0.0")
        self.assertNotIn(str(Path.home()), text)


if __name__ == "__main__":
    unittest.main()
