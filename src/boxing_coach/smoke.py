"""Reproducibility smoke test for the T03 engineering skeleton."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path

from .model_assets import AssetError, find_project_root, load_manifest, select_asset, verify_asset


EXPECTED_DISTRIBUTIONS = {
    "jsonschema": "4.26.0",
    "mediapipe": "1.1.1",
    "numpy": "2.5.4",
    "opencv-contrib-python": "5.0.0.93",
    "streamlit": "1.66.0",
}


class SmokeError(RuntimeError):
    """Raised when an engineering smoke check fails."""


def verify_python() -> None:
    if sys.version_info[:2] != (3, 12):
        raise SmokeError(
            f"Python 3.12 is required by the verified lock; got {sys.version_info.major}."
            f"{sys.version_info.minor}."
        )
    print(f"Python: {sys.version.split()[0]}")


def verify_distributions() -> None:
    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise SmokeError(f"Missing locked dependency: {distribution}") from exc
        if actual != expected:
            raise SmokeError(
                f"Dependency mismatch for {distribution}: expected {expected}, got {actual}."
            )
        print(f"Dependency: {distribution}=={actual}")


def verify_contract_fixtures(project_root: Path) -> None:
    command = [sys.executable, str(project_root / "tools" / "validate_contracts.py")]
    completed = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "contract validation returned a non-zero exit code"
        raise SmokeError(detail)


def verify_model_runtime(project_root: Path) -> None:
    import mediapipe as mp
    import numpy as np

    manifest = load_manifest(project_root)
    asset = select_asset(manifest)
    model_path = project_root / "models" / str(asset["filename"])
    verify_asset(model_path, asset)

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
    )
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.zeros((64, 64, 3), dtype=np.uint8),
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(image)
    print(f"Model runtime: verified synthetic blank frame, poses={len(result.pose_landmarks)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the locked runtime and synthetic contract fixtures."
    )
    parser.add_argument(
        "--with-model",
        action="store_true",
        help="Also verify the downloaded model hash and run a blank-frame model smoke test.",
    )
    parser.add_argument("--project-root", type=Path, help="Explicit checkout root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = find_project_root(args.project_root)
        print(f"Project root: {root}")
        verify_python()
        verify_distributions()
        verify_contract_fixtures(root)
        if args.with_model:
            verify_model_runtime(root)
    except (AssetError, SmokeError, OSError) as exc:
        print(f"SMOKE_ERROR: {exc}", file=sys.stderr)
        return 1
    print("T03_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
