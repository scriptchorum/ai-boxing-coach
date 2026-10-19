"""Download and verify untracked model assets from a pinned manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class AssetError(RuntimeError):
    """Raised when a model manifest or asset fails validation."""


def find_project_root(start: Path | None = None) -> Path:
    """Find the checkout root without relying on a machine-specific path."""

    candidates: list[Path] = []
    if start is not None:
        resolved = start.resolve()
        candidates.extend([resolved, *resolved.parents])
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    source_root = Path(__file__).resolve().parents[2]
    candidates.extend([source_root, *source_root.parents])

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "schemas" / "contracts.schema.json"
        ).is_file():
            return candidate
    raise AssetError("Cannot locate the project root containing pyproject.toml and schemas/.")


def load_manifest(project_root: Path) -> dict[str, Any]:
    manifest_path = project_root / "configs" / "model-assets.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetError(f"Cannot read model manifest: {manifest_path}") from exc

    if manifest.get("schema_version") != "1.0.0":
        raise AssetError("Unsupported model manifest schema_version.")
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise AssetError("Model manifest must contain at least one asset.")
    return manifest


def select_asset(manifest: dict[str, Any], asset_id: str | None = None) -> dict[str, Any]:
    selected_id = asset_id or manifest.get("default_asset_id")
    for asset in manifest["assets"]:
        if asset.get("asset_id") == selected_id:
            filename = Path(str(asset.get("filename", "")))
            if not filename.name or filename.name != str(filename):
                raise AssetError("Model filename must be a simple filename without directories.")
            return asset
    raise AssetError(f"Unknown model asset: {selected_id}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path: Path, asset: dict[str, Any]) -> None:
    if not path.is_file():
        raise AssetError(f"Model asset is missing: {path}")
    expected_size = int(asset["size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise AssetError(
            f"Model size mismatch for {path.name}: expected {expected_size}, got {actual_size}."
        )
    expected_hash = str(asset["sha256"]).lower()
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise AssetError(
            f"Model SHA-256 mismatch for {path.name}: expected {expected_hash}, got {actual_hash}."
        )


def sync_asset(project_root: Path, asset: dict[str, Any]) -> Path:
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / str(asset["filename"])

    if destination.exists():
        verify_asset(destination, asset)
        print(f"Model already verified: {destination}")
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        str(asset["url"]),
        headers={"User-Agent": "ai-boxing-coach-model-sync/0.1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        verify_asset(temporary, asset)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print(f"Downloaded and verified: {destination}")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download or verify model assets without adding weights to Git."
    )
    parser.add_argument("action", choices=("list", "sync", "verify"), nargs="?", default="list")
    parser.add_argument("--asset-id", help="Asset ID; defaults to the manifest default.")
    parser.add_argument("--project-root", type=Path, help="Explicit checkout root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        project_root = find_project_root(args.project_root)
        manifest = load_manifest(project_root)
        asset = select_asset(manifest, args.asset_id)
        path = project_root / "models" / str(asset["filename"])

        if args.action == "list":
            print(
                f"{asset['asset_id']} -> {asset['filename']} "
                f"({asset['size_bytes']} bytes, sha256={asset['sha256']})"
            )
        elif args.action == "sync":
            sync_asset(project_root, asset)
        else:
            verify_asset(path, asset)
            print(f"Model verified: {path}")
    except (AssetError, OSError, KeyError, ValueError, urllib.error.URLError) as exc:
        print(f"MODEL_ASSET_ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
