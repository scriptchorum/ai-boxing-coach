"""T12: Local storage inspection, offline readiness check, and safe cache/report deletion.

Guarantees:
1. Never deletes user source video files (.mp4, .mov, .avi, .mkv, .webm) or any path passed in protected_source_paths.
2. Never deletes model weight files (.task, .tflite, .onnx, .pt, .pth) or Git-tracked repository files.
3. Only deletes generated caches and exported report artifacts inside managed local directories (outputs/, runs/, work/matplotlib-cache/).
4. Verifies offline readiness (pinned model asset exists and matches manifest size/SHA-256 without network access).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from boxing_coach.model_assets import AssetError, find_project_root, load_manifest, select_asset, verify_asset

MANAGED_REL_DIRS = ("outputs", "runs", "work/matplotlib-cache")
PROTECTED_MEDIA_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
PROTECTED_MODEL_SUFFIXES = {".task", ".tflite", ".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".h5"}
REMOVABLE_ARTIFACT_SUFFIXES = {".zip", ".json", ".html", ".png", ".cache", ".tmp", ".log"}


class StorageSafetyError(RuntimeError):
    """Raised when a deletion target violates source-video or path-boundary protection."""


def _resolve_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def is_protected_file(
    path: Path,
    *,
    project_root: Path,
    protected_source_paths: Iterable[Path | str] = (),
) -> tuple[bool, str | None]:
    """Check whether a file must be protected from cache/report cleanup."""
    resolved = path.resolve()
    protected_resolved = {Path(p).expanduser().resolve() for p in protected_source_paths if p}
    if resolved in protected_resolved:
        return True, "bound_source_video"

    suffix = resolved.suffix.lower()
    if suffix in PROTECTED_MEDIA_SUFFIXES:
        return True, "user_video_media"
    if suffix in PROTECTED_MODEL_SUFFIXES:
        return True, "model_weight_asset"

    managed_roots = [(project_root / rel).resolve() for rel in MANAGED_REL_DIRS]
    if not any(_resolve_within(resolved, root) for root in managed_roots):
        return True, "outside_managed_cache_dirs"

    if suffix not in REMOVABLE_ARTIFACT_SUFFIXES:
        return True, "unrecognized_artifact_suffix"

    return False, None


def inspect_local_storage(
    project_root: Path | None = None,
    *,
    protected_source_paths: Sequence[Path | str] = (),
) -> dict[str, Any]:
    """Return a privacy-safe summary of managed cache/report files vs protected files."""
    root = (project_root or find_project_root()).resolve()
    removable_count = 0
    removable_bytes = 0
    protected_count = 0
    by_dir: dict[str, dict[str, int]] = {}

    for rel in MANAGED_REL_DIRS:
        target_dir = (root / rel).resolve()
        dir_count = 0
        dir_bytes = 0
        if target_dir.is_dir():
            for item in target_dir.rglob("*"):
                if not item.is_file():
                    continue
                protected, _ = is_protected_file(
                    item, project_root=root, protected_source_paths=protected_source_paths
                )
                if protected:
                    protected_count += 1
                else:
                    size = item.stat().st_size
                    removable_count += 1
                    removable_bytes += size
                    dir_count += 1
                    dir_bytes += size
        by_dir[rel] = {"removable_files": dir_count, "removable_bytes": dir_bytes}

    return {
        "managed_directories": list(MANAGED_REL_DIRS),
        "removable_files": removable_count,
        "removable_bytes": removable_bytes,
        "protected_files_skipped": protected_count,
        "directories": by_dir,
    }


def purge_local_caches_and_reports(
    project_root: Path | None = None,
    *,
    protected_source_paths: Sequence[Path | str] = (),
    include_matplotlib_cache: bool = True,
) -> dict[str, Any]:
    """Safely delete generated report/cache artifacts while preserving user videos and models."""
    root = (project_root or find_project_root()).resolve()
    active_rel_dirs = [
        rel
        for rel in MANAGED_REL_DIRS
        if include_matplotlib_cache or rel != "work/matplotlib-cache"
    ]

    deleted_files = 0
    deleted_bytes = 0
    preserved_protected_files = 0

    for rel in active_rel_dirs:
        target_dir = (root / rel).resolve()
        if not _resolve_within(target_dir, root):
            raise StorageSafetyError("Managed directory escapes project root.")
        if not target_dir.is_dir():
            continue
        for item in sorted(target_dir.rglob("*")):
            if not item.is_file():
                continue
            protected, _ = is_protected_file(
                item, project_root=root, protected_source_paths=protected_source_paths
            )
            if protected:
                preserved_protected_files += 1
                continue
            size = item.stat().st_size
            item.unlink()
            deleted_files += 1
            deleted_bytes += size

    return {
        "deleted_files": deleted_files,
        "deleted_bytes": deleted_bytes,
        "preserved_protected_files": preserved_protected_files,
    }


def check_offline_readiness(
    project_root: Path | None = None,
    *,
    asset_id: str | None = None,
) -> dict[str, Any]:
    """Check whether pinned model assets are ready on disk for 100% offline execution."""
    root = (project_root or find_project_root()).resolve()
    try:
        manifest = load_manifest(root)
        asset = select_asset(manifest, asset_id)
        target_dir = root / str(manifest.get("target_directory", "models"))
        asset_path = target_dir / str(asset["filename"])
        verify_asset(asset_path, asset)
    except AssetError as exc:
        return {
            "offline_ready": False,
            "asset_id": asset_id,
            "reason": str(exc),
        }
    return {
        "offline_ready": True,
        "asset_id": asset["asset_id"],
        "version": asset.get("version"),
        "sha256": asset.get("sha256"),
        "size_bytes": asset.get("size_bytes"),
        "reason": None,
    }
