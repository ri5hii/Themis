"""Artifact provenance: stamps, data hashes, and retrain backups.

Every trainable artifact (classifier, grounding gate, LoRA adapters) is
stamped with training time, git commit, package version, and a SHA-256 of
its training rows; retraining moves the previous artifact into
``models/backups/`` instead of overwriting it. analyze.py records which
artifact versions a run consumed under ``artifacts`` in its output.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_stamp() -> dict:
    """Current repo state: ``{'git_commit': short-hash, 'git_dirty': bool}``.

    Both values are None when the process is not inside a git repository.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    if commit.returncode != 0 or not commit.stdout.strip():
        return {"git_commit": None, "git_dirty": None}
    return {
        "git_commit": commit.stdout.strip(),
        "git_dirty": bool(dirty.stdout.strip()),
    }


def data_sha256(rows: list) -> str:
    """Deterministic hash over the canonical (sorted) serialization of rows.

    ``rows`` is a list of tuples/dicts; the same data always hashes to the
    same value regardless of input order.
    """
    canonical = json.dumps(
        sorted(rows, key=lambda r: json.dumps(r, ensure_ascii=False, sort_keys=True, default=str)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def package_version() -> str:
    """Installed legalrag package version; '0.0.0' when not importable."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("legalrag")
    except (PackageNotFoundError, ImportError, OSError):
        return "0.0.0"


def artifact_stamp(rows: list) -> dict:
    """Provenance stamp for one training run over ``rows``."""
    stamp = git_stamp()
    stamp["trained_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp["package_version"] = package_version()
    stamp["data_sha256"] = data_sha256(rows)
    return stamp


def backup_existing(
    out_dir: Path,
    filenames: list[str],
    backup_root: Path,
    kind: str,
) -> Path | None:
    """Move existing artifact files to ``backup_root/kind/<stamp>/``.

    Returns the backup directory, or None when nothing existed (fresh
    training, no previous artifact to preserve).
    """
    existing = [name for name in filenames if (out_dir / name).exists()]
    if not existing:
        return None
    rev = git_stamp().get("git_commit") or "nogit"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = backup_root / kind / f"{ts}-{rev}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in existing:
        (out_dir / name).replace(dest / name)
    return dest