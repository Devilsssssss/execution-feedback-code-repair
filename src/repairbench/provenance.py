from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: str | Path, repo_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _file_record(path: str | Path, repo_root: Path) -> dict[str, Any]:
    file_path = Path(path).resolve()
    return {
        "path": _display_path(file_path, repo_root),
        "sha256": sha256_file(file_path),
        "size_bytes": file_path.stat().st_size,
    }


def _git_record(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commit": None, "dirty": None}
    return {"available": True, "commit": commit, "dirty": bool(status.strip())}


def _source_record(repo_root: Path) -> dict[str, Any]:
    source_root = repo_root / "src" / "repairbench"
    files = {
        path.relative_to(repo_root).as_posix(): sha256_file(path)
        for path in sorted(source_root.glob("*.py"))
    }
    digest = hashlib.sha256()
    for name, file_hash in files.items():
        digest.update(f"{name}\0{file_hash}\n".encode())
    return {"sha256": digest.hexdigest(), "files": files}


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_identifier(value: Any) -> str | None:
    if value is None:
        return None
    identifier = str(value)
    try:
        parts = urlsplit(identifier)
    except ValueError:
        return identifier
    if parts.scheme and parts.netloc:
        host = parts.netloc.rsplit("@", 1)[-1]
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    return identifier


def _model_record(model: Any, run_name: str) -> dict[str, Any]:
    record: dict[str, Any] = {"kind": type(model).__name__, "run_name": run_name}
    for attribute in ("model_id", "adapter_id", "max_new_tokens", "device_map", "dtype"):
        value = getattr(model, attribute, None)
        if value is not None:
            record[attribute] = _safe_identifier(value)

    loaded_model = getattr(model, "_model", None)
    model_config = getattr(loaded_model, "config", None)
    if model_config is not None:
        record["resolved_model_commit"] = getattr(model_config, "_commit_hash", None)

    tokenizer = getattr(model, "_tokenizer", None)
    if tokenizer is not None:
        record["tokenizer_id"] = _safe_identifier(getattr(tokenizer, "name_or_path", None))
        tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
        record["resolved_tokenizer_commit"] = tokenizer_kwargs.get("_commit_hash")
    return record


def _executor_record(executor: Any, requested: str | None) -> dict[str, Any]:
    parameters = asdict(executor) if is_dataclass(executor) else {}
    return {
        "requested": requested,
        "kind": type(executor).__name__,
        "parameters": parameters,
    }


def _config_record(args: Any) -> dict[str, Any]:
    keys = (
        "command",
        "condition",
        "max_repairs",
        "max_new_tokens",
        "device_map",
        "limit",
        "seed",
        "shuffle_feedback_seed",
        "timeout",
        "allow_unsafe_local",
    )
    return {key: getattr(args, key, None) for key in keys}


def _code_environment(repo_root: Path) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("transformers", "tokenizers", "torch", "peft")
        },
        "git": _git_record(repo_root),
        "source": _source_record(repo_root),
    }


def build_evaluation_manifest(
    args: Any,
    model: Any,
    executor: Any,
    *,
    run_name: str,
    records_path: str | Path,
    summary_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    safe_run_name = _safe_identifier(run_name)
    return {
        "schema_version": 1,
        "run_id": str(uuid4()),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_name": safe_run_name,
        "inputs": {
            "tasks": _file_record(args.tasks, root),
            "bugs": _file_record(args.bugs, root),
        },
        "configuration": _config_record(args),
        "model": _model_record(model, safe_run_name or type(model).__name__),
        "executor": _executor_record(executor, getattr(args, "executor", None)),
        "environment": _code_environment(root),
        "outputs": {
            "records": _file_record(records_path, root),
            "summary": _file_record(summary_path, root),
        },
    }


def write_evaluation_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
