"""Shared SSH configuration for the optional remote experiment helpers.

Set QKT_SERVER_HOST, QKT_SERVER_USER, and QKT_SERVER_PASSWORD in the local
environment. Host keys are loaded from QKT_SERVER_KNOWN_HOSTS or the standard
~/.ssh/known_hosts file; unknown hosts are rejected.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SSHSettings:
    host: str
    username: str
    password: str
    port: int
    known_hosts: Path


def load_ssh_settings(
    environ: Mapping[str, str] | None = None, *, home: Path | None = None
) -> SSHSettings:
    """Load required connection settings without storing them in the repository."""
    values = os.environ if environ is None else environ
    required = {
        "QKT_SERVER_HOST": values.get("QKT_SERVER_HOST", "").strip(),
        "QKT_SERVER_USER": values.get("QKT_SERVER_USER", "").strip(),
        "QKT_SERVER_PASSWORD": values.get("QKT_SERVER_PASSWORD", ""),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"missing required environment variables: {', '.join(missing)}")

    try:
        port = int(values.get("QKT_SERVER_PORT", "22"))
    except ValueError as exc:
        raise ValueError("QKT_SERVER_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("QKT_SERVER_PORT must be between 1 and 65535")

    home_dir = Path.home() if home is None else Path(home)
    known_hosts = Path(
        values.get("QKT_SERVER_KNOWN_HOSTS", str(home_dir / ".ssh" / "known_hosts"))
    ).expanduser()
    if not known_hosts.is_file():
        raise ValueError(
            "SSH known_hosts file was not found; configure QKT_SERVER_KNOWN_HOSTS "
            "or add the server host key to ~/.ssh/known_hosts"
        )

    return SSHSettings(
        host=required["QKT_SERVER_HOST"],
        username=required["QKT_SERVER_USER"],
        password=required["QKT_SERVER_PASSWORD"],
        port=port,
        known_hosts=known_hosts.resolve(),
    )


def connect_server(
    settings: SSHSettings | None = None, *, paramiko_module: Any | None = None
) -> Any:
    """Connect using an explicit known_hosts file and reject unknown host keys."""
    if settings is None:
        settings = load_ssh_settings()
    if paramiko_module is None:
        try:
            import paramiko as paramiko_module
        except ImportError as exc:
            raise RuntimeError("install Paramiko to use the remote experiment helpers") from exc

    client = paramiko_module.SSHClient()
    try:
        client.load_host_keys(str(settings.known_hosts))
        client.set_missing_host_key_policy(paramiko_module.RejectPolicy())
        client.connect(
            settings.host,
            port=settings.port,
            username=settings.username,
            password=settings.password,
            timeout=15,
            allow_agent=False,
            look_for_keys=False,
        )
    except Exception as exc:
        with suppress(OSError):
            client.close()
        raise RuntimeError(
            "SSH connection failed; verify the host key, credentials, and server availability"
        ) from exc
    return client
