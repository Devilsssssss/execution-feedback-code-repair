"""Run a shell command on the training server via SSH (paramiko).

Usage:
  python scripts/remote_exec.py "<command>" [timeout_seconds]

Credentials come from the QKT_SERVER_PASSWORD environment variable so the
password is never embedded in command lines or logs.
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import paramiko


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "hostname"
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    password = os.environ.get("QKT_SERVER_PASSWORD")
    if not password:
        print("ERROR: set QKT_SERVER_PASSWORD first", file=sys.stderr)
        return 2
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "10.199.161.82",
        port=22,
        username="qianketong",
        password=password,
        timeout=15,
        allow_agent=False,
        look_for_keys=False,
    )
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    if err:
        print("[stderr]", err, file=sys.stderr, end="" if err.endswith("\n") else "\n")
    client.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
