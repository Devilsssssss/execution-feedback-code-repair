"""Run a shell command on the configured experiment server via SSH.

Usage:
  python scripts/remote_exec.py "<command>" [timeout_seconds]

Set QKT_SERVER_HOST, QKT_SERVER_USER, and QKT_SERVER_PASSWORD in the local
environment. Optional settings are QKT_SERVER_PORT and QKT_SERVER_KNOWN_HOSTS.
The default known-hosts file is ~/.ssh/known_hosts; unknown hosts are rejected.
"""

from __future__ import annotations

import sys

from remote_ssh import connect_server


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "hostname"
    try:
        timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        client = connect_server()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        code = stdout.channel.recv_exit_status()
        if out:
            print(out, end="" if out.endswith("\n") else "\n")
        if err:
            print("[stderr]", err, file=sys.stderr, end="" if err.endswith("\n") else "\n")
        return code
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
