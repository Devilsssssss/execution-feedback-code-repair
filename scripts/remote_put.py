"""Upload a local file to the configured experiment server via SFTP.

Usage:
  python scripts/remote_put.py <local_path> <remote_path>

Set QKT_SERVER_HOST, QKT_SERVER_USER, and QKT_SERVER_PASSWORD in the local
environment. Optional settings are QKT_SERVER_PORT and QKT_SERVER_KNOWN_HOSTS.
The default known-hosts file is ~/.ssh/known_hosts; unknown hosts are rejected.
"""

from __future__ import annotations

import sys

from remote_ssh import connect_server


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: remote_put.py <local> <remote>", file=sys.stderr)
        return 2
    local, remote = sys.argv[1], sys.argv[2]
    try:
        client = connect_server()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        sftp = client.open_sftp()
        try:
            sftp.put(local, remote)
        finally:
            sftp.close()
        print(f"uploaded {local} -> {remote}")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
