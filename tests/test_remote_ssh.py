import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from remote_ssh import connect_server, load_ssh_settings


class FakeClient:
    def __init__(self) -> None:
        self.loaded_host_keys: str | None = None
        self.host_key_policy = None
        self.connection = None
        self.closed = False

    def load_host_keys(self, filename: str) -> None:
        self.loaded_host_keys = filename

    def set_missing_host_key_policy(self, policy) -> None:
        self.host_key_policy = policy

    def connect(self, hostname: str, **kwargs) -> None:
        self.connection = (hostname, kwargs)

    def close(self) -> None:
        self.closed = True


class FakeParamiko:
    class RejectPolicy:
        pass

    def __init__(self) -> None:
        self.client = FakeClient()

    def SSHClient(self) -> FakeClient:
        return self.client


def _settings(tmp_path: Path):
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("example.invalid ssh-ed25519 AAAATESTKEY\n", encoding="utf-8")
    return load_ssh_settings(
        {
            "QKT_SERVER_HOST": "example.invalid",
            "QKT_SERVER_USER": "research-user",
            "QKT_SERVER_PASSWORD": "test-password",
            "QKT_SERVER_PORT": "2202",
            "QKT_SERVER_KNOWN_HOSTS": str(known_hosts),
        }
    )


def test_load_ssh_settings_reads_external_config_and_known_hosts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    assert settings.host == "example.invalid"
    assert settings.username == "research-user"
    assert settings.port == 2202
    assert settings.known_hosts == (tmp_path / "known_hosts").resolve()


def test_load_ssh_settings_rejects_missing_credentials_without_echoing_values(
    tmp_path: Path,
) -> None:
    secret = "must-not-appear-in-errors"
    with pytest.raises(ValueError) as error:
        load_ssh_settings({"QKT_SERVER_PASSWORD": secret}, home=tmp_path)

    assert secret not in str(error.value)
    assert "QKT_SERVER_HOST" in str(error.value)
    assert "QKT_SERVER_USER" in str(error.value)


def test_load_ssh_settings_requires_existing_known_hosts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="known_hosts file was not found"):
        load_ssh_settings(
            {
                "QKT_SERVER_HOST": "example.invalid",
                "QKT_SERVER_USER": "research-user",
                "QKT_SERVER_PASSWORD": "test-password",
            },
            home=tmp_path,
        )


def test_connect_server_loads_pinned_keys_and_rejects_unknown_hosts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    paramiko = FakeParamiko()

    client = connect_server(settings, paramiko_module=paramiko)

    assert client is paramiko.client
    assert client.loaded_host_keys == str(settings.known_hosts)
    assert isinstance(client.host_key_policy, FakeParamiko.RejectPolicy)
    assert client.connection == (
        "example.invalid",
        {
            "port": 2202,
            "username": "research-user",
            "password": "test-password",
            "timeout": 15,
            "allow_agent": False,
            "look_for_keys": False,
        },
    )


def test_connect_server_closes_client_and_hides_connection_details(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    paramiko = FakeParamiko()

    def fail_connect(hostname: str, **kwargs) -> None:
        raise OSError(f"authentication failed for {hostname} with test-password")

    paramiko.client.connect = fail_connect
    with pytest.raises(RuntimeError, match="SSH connection failed") as error:
        connect_server(settings, paramiko_module=paramiko)

    assert paramiko.client.closed
    assert "test-password" not in str(error.value)
    assert "example.invalid" not in str(error.value)
