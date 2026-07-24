"""Test-wide safety net: block real outbound network from the test process.

The unit suite is meant to run fully offline - the LLM-driven pieces (`run_task`,
`run_suite`, and the `llm_judge` grader's `Agent.run`) are always mocked. This
autouse fixture makes that a hard guarantee instead of a convention: if a test
ever reaches a real model client / HTTP endpoint (directly or via httpx/pydantic-ai),
the connection attempt raises instead of silently spending tokens.

Loopback is still allowed so Playwright's local driver/browser plumbing in
test_browser.py keeps working. A test that genuinely needs the network can opt out
with `@pytest.mark.allow_network`.
"""
from __future__ import annotations

import ipaddress
import socket

import pytest

_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex


def _is_local(address: object) -> bool:
    """True for connections we allow: AF_UNIX (str path) and loopback IPs."""
    if not isinstance(address, tuple):  # AF_UNIX and the like - local by nature
        return True
    host = address[0]
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Unresolved hostname rather than an IP - only localhost is local.
        return host in ("localhost", "localhost.localdomain")


def _guarded(real):
    def connect(self, address, *args, **kwargs):
        if not _is_local(address):
            raise RuntimeError(
                f"Blocked outbound network connection to {address!r} during tests. "
                "Unit tests must not make real network/LLM calls - mock run_task, "
                "run_suite, or grading.Agent.run. Add @pytest.mark.allow_network to "
                "opt out (see tests/conftest.py)."
            )
        return real(self, address, *args, **kwargs)

    return connect


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "allow_network: allow this test to make real outbound network connections."
    )


@pytest.fixture(autouse=True)
def _block_external_network(request, monkeypatch):
    if request.node.get_closest_marker("allow_network") is not None:
        return
    monkeypatch.setattr(socket.socket, "connect", _guarded(_REAL_CONNECT))
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded(_REAL_CONNECT_EX))
