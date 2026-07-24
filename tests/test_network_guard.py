import socket

import pytest


def test_external_connection_is_blocked():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="Blocked outbound network connection"):
            sock.connect(("example.com", 80))
    finally:
        sock.close()


def test_loopback_connection_is_allowed():
    # A loopback connect is permitted by the guard; it fails with a normal socket
    # error (nothing is listening), not the guard's RuntimeError.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.1)
    try:
        with pytest.raises(OSError) as exc_info:
            sock.connect(("127.0.0.1", 1))
        assert not isinstance(exc_info.value, RuntimeError)
    finally:
        sock.close()


@pytest.mark.allow_network
def test_allow_network_marker_disables_guard():
    # With the opt-out marker the guard is not installed, so an external connect
    # raises a normal socket error (DNS/refused/timeout), never the guard's RuntimeError.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.1)
    try:
        with pytest.raises(OSError) as exc_info:
            sock.connect(("192.0.2.1", 80))  # TEST-NET-1, guaranteed unroutable
        assert not isinstance(exc_info.value, RuntimeError)
    finally:
        sock.close()
