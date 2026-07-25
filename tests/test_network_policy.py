from __future__ import annotations

import socket

import pytest
from pytest_socket import SocketBlockedError


def test_external_network_sockets_are_blocked() -> None:
    with pytest.warns(UserWarning, match="tried to use socket"), pytest.raises(
        SocketBlockedError
    ):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
