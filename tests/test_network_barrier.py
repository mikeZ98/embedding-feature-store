"""The suite runs with the network walled off.

pytest-socket (``--disable-socket --allow-unix-socket`` in addopts) blocks
every non-AF_UNIX socket and the resolver entry points (``getaddrinfo``,
``gethostbyname``), so neither a connection nor a DNS lookup can leave the
process. AF_UNIX stays open because asyncio event loops need a
``socketpair()`` for their self-pipe.
"""

from __future__ import annotations

import asyncio
import re
import socket
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pytest_socket import SocketBlockedError

from embedding_feature_store.storage.qdrant_http import QdrantHttpStore

# pytest-socket warns before raising; the raise is what these tests assert.
pytestmark = pytest.mark.filterwarnings("ignore:A test tried to use socket:UserWarning")

INET_FAMILIES = [
    pytest.param(socket.AF_INET, id="AF_INET"),
    pytest.param(socket.AF_INET6, id="AF_INET6"),
]
SOCKET_TYPES = [
    pytest.param(socket.SOCK_STREAM, id="tcp"),
    pytest.param(socket.SOCK_DGRAM, id="udp"),
]


def test_barrier_is_enabled_for_the_whole_session(request: pytest.FixtureRequest) -> None:
    assert request.config.getoption("disable_socket") is True
    assert request.config.getoption("allow_unix_socket") is True


@pytest.mark.parametrize("family", INET_FAMILIES)
@pytest.mark.parametrize("kind", SOCKET_TYPES)
def test_inet_sockets_cannot_be_created(family: socket.AddressFamily, kind: int) -> None:
    with pytest.raises(SocketBlockedError):
        socket.socket(family, kind)


@pytest.mark.parametrize("address", [("127.0.0.1", 9), ("::1", 9)], ids=["ipv4", "ipv6"])
def test_blocking_connect_is_refused(address: tuple[str, int]) -> None:
    with pytest.raises(SocketBlockedError):
        socket.create_connection(address, timeout=0.1)


@pytest.mark.asyncio
async def test_asyncio_connections_are_refused() -> None:
    with pytest.raises(SocketBlockedError):
        await asyncio.open_connection("127.0.0.1", 9)


def leaves(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in leaves(sub)]
    return [exc]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        pytest.param("http://qdrant:6333", id="host-name"),
        pytest.param("http://127.0.0.1:6333", id="ip-literal"),
    ],
)
async def test_production_http_adapter_cannot_reach_the_network(base_url: str) -> None:
    """Without an injected mock transport, the real Qdrant client is stopped cold.

    A host name fails at resolution; an IP literal fails at socket creation,
    inside anyio's happy-eyeballs task group (hence the exception group).
    """
    async with QdrantHttpStore.connect(base_url, timeout_s=0.5) as store:
        with pytest.raises((SocketBlockedError, BaseExceptionGroup)) as exc:
            await store.collection_exists("anything")
    blocked = leaves(exc.value)
    assert blocked
    assert all(isinstance(leaf, SocketBlockedError) for leaf in blocked)


@pytest.mark.parametrize(
    ("resolve", "entry_point"),
    [
        pytest.param(
            lambda: socket.getaddrinfo("qdrant.example", 6333), "getaddrinfo", id="getaddrinfo"
        ),
        pytest.param(
            lambda: socket.getaddrinfo("127.0.0.1", 6333), "getaddrinfo", id="getaddrinfo-literal"
        ),
        pytest.param(
            lambda: socket.gethostbyname("qdrant.example"), "gethostbyname", id="gethostbyname"
        ),
    ],
)
def test_resolver_is_blocked(resolve: Callable[[], Any], entry_point: str) -> None:
    with pytest.raises(SocketBlockedError, match=re.escape(f"socket.{entry_point}")):
        resolve()


@pytest.mark.asyncio
async def test_httpx_never_resolves_names() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(SocketBlockedError):
            await client.get("http://qdrant.example:6333/collections")


def test_unix_sockets_are_allowed() -> None:
    left, right = socket.socketpair()
    with left, right:
        assert left.family == socket.AF_UNIX
        left.sendall(b"ping")
        assert right.recv(4) == b"ping"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        assert sock.family == socket.AF_UNIX


def test_event_loops_only_need_unix_socketpairs(monkeypatch: pytest.MonkeyPatch) -> None:
    families: list[socket.AddressFamily] = []
    real_socketpair = socket.socketpair

    def spy(*args: Any, **kwargs: Any) -> tuple[socket.socket, socket.socket]:
        pair = real_socketpair(*args, **kwargs)
        families.extend(s.family for s in pair)
        return pair

    monkeypatch.setattr(socket, "socketpair", spy)
    loop = asyncio.new_event_loop()
    try:
        assert loop.run_until_complete(asyncio.sleep(0, result="ran")) == "ran"
    finally:
        loop.close()
    assert families, "the loop's self-pipe is a socketpair"
    assert set(families) == {socket.AF_UNIX}
