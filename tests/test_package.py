"""Baseline package and test-harness invariants."""

from __future__ import annotations

import socket
from importlib import metadata, resources

import pytest
from pytest_socket import SocketBlockedError

import embedding_feature_store
from embedding_feature_store import __version__


def test_version_matches_distribution_metadata() -> None:
    assert __version__ == metadata.version("embedding-feature-store")


def test_public_api_surface_is_explicit() -> None:
    assert embedding_feature_store.__all__ == ["__version__"]


def test_ships_pep561_marker() -> None:
    marker = resources.files("embedding_feature_store").joinpath("py.typed")
    assert marker.is_file()


@pytest.mark.filterwarnings("ignore:A test tried to use socket.socket:UserWarning")
def test_network_sockets_are_blocked() -> None:
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_hypothesis_storage_is_project_local() -> None:
    from hypothesis.configuration import storage_directory  # noqa: PLC0415

    from tests.conftest import HYPOTHESIS_DIR  # noqa: PLC0415

    assert storage_directory("examples").path == HYPOTHESIS_DIR / "examples"
