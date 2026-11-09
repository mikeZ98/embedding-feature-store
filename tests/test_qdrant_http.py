"""Qdrant REST adapter against an offline emulator and hand-crafted responses."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import httpx
import pytest

from embedding_feature_store.core.errors import (
    ConflictError,
    FeatureStoreError,
    MalformedResponseError,
    NotFoundError,
    PermanentError,
    RateLimitedError,
    ServerError,
    StoreConnectionError,
    StoreTimeoutError,
    TransientError,
    ValidationFailedError,
)
from embedding_feature_store.core.ports import VectorStore
from embedding_feature_store.core.retry import Retrier
from embedding_feature_store.core.verification import ExpectationLedger, MigrationVerifier
from embedding_feature_store.schemas import (
    CreateAliasAction,
    DeleteAliasAction,
    DistanceMetric,
    PointRecord,
    RetryPolicy,
    VectorSpec,
)
from embedding_feature_store.storage.qdrant_http import QdrantHttpStore
from tests.fakes import RecordingSleep
from tests.qdrant_emulator import QdrantEmulator, error, ok

BASE = "http://qdrant.test:6333"
SPEC = VectorSpec(dimension=3, distance=DistanceMetric.DOT)

_conforms: VectorStore = QdrantHttpStore(httpx.AsyncClient())


def store_for(handler: Callable[[httpx.Request], httpx.Response]) -> QdrantHttpStore:
    return QdrantHttpStore.connect(BASE, transport=httpx.MockTransport(handler))


def emulated() -> tuple[QdrantHttpStore, QdrantEmulator]:
    emulator = QdrantEmulator()
    return store_for(emulator), emulator


def pt(pid: int, *vector: float, **payload: str | int | float | bool | None) -> PointRecord:
    return PointRecord(point_id=pid, vector=vector, payload=tuple(payload.items()))


def body(request: httpx.Request) -> object:
    return json.loads(request.content)


# ---------------------------------------------------------------------------
# Endpoints, end to end
# ---------------------------------------------------------------------------


class TestCollections:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("metric", list(DistanceMetric))
    async def test_create_describe_roundtrip(self, metric: DistanceMetric) -> None:
        store, emulator = emulated()
        spec = VectorSpec(dimension=8, distance=metric)
        assert not await store.collection_exists("c")
        await store.create_collection("c", spec)
        assert await store.collection_exists("c")
        info = await store.describe_collection("c")
        assert (info.name, info.spec, info.points_count) == ("c", spec, 0)
        create = emulator.requests[1]
        assert (create.method, create.url.path) == ("PUT", "/collections/c")
        assert body(create) == {
            "vectors": {
                "size": 8,
                "distance": {"cosine": "Cosine", "dot": "Dot", "euclid": "Euclid"}[metric],
            }
        }

    @pytest.mark.asyncio
    async def test_create_existing_is_conflict(self) -> None:
        store, _ = emulated()
        await store.create_collection("c", SPEC)
        with pytest.raises(ConflictError, match="already exists"):
            await store.create_collection("c", SPEC)

    @pytest.mark.asyncio
    async def test_delete_is_idempotent(self) -> None:
        store, emulator = emulated()
        await store.create_collection("c", SPEC)
        await store.delete_collection("c")
        await store.delete_collection("c")
        assert "c" not in emulator.collections

    @pytest.mark.asyncio
    async def test_describe_missing(self) -> None:
        store, _ = emulated()
        with pytest.raises(NotFoundError, match="doesn't exist"):
            await store.describe_collection("nope")


class TestPoints:
    @pytest.mark.asyncio
    async def test_upsert_count_scroll_delete(self) -> None:
        store, emulator = emulated()
        await store.create_collection("c", SPEC)
        points = [pt(i, float(i), 0.5, 0.1, tenant="acme", lsn=i, ok=True) for i in range(1, 8)]
        await store.upsert("c", points)
        assert await store.count("c") == 7

        upsert = emulator.requests[-2]
        assert upsert.url.params["wait"] == "true"
        assert body(upsert) == {
            "points": [
                {"id": p.point_id, "vector": list(p.vector), "payload": dict(p.payload)}
                for p in points
            ]
        }

        first = await store.scroll("c", offset=None, limit=3)
        assert [p.point_id for p in first.points] == [1, 2, 3]
        assert first.next_offset == 4
        assert first.points == tuple(points[:3])
        assert "offset" not in body(emulator.requests[-1])  # type: ignore[operator]
        second = await store.scroll("c", offset=first.next_offset, limit=10)
        assert second.points == tuple(points[3:])
        assert second.next_offset is None
        assert body(emulator.requests[-1]) == {
            "limit": 10,
            "offset": 4,
            "with_payload": True,
            "with_vector": True,
        }

        await store.delete_points("c", [2, 3, 99])
        delete = emulator.requests[-1]
        assert (delete.url.path, delete.url.params["wait"]) == (
            "/collections/c/points/delete",
            "true",
        )
        assert body(delete) == {"points": [2, 3, 99]}
        assert await store.count("c") == 5

    @pytest.mark.asyncio
    async def test_payload_types_survive_roundtrip(self) -> None:
        store, _ = emulated()
        await store.create_collection("c", SPEC)
        original = pt(1, 0.1, 0.2, 0.3, s="x", i=3, f=1.5, b=False, n=None)
        await store.upsert("c", [original])
        (back,) = (await store.scroll("c", offset=None, limit=1)).points
        assert back == original
        assert back.canonical_identity == original.canonical_identity

    @pytest.mark.asyncio
    async def test_empty_writes_send_nothing(self) -> None:
        store, emulator = emulated()
        await store.upsert("c", [])
        await store.delete_points("c", [])
        await store.update_aliases([])
        assert emulator.requests == []

    @pytest.mark.asyncio
    async def test_dimension_rejected_by_server(self) -> None:
        store, _ = emulated()
        await store.create_collection("c", SPEC)
        with pytest.raises(ValidationFailedError, match="Vector dimension error"):
            await store.upsert("c", [pt(1, 1.0)])

    @pytest.mark.asyncio
    async def test_verifier_runs_over_http(self) -> None:
        store, _ = emulated()
        await store.create_collection("c", SPEC)
        points = [pt(i, float(i), 1.0, 2.0, tenant="t") for i in range(25)]
        await store.upsert("c", points)
        ledger = ExpectationLedger()
        for p in points:
            ledger.record_upsert(p)
        report = await MigrationVerifier(store, page_size=4).require("c", SPEC, ledger)
        assert report.passed


class TestAliases:
    @pytest.mark.asyncio
    async def test_atomic_swap_endpoint(self) -> None:
        store, emulator = emulated()
        for name in ("old", "new"):
            await store.create_collection(name, SPEC)
        await store.update_aliases([CreateAliasAction(alias="live", collection="old")])
        assert await store.list_aliases() == {"live": "old"}

        await store.update_aliases(
            [DeleteAliasAction(alias="live"), CreateAliasAction(alias="live", collection="new")]
        )
        swap = emulator.requests[-1]
        assert (swap.method, swap.url.path) == ("POST", "/collections/aliases")
        assert body(swap) == {
            "actions": [
                {"delete_alias": {"alias_name": "live"}},
                {"create_alias": {"collection_name": "new", "alias_name": "live"}},
            ]
        }
        assert await store.list_aliases() == {"live": "new"}
        assert (await store.describe_collection("live")).name == "live"

    @pytest.mark.asyncio
    async def test_failed_swap_applies_nothing(self) -> None:
        store, _ = emulated()
        await store.create_collection("old", SPEC)
        await store.update_aliases([CreateAliasAction(alias="live", collection="old")])
        with pytest.raises(NotFoundError):
            await store.update_aliases(
                [
                    DeleteAliasAction(alias="live"),
                    CreateAliasAction(alias="live", collection="nope"),
                ]
            )
        assert await store.list_aliases() == {"live": "old"}


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, ValidationFailedError),
        (403, ValidationFailedError),
        (404, NotFoundError),
        (408, StoreTimeoutError),
        (409, ConflictError),
        (422, ValidationFailedError),
        (425, TransientError),
        (429, RateLimitedError),
        (500, ServerError),
        (502, ServerError),
        (503, ServerError),
        (504, ServerError),
    ],
)
async def test_status_mapping(status: int, kind: type[FeatureStoreError]) -> None:
    store = store_for(lambda _: error(status, "boom"))
    with pytest.raises(kind) as exc:
        await store.count("c")
    assert type(exc.value) is kind
    assert isinstance(exc.value, TransientError) is (status in {408, 425, 429} or status >= 500)
    assert f"POST /collections/c/points/count -> {status}: boom" in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("header", "expected"), [("1.5", 1.5), ("Wed, 21 Oct 2026", None), (None, None)]
)
async def test_retry_after_header(header: str | None, expected: float | None) -> None:
    headers = {} if header is None else {"Retry-After": header}
    store = store_for(
        lambda _: httpx.Response(429, headers=headers, json={"status": {"error": "slow"}})
    )
    with pytest.raises(RateLimitedError) as exc:
        await store.list_aliases()
    assert exc.value.retry_after == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "fragment"),
    [
        (httpx.Response(500, text="<html>gateway</html>"), "<html>gateway</html>"),
        (httpx.Response(500, json={"status": "error"}), ": error"),
        (httpx.Response(500, json={"status": {"error": "disk full"}}), ": disk full"),
    ],
)
async def test_error_message_extraction(response: httpx.Response, fragment: str) -> None:
    store = store_for(lambda _: response)
    with pytest.raises(ServerError, match=fragment):
        await store.list_aliases()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception", "kind"),
    [
        (httpx.ConnectTimeout("t"), StoreTimeoutError),
        (httpx.ReadTimeout("t"), StoreTimeoutError),
        (httpx.ConnectError("refused"), StoreConnectionError),
        (httpx.ReadError("reset by peer"), StoreConnectionError),
        (httpx.RemoteProtocolError("peer closed"), StoreConnectionError),
        (httpx.DecodingError("bad gzip"), MalformedResponseError),
    ],
)
async def test_transport_error_mapping(exception: Exception, kind: type[FeatureStoreError]) -> None:
    def raise_(_: httpx.Request) -> httpx.Response:
        raise exception

    with pytest.raises(kind) as exc:
        await store_for(raise_).collection_exists("c")
    assert exc.value.__cause__ is exception


@pytest.mark.asyncio
async def test_transient_failures_recovered_by_retrier() -> None:
    emulator = QdrantEmulator()
    failures = [error(503, "warming up"), httpx.Response(429, json={})]

    def flaky(request: httpx.Request) -> httpx.Response:
        return failures.pop(0) if failures else emulator(request)

    sleep = RecordingSleep()
    retrier = Retrier(RetryPolicy(max_attempts=5), sleep=sleep, seed=0)
    assert await retrier.run(store_for(flaky).list_aliases) == {}
    assert len(sleep.delays) == 2


@pytest.mark.asyncio
async def test_permanent_failures_not_retried() -> None:
    calls = 0

    def missing(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return error(404, "gone")

    with pytest.raises(NotFoundError):
        await Retrier(RetryPolicy(), sleep=RecordingSleep()).run(store_for(missing).list_aliases)
    assert calls == 1


# ---------------------------------------------------------------------------
# Malformed responses
# ---------------------------------------------------------------------------


def collection_body(distance: str = "Dot", size: int = 3, count: int | None = 0) -> object:
    return {
        "points_count": count,
        "config": {"params": {"vectors": {"size": size, "distance": distance}}},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "call"),
    [
        (httpx.Response(200, text="not json"), "count"),
        (ok({"total": 3}), "count"),
        (httpx.Response(200, json={"status": "ok"}), "count"),
        (ok(collection_body(distance="Manhattan")), "describe"),
        (ok(collection_body(size=0)), "describe"),
        (ok({"points": [{"id": "6f1e-uuid", "vector": [1.0]}]}), "scroll"),
        (ok({"points": [{"id": 1, "vector": {"text": [1.0]}}]}), "scroll"),
        (ok({"points": [{"id": 1, "vector": [1.0], "payload": {"k": {"nested": 1}}}]}), "scroll"),
        (ok({"points": [{"id": 2, "vector": [1.0]}, {"id": 1, "vector": [1.0]}]}), "scroll"),
        (ok({"points": [{"id": 1, "vector": []}]}), "scroll"),
    ],
)
async def test_malformed_bodies(response: httpx.Response, call: str) -> None:
    store = store_for(lambda _: response)
    calls: dict[str, Callable[[], Awaitable[object]]] = {
        "count": lambda: store.count("c"),
        "describe": lambda: store.describe_collection("c"),
        "scroll": lambda: store.scroll("c", offset=None, limit=10),
    }
    with pytest.raises(MalformedResponseError) as exc:
        await calls[call]()
    assert isinstance(exc.value, PermanentError)


@pytest.mark.asyncio
async def test_null_points_count_reads_as_zero() -> None:
    store = store_for(lambda _: ok(collection_body(count=None)))
    assert (await store.describe_collection("c")).points_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("call", ["upsert", "delete"])
async def test_unfinished_write_is_transient(call: str) -> None:
    store = store_for(lambda _: ok({"operation_id": 1, "status": "acknowledged"}))
    calls = {
        "upsert": lambda: store.upsert("c", [pt(1, 1.0, 2.0, 3.0)]),
        "delete": lambda: store.delete_points("c", [1]),
    }
    with pytest.raises(TransientError, match="not completed"):
        await calls[call]()


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_key_header_and_base_url() -> None:
    seen: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok({"aliases": []})

    async with QdrantHttpStore.connect(
        BASE, api_key="s3cret", transport=httpx.MockTransport(capture)
    ) as store:
        await store.list_aliases()
    assert seen[0].headers["api-key"] == "s3cret"
    assert str(seen[0].url) == f"{BASE}/aliases"


@pytest.mark.asyncio
async def test_no_api_key_header_by_default_and_close() -> None:
    seen: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok({"aliases": []})

    client = httpx.AsyncClient(base_url=BASE, transport=httpx.MockTransport(capture))
    store = QdrantHttpStore(client)
    await store.list_aliases()
    await store.aclose()
    assert "api-key" not in seen[0].headers
    assert client.is_closed
