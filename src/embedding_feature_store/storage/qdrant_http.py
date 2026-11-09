"""Async Qdrant REST adapter implementing :class:`VectorStore`.

Every transport or HTTP failure is translated into the core error taxonomy:

=====================  ==========================================
Condition              Raised
=====================  ==========================================
timeout                :class:`StoreTimeoutError` (transient)
connect/read/reset     :class:`StoreConnectionError` (transient)
408 / 425 / 429 / 5xx  transient (429 carries ``Retry-After``)
404                    :class:`NotFoundError`
409                    :class:`ConflictError`
other 4xx              :class:`ValidationFailedError`
undecodable response   :class:`MalformedResponseError`
=====================  ==========================================

The adapter never retries by itself; wrap calls in
:class:`~embedding_feature_store.core.retry.Retrier`. Writes use ``wait=true``
so a successful return means the change is applied and visible.

Points must use unsigned integer ids, a single unnamed vector and a flat
payload of scalars; anything else is reported as malformed.

Note: Qdrant re-normalises vectors of Cosine collections on write. Points
projected by :class:`~embedding_feature_store.core.projection.PointProjector`
are already unit length, so the stored float32 values round-trip unchanged up
to the last ulp; verification of Cosine collections assumes this.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Final, Literal, Self

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from embedding_feature_store.core.errors import (
    MalformedResponseError,
    NotFoundError,
    StoreConnectionError,
    StoreTimeoutError,
    TransientError,
    error_for_status,
)
from embedding_feature_store.schemas import (
    AliasAction,
    CollectionInfo,
    CreateAliasAction,
    DistanceMetric,
    PayloadValue,
    PointRecord,
    ScrollPage,
    VectorSpec,
)

__all__ = ["QdrantHttpStore"]

_DISTANCE_TO_WIRE: Final = {
    DistanceMetric.COSINE: "Cosine",
    DistanceMetric.DOT: "Dot",
    DistanceMetric.EUCLID: "Euclid",
}
_DISTANCE_FROM_WIRE: Final = {wire: metric for metric, wire in _DISTANCE_TO_WIRE.items()}
_WAIT: Final = {"wait": "true"}


# ---------------------------------------------------------------------------
# Wire models (response bodies); unknown fields are ignored.
# ---------------------------------------------------------------------------


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)


class _Exists(_Wire):
    exists: bool


class _VectorParams(_Wire):
    size: int
    distance: str


class _CollectionParams(_Wire):
    vectors: _VectorParams


class _CollectionConfig(_Wire):
    params: _CollectionParams


class _CollectionResult(_Wire):
    points_count: int | None = None
    config: _CollectionConfig


class _UpdateResult(_Wire):
    status: Literal["acknowledged", "completed"]


class _Count(_Wire):
    count: int


class _WirePoint(_Wire):
    id: int
    vector: list[float]
    payload: dict[str, PayloadValue] | None = None


class _ScrollResult(_Wire):
    points: list[_WirePoint]
    next_page_offset: int | None = None


class _Alias(_Wire):
    alias_name: str
    collection_name: str


class _Aliases(_Wire):
    aliases: list[_Alias]


class _Envelope[R](_Wire):
    result: R


class _ErrorStatus(_Wire):
    error: str


class _ErrorBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: _ErrorStatus | str | None = None


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _encode_point(point: PointRecord) -> dict[str, object]:
    return {"id": point.point_id, "vector": list(point.vector), "payload": dict(point.payload)}


def _encode_action(action: AliasAction) -> dict[str, object]:
    if isinstance(action, CreateAliasAction):
        return {"create_alias": {"collection_name": action.collection, "alias_name": action.alias}}
    return {"delete_alias": {"alias_name": action.alias}}


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return None


def _error_message(response: httpx.Response) -> str:
    prefix = f"{response.request.method} {response.request.url.path} -> {response.status_code}"
    try:
        status = _ErrorBody.model_validate_json(response.content).status
    except ValidationError:
        return f"{prefix}: {response.text[:200]}"
    detail = status.error if isinstance(status, _ErrorStatus) else status
    return f"{prefix}: {detail}"


class QdrantHttpStore:
    """:class:`VectorStore` over Qdrant's REST API."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    @classmethod
    def connect(
        cls,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> Self:
        headers = {} if api_key is None else {"api-key": api_key}
        client = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=timeout_s, transport=transport
        )
        return cls(client)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # -- transport -----------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: object = None,
        params: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        try:
            response = await self._client.request(method, path, json=json, params=params)
        except httpx.TimeoutException as exc:
            msg = f"{method} {path}: timed out"
            raise StoreTimeoutError(msg) from exc
        except httpx.TransportError as exc:
            msg = f"{method} {path}: {type(exc).__name__}: {exc}"
            raise StoreConnectionError(msg) from exc
        except httpx.HTTPError as exc:  # decoding, redirect loops: the response is unusable
            msg = f"{method} {path}: {type(exc).__name__}: {exc}"
            raise MalformedResponseError(msg) from exc
        if response.is_success:
            return response
        raise error_for_status(
            response.status_code, _error_message(response), retry_after=_retry_after(response)
        )

    async def _call[R](
        self,
        envelope: type[_Envelope[R]],
        method: str,
        path: str,
        *,
        json: object = None,
        params: Mapping[str, str] | None = None,
    ) -> R:
        response = await self._request(method, path, json=json, params=params)
        try:
            return envelope.model_validate_json(response.content).result
        except ValidationError as exc:
            msg = f"{method} {path}: unexpected response body: {exc}"
            raise MalformedResponseError(msg) from exc

    async def _write(self, method: str, path: str, body: object) -> None:
        outcome = await self._call(_Envelope[_UpdateResult], method, path, json=body, params=_WAIT)
        if outcome.status != "completed":
            msg = f"{method} {path}: write acknowledged but not completed despite wait=true"
            raise TransientError(msg)

    # -- VectorStore ----------------------------------------------------------

    async def collection_exists(self, name: str) -> bool:
        return (await self._call(_Envelope[_Exists], "GET", f"/collections/{name}/exists")).exists

    async def create_collection(self, name: str, spec: VectorSpec) -> None:
        body = {"vectors": {"size": spec.dimension, "distance": _DISTANCE_TO_WIRE[spec.distance]}}
        await self._call(_Envelope[bool], "PUT", f"/collections/{name}", json=body)

    async def delete_collection(self, name: str) -> None:
        """Idempotent: deleting a missing collection succeeds."""
        try:
            await self._call(_Envelope[bool], "DELETE", f"/collections/{name}")
        except NotFoundError:
            return

    async def describe_collection(self, name: str) -> CollectionInfo:
        result = await self._call(_Envelope[_CollectionResult], "GET", f"/collections/{name}")
        params = result.config.params.vectors
        distance = _DISTANCE_FROM_WIRE.get(params.distance)
        if distance is None:
            msg = f"collection {name!r} uses unsupported distance {params.distance!r}"
            raise MalformedResponseError(msg)
        try:
            return CollectionInfo(
                name=name,
                spec=VectorSpec(dimension=params.size, distance=distance),
                points_count=result.points_count or 0,
            )
        except ValidationError as exc:
            raise MalformedResponseError(str(exc)) from exc

    async def upsert(self, collection: str, points: Sequence[PointRecord]) -> None:
        if not points:
            return
        body = {"points": [_encode_point(p) for p in points]}
        await self._write("PUT", f"/collections/{collection}/points", body)

    async def delete_points(self, collection: str, point_ids: Sequence[int]) -> None:
        if not point_ids:
            return
        await self._write(
            "POST", f"/collections/{collection}/points/delete", {"points": list(point_ids)}
        )

    async def count(self, collection: str) -> int:
        path = f"/collections/{collection}/points/count"
        return (await self._call(_Envelope[_Count], "POST", path, json={"exact": True})).count

    async def scroll(self, collection: str, *, offset: int | None, limit: int) -> ScrollPage:
        body: dict[str, Any] = {"limit": limit, "with_payload": True, "with_vector": True}
        if offset is not None:
            body["offset"] = offset
        path = f"/collections/{collection}/points/scroll"
        result = await self._call(_Envelope[_ScrollResult], "POST", path, json=body)
        try:
            return ScrollPage(
                points=tuple(
                    PointRecord(
                        point_id=p.id,
                        vector=tuple(p.vector),
                        payload=tuple((p.payload or {}).items()),
                    )
                    for p in result.points
                ),
                next_offset=result.next_page_offset,
            )
        except ValidationError as exc:
            msg = f"scroll of {collection!r} returned invalid points: {exc}"
            raise MalformedResponseError(msg) from exc

    async def list_aliases(self) -> Mapping[str, str]:
        result = await self._call(_Envelope[_Aliases], "GET", "/aliases")
        return {a.alias_name: a.collection_name for a in result.aliases}

    async def update_aliases(self, actions: Sequence[AliasAction]) -> None:
        """One ``POST /collections/aliases`` call: Qdrant applies the actions atomically."""
        if not actions:
            return
        body = {"actions": [_encode_action(a) for a in actions]}
        await self._call(_Envelope[bool], "POST", "/collections/aliases", json=body)
