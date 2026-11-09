"""A tiny in-memory emulation of the Qdrant REST API for ``httpx.MockTransport``.

Covers exactly the endpoints the adapter uses, with Qdrant's response envelope,
error bodies, alias resolution in collection paths and atomic alias updates.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class _Collection:
    size: int
    distance: str
    points: dict[int, dict[str, Any]] = field(default_factory=dict)


def ok(result: object) -> httpx.Response:
    return httpx.Response(200, json={"result": result, "status": "ok", "time": 0.001})


def error(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"status": {"error": message}, "time": 0.0})


Route = Callable[["QdrantEmulator", httpx.Request, re.Match[str]], httpx.Response]


class QdrantEmulator:
    def __init__(self) -> None:
        self.collections: dict[str, _Collection] = {}
        self.aliases: dict[str, str] = {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        for method, pattern, route in _ROUTES:
            match = re.fullmatch(pattern, request.url.path)
            if match and request.method == method:
                return route(self, request, match)
        return error(404, f"no route for {request.method} {request.url.path}")

    def resolve(self, name: str) -> _Collection | None:
        return self.collections.get(self.aliases.get(name, name))

    # -- routes ----------------------------------------------------------------

    def exists(self, _: httpx.Request, m: re.Match[str]) -> httpx.Response:
        return ok({"exists": self.resolve(m["name"]) is not None})

    def create(self, request: httpx.Request, m: re.Match[str]) -> httpx.Response:
        if m["name"] in self.collections:
            return error(409, f"Collection `{m['name']}` already exists!")
        vectors = json.loads(request.content)["vectors"]
        self.collections[m["name"]] = _Collection(vectors["size"], vectors["distance"])
        return ok(True)

    def drop(self, _: httpx.Request, m: re.Match[str]) -> httpx.Response:
        if self.collections.pop(m["name"], None) is None:
            return error(404, f"Collection `{m['name']}` doesn't exist!")
        return ok(True)

    def describe(self, _: httpx.Request, m: re.Match[str]) -> httpx.Response:
        coll = self.resolve(m["name"])
        if coll is None:
            return error(404, f"Collection `{m['name']}` doesn't exist!")
        return ok(
            {
                "status": "green",
                "points_count": len(coll.points),
                "config": {"params": {"vectors": {"size": coll.size, "distance": coll.distance}}},
            }
        )

    def upsert(self, request: httpx.Request, m: re.Match[str]) -> httpx.Response:
        coll = self.resolve(m["name"])
        if coll is None:
            return error(404, "not found")
        for p in json.loads(request.content)["points"]:
            if len(p["vector"]) != coll.size:
                return error(400, "Wrong input: Vector dimension error")
            coll.points[p["id"]] = {"id": p["id"], "vector": p["vector"], "payload": p["payload"]}
        return ok({"operation_id": 1, "status": "completed"})

    def delete_points(self, request: httpx.Request, m: re.Match[str]) -> httpx.Response:
        coll = self.resolve(m["name"])
        if coll is None:
            return error(404, "not found")
        for pid in json.loads(request.content)["points"]:
            coll.points.pop(pid, None)
        return ok({"operation_id": 2, "status": "completed"})

    def count(self, _: httpx.Request, m: re.Match[str]) -> httpx.Response:
        coll = self.resolve(m["name"])
        if coll is None:
            return error(404, "not found")
        return ok({"count": len(coll.points)})

    def scroll(self, request: httpx.Request, m: re.Match[str]) -> httpx.Response:
        coll = self.resolve(m["name"])
        if coll is None:
            return error(404, "not found")
        body = json.loads(request.content)
        offset, limit = body.get("offset"), body["limit"]
        ids = sorted(i for i in coll.points if offset is None or i >= offset)
        page, rest = ids[:limit], ids[limit:]
        return ok(
            {
                "points": [coll.points[i] for i in page],
                "next_page_offset": rest[0] if rest else None,
            }
        )

    def list_aliases(self, _: httpx.Request, __: re.Match[str]) -> httpx.Response:
        return ok(
            {"aliases": [{"alias_name": a, "collection_name": c} for a, c in self.aliases.items()]}
        )

    def update_aliases(self, request: httpx.Request, _: re.Match[str]) -> httpx.Response:
        staged = dict(self.aliases)
        for action in json.loads(request.content)["actions"]:
            if "create_alias" in action:
                spec = action["create_alias"]
                if spec["collection_name"] not in self.collections:
                    return error(404, "Collection not found")  # nothing applied: atomic
                staged[spec["alias_name"]] = spec["collection_name"]
            else:
                name = action["delete_alias"]["alias_name"]
                if name not in staged:
                    return error(404, f"Alias {name} does not exist")
                del staged[name]
        self.aliases = staged
        return ok(True)


_C = r"/collections/(?P<name>[^/]+)"
_ROUTES: list[tuple[str, str, Route]] = [
    ("GET", "/aliases", QdrantEmulator.list_aliases),
    ("POST", "/collections/aliases", QdrantEmulator.update_aliases),
    ("GET", _C + "/exists", QdrantEmulator.exists),
    ("PUT", _C + "/points", QdrantEmulator.upsert),
    ("POST", _C + "/points/delete", QdrantEmulator.delete_points),
    ("POST", _C + "/points/count", QdrantEmulator.count),
    ("POST", _C + "/points/scroll", QdrantEmulator.scroll),
    ("PUT", _C, QdrantEmulator.create),
    ("DELETE", _C, QdrantEmulator.drop),
    ("GET", _C, QdrantEmulator.describe),
]
