import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import aiohttp
import requests
from requests.models import Response as SyncHTTPResponse

__all__ = [
    "AsyncHTTPResponse",
    "SyncHTTPResponse",
    "request",
    "get",
    "post",
    "get_json",
    "post_json",
    "arequest",
    "aget",
    "apost",
    "aget_json",
    "apost_json",
]


@dataclass(slots=True)
class AsyncHTTPResponse:
    status: int
    headers: Mapping[str, str]
    content: bytes
    url: str

    def raise_for_status(self) -> None:
        if 400 <= self.status:
            snippet = self.content[:200].decode(errors="replace")
            raise RuntimeError(f"HTTP {self.status} Error for {self.url}: {snippet}")

    async def text(self, encoding: str | None = None) -> str:
        if encoding is None:
            encoding = "utf-8"
        return self.content.decode(encoding, errors="replace")

    async def json(self) -> Any:
        return json.loads(self.content.decode(errors="replace"))


# ---------------- Sync helpers -----------------


def request(method: str, url: str, **kwargs) -> requests.Response:
    return requests.request(method.upper(), url, **kwargs)


def get(url: str, **kwargs) -> requests.Response:
    return requests.get(url, **kwargs)


def post(url: str, **kwargs) -> requests.Response:
    return requests.post(url, **kwargs)


def get_json(url: str, json: Any = None, **kwargs) -> Any:
    resp = get(url, json=json, **kwargs)
    resp.raise_for_status()
    return resp.json()


def post_json(url: str, json: Any = None, **kwargs) -> Any:
    resp = post(url, json=json, **kwargs)
    resp.raise_for_status()
    return resp.json()


# ---------------- Async helpers -----------------


async def arequest(method: str, url: str, **kwargs) -> AsyncHTTPResponse:
    async with aiohttp.ClientSession() as session:
        async with session.request(method.upper(), url, **kwargs) as resp:
            content = await resp.read()
            return AsyncHTTPResponse(
                status=resp.status,
                headers=dict(resp.headers),
                content=content,
                url=str(resp.url),
            )


async def aget(url: str, **kwargs) -> AsyncHTTPResponse:
    return await arequest("GET", url, **kwargs)


async def apost(url: str, **kwargs) -> AsyncHTTPResponse:
    return await arequest("POST", url, **kwargs)


async def aget_json(url: str, json: Any = None, **kwargs) -> Any:
    resp = await aget(url, json=json, **kwargs)
    resp.raise_for_status()
    return await resp.json()


async def apost_json(url: str, json: Any = None, **kwargs) -> Any:
    resp = await apost(url, json=json, **kwargs)
    resp.raise_for_status()
    return await resp.json()
