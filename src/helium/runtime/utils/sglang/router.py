import argparse
import json
import logging
import sys
from abc import abstractmethod
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

UNLIMITED: int = 9999999
DEFAULT_TIMEOUT = httpx.Timeout(timeout=24 * 60 * 60, connect=60)
DEFAULT_LIMITS = httpx.Limits(max_connections=UNLIMITED, max_keepalive_connections=256)


class BaseRouter:
    def __init__(self, worker_urls: list[str]) -> None:
        self.worker_urls = worker_urls

    @contextmanager
    @abstractmethod
    def get_worker(self) -> Iterator[str]:
        pass


class ShortestQueueRouter(BaseRouter):
    def __init__(self, worker_urls: list[str]) -> None:
        super().__init__(worker_urls)
        self.queue_lengths = [0] * len(worker_urls)

    @contextmanager
    def get_worker(self) -> Iterator[str]:
        min_index = self.queue_lengths.index(min(self.queue_lengths))
        self.queue_lengths[min_index] += 1
        try:
            yield self.worker_urls[min_index]
        finally:
            self.queue_lengths[min_index] -= 1


class RoundRobinRouter(BaseRouter):
    def __init__(self, worker_urls: list[str]) -> None:
        super().__init__(worker_urls)
        self.counter = 0

    @contextmanager
    def get_worker(self) -> Iterator[str]:
        worker = self.worker_urls[self.counter % len(self.worker_urls)]
        self.counter += 1
        yield worker


def _copy_headers(req: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for k, v in req.headers.items():
        lk = k.lower()
        if lk in {"host", "content-length"}:
            continue
        headers[k] = v
    return headers


def build_app(policy: str, worker_urls: list[str]) -> FastAPI:
    if not worker_urls:
        raise ValueError("worker_urls must be non-empty")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await app.state.client.aclose()

    app = FastAPI(lifespan=lifespan)

    client = httpx.AsyncClient(limits=DEFAULT_LIMITS, timeout=DEFAULT_TIMEOUT)
    app.state.client = client

    router: BaseRouter
    match policy:
        case "shortest_queue":
            router = ShortestQueueRouter(worker_urls)
        case "round_robin":
            router = RoundRobinRouter(worker_urls)
        case _:
            raise ValueError(f"unknown routing policy: {policy}")
    app.state.router = router

    def get_client(req: Request) -> httpx.AsyncClient:
        return req.app.state.client

    def get_router(req: Request) -> BaseRouter:
        return req.app.state.router

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    async def _proxy_json(
        req: Request, client: httpx.AsyncClient, worker_url: str, upstream_path: str
    ) -> Response:

        url = f"{worker_url}{upstream_path}"

        body = await req.body()
        headers = _copy_headers(req)
        method = req.method.upper()

        try:
            res = await client.request(method, url, content=body, headers=headers)
        except httpx.HTTPError as e:
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "upstream_url": url,
                    }
                },
            )

        response_headers: dict[str, str] = {}
        for k, v in res.headers.items():
            lk = k.lower()
            if lk in {"content-length", "transfer-encoding", "connection"}:
                continue
            response_headers[k] = v

        return Response(
            content=res.content,
            status_code=res.status_code,
            headers=response_headers,
            media_type=res.headers.get("content-type"),
        )

    async def _proxy_stream(
        req: Request, client: httpx.AsyncClient, worker_url: str, upstream_path: str
    ) -> Response:
        url = f"{worker_url}{upstream_path}"

        body = await req.body()
        headers = _copy_headers(req)
        method = req.method.upper()

        async def _iter():
            try:
                async with client.stream(
                    method, url, content=body, headers=headers
                ) as res:
                    async for chunk in res.aiter_bytes():
                        yield chunk
            except httpx.HTTPError as e:
                msg = str(e).replace("\n", " ")
                error_data = {
                    "error": {
                        "type": type(e).__name__,
                        "message": msg,
                        "upstream_url": url,
                    }
                }
                yield (f"data: {json.dumps(error_data)}\n\n".encode())
                yield b"data: [DONE]\n\n"

        return StreamingResponse(_iter(), media_type="text/event-stream")

    async def _proxy_openai(
        req: Request,
        client: httpx.AsyncClient,
        router: BaseRouter,
        upstream_path: str,
    ) -> Response:
        body = await req.json()
        with router.get_worker() as worker_url:
            if isinstance(body, dict) and body.get("stream") is True:
                return await _proxy_stream(req, client, worker_url, upstream_path)
            return await _proxy_json(req, client, worker_url, upstream_path)

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: Request,
        client: httpx.AsyncClient = Depends(get_client),
        router: BaseRouter = Depends(get_router),
    ) -> Response:
        return await _proxy_openai(req, client, router, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(
        req: Request,
        client: httpx.AsyncClient = Depends(get_client),
        router: BaseRouter = Depends(get_router),
    ) -> Response:
        return await _proxy_openai(req, client, router, "/v1/completions")

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("sglang-router")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4220)
    parser.add_argument(
        "--policy", choices=["shortest_queue", "round_robin"], default="shortest_queue"
    )
    parser.add_argument("--worker-urls", nargs="+", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--log-level", default="info")
    args, unknown = parser.parse_known_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info(
        "Starting py-router on %s:%s workers=%s, policy=%s",
        args.host,
        args.port,
        args.worker_urls,
        args.policy,
    )

    if unknown:
        logger.warning("Ignoring unknown arguments: %s", unknown)

    app = build_app(args.policy, args.worker_urls)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
