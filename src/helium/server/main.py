from argparse import ArgumentParser, Namespace
from contextlib import asynccontextmanager

import uvicorn
from fastapi import APIRouter, FastAPI

from helium import helium
from helium.runtime import HeliumServerConfig
from helium.server.routes import query

api_router = APIRouter()
api_router.include_router(query.router, tags=["query"])


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args()


args = parse_args()
server_config = HeliumServerConfig(
    host=args.host, port=args.port, benchmarking=args.benchmark
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with helium.serve_instance(config=server_config):
        yield


app = FastAPI(lifespan=lifespan)
app.include_router(api_router)


def run_api_server() -> None:
    uvicorn.run(app, host=server_config.host, port=server_config.port)


if __name__ == "__main__":
    run_api_server()
