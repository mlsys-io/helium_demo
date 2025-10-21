from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from httpx import ASGITransport, AsyncClient

from helium import graphs
from helium.runtime import HeliumServer
from helium.runtime.protocol import HeliumRequest, HeliumRequestConfig, HeliumResponse
from helium.server.routes import query
from helium.utils import unique_id

api_router = APIRouter()
api_router.include_router(query.router, tags=["query"])

app = FastAPI()
app.include_router(api_router)


@asynccontextmanager
async def api_server_client(
    server: HeliumServer,
) -> AsyncGenerator[AsyncClient, None]:
    assert server.is_started, "Server must be started."
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


async def invoke_api_server(
    client: AsyncClient,
    graph: graphs.Graph,
    request_config: HeliumRequestConfig,
    endpoint: str = "/request",
) -> HeliumResponse:
    name = unique_id()
    query = {name: graph.compile().serialize()}
    request = HeliumRequest(
        query=query,
        config=request_config,
    )
    resp = await client.post(endpoint, json=request.model_dump(mode="json"))
    response = HeliumResponse.model_validate(resp.json())
    response.outputs = response.outputs[name]
    return response
