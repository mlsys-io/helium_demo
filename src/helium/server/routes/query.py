from typing import Annotated

from fastapi import APIRouter, Depends

from helium import helium
from helium.runtime import HeliumServer
from helium.runtime.protocol import HeliumRequest, HeliumResponse

router = APIRouter()


def get_started_server():
    yield helium.get_started_instance()


@router.post("/request", tags=["request"])
async def handle_request(
    request: HeliumRequest, server: Annotated[HeliumServer, Depends(get_started_server)]
) -> HeliumResponse:
    return await server.request(request)
