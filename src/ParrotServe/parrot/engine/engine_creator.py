# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import json
from typing import Dict

from parrot.utils import get_logger
from parrot.constants import ENGINE_TYPE_BUILTIN, ENGINE_TYPE_OPENAI
from parrot.exceptions import ParrotEngineInternalError

from .llm_engine import LLMEngine
from .config import EngineConfig

# from .builtin.builtin_engine import BuiltinEngine
from .openai.openai_engine import OpenAIEngine


logger = get_logger("Engine Creator")


def create_engine(
    engine_config_path: str,
    connect_to_core: bool = True,
    override_args: Dict = {},
) -> LLMEngine:
    """Create an execution engine.

    NOTE(chaofan): We put this in an independent file to avoid circular imports.

    Args:
        engine_config_path: str. The path to the engine config file.
        connect_to_core: bool. Whether to connect to the Serve Core.
        override_args: Dict. The override arguments.

    Returns:
        LLMEngine. The created execution engine.
    """

    with open(engine_config_path) as f:
        engine_config = dict(json.load(f))

    if "device" in override_args:
        engine_config["instance"]["device"] = override_args["device"]
        override_args.pop("device")
    engine_config.update(override_args)

    if not EngineConfig.verify_config(engine_config):
        raise ParrotEngineInternalError(f"Invalid engine config: {engine_config}")

    engine_type = engine_config["engine_type"]

    # if engine_type == ENGINE_TYPE_BUILTIN:
    #     return BuiltinEngine(engine_config, connect_to_core)
    if engine_type == ENGINE_TYPE_OPENAI:
        return OpenAIEngine(engine_config, connect_to_core)
    else:
        raise ParrotEngineInternalError(f"Unsupported engine type: {engine_type}")
