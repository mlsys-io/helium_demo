import logging

_vllm_logger = logging.getLogger("vllm")


def init_child_logger(name: str) -> logging.Logger:
    return _vllm_logger.getChild(name)
