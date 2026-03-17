"""
Adapted from https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output.
"""

import inspect
import logging
from collections.abc import Callable
from typing import Any

from helium import envs

Logger = logging.Logger
LogLevel = int | str

_default_logger: Logger | None = None
_debug_logger: Logger | None = None


class ColorFormatter(logging.Formatter):
    """
    See the list of bash colors here:
    https://gist.github.com/JBlond/2fea43a3049b38287e5e9cefc87b2124
    """

    blue: str = "\x1b[0;34m"
    green: str = "\x1b[0;32m"
    yellow: str = "\x1b[0;33m"
    red: str = "\x1b[0;31m"
    bold_red: str = "\x1b[1;31m"
    reset: str = "\x1b[0m"
    format_str: str = "[%(name)s | %(levelname)s] %(message)s"

    FORMATS: dict[int, str] = {
        logging.DEBUG: blue + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class DebugColorFormatter(ColorFormatter):
    blue: str = "\x1b[0;94m"
    green: str = "\x1b[0;92m"
    yellow: str = "\x1b[0;93m"
    red: str = "\x1b[0;91m"
    bold_red: str = "\x1b[1;91m"
    reset: str = "\x1b[0m"
    format_str: str = "[%(name)s | %(levelname)s] (%(filename)s:%(lineno)d) %(message)s"

    FORMATS: dict[int, str] = {
        logging.DEBUG: blue + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }


def get_default_logger() -> Logger:
    global _default_logger

    if _default_logger is not None:
        return _default_logger

    log_level = envs.HELIUM_LOG_LEVEL

    logger = logging.getLogger("Helium")
    logger.setLevel(log_level)

    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(ColorFormatter())

    logger.addHandler(handler)

    logger.propagate = False

    _default_logger = logger
    return logger


def init_child_logger(
    name: str, logger: Logger | None = None, log_level: LogLevel | None = None
) -> Logger:
    if logger is None:
        logger = get_default_logger()
    logger = logger.getChild(name)
    if log_level is not None:
        logger.setLevel(log_level)
    return logger


def get_debug_logger() -> Logger:
    global _debug_logger

    if envs.DEBUG_MODE is None:
        raise ValueError(
            "Debug mode is disabled! "
            "Ensure that debug logging is used in the development environment."
        )

    if _debug_logger is not None:
        return _debug_logger

    log_level = logging.DEBUG

    logger = logging.getLogger("Debug")
    logger.setLevel(log_level)

    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(DebugColorFormatter())

    logger.addHandler(handler)

    logger.propagate = False

    _debug_logger = logger
    return logger


def log_on_exception(
    ignore: list[type[BaseException]] | None = None, is_method: bool = True
) -> Callable:
    def method_decorator(func):
        async def wrapper(self, *args, **kwargs) -> Any:
            try:
                return func(self, *args, **kwargs)
            except BaseException as e:
                logger: Logger | None = getattr(self, "logger", None)
                if logger is None or not isinstance(logger, Logger):
                    return
                if ignore and any(isinstance(e, exc) for exc in ignore):
                    return
                logger.exception(
                    "Exception '%s' occurred in function '%s' (called by '%s') (Detail: %s)",
                    e.__class__.__name__,
                    func.__name__,
                    inspect.stack()[1].function,
                    e,
                )
                raise

        return wrapper

    def decorator(func):
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except BaseException as e:
                logger = get_default_logger()
                logger.exception(
                    "Exception '%s' occurred in function '%s' (called by '%s') (Detail: %s)",
                    e.__class__.__name__,
                    func.__name__,
                    inspect.stack()[1].function,
                    e,
                )
                raise

        return wrapper

    return method_decorator if is_method else decorator


def log_on_exception_async(
    ignore: list[type[BaseException]] | None = None, is_method: bool = True
) -> Callable:
    def method_decorator(func):
        async def wrapper(self, *args, **kwargs) -> Any:
            try:
                return await func(self, *args, **kwargs)
            except BaseException as e:
                logger: Logger | None = getattr(self, "logger", None)
                if logger is None or not isinstance(logger, Logger):
                    return
                if ignore and any(isinstance(e, exc) for exc in ignore):
                    return
                logger.exception(
                    "Exception '%s' occurred in function '%s' (called by '%s') (Detail: %s)",
                    e.__class__.__name__,
                    func.__name__,
                    inspect.stack()[1].function,
                    e,
                )
                raise

        return wrapper

    def decorator(func):
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except BaseException as e:
                logger = get_default_logger()
                logger.exception(
                    "Exception '%s' occurred in function '%s' (called by '%s') (Detail: %s)",
                    e.__class__.__name__,
                    func.__name__,
                    inspect.stack()[1].function,
                    e,
                )
                raise

        return wrapper

    return method_decorator if is_method else decorator


def log_on_exception_async_generator(
    ignore: list[type[BaseException]] | None = None, is_method: bool = True
) -> Callable:
    def method_decorator(func):
        async def wrapper(self, *args, **kwargs) -> Any:
            try:
                async for ret in func(self, *args, **kwargs):
                    yield ret
            except BaseException as e:
                logger: Logger | None = getattr(self, "logger", None)
                if logger is None or not isinstance(logger, Logger):
                    return
                if ignore and any(isinstance(e, exc) for exc in ignore):
                    return
                logger.exception(
                    "Exception '%s' occurred in function '%s' (called by '%s') (Detail: %s)",
                    e.__class__.__name__,
                    func.__name__,
                    inspect.stack()[1].function,
                    e,
                )
                raise

        return wrapper

    def decorator(func):
        async def wrapper(*args, **kwargs) -> Any:
            try:
                async for ret in func(*args, **kwargs):
                    yield ret
            except BaseException as e:
                logger = get_default_logger()
                logger.exception(
                    "Exception '%s' occurred in function '%s' (called by '%s') (Detail: %s)",
                    e.__class__.__name__,
                    func.__name__,
                    inspect.stack()[1].function,
                    e,
                )
                raise

        return wrapper

    return method_decorator if is_method else decorator
