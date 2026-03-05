from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from pprint import pformat
from typing import Any, ParamSpec, TypeVar

TEE = "\u251c\u2500\u2500"
ELBOW = "\u2514\u2500\u2500"

# Stolen from flwr.common.logger
LOG_COLORS = {
    "DEBUG": "\033[94m",  # Blue
    "INFO": "\033[92m",  # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",  # Red
    "CRITICAL": "\033[95m",  # Magenta
    "RESET": "\033[0m",  # Reset to default
}

LOGGER_NAME = "FedBench"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.DEBUG)


class ColoredStreamHandler(logging.StreamHandler):  # type: ignore[type-arg]
    # Adapted from flwr.common.logger
    def format(self, record: logging.LogRecord) -> str:
        seperator = " " * (8 - len(record.levelname))
        log_fmt = (
            f"{LOG_COLORS[record.levelname]}"
            f"%(levelname)s %(asctime)s{LOG_COLORS['RESET']}"
            f": {seperator} %(message)s"
        )
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


handler = ColoredStreamHandler()
if log_level := os.getenv("FEDBENCH_LOG_LEVEL"):
    handler.setLevel(log_level.upper())
else:
    handler.setLevel(logging.INFO)

logger.addHandler(handler)


def log(
    source: str,
    message: str,
    level: int = logging.INFO,
    **kwargs: Any,
) -> None:

    msg = f"{source}: {message}" if source else message
    logger.log(level, msg, **kwargs)


log_debug = functools.partial(log, level=logging.DEBUG)
log_info = functools.partial(log, level=logging.INFO)
log_warning = functools.partial(log, level=logging.WARNING)
log_error = functools.partial(log, level=logging.ERROR)
log_critical = functools.partial(log, level=logging.CRITICAL)
pformat = functools.partial(pformat, indent=2, width=70, compact=True)


P = ParamSpec("P")
R = TypeVar("R")


def debug_calls(modulename: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            log_debug(modulename, f"Calling {func.__name__}")
            log_debug("", f"\t{TEE} args: {pformat(args)}")
            log_debug("", f"\t{TEE} kwargs: {pformat(kwargs)}")
            ret = func(*args, **kwargs)
            log_debug("", f"\t{ELBOW} return value: {pformat(ret)}")
            log_debug("", "")
            return ret

        return wrapper

    return decorator
