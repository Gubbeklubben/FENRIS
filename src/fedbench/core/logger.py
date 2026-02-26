import functools
import logging
import multiprocessing
import os
from logging.handlers import QueueHandler
from typing import Any

LOGGER_NAME = "FedBench"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.DEBUG)


def log(src: str, message: str, level: int = logging.INFO, **kwargs: Any) -> None:
    logger.log(level, f"{src}: {message}", **kwargs)


log_debug = functools.partial(log, level=logging.DEBUG)
log_info = functools.partial(log, level=logging.INFO)
log_warning = functools.partial(log, level=logging.WARNING)
log_error = functools.partial(log, level=logging.ERROR)
log_critical = functools.partial(log, level=logging.CRITICAL)


def add_queue_handler(queue: multiprocessing.Queue) -> None:  # type: ignore[type-arg]
    handler = QueueHandler(queue)
    if log_level := os.getenv("FEDBENCH_LOGLEVEL"):
        handler.setLevel(log_level.upper())
    else:
        handler.setLevel(logging.INFO)

    handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(handler)


_BOX_DRAWING = "\u251c\u2500\u2500"


# Quick and dirty, set and export env variable FLWR_LOG_LEVEL="DEBUG" to enable.
def log_calls(modulename):  # type: ignore[no-untyped-def]
    def decorator(func):  # type: ignore[no-untyped-def]
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            log_debug(modulename, "Calling {func.__name__}")
            log_debug(modulename, f"\t{_BOX_DRAWING} args: {args}")
            log_debug(modulename, f"\t{_BOX_DRAWING} kwargs: {kwargs}")
            ret = func(*args, **kwargs)
            log_debug(modulename, f"\t{_BOX_DRAWING} return value: {ret}")
            log_debug(modulename, "")
            return ret
        return wrapper
    return decorator