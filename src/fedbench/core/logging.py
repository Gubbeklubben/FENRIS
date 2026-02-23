from logging import DEBUG, INFO
from flwr.common.logger import log as _flwr_log


_BOX_DRAWING = "\u251c\u2500\u2500"


def log(header: str, message_lines: tuple[str, ...], level: int = INFO) -> None:
    if header:
        _flwr_log(level, header)

    for line in message_lines:
        _flwr_log(level, f"\t{_BOX_DRAWING} {line}")


# Quick and dirty, set and export env variable FLWR_LOG_LEVEL="DEBUG" to enable.
def log_calls(modulename):  # type: ignore[no-untyped-def]
    def decorator(func):  # type: ignore[no-untyped-def]
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            _flwr_log(DEBUG, f"{modulename}: Calling {func.__name__}")
            _flwr_log(DEBUG, f"\t{_BOX_DRAWING} args: {args}")
            _flwr_log(DEBUG, f"\t{_BOX_DRAWING} kwargs: {kwargs}")
            ret = func(*args, **kwargs)
            _flwr_log(DEBUG, f"\t{_BOX_DRAWING} return value: {ret}")
            _flwr_log(DEBUG, "")
            return ret
        return wrapper
    return decorator