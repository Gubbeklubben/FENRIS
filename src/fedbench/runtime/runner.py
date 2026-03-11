import uuid
from collections.abc import Iterable

from fedbench.config import Config
from fedbench.runtime.command import Command
from fedbench.runtime.eventbus import EventBus
from fedbench.core.events import (
    CommandCompleted,
    CommandStarted,
    Event,
    RunCompleted,
    RunFailed,
    RunStarted,
)
from fedbench.core.logger import log_debug, log_error
from fedbench.runtime.runcontext import RunContext


def run(config: Config, commands: Iterable[Command]) -> None:
    run_id = str(uuid.uuid4())
    eventbus = EventBus()
    eventbus.register(lambda event: log_debug("Event", event), (Event,))

    with eventbus:
        _run(run_id, config, commands, eventbus)


def _run(
    run_id: str,
    config: Config,
    commands: Iterable[Command],
    eventbus: EventBus,
) -> None:

    eventbus.emit(RunStarted(run_id))
    ctx = RunContext(run_id, config, eventbus)

    for command in commands:
        name = _infer_name(command)
        eventbus.emit(CommandStarted(name))
        try:
            command(ctx)
        except Exception as exc:
            eventbus.emit(RunFailed(run_id, name, str(type(exc)), str(exc)))
            log_error(
                __name__,
                f"Error executing command {name}",
                exc_info=True,
            )
            return
        else:
            eventbus.emit(CommandCompleted(name))

    eventbus.emit(RunCompleted(run_id))


def _infer_name(command: Command) -> str:
    if hasattr(command, "__name__"):
        name = str(command.__name__)
    else:
        name = type(command).__name__

    split = name.split("_")
    return split[0].capitalize() + "".join(s.capitalize() for s in split[1:])
