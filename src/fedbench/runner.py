import uuid
from collections.abc import Iterable

from fedbench.config import Config
from fedbench.core.eventbus import EventBus
from fedbench.core.events import (
    Event,
    RunStarted,
    RunCompleted,
    RunFailed,
    CommandStarted,
    CommandCompleted,
)
from fedbench.core.pipeline import Command
from fedbench.core.runcontext import RunContext


def run(
        config: Config,
        eventbus: EventBus,
        commands: Iterable[Command]) -> None:

    run_id = str(uuid.uuid4())
    eventbus.register(lambda e: print(e), (Event, ))

    with eventbus:
        eventbus.emit(RunStarted(run_id))

        ctx = RunContext(run_id, config, eventbus)
        for command in commands:
            eventbus.emit(CommandStarted(command.__name__))
            try:
                command(ctx)
            except Exception as exc:
                eventbus.emit(
                    RunFailed(
                        run_id,
                        command.__name__,
                        str(type(exc)),
                        str(exc))
                    )
                raise exc
            else:
                eventbus.emit(CommandCompleted(command.__name__))

        eventbus.emit(RunCompleted(run_id))