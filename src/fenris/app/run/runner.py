import uuid
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from fenris.app.run.command import Command
from fenris.app.run.eventbus import EventBus
from fenris.app.run.runcontext import RunContext
from fenris.app.run.scalability_collector import ScalabilityCollector
from fenris.config import Config
from fenris.core.events import (
    ClientReply,
    CommandCompleted,
    CommandStarted,
    Event,
    RunCompleted,
    RunFailed,
    RunStarted,
    ServerRequest,
    TrainEvalLoopCompleted,
    TrainEvalLoopStarted,
)
from fenris.core.logger import log_debug, log_error


def run(config: Config, commands: Iterable[Command]) -> None:
    eventbus = EventBus()
    eventbus.register(lambda event: log_debug("Event", event), (Event,))

    # Register the scalability collector before EventBus is opened
    collector = ScalabilityCollector()
    eventbus.register(
        observer=collector,
        event_types=(
            TrainEvalLoopStarted,
            TrainEvalLoopCompleted,
            ServerRequest,
            ClientReply,
        ),
    )
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dataset = Path(config.data.dataset).stem
    run_id = f"{timestamp}-{config.synthesizer}-{dataset}-{uuid.uuid4()}"
    ctx = RunContext(run_id, config, eventbus, collector)

    with eventbus:
        _run(ctx, commands)


def _run(ctx: RunContext, commands: Iterable[Command]) -> None:
    run_id = ctx.run_id
    eventbus = ctx.eventbus

    eventbus.emit(RunStarted(run_id))

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
            raise
        else:
            eventbus.emit(CommandCompleted(name))

    eventbus.emit(RunCompleted(run_id))


def _infer_name(command: Command) -> str:
    if hasattr(command, "__name__"):
        name = str(command.__name__)
    else:
        name = type(command).__name__

    split = name.split("_")
    return "".join(s.capitalize() for s in split)
