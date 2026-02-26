import multiprocessing
import sys
import uuid
from collections.abc import Iterable
from logging import StreamHandler
from logging.handlers import QueueListener

import fedbench.core.logger as fedbench_logger
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
from fedbench.core.logger import log_debug
from fedbench.core.pipeline import Command
from fedbench.core.runcontext import RunContext


def run(config: Config, commands: Iterable[Command]) -> None:
    run_id = str(uuid.uuid4())
    eventbus = EventBus()
    log_queue: multiprocessing.Queue = multiprocessing.Queue() # type:  ignore[type-arg]

    fedbench_logger.add_queue_handler(log_queue)
    eventbus.register(lambda event: log_debug("", event), (Event,))

    log_listener = QueueListener(log_queue, StreamHandler(sys.stdout))
    log_listener.start()
    try:
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
    finally:
        log_listener.stop()