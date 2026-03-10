import queue
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Protocol, Self, cast

from fedbench.core.events import Event
from fedbench.core.logger import log_error, log_warning


class BusState(Enum):
    INITIAL = 0
    OPEN = 1
    CLOSING = 2
    CLOSED = 3


class Observer(Protocol):
    def __call__(self, event: Event) -> None:
        pass


@dataclass
class _ObserverEntry:
    observer: Observer
    event_types: tuple[type[Event], ...]
    failures: int = 0


type _Observers = list[_ObserverEntry]
type _FrozenObservers = tuple[_ObserverEntry, ...]


class EventBus:
    def __init__(self) -> None:
        self._state = BusState.INITIAL
        self._observers: _Observers = []
        self._frozen_observers: _FrozenObservers = ()
        self._observer_thread: threading.Thread | None = None
        self._event_queue: queue.Queue[Event | None] = queue.Queue()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} <{self._state}>"

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:

        self.close()
        return None

    @property
    def state(self) -> BusState:
        return self._state

    def register(
        self,
        observer: Observer,
        event_types: Iterable[type[Event]],
    ) -> None:

        with self._lock:
            if self._state is not BusState.INITIAL:
                raise RuntimeError(f"{self}: Can not register observer.")

            if not event_types:
                return

            for event_type in event_types:
                if not issubclass(event_type, Event):
                    raise TypeError(f"Not a valid event type: {event_type}")

            self._observers.append(_ObserverEntry(observer, tuple(event_types)))

    def open(self) -> None:
        with self._lock:
            if self._state is not BusState.INITIAL:
                raise RuntimeError(f"{self}: Can not open.")

            # noinspection PyUnnecessaryCast
            self._frozen_observers = tuple(self._observers)
            self._observers.clear()
            self._observer_thread = threading.Thread(
                target=self._worker,
                name=self.__class__.__name__,
                daemon=False,
            )
            self._observer_thread.start()
            self._state = BusState.OPEN

    def emit(self, event: Event) -> None:
        if not isinstance(event, Event):
            raise TypeError(f"Not a valid event type: {event}")

        with self._lock:
            if self._state is not BusState.OPEN:
                raise RuntimeError(f"{self}: Can not emit event.")

            self._event_queue.put_nowait(event)

    def close(self) -> bool:
        if threading.current_thread() is self._observer_thread:
            raise RuntimeError(f"Attempt to close {self} from observer thread.")

        with self._lock:
            if self._state is BusState.INITIAL:
                raise RuntimeError(f"{self}: Can not close.")

            if self._state in (BusState.CLOSING, BusState.CLOSED):
                return False

            self._state = BusState.CLOSING  # No more events in

        self._event_queue.join()
        self._event_queue.put_nowait(None)
        # noinspection PyUnnecessaryCast
        cast(threading.Thread, self._observer_thread).join()

        with self._lock:
            self._state = BusState.CLOSED

        return True

    def _worker(self) -> None:
        while True:
            event = self._event_queue.get()
            if event is None:
                self._event_queue.task_done()
                return

            try:
                for entry in self._frozen_observers:
                    if entry.failures > 0:
                        log_warning(str(self), "Ignoring previously failed observer")
                        continue

                    if isinstance(event, entry.event_types):
                        # noinspection PyBroadException
                        try:
                            entry.observer(event)
                        except Exception:
                            log_error(
                                str(self), "Exception in observer: ", exc_info=True
                            )
                            entry.failures += 1
            finally:
                self._event_queue.task_done()
