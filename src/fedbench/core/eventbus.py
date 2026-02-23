import queue
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Protocol, Self, cast

from fedbench.core.events import Event


@dataclass(frozen=True)
class _BusClosed(Event):
    pass

_bus_closed = _BusClosed()


class BusState(Enum):
    INITIAL = 0
    OPEN    = 1
    CLOSING = 2
    CLOSED  = 3


class Observer(Protocol):
    def __call__(self, event: Event) -> None:
        pass


type _ObserverEntry = tuple[Observer, tuple[type[Event], ...]]
type _Observers = list[_ObserverEntry]
type _FrozenObservers = tuple[_ObserverEntry, ...]


class EventBus:
    def __init__(self) -> None:
        self._state = BusState.INITIAL
        self._observers: _Observers | None = []
        self._frozen_observers: _FrozenObservers | None = None
        self._observer_thread: threading.Thread | None = None
        self._event_queue: queue.Queue[Event] = queue.Queue()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} <{ self._state}>"

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None) -> bool | None:

        self.close()
        return None

    @property
    def state(self) -> BusState:
        return self._state

    def register(
            self,
            observer: Observer,
            event_types: Iterable[type[Event]]) -> None:

        with self._lock:
            if self._state is not BusState.INITIAL:
                raise RuntimeError(f"{self}: Can not register observer.")

            if not event_types:
                return

            for event_type in event_types:
                if not issubclass(event_type, Event):
                    raise TypeError(f"Not a valid event type: {event_type}")

            # noinspection PyUnnecessaryCast
            observers = cast(_Observers, self._observers)
            observers.append((observer, tuple(event_types)))

    def open(self) -> None:
        with self._lock:
            if self._state is not BusState.INITIAL:
                raise RuntimeError(f"{self}: Can not open.")

            # noinspection PyUnnecessaryCast
            self._frozen_observers = tuple(cast(_Observers, self._observers))
            self._observers = None
            self._observer_thread = threading.Thread(
                target=self._worker,
                name=self.__class__.__name__,
                daemon=False,
            )
            self._observer_thread.start()
            self._state = BusState.OPEN

    def emit(self, event: Event) -> None:
        with self._lock:
            if self._state is not BusState.OPEN:
                raise RuntimeError(f"{self}: Can not emit event.")

            self._event_queue.put_nowait(event)

    def close(self) -> bool:
        with self._lock:
            if self._state is BusState.INITIAL:
                raise RuntimeError(f"{self}: Can not close.")

            if self._state in (BusState.CLOSING, BusState.CLOSED):
                return False

            self._state = BusState.CLOSING # No more events in

        self._event_queue.join()
        self._event_queue.put_nowait(_bus_closed)
        # noinspection PyUnnecessaryCast
        cast(threading.Thread, self._observer_thread).join()

        with self._lock:
            self._state = BusState.CLOSED

        return True

    def _worker(self) -> None:
        while True:
            event = self._event_queue.get()
            if event is _bus_closed:
                self._event_queue.task_done()
                return

            # noinspection PyUnnecessaryCast
            observers = cast(_FrozenObservers, self._frozen_observers)
            try:
                for observer, event_types in observers:
                    if not isinstance(event, event_types):
                        continue
                    # noinspection PyBroadException
                    try:
                        observer(event)
                    except Exception as exc:
                        self._on_failing_observer(observer, exc)
            finally:
                self._event_queue.task_done()

    def _on_failing_observer(self, observer: Observer, exc: Exception) -> None:
        pass


