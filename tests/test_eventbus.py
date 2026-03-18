import queue
import threading
import time
from queue import Queue

import pytest

from fedbench.core.events import Event
from fedbench.runtime.eventbus import BusState, EventBus


class SomeEvent(Event):
    pass


def wait_for(predicate, timeout):
    start = time.perf_counter()
    while True:
        if predicate():
            return True

        if time.perf_counter() - start > timeout:
            return predicate()
        time.sleep(0.1)


@pytest.fixture
def default_observer():
    def _observer(*_):
        _observer.called = True

    _observer.called = False
    return _observer


@pytest.fixture
def event_bus():
    return EventBus()


def test_emitted_is_seen(event_bus):
    event = SomeEvent()
    seen = False

    def obs(e):
        nonlocal seen
        if e is event:
            seen = True

    event_bus.register(obs, (SomeEvent,))
    with event_bus:
        event_bus.emit(event)

    assert seen


def test_bad_observer(event_bus):
    def bad(*_):
        raise RuntimeError()

    def good(*_):
        good.called = "Hell yeah!"

    event_bus.register(bad, (SomeEvent,))
    event_bus.register(good, (SomeEvent,))
    with event_bus:
        event_bus.emit(SomeEvent())

    assert hasattr(good, "called")


def test_transitions(event_bus):
    observer_wait_for = threading.Event()

    def slow(*_):
        observer_wait_for.wait()

    assert event_bus.state is BusState.INITIAL
    event_bus.register(slow, (SomeEvent,))
    assert event_bus.state is BusState.INITIAL

    event_bus.open()
    assert event_bus.state is BusState.OPEN

    event_bus.emit(SomeEvent())
    t = threading.Thread(target=event_bus.close)
    t.start()
    assert wait_for(lambda: event_bus.state is BusState.CLOSING, timeout=2.0)

    observer_wait_for.set()
    t.join(3.0)
    assert event_bus.state is BusState.CLOSED


def test_concurrent_emitters(event_bus):
    total = 0
    num_producers = 10
    num_events_per_producer = 100
    obs_lock = threading.Lock()

    def obs(*_):
        nonlocal total
        with obs_lock:
            total += 1

    event_bus.register(obs, (SomeEvent,))

    def producer(n):
        for _ in range(n):
            event_bus.emit(SomeEvent())

    with event_bus:
        threads = [
            threading.Thread(target=producer, args=(num_events_per_producer,))
            for _ in range(num_producers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert total == (num_producers * num_events_per_producer)


def test_observer_called(event_bus, default_observer):
    event_bus.register(default_observer, (SomeEvent,))
    with event_bus:
        event_bus.emit(SomeEvent())

    assert default_observer.called


def test_init_state(event_bus, default_observer):
    event_bus.register(default_observer, (SomeEvent,))

    with pytest.raises(RuntimeError):
        event_bus.emit(SomeEvent())

    with pytest.raises(RuntimeError):
        event_bus.close()


def test_open_state(event_bus, default_observer):
    with event_bus:
        event_bus.emit(SomeEvent())

        with pytest.raises(RuntimeError):
            event_bus.register(default_observer, (SomeEvent,))

        with pytest.raises(RuntimeError):
            event_bus.open()


def test_closing_state(event_bus, default_observer):
    observer_wait_for = threading.Event()

    def slow(*_):
        observer_wait_for.wait()

    event_bus.register(slow, (SomeEvent,))
    event_bus.open()

    event_bus.emit(SomeEvent())
    t = threading.Thread(target=event_bus.close)
    t.start()
    assert wait_for(lambda: event_bus.state is BusState.CLOSING, timeout=2.0)

    with pytest.raises(RuntimeError):
        event_bus.register(lambda e: None, (SomeEvent,))

    with pytest.raises(RuntimeError):
        event_bus.open()

    with pytest.raises(RuntimeError):
        event_bus.emit(SomeEvent())

    assert not event_bus.close()

    observer_wait_for.set()
    t.join(3.0)


def test_closed_state(event_bus, default_observer):
    with event_bus:
        pass

    with pytest.raises(RuntimeError):
        event_bus.register(default_observer, (SomeEvent,))

    with pytest.raises(RuntimeError):
        event_bus.emit(SomeEvent())

    with pytest.raises(RuntimeError):
        event_bus.open()

    assert not event_bus.close()


def test_emit_none_raises(event_bus):
    with event_bus:
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            event_bus.emit(None)


def test_emit_any_raises(event_bus):
    with event_bus:
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            event_bus.emit(object())


def test_close_raises_from_observer_thread(event_bus):
    exc_queue: Queue[Exception] = queue.Queue()

    def observer(_):
        try:
            event_bus.close()
        except Exception as e:
            exc_queue.put_nowait(e)

    event_bus.register(observer, (SomeEvent,))
    with event_bus:
        event_bus.emit(SomeEvent())

    exc = exc_queue.get(timeout=3.0)
    assert isinstance(exc, RuntimeError)
