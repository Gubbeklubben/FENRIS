"""
Integration tests for FedNaughty — exhaustive scenario/point/exception matrix.

These tests are intentionally excluded from the standard pytest suite because
they run the full federated pipeline and are time-consuming.

To run explicitly:
    pytest tests/integration/test_fed_naughty.py -v
    pytest tests/integration/test_fed_naughty.py -v -k "corrupt"
    pytest tests/integration/test_fed_naughty.py -v -k "synth_sample"
    pytest tests/integration/test_fed_naughty.py -v -k "crash"
"""

import threading

import pytest

import fenris.app.run.runner as runner
from fenris.app.run.pipeline import pipeline
from fenris.config.builder import build_config

# ── Helpers ───────────────────────────────────────────────────────────

DATASET = "./datasets/breast_cancer.csv"


def _run(scenario: str, point: str, exception: str = "ValueError") -> None:
    """Build config and run the full pipeline for a single FedNaughty combination."""
    kwargs: dict = {
        "synthesizer": "fed_naughty",
        "coordinator": "fedavg",
        "partitioner": "iid_partitioner",
        "dataset": DATASET,
        "synthesizer_kwargs": {
            "scenario": scenario,
            "point": point,
        },
    }
    if scenario == "crash":
        kwargs["synthesizer_kwargs"]["exception"] = exception

    config = build_config(kwargs)
    runner.run(config, pipeline())


def _expect(
    scenario: str,
    point: str,
    *,
    raises: type[Exception] | None = None,
    caused_by: type[Exception] | None = None,
    match: str | None = None,
) -> pytest.param:
    test_id = f"{scenario}__{point}"
    return pytest.param(scenario, point, raises, caused_by, match, id=test_id)


def _expect_crash(
    point: str,
    exception: str,
    *,
    raises: type[Exception] | None = None,
) -> pytest.param:
    test_id = f"crash__{point}__{exception}"
    return pytest.param(point, exception, raises, id=test_id)


# ── Non-crash test matrix ─────────────────────────────────────────────

TEST_CASES = [
    _expect(
        "empty",
        "global_init",
        raises=ValueError,
        match="Cannot decode to fedavg.GlobalState. Payload is missing state.",
    ),
    _expect(
        "empty",
        "synth_train",
        raises=RuntimeError,
        caused_by=ValueError,
        match="Cannot decode to fedavg.ClientUpdate. Payload is missing state.",
    ),
    _expect(
        "empty",
        "synth_sample",
        raises=ValueError,
        match="DataFrame returned from <FedNaughty>.sample() is empty.",
    ),
    _expect(
        "corrupt",
        "global_init",
        raises=ValueError,
        match="Cannot decode to fedavg.GlobalState. Payload is missing state.",
    ),
    _expect(
        "corrupt",
        "synth_train",
        raises=RuntimeError,
        caused_by=ValueError,
        match="Cannot decode to fedavg.ClientUpdate. Payload is missing state.",
    ),
    _expect(
        "corrupt",
        "synth_sample",
        raises=ValueError,
        match=r"DataFrame returned from <FedNaughty>.sample() does not match schema",
    ),
    _expect(
        "wrong_type",
        "global_init",
        raises=TypeError,
        match="Invalid value type returned from <FedNaughty>.global_init()",
    ),
    _expect(
        "wrong_type",
        "synth_train",
        raises=RuntimeError,
        caused_by=ValueError,
        match="No replies, can not aggregate.",
        # The real root cause exception occurs in the clients:
        # TypeError("Invalid value type returned from <FedNaughty>.train() ...")
        # However, there's no simple way to test for this exception.
        # Ray catches it, but returns an empty client reply.
        # Instead, check for ValueError("No replies ...") raised by the server.
    ),
    _expect(
        "wrong_type",
        "synth_sample",
        raises=TypeError,
        match=r"<FedNaughty>.sample() must return a DataFrame",
    ),
    _expect(
        "nan_columns",
        "global_init",
        raises=ValueError,
        match="Scenario 'nan_columns' is not applicable at point 'global_init'.",
    ),
    _expect(
        "nan_columns",
        "synth_train",
        raises=ValueError,
        match="Scenario 'nan_columns' is not applicable at point 'synth_train'.",
    ),
    _expect(
        "nan_columns",
        "synth_sample",
        raises=None,  # We handle all-NaN columns gracefully by emitting NaN metrics
    ),
    _expect(
        "crash",
        "global_init",
        raises=ValueError,
    ),
    _expect(
        "crash",
        "synth_train",
        raises=RuntimeError,
        caused_by=ValueError,
    ),
    _expect(
        "crash",
        "synth_sample",
        raises=ValueError,
    ),
]


# ── Test functions ────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.parametrize("scenario,point,raises,caused_by,match", TEST_CASES)
def test_fed_naughty(scenario, point, raises, caused_by, match):
    if raises is None:
        _run(scenario, point)
        return

    thread_exceptions: list[BaseException] = []
    original_excepthook = threading.excepthook

    def capture_excepthook(args: threading.ExceptHookArgs) -> None:
        if args.exc_value is not None:
            thread_exceptions.append(args.exc_value)
        if caused_by is None:
            original_excepthook(args)

    threading.excepthook = capture_excepthook
    try:
        with pytest.raises(raises) as exc:
            _run(scenario, point)
    finally:
        threading.excepthook = original_excepthook

    if caused_by is None:
        if match is not None:
            assert match in str(exc.value)
    else:
        assert thread_exceptions, "Expected a thread exception"
        assert type(thread_exceptions[0]) is caused_by
        if match is not None:
            assert match in str(thread_exceptions[0])
