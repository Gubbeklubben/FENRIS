import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def ray_session():
    """Speed up back-to-back framework runs by initializing Ray once
    and keeping it loaded until the test suite is finished."""
    import ray

    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    ray.init(ignore_reinit_error=True)
    # Patch ray.shutdown to a no-op so Flower can't tear it down between tests
    original_shutdown = ray.shutdown
    ray.shutdown = lambda *args, **kwargs: None
    yield
    # Restore and actually shut down at end of session
    ray.shutdown = original_shutdown
    ray.shutdown()
