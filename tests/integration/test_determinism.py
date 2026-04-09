"""Integration tests: two runs with the same seed must produce identical synthetic.csv.

Run all synthesizers:
    pytest tests/integration/test_determinism.py -m integration

Run a single synthesizer (e.g. fed_tgan):
    pytest tests/integration/test_determinism.py -m integration -k fed_tgan
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import fedbench.runtime.runner as runner
from fedbench.config.builder import build_config
from fedbench.runtime.pipeline import pipeline
from fedbench.runtime.registry import Group

_DATASET = Path(__file__).parent.parent.parent / "datasets" / "breast_cancer.csv"


def _synthesizer_names() -> list[str]:
    return list(Group.SYNTHESIZERS.get_registry())


def _coordinator_for(synthesizer_name: str) -> str:
    """Return the first registered coordinator compatible with the synthesizer."""
    factory = Group.SYNTHESIZERS.get_registry().load(synthesizer_name)
    coord_registry = Group.COORDINATORS.get_registry()
    for coord_name in sorted(factory.SUPPORTS_COORDINATORS):
        if coord_name in coord_registry:
            return coord_name
    raise ValueError(
        f"No registered coordinator found for synthesizer {synthesizer_name!r}. "
        f"Supported: {factory.SUPPORTS_COORDINATORS}"
    )


def _run_once(synthesizer: str, coordinator: str, outputdir: Path) -> Path:
    """Execute one full pipeline run and return the path to synthetic.csv."""
    # build_config mutates its input (pops "seed"), so always pass a fresh dict.
    cli_input = {
        "synthesizer": synthesizer,
        "coordinator": coordinator,
        "partitioner": "iid_partitioner",
        "dataset": str(_DATASET),
        "outputdir": str(outputdir),
    }
    config = build_config(cli_input)
    runner.run(config, pipeline())

    matches = list(outputdir.glob("*/synthetic.csv"))
    assert len(matches) == 1, (
        f"Expected exactly one synthetic.csv under {outputdir}, found {matches}"
    )
    return matches[0]


@pytest.mark.integration
@pytest.mark.parametrize("synthesizer_name", _synthesizer_names())
def test_determinism(synthesizer_name: str, tmp_path: Path) -> None:
    """Two runs with the same default seed must produce bit-identical synthetic.csv."""
    coordinator_name = _coordinator_for(synthesizer_name)

    path_a = _run_once(synthesizer_name, coordinator_name, tmp_path / "run_a")
    path_b = _run_once(synthesizer_name, coordinator_name, tmp_path / "run_b")

    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    pd.testing.assert_frame_equal(df_a, df_b)
