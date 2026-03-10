import pandas as pd
import pytest

from fedbench.core.data import load_csv, PartitionedDataset, Partitioner
from fedbench.core.data.schemas import infer_schema
from fedbench.core.factory_registry import FactoryRegistry
from fedbench.partitioners import register_builtin_partitioners


@pytest.fixture
def built_in_partitioners():
    registry = FactoryRegistry(
        group=f"{__package__}.partitioners",
        product_cls=Partitioner
    )
    register_builtin_partitioners(registry)
    return registry


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "age": [25.0, 30.0, 45.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 99.0],
            "label": [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "cat": ["car", "truck", "car", "bike", "car", "truck", "car", "bike", "car", "truck", "car", "airplane"],
        }
    )

def test_load_csv_and_schema(sample_df, tmp_path):
    csv_path = tmp_path / "sample.csv"
    sample_df.to_csv(csv_path, index=False)

    df = load_csv(csv_path)
    schema = infer_schema(df)

    assert df.shape == sample_df.shape
    # schema is deterministic
    assert [c.kind for c in schema.columns] == [
        "integer",
        "continuous",
        "binary",
        "categorical",
    ]