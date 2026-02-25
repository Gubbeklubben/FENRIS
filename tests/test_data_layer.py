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
            "id": [1, 2, 3, 4],
            "age": [25.0, 30.0, 45.0, 50.0],
            "label": [0, 1, 1, 0],
            "cat": ["car", "truck", "car", "bike"],
        }
    )

def test_load_csv_and_schema(sample_df, tmp_path):
    csv_path = tmp_path / "sample.csv"
    sample_df.to_csv(csv_path, index=False)

    df, schema = load_csv(csv_path)

    assert df.shape == sample_df.shape
    # schema is deterministic
    assert [c.kind for c in schema.columns] == [
        "integer",
        "continuous",
        "binary",
        "categorical",
    ]

def test_iid_partitioning(sample_df, built_in_partitioners):
    ds = PartitionedDataset(
        sample_df,
        infer_schema(sample_df),
        partitioner=built_in_partitioners.call(
            name="iid-partitioner", num_partitions=2,
        ),
        test_size=0.2,
        seed=80085
    )
    assert ds.load_train_partition(0).get("id").values[0] == 1
    assert ds.load_train_partition(1).get("id").values[0] == 3
    assert ds.load_test_partition(0).get("id").values[0] == 2
    assert ds.load_test_partition(1).get("id").values[0] == 4