import pandas as pd
import pytest

from fedbench.builtins.partitioners.flwr_delegates import FlwrDelegatePartitioner
from fedbench.core.data import load_csv
from fedbench.core.data.schemas import infer_schema
from fedbench.runtime.partitioned_dataset import PartitionedDataset
from tests.fake_components import FakePartitionerRegistry


@pytest.fixture
def partitioners():
    return FakePartitionerRegistry()


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "age": [25.0, 30.0, 45.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 99.0],
            "label": [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "cat": [
                "car",
                "truck",
                "car",
                "bike",
                "car",
                "truck",
                "car",
                "bike",
                "car",
                "truck",
                "car",
                "airplane",
            ],
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


@pytest.fixture
def partitioned_dataset(sample_df):
    schema = infer_schema(sample_df)
    partitioner = FlwrDelegatePartitioner.with_iid_partitioner(num_partitions=2)
    return PartitionedDataset(sample_df, schema, partitioner, test_size=0.25, seed=42)


@pytest.mark.parametrize(
    "load_fn,kwargs",
    [
        ("load_global_holdout", {}),
        ("load_train_partition", {"partition_id": 0}),
        ("load_test_partition", {"partition_id": 0}),
    ],
)
def test_mutation_does_not_affect_source(partitioned_dataset, load_fn, kwargs):
    def load():
        return getattr(partitioned_dataset, load_fn)(**kwargs)

    result = load()
    original_values = load().copy()
    result.iloc[0, 0] = -9999
    assert load().equals(original_values)
