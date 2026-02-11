import pathlib
import pandas as pd
import pytest
from fedbench.data import loaders, partitioners, infer_schema
from fedbench.data.partitioners.horizontal_partition import deterministic_shard
from fedbench.data.partitioners.partitioned_dataset import PartitionedDataset
from fedbench.data.schemas import ColumnSchema

TEST_CSV = pathlib.Path(__file__).parent / "fixtures/sample.csv"

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

    df, schema = loaders.load_csv(csv_path)

    assert df.shape == sample_df.shape
    # schema is deterministic
    print([c.kind for c in schema.columns] )
    assert [c.kind for c in schema.columns] == [
        "integer",
        "continuous",
        "binary",
        "categorical",
    ]

def test_horizontal_partition(sample_df):
    shards = deterministic_shard(sample_df, 2, seed=42)
    assert len(shards) == 2
    # check that rows are distributed but shuffling deterministic
    assert shards[0].iloc[0]["id"] == 4  # deterministic assignment

def test_iid_partitioning(sample_df):
    ds = PartitionedDataset.using_iid_partitioning(sample_df, 2)
    assert ds.load_train_partition(0).get("id").values[0] == 1
    assert ds.load_train_partition(1).get("id").values[0] == 3
    assert ds.load_test_partition(0).get("id").values[0] == 2
    assert ds.load_test_partition(1).get("id").values[0] == 4