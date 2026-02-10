import pathlib
import pandas as pd
import pytest
from fedbench.data import loaders, partitioners, infer_schema
from fedbench.data.schemas import ColumnSchema

TEST_CSV = pathlib.Path(__file__).parent / "fixtures/sample.csv"

@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "age": [25, 30, 45, 50],
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
    assert [c.kind for c in schema.columns] == [
        "integer",
        "continuous",
        "binary",
        "categorical",
    ]

def test_horizontal_partition(sample_df):
    shards = partitioners.deterministic_shard(sample_df, 2, seed=42)
    assert len(shards) == 2
    # check that rows are distributed but shuffling deterministic
    assert shards[0].iloc[0]["id"] == 2  # deterministic assignment
