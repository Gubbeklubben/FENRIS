import pandas as pd
from datasets import Dataset
from fedbench.partitioners.flwr_delegates import FlwrDelegatePartitioner


def make_dummy_dataset(n: int = 100) -> Dataset:
    df = pd.DataFrame({"value": range(n), "label": [i % 3 for i in range(n)]})
    return Dataset.from_pandas(df)


def test_linear_partitioner():
    p = FlwrDelegatePartitioner.with_linear_partitioner(num_partitions=5)
    p._flwr_partitioner.dataset = make_dummy_dataset()
    assert p.num_partitions == 5


def test_square_partitioner():
    p = FlwrDelegatePartitioner.with_square_partitioner(num_partitions=5)
    p._flwr_partitioner.dataset = make_dummy_dataset()
    assert p.num_partitions == 5


def test_exponential_partitioner():
    p = FlwrDelegatePartitioner.with_exponential_partitioner(num_partitions=5)
    p._flwr_partitioner.dataset = make_dummy_dataset()
    assert p.num_partitions == 5


def test_dirichlet_partitioner():
    p = FlwrDelegatePartitioner.with_dirichlet_partitioner(
        num_partitions=5, partition_by="label", alpha=0.5
    )
    p._flwr_partitioner.dataset = make_dummy_dataset()
    assert p.num_partitions == 5


def test_pathological_partitioner():
    p = FlwrDelegatePartitioner.with_pathological_partitioner(
        num_partitions=5, partition_by="label", num_classes_per_partition=2
    )
    p._flwr_partitioner.dataset = make_dummy_dataset()
    assert p.num_partitions == 5


def test_shard_partitioner():
    p = FlwrDelegatePartitioner.with_shard_partitioner(
        num_partitions=5, partition_by="label"
    )
    p._flwr_partitioner.dataset = make_dummy_dataset()
    assert p.num_partitions == 5


def test_continuous_partitioner():
    p = FlwrDelegatePartitioner.with_continuous_partitioner(
        num_partitions=5, partition_by="value", strictness=0.7
    )
    p._flwr_partitioner.dataset = make_dummy_dataset()
    assert p.num_partitions == 5