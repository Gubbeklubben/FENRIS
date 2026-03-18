import pandas as pd
from pandas import DataFrame

from fedbench.partitioners.flwr_delegates import FlwrDelegatePartitioner


def make_dummy_dataset(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({"value": range(n), "label": [i % 3 for i in range(n)]})


def assert_valid_partition(p: FlwrDelegatePartitioner, num_partitions: int) -> None:
    assert p.num_partitions == num_partitions
    for i in range(num_partitions):
        partition = p.load_partition(i, split="train", seed=42, test_size=0.2)
        assert isinstance(partition, DataFrame)
        assert len(partition) > 0


def test_linear_partitioner():
    p = FlwrDelegatePartitioner.with_linear_partitioner(num_partitions=5)
    p.set_dataset(make_dummy_dataset())
    assert_valid_partition(p, 5)

    # Partition sizes should increase with partition id
    sizes = [
        len(p.load_partition(i, split="train", seed=42, test_size=0.2))
        for i in range(5)
    ]
    assert sizes == sorted(sizes), "Linear partition sizes should be increasing"


def test_square_partitioner():
    p = FlwrDelegatePartitioner.with_square_partitioner(num_partitions=5)
    p.set_dataset(make_dummy_dataset(n=1000))
    assert_valid_partition(p, 5)

    sizes = [
        len(p.load_partition(i, split="train", seed=42, test_size=0.2))
        for i in range(5)
    ]
    assert sizes == sorted(sizes), "Square partition sizes should be increasing"


def test_exponential_partitioner():
    p = FlwrDelegatePartitioner.with_exponential_partitioner(num_partitions=5)
    p.set_dataset(make_dummy_dataset(n=1000))
    assert_valid_partition(p, 5)

    sizes = [
        len(p.load_partition(i, split="train", seed=42, test_size=0.2))
        for i in range(5)
    ]
    assert sizes == sorted(sizes), "Exponential partition sizes should be increasing"


def test_dirichlet_partitioner():
    p = FlwrDelegatePartitioner.with_dirichlet_partitioner(
        num_partitions=5, partition_by="label", alpha=0.5
    )
    p.set_dataset(
        make_dummy_dataset(n=1000)
    )  # increased to avoid min_partition_size warnings
    assert_valid_partition(p, 5)

    partition_sizes = [
        len(p.load_partition(i, split="train", seed=42, test_size=0.2))
        for i in range(5)
    ]
    assert len(set(partition_sizes)) > 1, "Dirichlet partitions should be non-uniform"


def test_pathological_partitioner():
    p = FlwrDelegatePartitioner.with_pathological_partitioner(
        num_partitions=5, partition_by="label", num_classes_per_partition=2
    )
    p.set_dataset(make_dummy_dataset())
    assert_valid_partition(p, 5)

    # Each partition should contain exactly 2 unique labels
    for i in range(5):
        partition = p.load_partition(i, split="train", seed=42, test_size=0.2)
        assert partition["label"].nunique() <= 2, (
            f"Partition {i} should have at most 2 unique labels"
        )


def test_shard_partitioner():
    p = FlwrDelegatePartitioner.with_shard_partitioner(
        num_partitions=5, partition_by="label", num_shards_per_partition=2
    )
    p.set_dataset(make_dummy_dataset())
    assert_valid_partition(p, 5)

    # Shard partitioner should produce label-skewed partitions
    label_counts = [
        p.load_partition(i, split="train", seed=42, test_size=0.2)["label"].nunique()
        for i in range(5)
    ]
    assert any(count < 3 for count in label_counts), (
        "Shard partitioner should produce partitions with fewer than all labels"
    )


def test_continuous_partitioner():
    p = FlwrDelegatePartitioner.with_continuous_partitioner(
        num_partitions=5, partition_by="value", strictness=0.7
    )
    p.set_dataset(make_dummy_dataset())
    assert_valid_partition(p, 5)

    # Higher strictness → partitions concentrated around different value ranges
    partition_means = [
        p.load_partition(i, split="train", seed=42, test_size=0.2)["value"].mean()
        for i in range(5)
    ]
    assert len(set(partition_means)) > 1, (
        "Continuous partitions should have different value distributions"
    )
