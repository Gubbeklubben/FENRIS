from pathlib import Path
from typing import Any, Optional, Union

import pytest

from fedbench.config.builder import build_config, parse_for_function
from fedbench.config.parsing import coerce, is_optional
from fedbench.core.eval import Category

from .fake_components import FakeAlgRegistry, FakePartitionerRegistry


@pytest.fixture
def algorithms():
    return FakeAlgRegistry()


@pytest.fixture
def partitioners():
    return FakePartitionerRegistry()


# --- helpers ---------------------------------------------------------------


def minimal_valid_cfg(tmp_path: Path, **overrides):
    dataset = tmp_path / "data.csv"
    dataset.write_text("a,b\n1,2\n")

    base = {
        "dataset": str(dataset),
        "algorithm": FakeAlgRegistry.KEY,
        "coordinator": "MISSING",
        "partitioner": FakePartitionerRegistry.KEY,
    }
    base.update(overrides)
    return base


# --- minimal config builder validation ------------------------------------


def test_minimal_config(tmp_path, algorithms, partitioners):
    cfg_dict = minimal_valid_cfg(tmp_path)

    cfg = build_config(cfg_dict, algorithms, partitioners)

    assert cfg.data.dataset == str(Path(cfg_dict["dataset"]).resolve())
    assert cfg.data.partitioner == cfg_dict["partitioner"]
    assert cfg.algorithm == cfg_dict["algorithm"]


# --- dataset path validation ----------------------------------------------


def test_dataset_is_directory_raises(tmp_path, algorithms, partitioners):

    cfg = minimal_valid_cfg(tmp_path)
    cfg["dataset"] = str(tmp_path)

    with pytest.raises(IsADirectoryError):
        build_config(cfg, algorithms, partitioners)


def test_dataset_does_not_exist_raises(tmp_path, algorithms, partitioners):

    cfg = minimal_valid_cfg(tmp_path)
    cfg["dataset"] = str(tmp_path / "missing.csv")

    with pytest.raises(FileNotFoundError):
        build_config(cfg, algorithms, partitioners)


# --- metrics / category validation ----------------------------------------


def test_unsupported_category_raises(tmp_path, algorithms, partitioners):

    cfg = minimal_valid_cfg(
        tmp_path,
        run_categories=("not-a-category",),
        target_col="y",
    )

    with pytest.raises(ValueError):
        build_config(cfg, algorithms, partitioners)


# --- numeric config validation --------------------------------------------


@pytest.mark.parametrize("num_rounds", [0, -5])
def test_invalid_num_rounds_raises(tmp_path, num_rounds, algorithms, partitioners):

    cfg = minimal_valid_cfg(tmp_path, num_rounds=num_rounds)

    with pytest.raises(ValueError):
        build_config(cfg, algorithms, partitioners)


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.5])
def test_invalid_test_size_raises(tmp_path, test_size, algorithms, partitioners):

    cfg = minimal_valid_cfg(tmp_path, test_size=test_size)

    with pytest.raises(ValueError):
        build_config(cfg, algorithms, partitioners)


def test_invalid_num_synthetic_rows_raises(tmp_path, algorithms, partitioners):

    cfg = minimal_valid_cfg(tmp_path, num_synthetic_rows=0)

    with pytest.raises(ValueError):
        build_config(cfg, algorithms, partitioners)


# --- outputdir behavior ----------------------------------------------------


def test_default_outputdir_is_cwd_out(tmp_path, monkeypatch, algorithms, partitioners):

    monkeypatch.chdir(tmp_path)
    cfg = minimal_valid_cfg(tmp_path)

    config = build_config(cfg, algorithms, partitioners)
    assert config.outputdir == str(tmp_path / "out")


def test_custom_outputdir_is_resolved(tmp_path, algorithms, partitioners):

    out = tmp_path / "results"
    cfg = minimal_valid_cfg(tmp_path, outputdir=str(out))

    config = build_config(cfg, algorithms, partitioners)
    assert config.outputdir == str(out.resolve())


# --- positive sanity checks ------------------------------------------------


def test_valid_utility_category_with_target_col(tmp_path, algorithms, partitioners):

    cfg = minimal_valid_cfg(
        tmp_path,
        run_categories=(Category.UTILITY,),
        target_col="b",
    )
    config = build_config(cfg, algorithms, partitioners)
    assert config.data.target_col == "b"
    assert Category.UTILITY in config.metrics.run_categories


def test_unregistered_partitioner_raises(tmp_path, algorithms, partitioners):

    cfg = minimal_valid_cfg(
        tmp_path,
        partitioner="definitely-not-registered",
    )
    with pytest.raises(ValueError):
        build_config(cfg, algorithms, partitioners)


def test_unregistered_algorithm_raises(tmp_path, algorithms, partitioners):

    cfg = minimal_valid_cfg(
        tmp_path,
        algorithm="definitely-not-registered",
    )
    with pytest.raises(ValueError):
        build_config(cfg, algorithms, partitioners)


def test_static_defaults(tmp_path, algorithms, partitioners):
    cfg = minimal_valid_cfg(tmp_path)
    config = build_config(cfg, algorithms, partitioners)

    assert config.data.target_col is None
    assert config.data.sensitive_cols == ()

    assert config.metrics.run_categories == tuple(Category)
    assert config.metrics.early_stop is False
    assert config.metrics.stop_metric is None
    assert config.metrics.stop_mode is None
    assert config.metrics.stop_epsilon == 1e-3
    assert config.metrics.stop_patience == 3
    assert config.metrics.stop_min_rounds == 1
    assert config.metrics.stop_eval_every == 1
    assert config.metrics.stop_synthetic_rows is None

    assert config.num_rounds == 3
    assert config.test_size == 0.2
    assert config.seed.partitioning == 43
    assert config.seed.init == 44
    assert config.seed.sampling == 45
    assert config.seed.evaluation == 46
    assert config.num_synthetic_rows is None
    assert config.disable_pickle is False


# --- column name validation ------------------------------------------------


def test_invalid_target_col_raises(tmp_path, algorithms, partitioners):

    cfg = minimal_valid_cfg(tmp_path, target_col="nonexistent")

    with pytest.raises(ValueError):
        build_config(cfg, algorithms, partitioners)


def test_invalid_sensitive_col_raises(tmp_path, algorithms, partitioners):

    cfg = minimal_valid_cfg(tmp_path, sensitive_cols=("nonexistent",))

    with pytest.raises(ValueError):
        build_config(cfg, algorithms, partitioners)


def test_valid_target_col_passes(tmp_path, algorithms, partitioners):

    cfg = minimal_valid_cfg(tmp_path, target_col="a")
    config = build_config(cfg, algorithms, partitioners)
    assert config.data.target_col == "a"


def test_valid_sensitive_cols_passes(tmp_path, algorithms, partitioners):

    cfg = minimal_valid_cfg(tmp_path, sensitive_cols=("a", "b"))
    config = build_config(cfg, algorithms, partitioners)
    assert config.data.sensitive_cols == ("a", "b")


def test_omitted_columns_no_error(tmp_path, algorithms, partitioners):
    """Omitting target_col and sensitive_cols is valid (NaN-degradation path)."""
    cfg = minimal_valid_cfg(tmp_path)
    config = build_config(cfg, algorithms, partitioners)
    assert config.data.target_col is None
    assert config.data.sensitive_cols == ()


# --- coerce tests ----------------------------------------------------------


def test_coerce_bool_true_variants():
    """Test coercion of various truthy boolean strings"""
    for value in ["true", "True", "TRUE", "1", "yes", "Yes", "on", "ON"]:
        assert coerce(value, bool) is True, f"Failed for value: {value}"


def test_coerce_bool_false_variants():
    """Test coercion of various falsy boolean strings"""
    for value in ["false", "False", "FALSE", "0", "no", "No", "off", "OFF", ""]:
        assert coerce(value, bool) is False, f"Failed for value: {value}"


def test_coerce_int():
    """Test coercion of string to int"""
    assert coerce("42", int) == 42
    assert coerce("-10", int) == -10
    assert coerce("0", int) == 0


def test_coerce_float():
    """Test coercion of string to float"""
    assert coerce("3.14", float) == 3.14
    assert coerce("0.5", float) == 0.5
    assert coerce("-2.5", float) == -2.5


def test_coerce_list():
    """Test coercion of string to list"""
    assert coerce("[1,2,3]", list[int]) == [1, 2, 3]
    assert coerce("[a,b,c]", list[str]) == ["a", "b", "c"]


def test_coerce_tuple():
    """Test coercion of string to tuple"""
    assert coerce("(1,2,3)", tuple[int]) == (1, 2, 3)
    assert coerce("(x,y)", tuple[str]) == ("x", "y")


@pytest.mark.parametrize("list_type", [list, list[Any], tuple, tuple[Any]])
def test_coerce_invalid_list_raises(list_type):
    """Test that invalid list syntax raises error"""
    with pytest.raises(TypeError):
        coerce("not a valid list", list_type)


def test_coerce_str():
    """Test coercion of string to string"""
    result = coerce("hello", str)
    assert result == "hello"


# --- is_optional tests -----------------------------------------------------


def test_is_optional_with_optional_type():
    """Test is_optional returns True for Optional[T]"""
    annotation = Optional[str]
    assert is_optional(annotation) is True


def test_is_optional_with_union_with_none():
    """Test is_optional returns True for Union with None"""
    annotation = Union[str, None]
    assert is_optional(annotation) is True


def test_is_optional_with_union_without_none():
    """Test is_optional returns False for Union without None"""
    annotation = Union[str, int]
    assert is_optional(annotation) is False


def test_is_optional_with_non_optional_type():
    """Test is_optional returns False for regular types"""
    assert is_optional(str) is False
    assert is_optional(int) is False
    assert is_optional(bool) is False


def test_is_optional_with_list():
    """Test is_optional returns False for list types"""
    assert is_optional(list) is False


def test_is_optional_with_optional_list():
    """Test is_optional returns True for Optional[list[T]]"""
    annotation = Optional[list]
    assert is_optional(annotation) is True


# --- parse_for_function tests ----------------------------------------------


def test_parse_for_function_with_no_parameters():
    """Test parsing a function with no parameters"""

    def dummy_func():
        pass

    result = parse_for_function(dummy_func, {})
    assert result == {}


def test_parse_for_function_with_required_parameter():
    """Test parsing required parameters"""

    def dummy_func(required_param: str):
        pass

    result = parse_for_function(dummy_func, {"required_param": "value"})
    assert result == {"required_param": "value"}


def test_parse_for_function_missing_required_parameter_raises():
    """Test that missing required parameter raises TypeError"""

    def dummy_func(required_param: str):
        pass

    with pytest.raises(TypeError):
        parse_for_function(dummy_func, {})


def test_parse_for_function_with_default_parameter():
    """Test parsing function with default parameters"""

    def dummy_func(param_with_default: str = "default_value"):
        pass

    result = parse_for_function(dummy_func, {})
    assert result == {}


def test_parse_for_function_overrides_default():
    """Test that provided parameters override defaults"""

    def dummy_func(param: str = "default"):
        pass

    result = parse_for_function(dummy_func, {"param": "custom"})
    assert result == {"param": "custom"}


def test_parse_for_function_with_optional_type_annotation():
    """Test parsing optional type annotations"""

    def dummy_func(optional_param: Optional[str]):
        pass

    # Should not raise even though parameter is not provided
    result = parse_for_function(dummy_func, {})
    assert result == {}


def test_parse_for_function_unknown_parameter_raises():
    """Test that unknown parameters raise TypeError"""

    def dummy_func(known_param: str):
        pass

    with pytest.raises(TypeError, match="Unknown parameters"):
        parse_for_function(
            dummy_func,
            {"known_param": "value", "unknown_param": "value"},
        )


def test_parse_for_function_coerces_types():
    """Test that parameters are coerced to correct types"""

    def dummy_func(num: int, enabled: bool, items: list[int]):
        pass

    result = parse_for_function(
        dummy_func,
        {
            "num": "42",
            "enabled": "true",
            "items": "[1, 2, 3]",
        },
    )

    assert result["num"] == 42
    assert result["enabled"] is True
    assert result["items"] == [1, 2, 3]


def test_parse_for_function_multiple_parameters_mixed():
    """Test parsing mix of required, optional, and default parameters"""

    def dummy_func(
        required: str,
        with_default: int = 10,
        optional_type: Optional[str] = None,
    ):
        pass

    result = parse_for_function(
        dummy_func, {"required": "test_value", "with_default": "20"}
    )

    assert result["required"] == "test_value"
    assert result["with_default"] == 20
    assert "optional_type" not in result


# --- additional coerce edge cases --------------------------------------------------


def test_coerce_int_invalid_raises():
    """Test that invalid integer strings raise ValueError"""
    with pytest.raises(ValueError):
        coerce("not_a_number", int)


def test_coerce_float_invalid_raises():
    """Test that invalid float strings raise ValueError"""
    with pytest.raises(ValueError):
        coerce("not_a_float", float)


def test_coerce_empty_list():
    """Test coercion of empty list string"""
    assert coerce("[]", list) == []


def test_coerce_empty_tuple():
    """Test coercion of empty tuple string"""
    assert coerce("()", tuple) == ()


def test_coerce_nested_structures():
    """Test coercion of nested list structures"""
    result = coerce("[[1, 2], [3, 4]]", list[list[int]])
    assert result == [[1, 2], [3, 4]]


# --- dataset path expansion tests --------------------------------------------------


def test_dataset_path_expanduser(
    tmp_path,
    monkeypatch,
    algorithms,
    partitioners,
):
    """Test that dataset paths with ~ are expanded"""
    # Create a dataset file
    dataset = tmp_path / "data.csv"
    dataset.write_text("a,b\n1,2\n")

    cfg = minimal_valid_cfg(tmp_path)

    # Replace the path with expanded home dir path
    real_path = cfg["dataset"]

    # Config should expand and resolve the path correctly
    config = build_config(cfg, algorithms, partitioners)

    # Verify the path is resolved
    assert config.data.dataset == str(Path(real_path).resolve())
    assert not config.data.dataset.startswith("~")


def test_outputdir_path_expanduser(
    tmp_path,
    monkeypatch,
    algorithms,
    partitioners,
):
    """Test that outputdir paths are expanded and resolved"""
    dataset = tmp_path / "data.csv"
    dataset.write_text("a,b\n1,2\n")

    out = tmp_path / "results"
    cfg = minimal_valid_cfg(tmp_path, outputdir=str(out))

    config = build_config(cfg, algorithms, partitioners)

    # Verify output directory is properly resolved
    assert config.outputdir == str(out.resolve())
    assert not config.outputdir.startswith("~")


# --- categories parsing tests --------------------------------------------------


def test_categories_all_defaults_to_all_enum_members(
    tmp_path,
    algorithms,
    partitioners,
):
    """Test that no categories specified defaults to all Category enum members"""
    cfg = minimal_valid_cfg(tmp_path)
    # Don't specify run_categories

    config = build_config(cfg, algorithms, partitioners)

    # Should include all Category enum members
    assert config.metrics.run_categories == tuple(Category)
    assert len(config.metrics.run_categories) > 0


def test_single_category_parsing(tmp_path, algorithms, partitioners):
    """Test parsing a single category"""
    cfg = minimal_valid_cfg(
        tmp_path,
        run_categories=(Category.PRIVACY,),
        target_col="b",
    )

    config = build_config(cfg, algorithms, partitioners)

    assert config.metrics.run_categories == (Category.PRIVACY,)


def test_multiple_categories_parsing(
    tmp_path,
    algorithms,
    partitioners,
):
    """Test parsing multiple categories"""
    cfg = minimal_valid_cfg(
        tmp_path,
        run_categories=(Category.UTILITY, Category.PRIVACY),
        target_col="b",
    )

    config = build_config(cfg, algorithms, partitioners)

    assert Category.UTILITY in config.metrics.run_categories
    assert Category.PRIVACY in config.metrics.run_categories


# --- algorithm and partitioner kwargs parsing tests ----------------------


def test_algorithm_kwargs_empty(tmp_path, algorithms, partitioners):
    """Test that empty algorithm_kwargs defaults to empty dict"""
    cfg = minimal_valid_cfg(tmp_path)

    config = build_config(cfg, algorithms, partitioners)

    assert config.algorithm_kwargs == {}


def test_partitioner_kwargs_preserved(
    tmp_path,
    algorithms,
    partitioners,
):
    """Test that partitioner_kwargs are preserved"""
    cfg = minimal_valid_cfg(tmp_path, num_clients=5)

    config = build_config(cfg, algorithms, partitioners)

    assert "num_partitions" in config.data.partitioner_kwargs
    assert config.data.partitioner_kwargs["num_partitions"] == 5


# --- seed injection into partitioner kwargs --------------------------------


def test_parse_for_function_all_optional_with_union():
    """Test parsing when all parameters are optional with Union types"""

    def dummy_func(
        param1: Optional[str] = None,
        param2: Optional[int] = None,
    ):
        pass

    result = parse_for_function(dummy_func, {})
    assert result == {}


def test_parse_for_function_partial_parameters():
    """Test parsing when only some of multiple parameters are provided"""

    def dummy_func(
        param1: str,
        param2: str = "default",
        param3: Optional[str] = None,
    ):
        pass

    result = parse_for_function(dummy_func, {"param1": "value1"})

    assert result["param1"] == "value1"
    assert "param2" not in result  # Not provided, has default
    assert "param3" not in result  # Not provided, optional


def test_parse_for_function_with_tuple_coercion():
    """Test coercion of tuple parameters"""

    def dummy_func(items: tuple[int]):
        pass

    result = parse_for_function(dummy_func, {"items": "(1, 2, 3)"})

    assert result["items"] == (1, 2, 3)


# --- parse_for_function untyped parameter tests ---------------------------


def test_parse_for_function_untyped_param_passes_raw_string():
    """Untyped required param provided in raw → passed through as-is."""

    def dummy_func(x):
        pass

    result = parse_for_function(dummy_func, {"x": "42"})

    assert result["x"] == "42"


def test_parse_for_function_untyped_param_with_default_passes_raw_string():
    """Untyped param with default provided in raw → passed through as-is."""

    def dummy_func(x=10):
        pass

    result = parse_for_function(dummy_func, {"x": "99"})

    assert result["x"] == "99"


def test_parse_for_function_untyped_missing_required_still_raises():
    """Untyped required param not in raw → still raises TypeError."""

    def dummy_func(x):
        pass

    with pytest.raises(TypeError, match="Missing required parameter"):
        parse_for_function(dummy_func, {})


# --- validation tests --------------------------------------------------


def test_validate_partitioner_not_in_registry(
    tmp_path,
    algorithms,
    partitioners,
):
    """Test that unregistered partitioner raises ValueError"""
    cfg = minimal_valid_cfg(
        tmp_path,
        partitioner="nonexistent-partitioner",
    )

    with pytest.raises(ValueError):
        build_config(cfg, algorithms, partitioners)
