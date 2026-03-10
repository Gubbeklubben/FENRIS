"""Unit and integration tests for the fed_tgan_alt algorithm module.

Coverage:
  - FedTGANAlt constructor validation
  - Factory methods (create_coordinator / create_synthesizer)
  - Registry key resolution
  - Module-level helpers: _split_cat_num, _gumbel_softmax, _apply_activate,
    _cond_loss, _weighted_average_state_dicts, _sample_condvec_from_info
  - Generator / Residual / Discriminator forward passes
  - DataSampler: dim_cond_vec, sample_condvec, sample_original_condvec,
    sample_data (both branches), no-discrete-column paths
  - data_transformer: fit_local_continuous, fit_local_discrete,
    merge_vgm_models (all branches), merge_category_frequencies,
    GlobalDataTransformer (mixed / continuous-only / discrete-only),
    convert_column_name_value_to_id (happy path + errors)
  - weighting: compute_jsd, compute_wd, compute_client_weights
  - End-to-end smoke: fed_init → aggregate_fed_init → train → sample
  - Coordinator / Synthesizer properties and edge cases
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from fedbench.algorithms import register_builtin_algorithms
from fedbench.algorithms.fed_tgan_alt import FedTGANAlt
from fedbench.algorithms.fed_tgan_alt.data_sampler import DataSampler
from fedbench.algorithms.fed_tgan_alt.data_transformer import (
    GlobalDataTransformer,
    SpanInfo,
    fit_local_continuous,
    fit_local_discrete,
    merge_category_frequencies,
    merge_vgm_models,
)
from fedbench.algorithms.fed_tgan_alt.discriminator import Discriminator
from fedbench.algorithms.fed_tgan_alt.fed_tgan_alt import (
    FedTGANAltCoordinator,
    FedTGANAltSynthesizer,
    _apply_activate,
    _cond_loss,
    _gumbel_softmax,
    _sample_condvec_from_info,
    _split_cat_num,
    _weighted_average_state_dicts,
)
from fedbench.algorithms.fed_tgan_alt.generator import Generator, Residual
from fedbench.algorithms.fed_tgan_alt.weighting import (
    compute_client_weights,
    compute_jsd,
    compute_wd,
)
from fedbench.core.algorithm import Coordinator, Synthesizer
from fedbench.core.data.schemas import ColumnSchema, TableSchema, infer_schema
from fedbench.core.factory_registry import FactoryRegistry
from fedbench.core.update import Update


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
N = 60  # Small but enough for all VGM / GAN smoke tests


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Tiny mixed-type DataFrame: continuous, binary, categorical."""
    return pd.DataFrame(
        {
            "age": _RNG.normal(40, 10, N).astype(float),
            "income": _RNG.normal(50_000, 10_000, N).astype(float),
            "label": _RNG.choice([0, 1], N),
            "color": _RNG.choice(["red", "blue", "green"], N),
        }
    )


@pytest.fixture()
def sample_schema(sample_df):
    return infer_schema(sample_df)


@pytest.fixture()
def tiny_cfg() -> dict:
    """Minimal hyperparameter dict for fast smoke tests."""
    import torch

    return {
        "embedding-dim": 8,
        "generator-dim": [16],
        "discriminator-dim": [16],
        "generator-lr": 2e-4,
        "discriminator-lr": 2e-4,
        "batch-size": 10,
        "max-batches": 1,
        "discriminator-steps": 1,
        "pac": 2,
        "max-clusters": 3,
        "weight-threshold": 0.005,
        "log-frequency": True,
        "device": torch.device("cpu"),
    }


@pytest.fixture()
def tiny_algo() -> FedTGANAlt:
    return FedTGANAlt(
        embedding_dim=8,
        generator_dim=[16],
        discriminator_dim=[16],
        batch_size=10,
        pac=2,
        max_batches=1,
        max_clusters=3,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_resolves_fed_tgan_alt():
    """'fed_tgan_alt' key must resolve to FedTGANAlt from the builtin registry."""
    from fedbench.core.algorithm import Algorithm

    registry = FactoryRegistry(
        group="fedbench.algorithms",
        product_cls=Algorithm,
    )
    register_builtin_algorithms(registry)
    algo = registry.call("fed_tgan_alt")
    assert isinstance(algo, FedTGANAlt)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"embedding_dim": 0}, "embedding_dim"),
        ({"batch_size": 6, "pac": 4}, "divisible"),
        ({"batch_size": 1}, "even"),
        ({"pac": 0}, "pac"),
        ({"discriminator_steps": 0}, "discriminator_steps"),
        ({"max_batches": 0}, "max_batches"),
        ({"max_clusters": 0}, "max_clusters"),
        ({"generator_lr": 0}, "generator_lr"),
        ({"discriminator_lr": 0.2}, "discriminator_lr"),
    ],
)
def test_constructor_rejects_bad_params(kwargs, match):
    base = dict(batch_size=10, pac=2, embedding_dim=8)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        FedTGANAlt(**base)


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------


def test_create_coordinator_returns_coordinator(tiny_algo):
    coordinator = tiny_algo.create_coordinator()
    assert isinstance(coordinator, Coordinator)
    assert isinstance(coordinator, FedTGANAltCoordinator)


def test_create_synthesizer_returns_synthesizer(tiny_algo):
    synthesizer = tiny_algo.create_synthesizer()
    assert isinstance(synthesizer, Synthesizer)
    assert isinstance(synthesizer, FedTGANAltSynthesizer)


# ---------------------------------------------------------------------------
# data_transformer: fit_local_continuous
# ---------------------------------------------------------------------------


def test_fit_local_continuous_returns_vgm_keys():
    data = _RNG.normal(0, 1, 100)
    result = fit_local_continuous(data, max_clusters=3)
    assert set(result.keys()) == {"means", "covariances", "weights"}


def test_fit_local_continuous_empty_data():
    result = fit_local_continuous(np.array([]), max_clusters=3)
    assert result == {"means": [], "covariances": [], "weights": []}


def test_fit_local_continuous_lengths_match():
    data = _RNG.normal(5, 2, 80)
    result = fit_local_continuous(data, max_clusters=4)
    assert len(result["means"]) == len(result["covariances"]) == len(result["weights"])


# ---------------------------------------------------------------------------
# data_transformer: fit_local_discrete
# ---------------------------------------------------------------------------


def test_fit_local_discrete_counts():
    series = pd.Series(["a", "b", "a", "c", "a"])
    result = fit_local_discrete(series)
    assert result["a"] == 3
    assert result["b"] == 1
    assert result["c"] == 1


def test_fit_local_discrete_string_cast():
    series = pd.Series([1, 2, 1, 2, 1])
    result = fit_local_discrete(series)
    assert all(isinstance(k, str) for k in result)


# ---------------------------------------------------------------------------
# data_transformer: merge_category_frequencies
# ---------------------------------------------------------------------------


def test_merge_category_frequencies_sums_counts():
    local = [{"red": 3, "blue": 2}, {"red": 1, "blue": 5, "green": 4}]
    merged = merge_category_frequencies(local)
    assert merged == {"red": 4, "blue": 7, "green": 4}


def test_merge_category_frequencies_empty_input():
    assert merge_category_frequencies([]) == {}


def test_merge_category_frequencies_single_client():
    local = [{"cat": 10}]
    assert merge_category_frequencies(local) == {"cat": 10}


# ---------------------------------------------------------------------------
# data_transformer: merge_vgm_models
# ---------------------------------------------------------------------------


def _simple_vgm(mean: float = 0.0) -> dict:
    return {"means": [mean], "covariances": [1.0], "weights": [1.0]}


def test_merge_vgm_models_returns_keys():
    result = merge_vgm_models([_simple_vgm(), _simple_vgm(5.0)], [30, 30])
    assert set(result.keys()) == {"means", "covariances", "weights"}


def test_merge_vgm_models_empty_vgms():
    """Clients with empty VGM params (no valid components) should not crash."""
    empty = {"means": [], "covariances": [], "weights": []}
    result = merge_vgm_models([empty, empty], [10, 10])
    assert result == {"means": [], "covariances": [], "weights": []}


def test_merge_vgm_models_cap_scales_down():
    """max_total_samples cap should not crash and output is still a valid VGM."""
    vgms = [_simple_vgm(i) for i in range(5)]
    counts = [100_000] * 5  # raw total = 500_000 > default cap
    result = merge_vgm_models(vgms, counts, max_total_samples=1_000)
    assert len(result["means"]) == len(result["weights"])


# ---------------------------------------------------------------------------
# GlobalDataTransformer: round-trip
# ---------------------------------------------------------------------------


@pytest.fixture()
def fitted_transformer(sample_df) -> GlobalDataTransformer:
    """Build a GlobalDataTransformer fitted on sample_df's columns."""
    # Fit local VGMs for continuous columns
    cont_vgm_age = fit_local_continuous(sample_df["age"].to_numpy(), max_clusters=3)
    cont_vgm_inc = fit_local_continuous(sample_df["income"].to_numpy(), max_clusters=3)

    global_vgms = {"age": cont_vgm_age, "income": cont_vgm_inc}
    global_categories = {
        "label": ["0", "1"],
        "color": ["blue", "green", "red"],
    }

    transformer = GlobalDataTransformer()
    transformer.fit_global(
        column_order=["label", "color", "age", "income"],
        column_types={
            "label": "discrete",
            "color": "discrete",
            "age": "continuous",
            "income": "continuous",
        },
        global_vgms=global_vgms,
        global_categories=global_categories,
    )
    return transformer


def test_transformer_output_dimensions(fitted_transformer):
    assert fitted_transformer.output_dimensions > 0


def test_transformer_transform_shape(sample_df, fitted_transformer):
    encoded = fitted_transformer.transform(
        sample_df.assign(label=sample_df["label"].astype(str))
    )
    assert encoded.shape[0] == len(sample_df)
    assert encoded.shape[1] == fitted_transformer.output_dimensions


def test_transformer_inverse_transform_columns(sample_df, fitted_transformer):
    df = sample_df.assign(label=sample_df["label"].astype(str))
    encoded = fitted_transformer.transform(df)
    recovered = fitted_transformer.inverse_transform(encoded)
    assert set(["label", "color", "age", "income"]).issubset(set(recovered.columns))
    assert len(recovered) == len(sample_df)


# ---------------------------------------------------------------------------
# compute_client_weights
# ---------------------------------------------------------------------------


def test_compute_client_weights_single_client():
    weights = compute_client_weights(
        cat_freqs=[{"color": {"red": 5, "blue": 3}}],
        cont_vgms=[{"age": _simple_vgm()}],
        client_sample_counts=[8],
        cat_columns=["color"],
        cont_columns=["age"],
    )
    assert len(weights) == 1
    assert abs(weights[0] - 1.0) < 1e-9


def test_compute_client_weights_sum_to_one():
    weights = compute_client_weights(
        cat_freqs=[
            {"color": {"red": 5, "blue": 3}},
            {"color": {"red": 2, "blue": 8}},
        ],
        cont_vgms=[
            {"age": _simple_vgm(40.0)},
            {"age": _simple_vgm(50.0)},
        ],
        client_sample_counts=[8, 10],
        cat_columns=["color"],
        cont_columns=["age"],
    )
    assert len(weights) == 2
    assert abs(sum(weights) - 1.0) < 1e-6


def test_compute_client_weights_missing_column():
    """Client missing a continuous column must not crash (regression: zip bug)."""
    weights = compute_client_weights(
        cat_freqs=[
            {"color": {"red": 5}},
            {"color": {"blue": 5}},
        ],
        cont_vgms=[
            {"age": _simple_vgm()},  # client 0 has 'age'
            {},  # client 1 missing 'age'
        ],
        client_sample_counts=[10, 10],
        cat_columns=["color"],
        cont_columns=["age"],
    )
    assert len(weights) == 2
    assert all(w >= 0 for w in weights)


def test_compute_client_weights_no_columns():
    """No cat/cont columns → uniform weights, no crash."""
    weights = compute_client_weights(
        cat_freqs=[{}, {}],
        cont_vgms=[{}, {}],
        client_sample_counts=[5, 5],
        cat_columns=[],
        cont_columns=[],
    )
    assert weights == [0.5, 0.5]


# ---------------------------------------------------------------------------
# End-to-end smoke: configure_fed_init → fed_init → aggregate_fed_init
#                  → aggregate_train of one train round → sample
# ---------------------------------------------------------------------------


@pytest.fixture()
def e2e_state(sample_df, sample_schema, tiny_cfg):
    """Run a full fed-init handshake with two synthetic clients.

    Returns the coordinator global state ready for a train round.
    """
    client_ids = [0, 1]
    coordinator = FedTGANAltCoordinator(tiny_cfg)
    synthesizer = FedTGANAltSynthesizer(tiny_cfg)

    # Split sample_df into two halves to simulate two clients
    half = len(sample_df) // 2
    client_data = {0: sample_df.iloc[:half].reset_index(drop=True),
                   1: sample_df.iloc[half:].reset_index(drop=True)}

    # configure_fed_init → yields (client_id, config_update)
    init_requests = list(coordinator.configure_fed_init(
        seed=42,
        schema=sample_schema,
        client_ids=client_ids,
    ))
    assert len(init_requests) == 2

    # Each client runs fed_init
    init_replies = []
    for cid, request in init_requests:
        reply = synthesizer.fed_init(request, 42, sample_schema, client_data[cid])
        init_replies.append((cid, reply))

    # Coordinator aggregates
    coordinator.aggregate_fed_init(init_replies)

    return coordinator, synthesizer


def test_fed_init_reply_keys(sample_df, sample_schema, tiny_cfg):
    """fed_init must return 'cat-freqs', 'cont-vgms', and 'init-extras'."""
    coordinator = FedTGANAltCoordinator(tiny_cfg)
    synthesizer = FedTGANAltSynthesizer(tiny_cfg)

    # We only need the first request to test fed_init reply structure
    init_requests = list(coordinator.configure_fed_init(
        seed=0, schema=sample_schema, client_ids=[0]
    ))
    request = init_requests[0][1]
    reply = synthesizer.fed_init(request, 0, sample_schema, sample_df)

    assert "cat-freqs" in reply.objects
    assert "cont-vgms" in reply.objects
    assert "init-extras" in reply.extras
    assert "num-samples" in reply.extras["init-extras"]


def test_aggregate_fed_init_builds_transformer(e2e_state):
    """After aggregate_fed_init, coordinator must have a global transformer."""
    coordinator, _ = e2e_state
    assert coordinator._transformer is not None
    assert coordinator._transformer.output_dimensions > 0


def test_aggregate_fed_init_builds_global_state(e2e_state):
    """global_state must include generator/discriminator arrays and model-info."""
    coordinator, _ = e2e_state
    state = coordinator.global_state
    assert state is not None
    assert "generator" in state.arrays
    assert "discriminator" in state.arrays
    assert "model-info" in state.extras


def test_train_returns_g_and_d_arrays(e2e_state, sample_df, sample_schema):
    """synthesizer.train must return 'generator' and 'discriminator' in arrays."""
    coordinator, synthesizer = e2e_state
    global_state = coordinator.global_state

    reply = synthesizer.train(global_state, sample_df)

    assert "generator" in reply.arrays
    assert "discriminator" in reply.arrays


def test_sample_output_shape(e2e_state, sample_df, sample_schema, tiny_cfg):
    """synthesizer.sample must return a DataFrame with correct shape and columns."""
    coordinator, synthesizer = e2e_state

    # Run one train round to update G/D weights
    global_state = coordinator.global_state
    train_reply = synthesizer.train(global_state, sample_df)
    coordinator.aggregate_train([(0, train_reply)])

    # Now sample
    final_state = coordinator.global_state
    result = synthesizer.sample(final_state, num_rows=12, seed=1)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 12
    # All original columns should be present
    assert set(sample_df.columns).issubset(set(result.columns))


def test_sample_binary_column_values(e2e_state, sample_df):
    """Binary column 'label' must only contain values present in training data."""
    coordinator, synthesizer = e2e_state
    global_state = coordinator.global_state
    train_reply = synthesizer.train(global_state, sample_df)
    coordinator.aggregate_train([(0, train_reply)])

    result = synthesizer.sample(coordinator.global_state, num_rows=20, seed=2)

    # label is a binary column; values should be 0 or 1 (possibly as strings/ints)
    unique_values = set(str(v) for v in result["label"].unique())
    assert unique_values.issubset({"0", "1"})


# ---------------------------------------------------------------------------
# _split_cat_num
# ---------------------------------------------------------------------------


def test_split_cat_num_mixed_schema():
    schema = TableSchema(
        (
            ColumnSchema("age", "continuous"),
            ColumnSchema("score", "integer"),
            ColumnSchema("flag", "binary"),
            ColumnSchema("color", "categorical"),
        )
    )
    cat, num = _split_cat_num(schema)
    assert set(cat) == {"flag", "color"}
    assert set(num) == {"age", "score"}


def test_split_cat_num_all_continuous():
    schema = TableSchema(
        (ColumnSchema("a", "continuous"), ColumnSchema("b", "integer"))
    )
    cat, num = _split_cat_num(schema)
    assert cat == []
    assert set(num) == {"a", "b"}


def test_split_cat_num_all_categorical():
    schema = TableSchema(
        (ColumnSchema("x", "binary"), ColumnSchema("y", "categorical"))
    )
    cat, num = _split_cat_num(schema)
    assert set(cat) == {"x", "y"}
    assert num == []


# ---------------------------------------------------------------------------
# _gumbel_softmax
# ---------------------------------------------------------------------------


def test_gumbel_softmax_output_shape():
    logits = torch.zeros(8, 5)
    out = _gumbel_softmax(logits, tau=0.2)
    assert out.shape == (8, 5)


def test_gumbel_softmax_rows_sum_to_one():
    logits = torch.randn(16, 4)
    out = _gumbel_softmax(logits, tau=0.2)
    row_sums = out.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(16), atol=1e-5)


def test_gumbel_softmax_clamps_large_logits():
    """Values far outside [-20, 20] should not cause NaN."""
    logits = torch.full((4, 3), 1e6)
    out = _gumbel_softmax(logits)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# _apply_activate
# ---------------------------------------------------------------------------


def _make_output_info() -> list[list[SpanInfo]]:
    """Minimal output_info: one continuous col (1+2) and one discrete col (3)."""
    return [
        [SpanInfo(1, "tanh"), SpanInfo(2, "softmax")],
        [SpanInfo(3, "softmax")],
    ]


def test_apply_activate_output_shape():
    output_info = _make_output_info()
    total_dim = sum(s.dim for col in output_info for s in col)
    data = torch.randn(8, total_dim)
    out = _apply_activate(data, output_info)
    assert out.shape == (8, total_dim)


def test_apply_activate_tanh_range():
    output_info = [[SpanInfo(4, "tanh")]]
    data = torch.full((4, 4), 10.0)
    out = _apply_activate(data, output_info)
    assert (out <= 1.0).all() and (out >= -1.0).all()


# ---------------------------------------------------------------------------
# _cond_loss
# ---------------------------------------------------------------------------


def test_cond_loss_returns_scalar():
    output_info = [[SpanInfo(1, "tanh"), SpanInfo(2, "softmax")], [SpanInfo(3, "softmax")]]
    batch = 10
    fake = torch.randn(batch, 6)
    cond = torch.zeros(batch, 5)
    cond[:, 0] = 1.0  # pick first category of first discrete column
    mask = torch.zeros(batch, 2)
    mask[:, 0] = 1.0
    loss = _cond_loss(fake, cond, mask, output_info)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def test_cond_loss_pure_continuous_is_zero():
    """With no discrete columns the loss must be exactly 0."""
    output_info = [[SpanInfo(4, "tanh")]]
    batch = 8
    fake = torch.randn(batch, 4)
    cond = torch.zeros(batch, 0)
    mask = torch.zeros(batch, 0)
    loss = _cond_loss(fake, cond, mask, output_info)
    assert loss.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _weighted_average_state_dicts
# ---------------------------------------------------------------------------


def _make_state_dict(val: float) -> dict[str, torch.Tensor]:
    return {"w": torch.full((3, 3), val), "b": torch.full((3,), val)}


def test_weighted_average_state_dicts_uniform():
    sds = [_make_state_dict(1.0), _make_state_dict(3.0)]
    result = _weighted_average_state_dicts(sds, [0.5, 0.5])
    assert torch.allclose(result["w"], torch.full((3, 3), 2.0))


def test_weighted_average_state_dicts_tilted():
    sds = [_make_state_dict(0.0), _make_state_dict(4.0)]
    result = _weighted_average_state_dicts(sds, [0.25, 0.75])
    assert torch.allclose(result["b"], torch.full((3,), 3.0))


def test_weighted_average_state_dicts_same_keys():
    sds = [_make_state_dict(1.0), _make_state_dict(2.0)]
    result = _weighted_average_state_dicts(sds, [0.5, 0.5])
    assert set(result.keys()) == {"w", "b"}


# ---------------------------------------------------------------------------
# _sample_condvec_from_info
# ---------------------------------------------------------------------------


def _discrete_output_info(n_cats: int) -> list[list[SpanInfo]]:
    return [[SpanInfo(n_cats, "softmax")]]


def test_sample_condvec_from_info_shape():
    output_info = _discrete_output_info(4)
    out = _sample_condvec_from_info(output_info, batch_size=8)
    assert out.shape == (8, 4)


def test_sample_condvec_from_info_one_hot():
    output_info = _discrete_output_info(5)
    out = _sample_condvec_from_info(output_info, batch_size=32)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_sample_condvec_from_info_with_probs():
    output_info = _discrete_output_info(3)
    probs = np.array([0.7, 0.2, 0.1])
    out = _sample_condvec_from_info(output_info, batch_size=100, category_probs=probs)
    assert out.shape == (100, 3)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_sample_condvec_from_info_no_discrete():
    output_info = [[SpanInfo(4, "tanh")]]
    out = _sample_condvec_from_info(output_info, batch_size=8)
    assert out.shape == (8, 0)


# ---------------------------------------------------------------------------
# Generator / Residual
# ---------------------------------------------------------------------------


def test_residual_output_dim():
    block = Residual(10, 20)
    x = torch.randn(4, 10)
    out = block(x)
    assert out.shape == (4, 30)  # 20 (new) + 10 (skip)


def test_generator_output_shape():
    g = Generator(embedding_dim=16, generator_dim=[32, 32], data_dim=10)
    x = torch.randn(8, 16)
    out = g(x)
    assert out.shape == (8, 10)


def test_generator_no_nan_output():
    g = Generator(embedding_dim=8, generator_dim=[16], data_dim=6)
    x = torch.randn(4, 8)
    out = g(x)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------


def test_discriminator_output_shape():
    d = Discriminator(input_dim=8, discriminator_dim=[16], pac=2)
    x = torch.randn(8, 8)  # batch must be divisible by pac
    out = d(x)
    assert out.shape == (4, 1)  # batch/pac rows


def test_discriminator_gradient_penalty_scalar():
    d = Discriminator(input_dim=6, discriminator_dim=[16], pac=2)
    real = torch.randn(4, 6)
    fake = torch.randn(4, 6)
    pen = d.calc_gradient_penalty(real, fake, device="cpu")
    assert pen.shape == ()
    assert not torch.isnan(pen)


def test_discriminator_gradient_penalty_positive():
    d = Discriminator(input_dim=6, discriminator_dim=[16], pac=2)
    real = torch.randn(4, 6)
    fake = torch.randn(4, 6)
    pen = d.calc_gradient_penalty(real, fake, device="cpu")
    assert pen.item() >= 0.0


# ---------------------------------------------------------------------------
# DataSampler
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_encoded_data() -> tuple[np.ndarray, list[list[SpanInfo]]]:
    """4-col encoded matrix: 1 continuous (tanh+2-way softmax) + 1 discrete (3-way)."""
    rng = np.random.default_rng(7)
    n = 40
    # continuous: normalized value + mode one-hot (2 categories)
    norm = rng.uniform(-0.9, 0.9, (n, 1)).astype(np.float32)
    mode = np.zeros((n, 2), dtype=np.float32)
    mode[np.arange(n), rng.integers(0, 2, n)] = 1.0
    # discrete: 3 categories one-hot
    disc = np.zeros((n, 3), dtype=np.float32)
    disc[np.arange(n), rng.integers(0, 3, n)] = 1.0
    data = np.concatenate([norm, mode, disc], axis=1)
    output_info = [
        [SpanInfo(1, "tanh"), SpanInfo(2, "softmax")],
        [SpanInfo(3, "softmax")],
    ]
    return data, output_info


def test_data_sampler_dim_cond_vec(tiny_encoded_data):
    data, output_info = tiny_encoded_data
    sampler = DataSampler(data, output_info)
    assert sampler.dim_cond_vec() == 3  # only the 3-way discrete column


def test_data_sampler_sample_condvec_shape(tiny_encoded_data):
    data, output_info = tiny_encoded_data
    sampler = DataSampler(data, output_info)
    result = sampler.sample_condvec(16)
    assert result is not None
    cond, mask, col_ids, cat_ids = result
    assert cond.shape == (16, 3)
    assert mask.shape == (16, 1)
    assert col_ids.shape == (16,)
    assert cat_ids.shape == (16,)


def test_data_sampler_sample_condvec_one_hot(tiny_encoded_data):
    data, output_info = tiny_encoded_data
    sampler = DataSampler(data, output_info)
    cond, _, _, _ = sampler.sample_condvec(64)
    assert np.allclose(cond.sum(axis=1), 1.0)


def test_data_sampler_sample_condvec_returns_none_if_no_discrete():
    """DataSampler with only continuous columns must return None from sample_condvec."""
    rng = np.random.default_rng(0)
    data = rng.uniform(-1, 1, (20, 3)).astype(np.float32)
    output_info = [[SpanInfo(1, "tanh"), SpanInfo(2, "softmax")]]
    sampler = DataSampler(data, output_info)
    assert sampler.sample_condvec(8) is None


def test_data_sampler_sample_original_condvec(tiny_encoded_data):
    data, output_info = tiny_encoded_data
    sampler = DataSampler(data, output_info)
    out = sampler.sample_original_condvec(20)
    assert out is not None
    assert out.shape == (20, 3)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_data_sampler_sample_original_condvec_no_discrete():
    rng = np.random.default_rng(0)
    data = rng.uniform(-1, 1, (20, 1)).astype(np.float32)
    output_info = [[SpanInfo(1, "tanh")]]
    sampler = DataSampler(data, output_info)
    assert sampler.sample_original_condvec(8) is None


def test_data_sampler_sample_data_unconditional(tiny_encoded_data):
    data, output_info = tiny_encoded_data
    sampler = DataSampler(data, output_info)
    rows = sampler.sample_data(data, n=10, col=None, opt=None)
    assert rows.shape == (10, data.shape[1])


def test_data_sampler_sample_data_conditional(tiny_encoded_data):
    data, output_info = tiny_encoded_data
    sampler = DataSampler(data, output_info)
    cond, _, col_ids, cat_ids = sampler.sample_condvec(12)
    rows = sampler.sample_data(data, n=0, col=col_ids, opt=cat_ids)
    assert rows.shape == (12, data.shape[1])


# ---------------------------------------------------------------------------
# weighting: compute_jsd / compute_wd
# ---------------------------------------------------------------------------


def test_compute_jsd_identical_distributions():
    p = np.array([0.5, 0.5])
    assert compute_jsd(p, p) == pytest.approx(0.0, abs=1e-9)


def test_compute_jsd_maximum_divergence():
    """Disjoint distributions maximise JSD (= ln 2 ≈ 0.693 in nats)."""
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert compute_jsd(p, q) == pytest.approx(np.log(2), abs=1e-6)


def test_compute_jsd_in_unit_interval():
    rng = np.random.default_rng(5)
    p = rng.dirichlet([1, 2, 3])
    q = rng.dirichlet([3, 2, 1])
    assert 0.0 <= compute_jsd(p, q) <= 1.0


def test_compute_wd_same_distribution():
    a = np.linspace(0, 1, 50)
    assert compute_wd(a, a) == pytest.approx(0.0, abs=1e-9)


def test_compute_wd_shifted_distributions():
    a = np.zeros(50)
    b = np.ones(50)
    assert compute_wd(a, b) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# GlobalDataTransformer: edge-case variants
# ---------------------------------------------------------------------------


@pytest.fixture()
def continuous_only_transformer() -> GlobalDataTransformer:
    vgm = fit_local_continuous(np.random.default_rng(1).normal(0, 1, 80))
    t = GlobalDataTransformer()
    t.fit_global(
        column_order=["x", "y"],
        column_types={"x": "continuous", "y": "continuous"},
        global_vgms={"x": vgm, "y": vgm},
        global_categories={},
    )
    return t


@pytest.fixture()
def discrete_only_transformer() -> GlobalDataTransformer:
    t = GlobalDataTransformer()
    t.fit_global(
        column_order=["c"],
        column_types={"c": "discrete"},
        global_vgms={},
        global_categories={"c": ["a", "b", "c"]},
    )
    return t


def test_transformer_continuous_only_round_trip(continuous_only_transformer):
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.normal(0, 1, 20), "y": rng.normal(5, 2, 20)})
    encoded = continuous_only_transformer.transform(df)
    assert encoded.shape[1] == continuous_only_transformer.output_dimensions
    recovered = continuous_only_transformer.inverse_transform(encoded)
    assert list(recovered.columns) == ["x", "y"]
    assert len(recovered) == 20


def test_transformer_discrete_only_round_trip(discrete_only_transformer):
    df = pd.DataFrame({"c": ["a", "b", "c", "a", "b"]})
    encoded = discrete_only_transformer.transform(df)
    assert encoded.shape == (5, 3)
    recovered = discrete_only_transformer.inverse_transform(encoded)
    assert list(recovered.columns) == ["c"]
    assert set(recovered["c"].unique()).issubset({"a", "b", "c"})


def test_transformer_unknown_category_handled(discrete_only_transformer):
    """Unknown categories (handle_unknown='ignore') must not raise."""
    df = pd.DataFrame({"c": ["a", "z", "b"]})  # 'z' is unknown
    encoded = discrete_only_transformer.transform(df)
    assert encoded.shape == (3, 3)


# ---------------------------------------------------------------------------
# GlobalDataTransformer.convert_column_name_value_to_id
# ---------------------------------------------------------------------------


@pytest.fixture()
def id_transformer() -> GlobalDataTransformer:
    t = GlobalDataTransformer()
    t.fit_global(
        column_order=["color", "size"],
        column_types={"color": "discrete", "size": "discrete"},
        global_vgms={},
        global_categories={"color": ["blue", "green", "red"], "size": ["L", "M", "S"]},
    )
    return t


def test_convert_column_name_value_to_id_happy_path(id_transformer):
    result = id_transformer.convert_column_name_value_to_id("color", "green")
    assert "discrete_column_id" in result
    assert "column_id" in result
    assert "value_id" in result
    assert isinstance(result["value_id"], int)


def test_convert_column_name_value_to_id_correct_value(id_transformer):
    result = id_transformer.convert_column_name_value_to_id("color", "red")
    cats = ["blue", "green", "red"]
    assert result["value_id"] == cats.index("red")


def test_convert_column_name_value_to_id_unknown_column(id_transformer):
    with pytest.raises(ValueError, match="not found"):
        id_transformer.convert_column_name_value_to_id("weight", "heavy")


def test_convert_column_name_value_to_id_unknown_value(id_transformer):
    with pytest.raises(ValueError, match="not found"):
        id_transformer.convert_column_name_value_to_id("color", "purple")


# ---------------------------------------------------------------------------
# merge_vgm_models: single-client branch
# ---------------------------------------------------------------------------


def test_merge_vgm_models_single_client():
    vgm = {"means": [0.0, 1.0], "covariances": [1.0, 1.0], "weights": [0.6, 0.4]}
    result = merge_vgm_models([vgm], [50])
    assert set(result.keys()) == {"means", "covariances", "weights"}
    assert len(result["means"]) > 0


# ---------------------------------------------------------------------------
# Coordinator / Synthesizer properties
# ---------------------------------------------------------------------------


def test_coordinator_arrays_to_ml_framework_map(tiny_cfg):
    coord = FedTGANAltCoordinator(tiny_cfg)
    mapping = coord.arrays_to_ml_framework_map
    assert mapping is not None
    assert mapping["generator"] == "torch"
    assert mapping["discriminator"] == "torch"


def test_synthesizer_arrays_to_ml_framework_map(tiny_cfg):
    synth = FedTGANAltSynthesizer(tiny_cfg)
    mapping = synth.arrays_to_ml_framework_map
    assert mapping is not None
    assert "generator" in mapping


def test_aggregate_train_weight_mismatch_fallback(e2e_state, sample_df):
    """When weight count mismatches client count, aggregate_train falls back to uniform."""
    coordinator, synthesizer = e2e_state
    global_state = coordinator.global_state
    train_reply = synthesizer.train(global_state, sample_df)

    # Deliberately corrupt weights to trigger the fallback branch
    coordinator._client_weights = [0.5, 0.5, 0.5]  # 3 weights for 1 client
    coordinator.aggregate_train([(0, train_reply)])

    # The aggregation should succeed (uniform fallback used)
    new_state = coordinator.global_state
    assert "generator" in new_state.arrays


def test_aggregate_train_empty_raises(e2e_state):
    """aggregate_train must raise ValueError when given no replies."""
    coordinator, _ = e2e_state
    with pytest.raises(ValueError, match="No replies"):
        coordinator.aggregate_train(iter([]))


# ---------------------------------------------------------------------------
# train reply metrics
# ---------------------------------------------------------------------------


def test_train_reply_contains_metrics(e2e_state, sample_df):
    """train reply must contain loss-g, loss-d, num-samples in metrics."""
    coordinator, synthesizer = e2e_state
    reply = synthesizer.train(coordinator.global_state, sample_df)
    metrics = reply.metrics["metrics"]
    assert "loss-g" in metrics
    assert "loss-d" in metrics
    assert "num-samples" in metrics
    assert metrics["num-samples"] == len(sample_df)


# ---------------------------------------------------------------------------
# sample: integer column rounding and schema column order
# ---------------------------------------------------------------------------


def test_sample_integer_column_is_int(tiny_cfg):
    """Integer-kind columns in the schema must be returned as dtype int."""
    rng = np.random.default_rng(99)
    n = 60
    df = pd.DataFrame({
        "count": rng.integers(1, 20, n),
        "label": rng.choice(["a", "b"], n),
    })
    schema = infer_schema(df)
    coordinator = FedTGANAltCoordinator(tiny_cfg)
    synthesizer = FedTGANAltSynthesizer(tiny_cfg)

    init_requests = list(coordinator.configure_fed_init(42, schema, [0]))
    reply = synthesizer.fed_init(init_requests[0][1], 42, schema, df)
    coordinator.aggregate_fed_init([(0, reply)])

    train_reply = synthesizer.train(coordinator.global_state, df)
    coordinator.aggregate_train([(0, train_reply)])

    result = synthesizer.sample(coordinator.global_state, num_rows=10, seed=3)
    assert result["count"].dtype in (int, np.int64, np.int32)


def test_sample_restores_schema_column_order(e2e_state, sample_df, sample_schema):
    """Sampled DataFrame columns must follow the original schema order."""
    coordinator, synthesizer = e2e_state
    train_reply = synthesizer.train(coordinator.global_state, sample_df)
    coordinator.aggregate_train([(0, train_reply)])

    result = synthesizer.sample(coordinator.global_state, num_rows=10, seed=4)
    schema_cols = [c.name for c in sample_schema.columns]
    result_cols = list(result.columns)
    assert result_cols == [c for c in schema_cols if c in result_cols]


# ---------------------------------------------------------------------------
# End-to-end: continuous-only dataset (no condvec path)
# ---------------------------------------------------------------------------


def test_e2e_continuous_only_dataset(tiny_cfg):
    """Algorithm must work on a dataset with no categorical / binary columns."""
    rng = np.random.default_rng(11)
    n = 60
    df = pd.DataFrame({
        "x": rng.normal(0, 1, n).astype(float),
        "y": rng.normal(5, 2, n).astype(float),
    })
    schema = infer_schema(df)
    coordinator = FedTGANAltCoordinator(tiny_cfg)
    synthesizer = FedTGANAltSynthesizer(tiny_cfg)

    init_requests = list(coordinator.configure_fed_init(0, schema, [0]))
    init_reply = synthesizer.fed_init(init_requests[0][1], 0, schema, df)
    coordinator.aggregate_fed_init([(0, init_reply)])

    train_reply = synthesizer.train(coordinator.global_state, df)
    coordinator.aggregate_train([(0, train_reply)])

    result = synthesizer.sample(coordinator.global_state, num_rows=15, seed=5)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 15
    assert set(df.columns).issubset(set(result.columns))


# ---------------------------------------------------------------------------
# End-to-end: multi-round training converges without errors
# ---------------------------------------------------------------------------


def test_e2e_multi_round_training(sample_df, sample_schema, tiny_cfg):
    """Three federated rounds of training must complete without error."""
    client_ids = [0, 1]
    coordinator = FedTGANAltCoordinator(tiny_cfg)
    synthesizer = FedTGANAltSynthesizer(tiny_cfg)

    half = len(sample_df) // 2
    client_data = {
        0: sample_df.iloc[:half].reset_index(drop=True),
        1: sample_df.iloc[half:].reset_index(drop=True),
    }

    init_requests = list(coordinator.configure_fed_init(0, sample_schema, client_ids))
    init_replies = [
        (cid, synthesizer.fed_init(req, 0, sample_schema, client_data[cid]))
        for cid, req in init_requests
    ]
    coordinator.aggregate_fed_init(init_replies)

    for _ in range(3):
        train_replies = [
            (cid, synthesizer.train(coordinator.global_state, client_data[cid]))
            for cid in client_ids
        ]
        coordinator.aggregate_train(train_replies)

    result = synthesizer.sample(coordinator.global_state, num_rows=10, seed=9)
    assert len(result) == 10
