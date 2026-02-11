from fedbench.eval.evaluators.fidelity_extended import ExtendedFidelityEvaluator

def test_extended_fidelity_metrics(eval_ctx):
    ev = ExtendedFidelityEvaluator()
    out = ev.evaluate(eval_ctx)

    assert "fidelity_ext.ks_mean" in out
    assert "fidelity_ext.wasserstein_mean" in out
    assert "fidelity_ext.t_stat_mean_abs" in out

    assert out["fidelity_ext.ks_mean"] >= 0
    assert out["fidelity_ext.wasserstein_mean"] >= 0
