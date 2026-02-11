import math

from fedbench.eval.evaluators.fidelity import BasicFidelityEvaluator


def test_fidelity_outputs_present(eval_ctx):
    ev = BasicFidelityEvaluator()
    out = ev.evaluate(eval_ctx)

    assert "fidelity.mean_abs_diff" in out
    assert "fidelity.std_abs_diff" in out
    assert "fidelity.categorical_tv_mean" in out
    assert "fidelity.corr_fro_diff" in out


def test_fidelity_numeric_values(eval_ctx):
    ev = BasicFidelityEvaluator()
    out = ev.evaluate(eval_ctx)

    assert out["fidelity.mean_abs_diff"] >= 0
    assert out["fidelity.std_abs_diff"] >= 0
    assert out["fidelity.categorical_tv_mean"] >= 0
    assert not math.isnan(out["fidelity.corr_fro_diff"])
