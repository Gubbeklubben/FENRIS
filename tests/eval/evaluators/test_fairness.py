from fedbench.eval.evaluators.fairness import FairnessEvaluator


def test_fairness_placeholder(eval_ctx):
    ev = FairnessEvaluator()
    out = ev.evaluate(eval_ctx)

    assert out == {"fairness.enabled": 0.0}
