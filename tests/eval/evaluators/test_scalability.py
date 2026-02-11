from fedbench.eval.evaluators.scalability import ScalabilityEvaluator

def test_scalability_empty(eval_ctx):
    ev = ScalabilityEvaluator()
    out = ev.evaluate(eval_ctx)

    assert out == {}
