from fedbench.eval.suite import EvaluationSuite

def test_metrics_json_shape(eval_ctx):
    suite = EvaluationSuite.default(
        ["fidelity", "utility", "privacy"]
    )
    out = suite.evaluate(eval_ctx)

    for k, v in out.items():
        assert isinstance(k, str)
        assert v is None or isinstance(v, float)
