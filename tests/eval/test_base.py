from fedbench.eval.evaluators.base import Evaluator

class DummyEvaluator(Evaluator):
    name = "dummy"

    def _evaluate(self, ctx):
        return {"a": 1, "b": None}


def test_evaluator_prefixing(eval_ctx):
    ev = DummyEvaluator()
    out = ev.evaluate(eval_ctx)

    assert out == {
        "dummy.a": 1.0,
        "dummy.b": None,
    }
