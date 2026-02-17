from fedbench.eval.evaluators.base import Evaluator

class DummyEvaluator(Evaluator):
    @property
    def category(self):
        return "dummy"
    def evaluate(self, ctx):
        return 1.5


def test_evaluator_prefixing(eval_ctx):
    ev = DummyEvaluator()
    out = ev.evaluate(eval_ctx)

    assert out == 1.5
