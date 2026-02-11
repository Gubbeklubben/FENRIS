from fedbench.eval.evaluators.base import Evaluator
from fedbench.eval.suite import EvaluationSuite


class Ev1(Evaluator):
    name = "e1"
    def _evaluate(self, ctx):
        return {"x": 1}


class Ev2(Evaluator):
    name = "e2"
    def _evaluate(self, ctx):
        return {"y": 2}


def test_suite_composition(eval_ctx):
    suite = EvaluationSuite([Ev1(), Ev2()])
    out = suite.evaluate(eval_ctx)

    assert out == {
        "e1.x": 1.0,
        "e2.y": 2.0,
    }
