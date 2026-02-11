from fedbench.eval.context import EvalContext
from fedbench.eval.evaluators.utility import TSTREvaluator

def test_tstr_binary_classification(eval_ctx):
    ev = TSTREvaluator()
    out = ev.evaluate(eval_ctx)

    # binary label → AUC
    assert "utility.tstr_auc" in out
    assert 0.0 <= out["utility.tstr_auc"] <= 1.0


def test_tstr_no_target(schema, split_data, synthetic_data):
    train, test = split_data

    ctx = EvalContext(
        schema=schema,
        train_df=train,
        test_df=test,
        synthetic_df=synthetic_data,
        target_column=None,
    )

    ev = TSTREvaluator()
    out = ev.evaluate(ctx)

    assert out == {}
