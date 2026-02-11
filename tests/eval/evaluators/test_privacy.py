from fedbench.eval.evaluators.privacy import PrivacyEvaluator

def test_privacy_metrics_present(eval_ctx):
    ev = PrivacyEvaluator()
    out = ev.evaluate(eval_ctx)

    expected = {
        "privacy.exact_row_match_rate_train",
        "privacy.exact_row_match_any",
        "privacy.partial_match_rate_top1",
        "privacy.partial_match_rate_top2",
        "privacy.partial_match_rate_top3",
        "privacy.partial_match_any",
    }

    assert expected.issubset(out.keys())


def test_privacy_exact_match_zero(eval_ctx):
    ev = PrivacyEvaluator()
    out = ev.evaluate(eval_ctx)

    assert out["privacy.exact_row_match_rate_train"] == 0.0
