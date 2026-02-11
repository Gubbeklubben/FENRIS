def test_eval_context_fields(eval_ctx):
    assert eval_ctx.train_df is not None
    assert eval_ctx.synthetic_df is not None
    assert eval_ctx.target_column == "label"
    assert "cat" in eval_ctx.sensitive_columns
