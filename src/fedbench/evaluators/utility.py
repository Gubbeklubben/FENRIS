import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline

from fedbench.core.eval import Evaluator, EvalContext
from fedbench.util.metrics import make_tabular_preprocessor


class TSTREvaluator(Evaluator):
    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        if ctx.target_column is None or ctx.test_df is None:
            return {}

        D_syn = ctx.synthetic_df
        D_test = ctx.test_df
        y_col = ctx.target_column

        X_syn = D_syn.drop(columns=[y_col])
        y_syn = D_syn[y_col]
        X_test = D_test.drop(columns=[y_col])
        y_test = D_test[y_col]

        preprocessor = make_tabular_preprocessor(X_syn)

        match ctx.schema.kind_of(y_col):

            # Binary classification
            case "binary":
                model = LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=ctx.seed
                )
                pipe = Pipeline([
                    ("pre", preprocessor),
                    ("model", model)
                ])

                pipe.fit(X_syn, y_syn)
                proba = pipe.predict_proba(X_test)[:, 1]

                return {
                    "tstr_auc": roc_auc_score(y_test, proba)
                }

            # Multiclass classification
            case "categorical":
                model = LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=ctx.seed
                )
                pipe = Pipeline([
                    ("pre", preprocessor),
                    ("model", model)
                ])

                pipe.fit(X_syn, y_syn)
                pred = pipe.predict(X_test)

                return {
                    "tstr_accuracy": accuracy_score(y_test, pred)
                }

            # Regression
            case _:
                model = Ridge(random_state=ctx.seed)
                pipe = Pipeline([
                    ("pre", preprocessor),
                    ("model", model)
                ])

                pipe.fit(X_syn, y_syn)
                pred = pipe.predict(X_test)

                return {
                    "tstr_rmse": np.sqrt(mean_squared_error(y_test, pred))
                }