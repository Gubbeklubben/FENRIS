import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fedbench.core.eval import Evaluator, EvalContext


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

        # column types
        num_cols = X_syn.select_dtypes(include="number").columns
        cat_cols = X_syn.select_dtypes(exclude="number").columns

        # preprocessing
        pre = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), num_cols),
                ("cat", Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))
                ]), cat_cols),
            ],
            remainder="drop"
        )

        # Detect task type
        n_unique = y_syn.nunique()
        if n_unique < 2:
            return {} # Degenerate case, can't evaluate

        is_classification = (
            y_syn.dtype == "object"
            or y_syn.dtype.name == "category"
            or y_syn.dtype.kind in "biu"
        )

        # Binary classification
        if is_classification and n_unique == 2:
            model = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=0
            )
            pipe = Pipeline([
                ("pre", pre),
                ("model", model)
            ])

            pipe.fit(X_syn, y_syn)
            proba = pipe.predict_proba(X_test)[:, 1]

            return {
                "tstr_auc": roc_auc_score(y_test, proba)
            }

        # Multiclass classification
        elif is_classification and n_unique > 2:
            model = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=0
            )
            pipe = Pipeline([
                ("pre", pre),
                ("model", model)
            ])

            pipe.fit(X_syn, y_syn)
            pred = pipe.predict(X_test)

            return {
                "tstr_accuracy": accuracy_score(y_test, pred)
            }

        # Regression
        else:
            model = Ridge(random_state=0)
            pipe = Pipeline([
                ("pre", pre),
                ("model", model)
            ])

            pipe.fit(X_syn, y_syn)
            pred = pipe.predict(X_test)

            return {
                "tstr_rmse": np.sqrt(mean_squared_error(y_test, pred))
            }