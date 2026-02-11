import numpy as np
from typing import Dict

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

from .base import Evaluator
from ..context import EvalContext


class TSTREvaluator(Evaluator):
    name = "utility"

    def _evaluate(self, ctx: EvalContext) -> Dict[str, float]:
        if ctx.target_column is None or ctx.test_df is None:
            return {}

        syn = ctx.synthetic_df
        test = ctx.test_df
        y = ctx.target_column

        Xs, ys = syn.drop(columns=[y]), syn[y]
        Xt, yt = test.drop(columns=[y]), test[y]

        num_cols = Xs.select_dtypes(include="number").columns
        cat_cols = Xs.select_dtypes(exclude="number").columns

        pre = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ])

        # classification vs regression
        if ys.nunique() <= 2:
            clf = Pipeline([("pre", pre),
                            ("model", LogisticRegression(max_iter=1000))])
            clf.fit(Xs, ys)
            if hasattr(clf["model"], "predict_proba"):
                proba = clf.predict_proba(Xt)[:, 1]
                return {"tstr_auc": roc_auc_score(yt, proba)}
            else:
                pred = clf.predict(Xt)
                return {"tstr_accuracy": accuracy_score(yt, pred)}
        else:
            reg = Pipeline([("pre", pre),
                            ("model", Ridge())])
            reg.fit(Xs, ys)
            pred = reg.predict(Xt)
            return {"tstr_rmse": np.sqrt(mean_squared_error(yt, pred))}
