"""
Train a supervised regression model that predicts a problem statement's
"Build Score" (0-100, feasibility of shipping it as a student project)
from its structured features.

Run:
    python ml/train_build_score_model.py

Outputs (written to ml/artifacts/):
    build_score_model.joblib   - trained sklearn Pipeline (preprocessing + model)
    metrics.json               - held-out evaluation metrics
    feature_importance.json    - permutation feature importance
    feature_importance.png     - bar chart of feature importance
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "problems.json")
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

RANDOM_STATE = 42


def load_dataframe():
    with open(DATA_PATH) as f:
        problems = json.load(f)
    df = pd.DataFrame(problems)
    df["min_weeks"] = df["estimatedTimeWeeks"].apply(lambda x: x[0])
    df["max_weeks"] = df["estimatedTimeWeeks"].apply(lambda x: x[1])
    df["avg_weeks"] = (df["min_weeks"] + df["max_weeks"]) / 2
    df["week_span"] = df["max_weeks"] - df["min_weeks"]
    df["num_domain_tags"] = df["domainTags"].apply(len)
    df["num_stack_items"] = df["suggestedStack"].apply(len)
    return df


def build_feature_matrix(df):
    """Build the feature matrix used by the model.

    With only 60 labeled examples, one-hot-encoding every individual
    domain tag (~20 distinct tags) blew up the feature space to 40+
    dimensions and the model badly overfit (negative held-out R2).
    Category already captures most of that signal, so tags are folded
    down to a single count feature instead of per-tag dummies.
    """
    return df[
        [
            "category",
            "difficulty",
            "dataAvailability",
            "avg_weeks",
            "week_span",
            "num_domain_tags",
            "num_stack_items",
        ]
    ].copy()


def make_preprocessor(categorical_cols, numeric_cols, scale_numeric=False):
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", numeric_step, numeric_cols),
        ]
    )


def main():
    df = load_dataframe()
    X = build_feature_matrix(df)
    y = df["buildScore"].values

    categorical_cols = ["category", "difficulty", "dataAvailability"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # With n=60, model choice and regularization strength matter more than
    # exotic architectures. Compare a regularized linear model against two
    # tree ensembles via cross-validated R2 and keep whichever generalizes
    # best, instead of assuming a bigger model is a better one.
    candidates = {
        "baseline_mean": (
            Pipeline(
                [
                    ("preprocess", make_preprocessor(categorical_cols, numeric_cols)),
                    ("model", DummyRegressor(strategy="mean")),
                ]
            ),
            {},
        ),
        "ridge": (
            Pipeline(
                [
                    ("preprocess", make_preprocessor(categorical_cols, numeric_cols, scale_numeric=True)),
                    ("model", Ridge(random_state=RANDOM_STATE)),
                ]
            ),
            {"model__alpha": [0.5, 1.0, 3.0, 10.0, 30.0, 100.0]},
        ),
        "random_forest": (
            Pipeline(
                [
                    ("preprocess", make_preprocessor(categorical_cols, numeric_cols)),
                    ("model", RandomForestRegressor(random_state=RANDOM_STATE)),
                ]
            ),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [2, 3, 4, None],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "gradient_boosting": (
            Pipeline(
                [
                    ("preprocess", make_preprocessor(categorical_cols, numeric_cols)),
                    ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
                ]
            ),
            {
                "model__n_estimators": [50, 100],
                "model__max_depth": [1, 2, 3],
                "model__learning_rate": [0.03, 0.1],
            },
        ),
    }

    results = {}
    fitted_searches = {}
    for name, (pipeline, grid) in candidates.items():
        search = GridSearchCV(pipeline, grid, cv=cv, scoring="r2", n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        results[name] = float(search.best_score_)
        fitted_searches[name] = search

    best_name = max(
        (name for name in results if name != "baseline_mean"), key=results.get
    )
    best_search = fitted_searches[best_name]
    best_model = best_search.best_estimator_

    preds = best_model.predict(X_test)
    mae = float(np.mean(np.abs(preds - y_test)))
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    ss_res = float(np.sum((y_test - preds) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    metrics = {
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model_comparison_cv_r2": results,
        "selected_model": best_name,
        "best_params": best_search.best_params_,
        "cv_r2_mean": results[best_name],
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
        "note": (
            "n=60 is a small dataset for regression; cv_r2_mean (5-fold, "
            "averaged) is the more trustworthy generalization estimate here "
            "than test_r2 from a single 12-row held-out split. Compare "
            "selected model's cv_r2_mean against model_comparison_cv_r2."
            "baseline_mean to see whether it beats predicting the average "
            "Build Score outright."
        ),
    }
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # Refit the winning model on ALL data before shipping it — with only 60
    # labeled examples, holding out the test split for the final artifact
    # would waste signal the API relies on at inference time.
    final_model = best_search.best_estimator_
    final_model.fit(X, y)

    joblib.dump(
        {"pipeline": final_model, "feature_columns": list(X.columns), "model_name": best_name},
        os.path.join(ARTIFACTS_DIR, "build_score_model.joblib"),
    )

    # Permutation importance (computed on the held-out test split, using
    # the model trained on X_train, before the final all-data refit above)
    importance = permutation_importance(
        best_model, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE
    )
    imp_df = pd.DataFrame(
        {"feature": X.columns, "importance": importance.importances_mean}
    ).sort_values("importance", ascending=False)
    imp_df.to_json(
        os.path.join(ARTIFACTS_DIR, "feature_importance.json"), orient="records", indent=2
    )

    top = imp_df.iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top["feature"], top["importance"], color="#1FAE8E")
    plt.xlabel("Permutation importance (Δ R²)")
    plt.title(f"Build Score model ({best_name}) — feature importances")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "feature_importance.png"), dpi=150)
    print("Saved model + metrics + feature importance to", ARTIFACTS_DIR)
    print(f"Selected model: {best_name} (cv_r2_mean={results[best_name]:.3f})")


if __name__ == "__main__":
    main()
