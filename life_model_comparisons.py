import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from summaries import (
    comparison_metrics,
    create_comparison_summary_table,
    create_kfold_summary_table,
    group_kfold_comparison_metrics,
)

TARGET_COL = "Life expectancy "
GROUP_COL = "Country"
EXCLUDED_COLS = [TARGET_COL, GROUP_COL, "Year", "Status"]


def analyze_no_pca(X, y, groups):
    model = LinearRegression()
    model.fit(X, y)

    comparison = comparison_metrics(model, X, y)
    comparison["Model"] = "Linear Regression"

    kfold = group_kfold_comparison_metrics(model, X, y, groups)
    kfold["Model"] = "Linear Regression"

    return comparison, kfold


def analyze_with_pca(X, y, groups):
    model_pca = make_pipeline(StandardScaler(), PCA(n_components=0.8), LinearRegression())
    model_pca.fit(X, y)

    comparison = comparison_metrics(model_pca, X, y)
    comparison["Model"] = "PCA + Linear Regression"

    kfold = group_kfold_comparison_metrics(model_pca, X, y, groups)
    kfold["Model"] = "PCA + Linear Regression"

    return comparison, kfold


def analyze_lgbm(X, y, groups):
    X_values = X.to_numpy()
    model_lgbm = LGBMRegressor(n_estimators=100, random_state=14)
    model_lgbm.fit(X_values, y)

    comparison = comparison_metrics(model_lgbm, X_values, y)
    comparison["Model"] = "LightGBM"

    kfold = group_kfold_comparison_metrics(model_lgbm, X_values, y, groups)
    kfold["Model"] = "LightGBM"

    return comparison, kfold


def analyze_mlp(X, y, groups):
    model_mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(32, 16), solver="lbfgs", max_iter=5000, random_state=14),
    )
    model_mlp.fit(X, y)

    comparison = comparison_metrics(model_mlp, X, y)
    comparison["Model"] = "MLP Regressor"

    kfold = group_kfold_comparison_metrics(model_mlp, X, y, groups)
    kfold["Model"] = "MLP Regressor"

    return comparison, kfold


def main():
    data_df = pd.read_csv("life.csv").dropna() # SOURCE: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who

    X = data_df[[col for col in data_df.columns if col not in EXCLUDED_COLS]]
    y = data_df[TARGET_COL]
    groups = data_df[GROUP_COL]

    metrics_no_pca, kfold_no_pca = analyze_no_pca(X, y, groups)
    metrics_with_pca, kfold_with_pca = analyze_with_pca(X, y, groups)
    metrics_lgbm, kfold_lgbm = analyze_lgbm(X, y, groups)
    metrics_mlp, kfold_mlp = analyze_mlp(X, y, groups)

    create_comparison_summary_table(metrics_no_pca, metrics_with_pca, metrics_lgbm, metrics_mlp)
    create_kfold_summary_table(kfold_no_pca, kfold_with_pca, kfold_lgbm, kfold_mlp)


if __name__ == "__main__":
    main()
