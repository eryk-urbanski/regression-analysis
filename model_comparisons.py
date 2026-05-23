import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMRegressor

from summaries import (
    comparison_metrics,
    create_comparison_summary_table,
    create_kfold_summary_table,
    kfold_comparison_metrics,
)
from config import TARGET_COLS, ID_COLS

# TARGET_COLS = ["Life expectancy "]
# ID_COLS = ["Country", "Year", "Status"]
# TARGET_COLS = ["percentProtein"]
# ID_COLS = [""]

def analyze_no_pca(X, y):
    model = LinearRegression()
    model.fit(X, y)

    comparison = comparison_metrics(model, X, y)
    comparison["Model"] = "Linear Regression"

    kfold = kfold_comparison_metrics(model, X, y)
    kfold["Model"] = "Linear Regression"

    return comparison, kfold


def analyze_with_pca(X, y):
    model_pca = make_pipeline(StandardScaler(), PCA(n_components=0.8), LinearRegression())
    model_pca.fit(X, y)

    comparison = comparison_metrics(model_pca, X, y)
    comparison["Model"] = "PCA + Linear Regression"

    kfold = kfold_comparison_metrics(model_pca, X, y)
    kfold["Model"] = "PCA + Linear Regression"

    return comparison, kfold


def analyze_lgbm(X, y):
    X_values = X.to_numpy()
    model_lgbm = LGBMRegressor(n_estimators=100, random_state=14)
    model_lgbm.fit(X_values, y)

    comparison = comparison_metrics(model_lgbm, X_values, y)
    comparison["Model"] = "LightGBM"

    kfold = kfold_comparison_metrics(model_lgbm, X_values, y)
    kfold["Model"] = "LightGBM"

    return comparison, kfold


def main():
    data_df = pd.read_csv("data_ideal_188834.csv")
    # data_df = pd.read_csv("life.csv") # SOURCE: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
    # data_df = pd.read_csv("absopr.csv") 
    data_df = data_df.dropna()

    X = data_df[[col for col in data_df.columns if col not in TARGET_COLS + ID_COLS]]
    y = data_df[TARGET_COLS[0]]

    metrics_no_pca, kfold_no_pca = analyze_no_pca(X, y)
    metrics_with_pca, kfold_with_pca = analyze_with_pca(X, y)
    metrics_lgbm, kfold_lgbm = analyze_lgbm(X, y)

    create_comparison_summary_table(metrics_no_pca, metrics_with_pca, metrics_lgbm)
    create_kfold_summary_table(kfold_no_pca, kfold_with_pca, kfold_lgbm)


if __name__ == "__main__":
    main()
