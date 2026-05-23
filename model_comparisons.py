import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from summaries import comparison_metrics, create_comparison_summary_table
# from config import TARGET_COLS, ID_COLS

TARGET_COLS = ["Life expectancy "]
ID_COLS = ["Country", "Year", "Status"]

def analyze_no_pca(data_df, X, y):
    model = LinearRegression()
    model.fit(X, y)

    return comparison_metrics(model, X, y)


def analyze_with_pca(X, y):
    model_pca = make_pipeline(StandardScaler(), PCA(n_components=0.8), LinearRegression())
    model_pca.fit(X, y)

    return comparison_metrics(model_pca, X, y)


def main():
    # data_df = pd.read_csv("data_ideal_188834.csv")
    data_df = pd.read_csv("life.csv") # SOURCE: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
    data_df = data_df.dropna()

    X = data_df[[col for col in data_df.columns if col not in TARGET_COLS + ID_COLS]]
    y = data_df[TARGET_COLS]

    metrics_no_pca = analyze_no_pca(data_df, X, y)
    metrics_with_pca = analyze_with_pca(X, y)

    create_comparison_summary_table(metrics_no_pca, metrics_with_pca)


if __name__ == "__main__":
    main()
