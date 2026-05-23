import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from config import TARGET_COLS, ID_COLS, PREDICTOR_COLS
from stat_tests import shapiro_wilk_test, levene_test
from summaries import create_regression_summary_table, robust_model_summary, summary_OLS
from utils import calc_residuals
from plotting import (
    scatter_with_regression,
    scatter_with_model,
    diagnostic_plots
)
from regression_models import bootstrap


def main():
    data_df = pd.read_csv("data_ideal_188834.csv")
    # data_df = pd.read_csv("data_outliers_188834.csv")
    
    X = data_df[PREDICTOR_COLS]
    y = data_df[TARGET_COLS]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    ols_results = summary_OLS(model, X, y)
    diagnostic_plots(y, y_pred)

    residuals = calc_residuals(y, y_pred)
    shapiro_wilk_test(residuals)
    levene_test(y_pred, residuals)

    scatter_with_regression(
        data=data_df,
        x_col="hours_studied",
        y_col="exam_score",
        title="Exam Score vs Hours Studied with Linear Regression",
        x_label="Hours Studied",
        y_label="Exam Score (0--100)"
    )

    bootstrap_results = bootstrap(
        data=data_df,
        x_col="hours_studied",
        y_col="exam_score",
        n_boot=2000,
        seed=14
    )

    m_estimators_results = robust_model_summary(
        data=data_df,
        x_col="hours_studied",
        y_col="exam_score"
    )

    create_regression_summary_table(
        ols_results=ols_results,
        bootstrap_results=bootstrap_results,
        m_estimators_results=m_estimators_results,
    )

    scatter_with_model(
        data=data_df,
        x_col="hours_studied",
        y_col="exam_score",
        model=m_estimators_results["model"],
        title="Exam Score vs Hours Studied with M-estimators Regression",
        x_label="Hours Studied",
        y_label="Exam Score (0--100)",
        line_color="darkgreen",
    )


if __name__ == "__main__":
    main()
