import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from config import TARGET_COLS, ID_COLS, PREDICTOR_COLS


def summary_OLS(model, X, y):
    y_pred = model.predict(X)

    b0 = model.intercept_[0]
    b1= model.coef_[0]
    slope_se = np.sqrt(np.sum((y - y_pred) ** 2) / (len(X) - 2)) / np.sqrt(np.sum((X - np.mean(X)) ** 2))
    p_value = 2 * (1 - stats.t.cdf(np.abs(b1 / slope_se), df=len(X) - 2))
    r2 = model.score(X, y)
    adj_r2 = 1 - (1 - r2) * (len(X) - 1) / (len(X) - X.shape[1] - 1)

    print(f"OLS Regression Results:")
    print(f"Intercept (b0): {b0}")
    print(f"Slope (b1 coefficient): {b1}")
    print(f"Standard Error of Slope: {slope_se}")
    print(f"p-value: {p_value}")
    print(f"R-squared: {r2}")
    print(f"Adjusted R-squared: {adj_r2}")


def add_panel_label(ax, label):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )


def qq_plot(residuals, ax):
    sm.qqplot(residuals, line='q', ax=ax)
    line = ax.lines[-1]
    line.set_color("red")
    line.set_linewidth(2)
    ax.set_title("Q-Q Plot of Residuals")
    add_panel_label(ax, "A")


def histogram_residuals(residuals, ax):
    ax.hist(residuals, bins=10, edgecolor='black')
    ax.set_title("Histogram of Residuals")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Frequency")
    add_panel_label(ax, "B")


def residuals_vs_fitted_plot(y_pred, residuals, ax, title="Residuals vs Fitted Values"):
    """
    Plots residuals vs fitted values (equivalent to R: plot(fitted, residuals) + abline(h=0))
    
    Parameters:
    - y_true: ground truth values
    - y_pred: predicted values (model.predict(X))
    - title: plot title
    """
    ax.scatter(y_pred, residuals)

    ax.set_title(title)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")

    # horizontal reference line at 0
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
    add_panel_label(ax, "C")

    return residuals


def scale_location_plot(y_pred, residuals, ax, title="Scale-Location Plot"):
    """
    Scale-location plot (homoskedasticity check).
    Equivalent to R:
    plot(fitted(model), sqrt(abs(residuals(model))))
    
    Parameters:
    - y_true: ground truth values
    - y_pred: predicted values
    - title: plot title
    """
    scaled_residuals = np.sqrt(np.abs(residuals))

    ax.scatter(y_pred, scaled_residuals)

    ax.set_title(title)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("sqrt(|Residuals|)")
    add_panel_label(ax, "D")

    return scaled_residuals


def diagnostic_plots(y, y_pred):
    residuals = np.asarray(y).flatten() - np.asarray(y_pred).flatten()
    fitted_values = np.asarray(y_pred).flatten()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    qq_plot(residuals, axes[0])
    histogram_residuals(residuals, axes[1])
    residuals_vs_fitted_plot(fitted_values, residuals, axes[2])
    scale_location_plot(fitted_values, residuals, axes[3])

    fig.tight_layout()
    plt.show()
    

def main():
    data_df = pd.read_csv("data_ideal_188834.csv")
    
    X = data_df[PREDICTOR_COLS]
    y = data_df[TARGET_COLS]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    summary_OLS(model, X, y)
    diagnostic_plots(y, y_pred)


if __name__ == "__main__":
    main()
