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

    ci_lower = b1 - 1.96 * slope_se
    ci_upper = b1 + 1.96 * slope_se

    print("\n" + "-" * 50)
    print(f"OLS Regression Results:")
    print("-" * 50)
    print(f"Intercept (b0): {b0}")
    print(f"Slope (b1 coefficient): {b1}")
    print(f"Standard Error of Slope: {slope_se}")
    print(f"p-value: {p_value}")
    print(f"R-squared: {r2}")
    print(f"Adjusted R-squared: {adj_r2}")

    print(f"95% Confidence Interval for Slope: [{ci_lower}, {ci_upper}]")


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
    residuals = calc_residuals(y, y_pred)
    fitted_values = np.asarray(y_pred).flatten()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    qq_plot(residuals, axes[0])
    histogram_residuals(residuals, axes[1])
    residuals_vs_fitted_plot(fitted_values, residuals, axes[2])
    scale_location_plot(fitted_values, residuals, axes[3])

    fig.tight_layout()
    plt.show()


def calc_residuals(y, y_pred):
    return np.asarray(y).flatten() - np.asarray(y_pred).flatten()


def shapiro_wilk_test(residuals):
    stat, p_value = stats.shapiro(residuals)
    print("\n" + "-" * 50)
    print("Shapiro-Wilk Test for Normality of Residuals")
    print("-" * 50)
    print(f"W = {stat}\np-value = {p_value}")
    if p_value > 0.05:
        print("Residuals are likely normally distributed (fail to reject H0).")
    else:
        print("Residuals are not normally distributed (reject H0).")


def levene_test(y_pred, residuals, n_groups=4):
    groups = pd.cut(y_pred.squeeze(), bins=n_groups)
    grouped_residuals = [
        residuals[groups == g]
        for g in groups.categories
    ]
    stat, p_value = stats.levene(*grouped_residuals)
    print("\n" + "-" * 50)
    print("Levene's Test for Homogeneity of Variance")
    print("-" * 50)
    print(f"W = {stat}\np-value = {p_value}")
    if p_value > 0.05:
        print("Residuals have equal variances (fail to reject H0).")
    else:
        print("Residuals do not have equal variances (reject H0).")


def scatter_with_regression(data, x_col, y_col, title=None,
                            x_label=None, y_label=None,
                            point_color="steelblue",
                            line_color="red",
                            alpha=0.7,
                            ci=95):
    """
    Generic ggplot2-like scatter + linear regression plot.

    Parameters:
    - data: pandas DataFrame
    - x_col: name of x variable column
    - y_col: name of y variable column
    - title: plot title (optional)
    - x_label: x-axis label (optional)
    - y_label: y-axis label (optional)
    """

    plt.figure()

    sns.regplot(
        data=data,
        x=x_col,
        y=y_col,
        scatter_kws={
            "color": point_color,
            "s": 60,
            "alpha": alpha
        },
        line_kws={
            "color": line_color
        },
        ci=ci
    )

    if title:
        plt.title(title, fontweight="bold")

    plt.xlabel(x_label if x_label else x_col)
    plt.ylabel(y_label if y_label else y_col)

    plt.show()
    

def bootstrap(data, x_col, y_col, n_boot=2000, seed=None):
    """
    Bootstrap linear regression coefficients (intercept + slope)
    and confidence interval for slope.

    Parameters:
    - data: pandas DataFrame
    - x_col: predictor column name
    - y_col: target column name
    - n_boot: number of bootstrap samples
    - seed: random seed for reproducibility

    Returns:
    - dictionary with results
    """

    if seed is not None:
        np.random.seed(seed)

    X = data[x_col].values.reshape(-1, 1)
    y = data[y_col].values

    coefs = np.zeros((n_boot, 2))  # [intercept, slope]

    n = len(data)

    for i in range(n_boot):
        idx = np.random.randint(0, n, n)

        X_sample = X[idx]
        y_sample = y[idx]

        model = LinearRegression()
        model.fit(X_sample, y_sample)

        coefs[i, 0] = model.intercept_
        coefs[i, 1] = model.coef_[0]

    # statistics
    intercept_mean = np.mean(coefs[:, 0])
    slope_mean = np.mean(coefs[:, 1])

    slope_std = np.std(coefs[:, 1], ddof=1)

    # percentile CI (like R boot.ci(type="perc"))
    ci_lower = np.percentile(coefs[:, 1], 2.5)
    ci_upper = np.percentile(coefs[:, 1], 97.5)

    # print results
    print("\n" + "-" * 50)
    print("Bootstrap Linear Regression Results")
    print("-" * 50)

    print(f"Intercept (mean): {intercept_mean:.6f}")
    print(f"Slope (mean): {slope_mean:.6f}")
    print(f"Slope Std Error: {slope_std:.6f}")

    print(f"95% CI for slope: [{ci_lower:.6f}, {ci_upper:.6f}]")

    return {
        "intercept_mean": intercept_mean,
        "slope_mean": slope_mean,
        "slope_std": slope_std,
        "ci_95_slope": (ci_lower, ci_upper),
        "bootstrap_coefficients": coefs
    }


def main():
    data_df = pd.read_csv("data_ideal_188834.csv")
    
    X = data_df[PREDICTOR_COLS]
    y = data_df[TARGET_COLS]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    summary_OLS(model, X, y)
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


if __name__ == "__main__":
    main()
