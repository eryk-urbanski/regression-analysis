import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from config import TARGET_COLS, ID_COLS, PREDICTOR_COLS


def _as_scalar(value):
    return np.asarray(value).reshape(-1)[0]


def summary_OLS(model, X, y):
    y_pred = model.predict(X)

    b0 = _as_scalar(model.intercept_)
    b1 = _as_scalar(model.coef_)
    slope_se = np.sqrt(np.sum((y - y_pred) ** 2) / (len(X) - 2)) / np.sqrt(np.sum((X - np.mean(X)) ** 2))
    slope_se = _as_scalar(slope_se)
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

    return {
        "b0": b0,
        "b1": b1,
        "slope_se": slope_se,
        "p_value": _as_scalar(p_value),
        "r2": r2,
        "adj_r2": adj_r2,
        "ci_95_slope": (ci_lower, ci_upper)
    }


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


def _predict_for_plot(model, X_values):
    try:
        predictions = model.predict(X_values)
    except (TypeError, ValueError):
        X_with_const = sm.add_constant(X_values, has_constant="add")
        predictions = model.predict(X_with_const)

    return np.asarray(predictions).reshape(-1)


def _confidence_band_for_plot(model, X_values, ci=95):
    if not hasattr(model, "cov_params"):
        return None, None

    covariance = np.asarray(model.cov_params())
    if covariance.ndim != 2:
        return None, None

    n_features = X_values.shape[1]
    if covariance.shape == (n_features + 1, n_features + 1):
        design_matrix = sm.add_constant(X_values, has_constant="add")
    elif covariance.shape == (n_features, n_features):
        design_matrix = X_values
    else:
        return None, None

    standard_errors = np.sqrt(np.einsum("ij,jk,ik->i", design_matrix, covariance, design_matrix))
    z_value = stats.norm.ppf(0.5 + ci / 200)
    fitted_values = _predict_for_plot(model, X_values)

    lower = fitted_values - z_value * standard_errors
    upper = fitted_values + z_value * standard_errors

    return lower, upper


def scatter_with_model(data, x_col, y_col, model, title=None,
                       x_label=None, y_label=None,
                       point_color="steelblue",
                       line_color="red",
                       alpha=0.7,
                       ci=95,
                       band_alpha=0.2):
    """
    Scatter plot with fitted line for any model exposing predict().

    Parameters:
    - data: pandas DataFrame
    - x_col: name of x variable column
    - y_col: name of y variable column
    - model: fitted model object with predict()
    - title: plot title (optional)
    - x_label: x-axis label (optional)
    - y_label: y-axis label (optional)
    """

    x_values = data[x_col].to_numpy()
    y_values = data[y_col].to_numpy()

    order = np.argsort(x_values)
    x_sorted = x_values[order].reshape(-1, 1)
    y_pred_sorted = _predict_for_plot(model, x_sorted)
    ci_lower, ci_upper = _confidence_band_for_plot(model, x_sorted, ci=ci)

    plt.figure()
    plt.scatter(
        x_values,
        y_values,
        color=point_color,
        s=60,
        alpha=alpha,
    )
    if ci_lower is not None and ci_upper is not None:
        plt.fill_between(
            x_sorted.ravel(),
            ci_lower,
            ci_upper,
            color=line_color,
            alpha=band_alpha,
        )
    plt.plot(x_sorted.ravel(), y_pred_sorted, color=line_color, linewidth=2)

    if title:
        plt.title(title, fontweight="bold")

    plt.xlabel(x_label if x_label else x_col)
    plt.ylabel(y_label if y_label else y_col)

    plt.show()


def scatter_with_regression(data, x_col, y_col, title=None,
                            x_label=None, y_label=None,
                            point_color="steelblue",
                            line_color="red",
                            alpha=0.7,
                            ci=95):
    """
    Scatter plot with an OLS regression line and optional confidence band.
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


def robust_model_summary(data, x_col, y_col):
    """
    Robust linear regression summary (R: rlm + psi.huber equivalent).
    """

    # Prepare data
    X = data[x_col].values
    y = data[y_col].values

    # Add intercept (like R formula)
    X = sm.add_constant(X)

    # Robust linear model (Huber M-estimator)
    model = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()

    # Coefficients (like coef(model))
    coef = model.params

    # Standard error of slope (like summary$coefficients["x", "Std. Error"])
    se_slope = model.bse[1]

    # Print results
    print("\n" + "-" * 50)
    print("Robust Linear Regression Results")
    print("-" * 50)

    print(f"Intercept: {coef[0]:.6f}")
    print(f"Slope (b1 coefficient): {coef[1]:.6f}")
    print(f"Standard Error of Slope: {se_slope:.6f}")
    print(f"t statistic for slope: {coef[1] / se_slope:.6f}")

    return {
        "model": model,
        "coef": coef,
        "se_slope": se_slope,
        "t_stat_slope": coef[1] / se_slope if se_slope != 0 else 0
    }


def create_regression_summary_table(ols_results, bootstrap_results, m_estimators_results):
    ols_ci = ols_results["ci_95_slope"]
    bootstrap_ci = bootstrap_results["ci_95_slope"]

    ols_p_to_print = "p<.001" if ols_results["p_value"] < 0.001 else f"p={ols_results['p_value']:.6f}"

    summary_df = pd.DataFrame(
        [
            {
                "Method": "OLS",
                "Intercept": ols_results["b0"],
                "Slope": ols_results["b1"],
                "SE(Slope)": ols_results["slope_se"],
                "Significance": f"{ols_p_to_print}; CI95%=[{ols_ci[0]:.6f}, {ols_ci[1]:.6f}]",
            },
            {
                "Method": "Bootstrap",
                "Intercept": bootstrap_results["intercept_mean"],
                "Slope": bootstrap_results["slope_mean"],
                "SE(Slope)": bootstrap_results["slope_std"],
                "Significance": f"CI95%=[{bootstrap_ci[0]:.6f}, {bootstrap_ci[1]:.6f}]",
            },
            {
                "Method": "M-estimators",
                "Intercept": m_estimators_results["coef"][0],
                "Slope": m_estimators_results["coef"][1],
                "SE(Slope)": m_estimators_results["se_slope"],
                "Significance": f"t={m_estimators_results['t_stat_slope']:.6f}",
            },
        ]
    )

    print("\nRegression summary table")
    print(summary_df.to_string(index=False))

    return summary_df


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
