import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from utils import _as_scalar


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