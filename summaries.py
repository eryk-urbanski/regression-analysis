import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold

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

    X = data[x_col].values
    y = data[y_col].values
    X = sm.add_constant(X)

    model = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()

    coef = model.params
    se_slope = model.bse[1]

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

    n = len(X)
    p = X.shape[1]

    b0 = _as_scalar(model.intercept_)
    b1 = _as_scalar(model.coef_)

    rss = np.sum((y - y_pred) ** 2)

    slope_se = np.sqrt(rss / (n - 2)) / np.sqrt(np.sum((X - np.mean(X)) ** 2))
    slope_se = _as_scalar(slope_se)

    p_value = 2 * (1 - stats.t.cdf(np.abs(b1 / slope_se), df=n - 2))
    p_value_to_print = "p<.001" if p_value < 0.001 else f"p={p_value:.6f}"

    r2 = model.score(X, y)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    ci_lower = b1 - 1.96 * slope_se
    ci_upper = b1 + 1.96 * slope_se

    k = p + 1
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)

    print("\n" + "-" * 50)
    print("OLS Regression Results:")
    print("-" * 50)
    print(f"Intercept (b0): {b0}")
    print(f"Slope (b1 coefficient): {b1}")
    print(f"Standard Error of Slope: {slope_se}")
    print(f"p-value: {p_value_to_print}")
    print(f"R-squared: {r2}")
    print(f"Adjusted R-squared: {adj_r2}")
    print(f"AIC: {aic}")
    print(f"BIC: {bic}")
    print(f"95% Confidence Interval for Slope: [{ci_lower}, {ci_upper}]")

    return {
        "b0": b0,
        "b1": b1,
        "slope_se": slope_se,
        "p_value": _as_scalar(p_value),
        "r2": r2,
        "adj_r2": adj_r2,
        "aic": aic,
        "bic": bic,
        "ci_95_slope": (ci_lower, ci_upper)
    }


def comparison_metrics(model, X, y):
    y_true = np.asarray(y).ravel()
    y_pred = np.asarray(model.predict(X)).ravel()

    n = len(X)
    p = X.shape[1]

    rss = np.sum((y_true - y_pred) ** 2)

    r2 = model.score(X, y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    k = p + 1
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)

    return {
        "R-squared": float(r2),
        "Adjusted R-squared": float(adj_r2),
        "AIC": float(aic),
        "BIC": float(bic)
    }


def _cross_val_ape_metrics(model, X_array, y_array, splits):
    absolute_percentage_errors = []
    fold_mean_apes = []

    for train_idx, test_idx in splits:
        fold_model = clone(model)
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]

        fold_model.fit(X_train, y_train)
        y_pred = np.asarray(fold_model.predict(X_test)).ravel()

        non_zero_mask = y_test != 0
        if np.any(non_zero_mask):
            fold_errors = np.abs(
                (y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask]
            ) * 100
            absolute_percentage_errors.extend(fold_errors.tolist())
            fold_mean_apes.append(float(np.mean(fold_errors)))

    if not absolute_percentage_errors:
        raise ValueError("Cannot calculate percentage error because all target values are zero.")

    fold_mean_apes_array = np.asarray(fold_mean_apes, dtype=float)
    if len(fold_mean_apes_array) > 1:
        fold_mean_ape_std = float(np.std(fold_mean_apes_array, ddof=1))
        fold_mean_ape_se = float(fold_mean_ape_std / np.sqrt(len(fold_mean_apes_array)))
    else:
        fold_mean_ape_std = 0.0
        fold_mean_ape_se = 0.0

    return {
        "Mean APE (%)": float(np.mean(absolute_percentage_errors)),
        "Median APE (%)": float(np.median(absolute_percentage_errors)),
        "Fold Mean APE SD (%)": fold_mean_ape_std,
        "Fold Mean APE SE (%)": fold_mean_ape_se,
        "Evaluated folds": int(len(fold_mean_apes_array)),
    }


def kfold_comparison_metrics(model, X, y, n_splits=5):
    X_array = np.asarray(X)
    y_array = np.asarray(y).ravel()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    return _cross_val_ape_metrics(model, X_array, y_array, kf.split(X_array))


def group_kfold_comparison_metrics(model, X, y, groups, n_splits=5):
    X_array = np.asarray(X)
    y_array = np.asarray(y).ravel()
    groups_array = np.asarray(groups).ravel()

    if len(groups_array) != len(y_array):
        raise ValueError("groups must have the same number of rows as X and y.")

    unique_groups = np.unique(groups_array)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"GroupKFold with n_splits={n_splits} requires at least {n_splits} distinct groups, "
            f"but got {len(unique_groups)}."
        )

    gkf = GroupKFold(n_splits=n_splits)

    return _cross_val_ape_metrics(model, X_array, y_array, gkf.split(X_array, y_array, groups_array))


def create_comparison_summary_table(*metrics_dicts):
    summary_df = pd.DataFrame(metrics_dicts)
    if "Model" in summary_df.columns:
        summary_df = summary_df[["Model"] + [col for col in summary_df.columns if col != "Model"]]
    print("\nModel Comparison Summary")
    print(summary_df.to_string(index=False))
    return summary_df


def create_kfold_summary_table(*metrics_dicts):
    summary_df = pd.DataFrame(metrics_dicts)
    ordered_columns = [
        "Model",
        "Mean APE (%)",
        "Median APE (%)",
        "Fold Mean APE SD (%)",
        "Fold Mean APE SE (%)",
        "Evaluated folds",
    ]
    present_ordered_columns = [col for col in ordered_columns if col in summary_df.columns]
    remaining_columns = [col for col in summary_df.columns if col not in present_ordered_columns]
    summary_df = summary_df[present_ordered_columns + remaining_columns]
    print("\nK-Fold Model Assessment Summary")
    print(summary_df.to_string(index=False))
    return summary_df
