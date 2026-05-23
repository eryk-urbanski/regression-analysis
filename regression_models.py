import numpy as np
from sklearn.linear_model import LinearRegression


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