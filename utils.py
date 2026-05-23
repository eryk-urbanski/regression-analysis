import numpy as np
import statsmodels.api as sm
from scipy import stats

def _as_scalar(value):
    return np.asarray(value).reshape(-1)[0]


def calc_residuals(y, y_pred):
    return np.asarray(y).flatten() - np.asarray(y_pred).flatten()


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