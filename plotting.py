import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

from utils import _predict_for_plot, _confidence_band_for_plot, calc_residuals


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
