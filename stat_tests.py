from scipy import stats
import pandas as pd

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