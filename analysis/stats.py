"""
Reusable statistical analysis functions: correlation testing, regression
modeling, and residual-based outlier detection. Used by both the
exploratory notebook and the Streamlit dashboard, so the two always agree.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def correlation_matrix_with_significance(df, columns):
    """Returns (correlation_matrix, p_value_matrix) for the given columns.

    Uses pairwise-complete Pearson correlation - rows with a missing value
    in either column are dropped for that pair only, so one sparsely
    reported indicator doesn't shrink every other pair's sample size.
    """
    n = len(columns)
    corr = pd.DataFrame(np.ones((n, n)), index=columns, columns=columns)
    pvals = pd.DataFrame(np.zeros((n, n)), index=columns, columns=columns)

    for i, col_a in enumerate(columns):
        for j, col_b in enumerate(columns):
            if i == j:
                continue
            paired = df[[col_a, col_b]].dropna()
            if len(paired) < 3:
                corr.loc[col_a, col_b] = np.nan
                pvals.loc[col_a, col_b] = np.nan
                continue
            r, p = stats.pearsonr(paired[col_a], paired[col_b])
            corr.loc[col_a, col_b] = r
            pvals.loc[col_a, col_b] = p

    return corr, pvals


def fit_regression(df, outcome, predictors, id_cols=None):
    """Fits an OLS regression of `outcome` on `predictors`.

    id_cols (e.g. ["country"]) are carried through into the returned
    cleaned DataFrame even though they aren't part of the model, so
    downstream functions like residual_ranking can identify each row.

    Returns the fitted statsmodels result object (has .summary(),
    .rsquared, .params, .pvalues, etc.) along with the cleaned data
    actually used (rows with any missing value in outcome/predictors
    dropped - id_cols are not required to be non-null).
    """
    id_cols = id_cols or []
    cols = [outcome] + predictors
    required = df[cols].dropna().index
    clean = df.loc[required, id_cols + cols]

    X = sm.add_constant(clean[predictors])
    y = clean[outcome]

    model = sm.OLS(y, X).fit()
    return model, clean


def residual_ranking(df, model, clean_data, outcome, predictors, id_cols):
    """Ranks observations by how far they fall above/below the regression
    prediction - i.e. which countries over- or under-perform what the
    model would predict given their other indicators.

    Returns a DataFrame sorted by residual (largest positive = most above
    the predicted trend, largest negative = most below it).
    """
    X = sm.add_constant(clean_data[predictors])
    predicted = model.predict(X)
    residuals = clean_data[outcome] - predicted

    result = clean_data[id_cols + [outcome]].copy()
    result["predicted"] = predicted
    result["residual"] = residuals
    result["abs_residual"] = residuals.abs()

    return result.sort_values("residual", ascending=False).reset_index(drop=True)


def summarize_top_correlates(corr_matrix, pval_matrix, outcome, alpha=0.05):
    """Returns the predictors most correlated with `outcome`, sorted by
    absolute correlation strength, restricted to statistically significant
    results at the given alpha level.
    """
    if outcome not in corr_matrix.columns:
        return pd.DataFrame(columns=["indicator", "correlation", "p_value"])

    rows = []
    for col in corr_matrix.columns:
        if col == outcome:
            continue
        r = corr_matrix.loc[outcome, col]
        p = pval_matrix.loc[outcome, col]
        if pd.notna(r) and pd.notna(p) and p < alpha:
            rows.append({"indicator": col, "correlation": r, "p_value": p})

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.reindex(out["correlation"].abs().sort_values(ascending=False).index).reset_index(drop=True)
