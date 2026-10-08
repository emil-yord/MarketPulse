"""
Market Pulse - an interactive dashboard exploring statistical relationships
between economic indicators and life expectancy across EU countries,
with a focus on where Bulgaria sits relative to the trend.

Run with:
    streamlit run dashboard/app.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import stats as mystats
import fetch_data

st.set_page_config(page_title="Market Pulse", page_icon="📊", layout="wide")

INDICATOR_LABELS = {
    "gdp_growth_pct": "GDP Growth (%)",
    "inflation_pct": "Inflation (%)",
    "unemployment_pct": "Unemployment (%)",
    "life_expectancy_years": "Life Expectancy (years)",
    "education_spend_pct_gdp": "Education Spending (% of GDP)",
    "gni_per_capita_usd": "GNI per Capita (US$)",
    "gini_index": "Gini Index (inequality)",
}

HIGHLIGHT_COUNTRY = "Bulgaria"


@st.cache_data(show_spinner=False)
def fetch_live_data():
    """Pulls real World Bank data at app startup. Used so a deployed demo
    (e.g. on Streamlit Cloud, where no one has run fetch_data.py by hand)
    still shows real data instead of only ever showing the synthetic
    sample. Cached for the life of the app process - not re-fetched on
    every page interaction.
    """
    return fetch_data.fetch_all_indicators(verbose=False)


@st.cache_data
def load_data():
    real_path = "data/processed/indicators.csv"
    sample_path = "data/sample_demo_data.csv"

    if os.path.exists(real_path):
        return pd.read_csv(real_path), False

    # No cached real data on disk - try fetching it live before falling
    # back to the synthetic sample. This is what makes a deployed public
    # demo show real data without a manual setup step.
    try:
        with st.spinner("Fetching live data from the World Bank API (first load only)..."):
            df = fetch_live_data()
        if df is not None and len(df) > 0:
            return df, False
    except Exception as e:
        print(f"Live fetch failed, falling back to sample data: {e}")

    if os.path.exists(sample_path):
        return pd.read_csv(sample_path), True
    else:
        st.error(
            "No data found and the live World Bank fetch failed. Run "
            "`python data/generate_sample_data.py` for a quick preview, or "
            "`python data/fetch_data.py` for real data."
        )
        st.stop()


def main():
    df, is_sample = load_data()

    st.title("📊 Market Pulse")
    st.caption(
        "Exploring statistical drivers of life expectancy across EU countries, "
        f"with {HIGHLIGHT_COUNTRY} highlighted against the trend."
    )

    if is_sample:
        st.warning(
            "⚠️ Showing **synthetic demo data** (not real World Bank figures) so the "
            "dashboard can be explored immediately. Run `python data/fetch_data.py` "
            "to pull and switch to real data.",
            icon="⚠️",
        )

    indicator_cols = [c for c in INDICATOR_LABELS if c in df.columns]

    # ---- Sidebar controls ----
    st.sidebar.header("Controls")

    years = sorted(df["year"].unique())
    year = st.sidebar.select_slider("Year", options=years, value=years[-1])

    outcome = "life_expectancy_years"
    predictor_options = [c for c in indicator_cols if c != outcome]
    predictors = st.sidebar.multiselect(
        "Predictors to include in the regression",
        options=predictor_options,
        default=predictor_options[:3],
        format_func=lambda c: INDICATOR_LABELS[c],
    )

    year_df = df[df["year"] == year].dropna(subset=[outcome])

    if len(predictors) == 0:
        st.info("Select at least one predictor in the sidebar to run the analysis.")
        return

    # ---- Correlation heatmap ----
    st.subheader("Correlation matrix")
    corr, pval = mystats.correlation_matrix_with_significance(
        year_df, [outcome] + predictors
    )
    display_corr = corr.rename(index=INDICATOR_LABELS, columns=INDICATOR_LABELS)

    fig_heat = px.imshow(
        display_corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig_heat.update_layout(height=400)
    st.plotly_chart(fig_heat, use_container_width=True)

    top = mystats.summarize_top_correlates(corr, pval, outcome)
    if len(top) > 0:
        st.caption(
            "Statistically significant correlates with life expectancy (p < 0.05), "
            "strongest first: "
            + ", ".join(f"{INDICATOR_LABELS.get(r.indicator, r.indicator)} (r={r.correlation:.2f})"
                         for r in top.itertuples())
        )

    # ---- Regression ----
    st.subheader("Regression model")
    model, clean = mystats.fit_regression(year_df, outcome, predictors, id_cols=["country"])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("R² (variance explained)", f"{model.rsquared:.3f}")
        st.metric("Countries in model", len(clean))

    with col2:
        coef_df = pd.DataFrame({
            "Predictor": [INDICATOR_LABELS.get(p, p) for p in predictors],
            "Coefficient": [model.params[p] for p in predictors],
            "p-value": [model.pvalues[p] for p in predictors],
        })
        coef_df["Significant (p<0.05)"] = coef_df["p-value"] < 0.05
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

    # ---- Scatter: outcome vs each predictor, country highlighted ----
    st.subheader(f"{HIGHLIGHT_COUNTRY} vs. the trend")
    scatter_predictor = st.selectbox(
        "Plot life expectancy against:",
        options=predictors,
        format_func=lambda c: INDICATOR_LABELS[c],
    )

    plot_df = year_df.dropna(subset=[outcome, scatter_predictor]).copy()
    plot_df["is_highlight"] = plot_df["country"] == HIGHLIGHT_COUNTRY

    fig_scatter = px.scatter(
        plot_df,
        x=scatter_predictor,
        y=outcome,
        color="is_highlight",
        color_discrete_map={True: "#D64545", False: "#4C72B0"},
        hover_name="country",
        trendline="ols",
        labels={
            scatter_predictor: INDICATOR_LABELS[scatter_predictor],
            outcome: INDICATOR_LABELS[outcome],
        },
    )
    fig_scatter.update_traces(marker=dict(size=10))
    fig_scatter.update_layout(showlegend=False, height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ---- Residual ranking ----
    st.subheader("Who over/under-performs the model?")
    st.caption(
        "Countries whose actual life expectancy is far from what the regression "
        "predicts from their other indicators - a large positive residual means "
        "life expectancy is higher than the model expects, given everything else."
    )

    ranking = mystats.residual_ranking(year_df, model, clean, outcome, predictors, ["country"])
    ranking_display = ranking.rename(columns={
        outcome: INDICATOR_LABELS[outcome],
        "predicted": "Predicted",
        "residual": "Residual",
    })[["country", INDICATOR_LABELS[outcome], "Predicted", "Residual"]]

    def highlight_row(row):
        if row["country"] == HIGHLIGHT_COUNTRY:
            return ["background-color: #FFE8E8"] * len(row)
        return [""] * len(row)

    st.dataframe(
        ranking_display.style.apply(highlight_row, axis=1).format(
            {INDICATOR_LABELS[outcome]: "{:.1f}", "Predicted": "{:.1f}", "Residual": "{:+.2f}"}
        ),
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    bg_rank = ranking[ranking["country"] == HIGHLIGHT_COUNTRY]
    if len(bg_rank) == 1:
        rank_position = ranking.index[ranking["country"] == HIGHLIGHT_COUNTRY][0] + 1
        residual_val = bg_rank.iloc[0]["residual"]
        direction = "above" if residual_val > 0 else "below"
        st.info(
            f"**{HIGHLIGHT_COUNTRY}** ranks **#{rank_position} of {len(ranking)}** "
            f"by residual, sitting **{abs(residual_val):.2f} years {direction}** "
            "what the model predicts given its other indicators."
        )


if __name__ == "__main__":
    main()
