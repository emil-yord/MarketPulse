# Market Pulse — Statistical Drivers of Life Expectancy Across the EU

A data analysis project asking a specific question with real public data:
**which economic and social indicators correlate most strongly with life
expectancy across EU countries, and where does Bulgaria sit relative to
what the overall trend predicts?**

**Stack:** Python · pandas · scipy/statsmodels · matplotlib/seaborn ·
Plotly · Streamlit · Jupyter

---

## Why this project

Unlike a generic "Titanic" or "housing prices" notebook, this pulls real
data from a live public API and is built around an actual analytical
question rather than just demonstrating that a chart can be drawn. It
exercises three distinct skills:

- **Data collection** — pulling and tidying real data from the World
  Bank's open REST API (no key required)
- **Statistics** — correlation significance testing, OLS regression, and
  residual analysis to identify which countries over/under-perform what
  the model predicts
- **Communication** — the same analysis is presented two ways: a Jupyter
  notebook for the full exploratory process, and a Streamlit dashboard for
  anyone to explore interactively without reading code

## Project structure

```
market-pulse/
├── data/
│   ├── fetch_data.py           Pulls real indicators from the World Bank API
│   ├── generate_sample_data.py Generates synthetic data for offline preview
│   ├── sample_demo_data.csv    Synthetic data (committed, so the dashboard
│   │                           runs immediately — see note below)
│   └── processed/              Real fetched data lands here (gitignored)
├── analysis/
│   └── stats.py                 Correlation, regression, and residual-ranking
│                                 functions shared by the notebook and dashboard
├── notebooks/
│   ├── build_notebook.py        Generates the notebook (for reproducibility)
│   └── exploratory_analysis.ipynb  The actual analysis walkthrough
└── dashboard/
    └── app.py                   Interactive Streamlit dashboard
```

### A note on the sample data

`data/sample_demo_data.csv` is **synthetic** — randomly generated with a
loosely plausible correlation structure baked in, purely so the dashboard
and notebook have something to run against immediately. It is clearly
labeled as such everywhere it's used (a warning banner in the dashboard,
comments in the code) and should never be read as real. Run
`python data/fetch_data.py` to replace it with actual World Bank data —
real analysis only happens on real data.

## Getting started

Requires Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Option A — explore immediately with sample data

```bash
python data/generate_sample_data.py   # regenerates data/sample_demo_data.csv
streamlit run dashboard/app.py
```

### Option B — pull real data (recommended before sharing this project)

```bash
python data/fetch_data.py             # takes ~10 seconds, hits the World Bank API
streamlit run dashboard/app.py        # automatically prefers real data if present
```

### Run the notebook

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

Run all cells top to bottom. The last cell is intentionally left for you
to fill in — the actual written conclusion from reading the real output,
which is the part that makes this read as analysis rather than a pipeline.

## What the dashboard shows

- A correlation matrix across all indicators, with significance testing
  (not just raw correlation — which ones are actually statistically
  meaningful at this sample size)
- An OLS regression of life expectancy on selectable predictors, with
  coefficients, p-values, and R²
- A scatter plot of life expectancy vs. any chosen predictor, with
  Bulgaria highlighted against the fitted trend line
- A full residual ranking — which countries most over/under-perform what
  the model predicts, with Bulgaria's position called out directly

## Possible extensions

- Expand beyond the EU27 for a global comparison
- Add a time-series view — how has Bulgaria's position relative to the
  trend changed over the 2008-2022 window?
- Swap in the World Happiness Report dataset for a direct well-being
  measure instead of life expectancy as the outcome variable
- Deploy the dashboard on Streamlit Community Cloud for a live demo link
