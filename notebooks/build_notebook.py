"""
Builds notebooks/exploratory_analysis.ipynb as raw JSON (no nbformat
dependency needed to generate it). Run once: python notebooks/build_notebook.py
"""

import json


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md("""# Market Pulse: Statistical Drivers of Life Expectancy Across the EU

**Question:** Which economic and social indicators correlate most strongly with
life expectancy across EU countries, and where does Bulgaria sit relative to
what the overall trend would predict?

**Data:** World Bank Open Data API (2008-2022), 27 EU countries, 7 indicators.
See `data/fetch_data.py` for the collection script.
"""),

    code("""import sys
sys.path.insert(0, '../analysis')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import stats as mystats

sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns', None)
"""),

    md("""## 1. Load the data

Run `python data/fetch_data.py` first to pull real data from the World Bank
API. Falls back to the synthetic sample dataset if the real one hasn't been
fetched yet, purely so this notebook has something to run against."""),

    code("""import os

if os.path.exists('../data/processed/indicators.csv'):
    df = pd.read_csv('../data/processed/indicators.csv')
    print('Using real World Bank data')
else:
    df = pd.read_csv('../data/sample_demo_data.csv')
    print('⚠️  Using SYNTHETIC sample data — run data/fetch_data.py for real results')

df.head()
"""),

    code("""df.describe()
"""),

    md("""## 2. Focus on the most recent complete year

Cross-sectional analysis is easier to reason about than a full panel, and
most of these indicators move slowly year to year anyway."""),

    code("""latest_year = df['year'].max()
year_df = df[df['year'] == latest_year].copy()
print(f'Analyzing {latest_year}: {year_df["country"].nunique()} countries')
year_df.sort_values('life_expectancy_years', ascending=False)[['country', 'life_expectancy_years']]
"""),

    md("""## 3. Correlation matrix

How does each indicator relate to every other indicator, and specifically
to life expectancy?"""),

    code("""indicator_cols = [
    'life_expectancy_years', 'gni_per_capita_usd', 'unemployment_pct',
    'gdp_growth_pct', 'inflation_pct', 'education_spend_pct_gdp', 'gini_index'
]

corr, pval = mystats.correlation_matrix_with_significance(year_df, indicator_cols)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title(f'Indicator correlation matrix ({latest_year})')
plt.tight_layout()
plt.show()
"""),

    code("""top_correlates = mystats.summarize_top_correlates(corr, pval, 'life_expectancy_years')
top_correlates
"""),

    md("""## 4. Regression: what predicts life expectancy?

Fit an OLS model using the indicators most plausibly causally related to
health outcomes — income, unemployment, and education spending — and see
how much variance they jointly explain."""),

    code("""predictors = ['gni_per_capita_usd', 'unemployment_pct', 'education_spend_pct_gdp']

model, clean = mystats.fit_regression(year_df, 'life_expectancy_years', predictors, id_cols=['country'])
print(model.summary())
"""),

    md("""## 5. Where does Bulgaria sit relative to the model?

The residual — actual minus predicted life expectancy — shows which
countries are over- or under-performing what their economic indicators
alone would suggest."""),

    code("""ranking = mystats.residual_ranking(year_df, model, clean, 'life_expectancy_years', predictors, ['country'])
ranking
"""),

    code("""fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#D64545' if c == 'Bulgaria' else '#4C72B0' for c in ranking['country']]
ax.barh(ranking['country'], ranking['residual'], color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Residual (years above/below model prediction)')
ax.set_title('Who over/under-performs the life-expectancy model?')
plt.tight_layout()
plt.show()
"""),

    md("""## 6. Scatter: life expectancy vs. GNI per capita, Bulgaria highlighted"""),

    code("""fig, ax = plt.subplots(figsize=(9, 6))
is_bg = year_df['country'] == 'Bulgaria'

ax.scatter(year_df.loc[~is_bg, 'gni_per_capita_usd'], year_df.loc[~is_bg, 'life_expectancy_years'],
           s=60, color='#4C72B0', label='Other EU countries')
ax.scatter(year_df.loc[is_bg, 'gni_per_capita_usd'], year_df.loc[is_bg, 'life_expectancy_years'],
           s=120, color='#D64545', label='Bulgaria', zorder=5)

ax.set_xlabel('GNI per capita (US$)')
ax.set_ylabel('Life expectancy (years)')
ax.set_title(f'Life expectancy vs. income, {latest_year}')
ax.legend()
plt.tight_layout()
plt.show()
"""),

    md("""## Findings

*(Fill this in once you've run the notebook against real data — write 3-5
sentences on what the regression and residual ranking actually show. This
is the part that makes the project read as analysis rather than just a
pipeline: what did you conclude, and does it hold up to a second look?)*
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("notebooks/exploratory_analysis.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Wrote notebooks/exploratory_analysis.ipynb")
