"""
Generates a small SYNTHETIC dataset with the same shape as fetch_data.py's
output, so the dashboard and notebook can be explored immediately without
needing network access first.

This is fake data with a loosely plausible correlation structure baked in
via a simple linear model plus noise - it is NOT real World Bank data and
should never be treated or cited as such. Run fetch_data.py to replace it
with the real thing.
"""

import numpy as np
import pandas as pd

EU27 = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "HR": "Croatia",
    "CY": "Cyprus", "CZ": "Czechia", "DK": "Denmark", "EE": "Estonia",
    "FI": "Finland", "FR": "France", "DE": "Germany", "GR": "Greece",
    "HU": "Hungary", "IE": "Ireland", "IT": "Italy", "LV": "Latvia",
    "LT": "Lithuania", "LU": "Luxembourg", "MT": "Malta", "NL": "Netherlands",
    "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SK": "Slovakia",
    "SI": "Slovenia", "ES": "Spain", "SE": "Sweden",
}

YEARS = range(2008, 2023)

rng = np.random.default_rng(seed=42)


def main():
    rows = []

    # Give each country a fixed "development level" so indicators correlate
    # sensibly within a country and across years, instead of being pure noise.
    dev_level = {code: rng.uniform(0.2, 1.0) for code in EU27}

    for code, name in EU27.items():
        level = dev_level[code]
        for year in YEARS:
            trend = (year - 2008) * 0.3  # mild overall improvement over time

            gni = 8000 + level * 55000 + rng.normal(0, 2000) + trend * 300
            life_exp = 72 + level * 10 + rng.normal(0, 0.8) + trend * 0.1
            unemployment = max(2, 14 - level * 9 + rng.normal(0, 1.5))
            gdp_growth = rng.normal(1.5 + level * 0.5, 2.0)
            inflation = rng.normal(2.0 - level * 0.3, 1.2)
            education_spend = 3.5 + level * 2.5 + rng.normal(0, 0.4)
            gini = max(22, 38 - level * 10 + rng.normal(0, 2))

            rows.append({
                "country_code": code,
                "country": name,
                "year": year,
                "gdp_growth_pct": round(gdp_growth, 2),
                "inflation_pct": round(inflation, 2),
                "unemployment_pct": round(unemployment, 2),
                "life_expectancy_years": round(life_exp, 1),
                "education_spend_pct_gdp": round(education_spend, 2),
                "gni_per_capita_usd": round(gni, 0),
                "gini_index": round(gini, 1),
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/sample_demo_data.csv", index=False)
    print(f"Wrote {len(df)} synthetic rows to data/sample_demo_data.csv")
    print("Reminder: this is fake data for UI preview only - run fetch_data.py for the real thing.")


if __name__ == "__main__":
    main()
