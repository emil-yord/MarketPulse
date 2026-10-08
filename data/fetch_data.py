"""
Pulls economic and well-being indicators for EU countries from the World
Bank's public API (no key required) and saves a tidy, analysis-ready CSV.

Usage:
    python data/fetch_data.py

Output:
    data/processed/indicators.csv  - one row per (country, year), one
                                      column per indicator
"""

import time
import requests
import pandas as pd

API_BASE = "https://api.worldbank.org/v2"

# EU27 country codes (World Bank 2-letter codes)
EU27 = [
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR",
    "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
    "SI", "ES", "SE"
]

# World Bank indicator codes -> friendly column names
INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "FP.CPI.TOTL.ZG": "inflation_pct",
    "SL.UEM.TOTL.ZS": "unemployment_pct",
    "SP.DYN.LE00.IN": "life_expectancy_years",
    "SE.XPD.TOTL.GD.ZS": "education_spend_pct_gdp",
    "NY.GNP.PCAP.CD": "gni_per_capita_usd",
    "SI.POV.GINI": "gini_index",
}

START_YEAR = 2008
END_YEAR = 2022


def fetch_indicator(indicator_code, countries, start_year, end_year):
    """Fetches one indicator for a list of countries across a year range.

    The World Bank API paginates and returns up to ~32k rows per page by
    default; one page is plenty for 27 countries x ~15 years.
    """
    country_str = ";".join(countries)
    url = f"{API_BASE}/country/{country_str}/indicator/{indicator_code}"
    params = {
        "format": "json",
        "date": f"{start_year}:{end_year}",
        "per_page": 2000,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if len(payload) < 2 or payload[1] is None:
        print(f"  warning: no data returned for {indicator_code}")
        return pd.DataFrame(columns=["country_code", "country", "year", "value"])

    rows = []
    for entry in payload[1]:
        rows.append({
            "country_code": entry["countryiso3code"],
            "country": entry["country"]["value"],
            "year": int(entry["date"]),
            "value": entry["value"],
        })

    return pd.DataFrame(rows)


def fetch_all_indicators(verbose=True):
    """Fetches every indicator in INDICATORS for every EU27 country and
    returns one merged, tidy DataFrame (one row per country+year).

    Separated from main() so the dashboard can call this directly on
    startup (e.g. on Streamlit Cloud, where there's no separate step to
    run fetch_data.py first) instead of only via the command line.
    """
    if verbose:
        print(f"Fetching {len(INDICATORS)} indicators for {len(EU27)} EU countries "
              f"({START_YEAR}-{END_YEAR})...")

    frames = []
    for code, col_name in INDICATORS.items():
        if verbose:
            print(f"  fetching {col_name} ({code})...")
        df = fetch_indicator(code, EU27, START_YEAR, END_YEAR)
        df = df.rename(columns={"value": col_name})
        df = df[["country_code", "country", "year", col_name]]
        frames.append(df)
        time.sleep(0.3)  # be polite to the free public API

    # Merge all indicators into one wide table, keyed on country + year
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on=["country_code", "country", "year"], how="outer")

    return merged.sort_values(["country", "year"]).reset_index(drop=True)


def main():
    merged = fetch_all_indicators()

    out_path = "data/processed/indicators.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved {len(merged)} rows to {out_path}")
    print(f"Countries: {merged['country'].nunique()}  Years: {merged['year'].min()}-{merged['year'].max()}")


if __name__ == "__main__":
    main()
