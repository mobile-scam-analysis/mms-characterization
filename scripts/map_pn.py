#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

import phonenumbers
from phonenumbers import region_code_for_number
import pycountry

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "Liberation Serif"]
})

# Change path to CSV and download shapefiles to draw map
INPUT_CSV = "data.csv"
PHONE_COL = "Phone Number"
SHAPEFILE = "shapefiles/countries/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
OUT_PNG   = "phone_number_map_new.pdf"


def alpha2_to_alpha3(alpha2: str):
    try:
        c = pycountry.countries.get(alpha_2=alpha2)
        return c.alpha_3 if c else None
    except Exception:
        return None


def extract_numbers(text: str):
    """Return ALL phone-like tokens from a text cell (no dedup)."""
    if not isinstance(text, str):
        return []
    return re.findall(r'\+?\d[\d\s\-()]{7,}\d', text)


def parse_number_safely(raw: str):
    s = str(raw).strip()

    if s.startswith("+"):
        cleaned = "+" + re.sub(r"\D", "", s[1:])
    else:
        cleaned = re.sub(r"\D", "", s)

    if not cleaned or cleaned == "+":
        return None

    try:
        if cleaned.startswith("+"):
            num = phonenumbers.parse(cleaned, None)
            return num if phonenumbers.is_possible_number(num) else None

        if len(cleaned) == 10:
            num = phonenumbers.parse(cleaned, "US")
            return num if phonenumbers.is_possible_number(num) else None

        if len(cleaned) == 11 and cleaned.startswith("1"):
            num = phonenumbers.parse("+" + cleaned, None)
            return num if phonenumbers.is_possible_number(num) else None

        if len(cleaned) > 11:
            num = phonenumbers.parse("+" + cleaned, None)
            return num if phonenumbers.is_possible_number(num) else None

        num = phonenumbers.parse(cleaned, "US")
        return num if phonenumbers.is_possible_number(num) else None

    except Exception:
        return None


df = pd.read_csv(INPUT_CSV, low_memory=False)
phone_series = df.get(PHONE_COL, pd.Series(dtype=str)).dropna().astype(str)

country_counts = defaultdict(int)

nanp_count = 0
non_nanp_count = 0

for row in phone_series:
    for raw in extract_numbers(row):
        num = parse_number_safely(raw)
        if not num:
            continue

        alpha2 = region_code_for_number(num)
        if not alpha2:
            continue

        country_counts[alpha2] += 1

        # NANP = US + CA
        if alpha2 in {"US", "CA"}:
            nanp_count += 1
        else:
            non_nanp_count += 1


total_extracted = sum(len(extract_numbers(x)) for x in phone_series)
total_valid = int(sum(country_counts.values()))

print(f"\nTotal phone numbers extracted: {total_extracted}")
print(f"Total valid phone numbers mapped to countries: {total_valid}")


print("\n=== NANP vs Non-NANP ===")
print(f"NANP (US + CA):     {nanp_count:,} ({nanp_count / max(total_valid,1):.2%})")
print(f"Non-NANP:           {non_nanp_count:,} ({non_nanp_count / max(total_valid,1):.2%})")
print(f"Check (sum):        {nanp_count + non_nanp_count:,}\n")


def alpha2_to_name(alpha2: str) -> str:
    try:
        c = pycountry.countries.get(alpha_2=alpha2)
        return c.name if c else alpha2
    except Exception:
        return alpha2


country_table = (
    pd.DataFrame(
        [
            {
                "alpha2": code,
                "alpha3": alpha2_to_alpha3(code),
                "country_name": alpha2_to_name(code),
                "count": cnt,
            }
            for code, cnt in country_counts.items()
        ]
    )
    .dropna(subset=["alpha3"])
    .sort_values("count", ascending=False)
    .reset_index(drop=True)
)

grand_total = country_table["count"].sum()
country_table["share"] = country_table["count"] / max(grand_total, 1)

print("\n=== Phone numbers by country (top 10) ===")
print(
    country_table.loc[:, ["alpha2", "country_name", "count", "share"]]
    .head(10)
    .to_string(index=False, formatters={"share": "{:.2%}".format})
)

print(f"\nCountries observed: {len(country_table)}")
print(f"Total valid numbers: {grand_total:,}")



country_df = (
    pd.DataFrame(
        [{"ISO_A3": alpha2_to_alpha3(code), "count": cnt} for code, cnt in country_counts.items()]
    )
    .dropna(subset=["ISO_A3"])
)

world = gpd.read_file(SHAPEFILE)
world = world.merge(country_df, left_on="ADM0_A3", right_on="ISO_A3", how="left")
world = world[world["CONTINENT"] != "Antarctica"].copy()
world["count"] = world["count"].astype(float)

counts = world["count"].dropna()
if counts.empty:
    raise ValueError("No country counts to plot (all NaN).")

vmin = max(1.0, float(counts.min()))
vmax = float(counts.max())

fig, ax = plt.subplots(figsize=(12,8))

norm = LogNorm(vmin=vmin, vmax=vmax)

world.plot(
    column="count",
    cmap="coolwarm",
    linewidth=0.7,
    edgecolor="#222222",
    ax=ax,
    norm=norm,
    missing_kwds={"color": "white", "label": "No data"},
)

sm = ScalarMappable(norm=norm, cmap="coolwarm")
sm.set_array([])
sm.set_clim(vmin, vmax)
cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)

# --- clean colorbar ticks: drop 10,000 if max is close ---
candidate_ticks = [1, 10, 100, 1_000, 10_000]

vmin_i = int(vmin)
vmax_i = int(vmax)

# keep only ticks inside range
ticks = [t for t in candidate_ticks if vmin_i <= t <= vmax_i]

# If 10,000 is close to vmax, remove it
if 10_000 in ticks:
    # "close" = within ~20% of vmax
    if abs(vmax_i - 10_000) / vmax_i < 0.20:
        ticks.remove(10_000)

# Always include the true max
if vmax_i not in ticks:
    ticks.append(vmax_i)

ticks = sorted(ticks)

cbar.set_ticks(ticks)
cbar.ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f"{int(x):,}")
)

cbar.minorticks_off()
cbar.ax.tick_params(labelsize=16, pad=6)

cbar.set_label("Phone Number Count (per country)", fontsize=20, fontweight="bold")
cbar.ax.tick_params(labelsize=16)

ax.axis("off")
plt.tight_layout(pad=0)
plt.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0)
plt.show()