import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm 
import matplotlib as mpl  
from matplotlib import rcParams
import re
from dateutil import parser as du_parser

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times", "DejaVu Serif", "Liberation Serif"]
rcParams["axes.unicode_minus"] = False

_EPOCH_RE = re.compile(r"\b(\d{10,13})\b")

_KNOWN_FORMATS = [
    "%m/%d/%y %H:%M",        
    "%m/%d/%y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",    
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
]

def _parse_one_ts(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return pd.NaT
    s = str(s).strip()
    if not s:
        return pd.NaT

    m = _EPOCH_RE.search(s)
    if m:
        v = int(m.group(1))
        unit = "ms" if v >= 10**12 else "s"
        return pd.to_datetime(v, unit=unit, utc=True, errors="coerce")

    for fmt in _KNOWN_FORMATS:
        try:
            dt = pd.to_datetime(s, format=fmt, errors="raise")
            # localize naive timestamps to UTC
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
            else:
                dt = dt.tz_convert("UTC")
            return dt
        except Exception:
            pass

    try:
        dt = du_parser.parse(s, fuzzy=True)
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    except Exception:
        return pd.NaT


def _latest_valid_ts(cell):
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return pd.NaT
    parts = [p.strip() for p in str(cell).split(";") if p.strip()]
    parsed = [_parse_one_ts(p) for p in parts]
    parsed = [p for p in parsed if not pd.isna(p)]
    return max(parsed) if parsed else pd.NaT




# --- Configuration ---
CATEGORY_COLUMN = 'cluster_name'
TIMESTAMP_COLUMN = 'timestamp'
COLOR_PALETTE_CLICK = ['#ff3902', '#05badd', '#ffb404', '#2b4871', '#C6EBBE']
REPLY_COLOR_OVERRIDE = '#c96cc7'

NAME_MAP = {
    "Fake Job": "Fake Job *",
    "Romance": "Romance *",
    "Wrong Number": "Wrong\nNumber *",
    "Account Verification/Payment": "Account Verification/\nPayment",
    "Gift/Prize": "Gift/\nPrize",
    "Toll/DMV": "Toll/DMV",
    "Postal": "Postal",
}

REPLY_BASED = [
    "Fake Job *",
    "Romance *",
    "Wrong\nNumber *",
]
CLICK_BASED = [
    "Account Verification/\nPayment",
    "E-Commerce",
    "Toll/DMV",
    "Postal",
    "Gift/\nPrize"
]

CLICK_BASED_FOR_AVG = CLICK_BASED
CLICK_BASED_FOR_AVG_EXCL_POSTAL = [
    "Account Verification/\nPayment",
    "E-Commerce",
    "Gift/\nPrize",
]

def calculate_and_save_group_cagr(plot_df, output_file):
    print("\n--- Calculating Compounded Annual Growth Rate (CAGR) (Excluding 2020) ---")

    available_years = plot_df.index.to_list()
    new_start_year = next((year for year in available_years if year >= 2021), None)

    if new_start_year is None:
        print("CAGR requires data starting from at least 2021 to exclude 2020.")
        return

    end_year = plot_df.index.max()
    N = end_year - new_start_year

    if N <= 0:
        print(f"CAGR requires at least two years of data after {new_start_year}.")
        return

    all_cols_present = plot_df.columns.tolist()

    reply_cols = [c for c in all_cols_present if c in REPLY_BASED]
    click_cols_incl = [c for c in all_cols_present if c in CLICK_BASED]
    click_cols_excl = [c for c in all_cols_present if c in CLICK_BASED_FOR_AVG_EXCL_POSTAL]
    category_cols = all_cols_present

    cagr_results = {}

    def calculate_group_cagr(cols):
        if not cols:
            return np.nan

        is_single_category = isinstance(cols, str) or (isinstance(cols, list) and len(cols) == 1)

        if is_single_category:
            col_name = cols if isinstance(cols, str) else cols[0]
            start_count = plot_df.loc[new_start_year, col_name]
            end_count = plot_df.loc[end_year, col_name]
        else:
            start_count = plot_df.loc[new_start_year, cols].sum()
            end_count = plot_df.loc[end_year, cols].sum()

        if start_count == 0:
            return np.inf if end_count > 0 else 0.0

        cagr = ((end_count / start_count) ** (1 / N) - 1) * 100
        return cagr

    cagr_results['Reply-Based CAGR'] = calculate_group_cagr(reply_cols)
    cagr_results['Click-Based CAGR (Incl. Postal)'] = calculate_group_cagr(click_cols_incl)
    cagr_results['Click-Based CAGR (Excl. Postal)'] = calculate_group_cagr(click_cols_excl)

    category_cagrs = {f'{col} CAGR': calculate_group_cagr(col) for col in category_cols}
    combined_results = {**cagr_results, **category_cagrs}

    cagr_df = pd.Series(combined_results).to_frame(f'CAGR {new_start_year}-{end_year}')
    cagr_df.columns.name = 'Group/Category'

    def format_cagr(x):
        if x == np.inf:
            return 'inf%'
        if pd.isna(x):
            return 'N/A'
        return f'{x:.2f}%'

    cagr_formatted = cagr_df.applymap(format_cagr)

    try:
        with open(output_file, 'a') as f:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("Compounded Annual Growth Rate (CAGR) (Excluding 2020)\n")
            f.write(f"Period used: {new_start_year} to {end_year} ({N} growth periods)\n")
            f.write("-" * 80 + "\n")
            f.write(cagr_formatted.to_string())
        print(f"Overall and Category CAGR (Excl. 2020) appended to {output_file}")
    except Exception as e:
        print(f"Error saving CAGR data to file: {e}")

def calculate_and_print_group_growth(plot_df):
    print("\n--- Calculating OVERALL SIMPLE AVERAGE Yearly Growth by Scam Type ---")

    growth_df = plot_df.pct_change(axis=0) * 100
    avg_category_growth = growth_df.replace([np.inf, -np.inf], np.nan).mean(axis=0)

    reply_cols_present = [c for c in avg_category_growth.index if c in REPLY_BASED]
    click_cols_incl_postal = [c for c in avg_category_growth.index if c in CLICK_BASED]
    click_cols_excl_postal = [c for c in avg_category_growth.index if c in CLICK_BASED_FOR_AVG_EXCL_POSTAL]

    if reply_cols_present:
        reply_group_avg = avg_category_growth[reply_cols_present].mean()
        print(f"Reply-Based Scams Overall Simple Average Growth: {reply_group_avg:.2f}%")
    else:
        print("Reply-Based Scams: No data present.")

    if click_cols_incl_postal:
        click_group_avg_incl = avg_category_growth[click_cols_incl_postal].mean()
        print(f"Click-Based Scams Overall Simple Average Growth (Incl. Postal): {click_group_avg_incl:.2f}%")
    else:
        print("Click-Based Scams (Incl. Postal): No data present.")

    if click_cols_excl_postal:
        click_group_avg_excl = avg_category_growth[click_cols_excl_postal].mean()
        print(f"Click-Based Scams Overall Simple Average Growth (Excl. Postal): {click_group_avg_excl:.2f}%")
    else:
        print("Click-Based Scams (Excl. Postal): No data present.")

    print("-" * 60)

def calculate_and_save_yearly_group_growth(plot_df, output_file):
    print("\n--- Calculating Yearly Simple Average Growth by Scam Type ---")

    growth_df = plot_df.pct_change(axis=0) * 100
    growth_df_clean = growth_df.replace([np.inf, -np.inf], np.nan)

    all_cols = growth_df_clean.columns.tolist()
    reply_cols = [c for c in all_cols if c in REPLY_BASED]
    click_cols_incl = [c for c in all_cols if c in CLICK_BASED]
    click_cols_excl = [c for c in all_cols if c in CLICK_BASED_FOR_AVG_EXCL_POSTAL]

    reply_yearly_avg = growth_df_clean[reply_cols].mean(axis=1) if reply_cols else pd.Series(np.nan, index=growth_df_clean.index)
    click_yearly_avg_incl = growth_df_clean[click_cols_incl].mean(axis=1) if click_cols_incl else pd.Series(np.nan, index=growth_df_clean.index)
    click_yearly_avg_excl = growth_df_clean[click_cols_excl].mean(axis=1) if click_cols_excl else pd.Series(np.nan, index=growth_df_clean.index)

    group_summary_df = pd.DataFrame({
        'Reply-Based Avg': reply_yearly_avg,
        'Click-Based Avg (Incl. Postal)': click_yearly_avg_incl,
        'Click-Based Avg (Excl. Postal)': click_yearly_avg_excl
    })
    group_summary_df.index.name = 'Year'

    def format_group_growth(x):
        return f'{x:.2f}%' if pd.notna(x) else 'N/A'

    group_summary_formatted = group_summary_df.applymap(format_group_growth)

    if group_summary_formatted.index.min() == 2021:
        na_row = pd.Series('N/A', index=group_summary_formatted.columns, name=2020)
        group_summary_formatted = pd.concat([pd.DataFrame([na_row]), group_summary_formatted]).sort_index()

    try:
        with open(output_file, 'a') as f:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("Yearly Simple Average Growth (%) by Scam Type\n")
            f.write("Note: This table shows the simple arithmetic mean of category growth within each group, per year.\n")
            f.write("-" * 80 + "\n")
            f.write(group_summary_formatted.to_string())
        print(f"Yearly group simple average growth appended to {output_file}")
    except Exception as e:
        print(f"Error saving yearly group growth data to file: {e}")

def calculate_yearly_growth(plot_df, output_file="category_yearly_growth_new.txt"):
    print("\n--- Calculating Individual Category Yearly Growth % ---")

    growth_df = plot_df.pct_change(axis=0) * 100

    def format_growth(x):
        if pd.isna(x):
            return 'N/A'
        elif x == np.inf:
            return 'inf%'
        else:
            return f'{x:.2f}%'

    growth_df_formatted = growth_df.applymap(format_growth)

    if plot_df.index.min() == 2021:
        na_row = pd.Series('N/A', index=growth_df_formatted.columns, name=2020)
        growth_df_formatted = pd.concat([pd.DataFrame([na_row]), growth_df_formatted]).sort_index()

    growth_df_formatted.index = growth_df_formatted.index.astype(str)
    growth_df_formatted.index.name = "Year"

    try:
        with open(output_file, 'w') as f:
            f.write("Yearly Growth Percentage (%) of Scam Categories\n")
            f.write("Note: 'inf%' indicates growth from zero counts in the previous year.\n")
            f.write("-" * 80 + "\n")
            f.write(growth_df_formatted.to_string())
        print(f"Individual category yearly growth percentages saved to {output_file}")
    except Exception as e:
        print(f"Error saving growth data to file: {e}")

def calculate_cagr(plot_df: pd.DataFrame, cols, start_year: int, end_year: int):
    if isinstance(cols, str):
        cols = [cols]

    if start_year not in plot_df.index or end_year not in plot_df.index:
        return np.nan

    start_count = plot_df.loc[start_year, cols].sum()
    end_count = plot_df.loc[end_year, cols].sum()

    N = end_year - start_year
    if N <= 0:
        return np.nan

    if start_count == 0:
        return np.inf if end_count > 0 else 0.0

    cagr = (end_count / start_count) ** (1 / N) - 1
    return cagr * 100

def write_explicit_cagr_to_file(plot_df, output_file):
    years = sorted(plot_df.index.tolist())
    start_year = next((y for y in years if y >= 2021), None)
    end_year = years[-1] if years else None

    if start_year is None or end_year is None or end_year <= start_year:
        print("Not enough data to compute explicit CAGR.")
        return

    lines = []
    lines.append("\n\n" + "=" * 80)
    lines.append("CAGR (Explicit Helper Calculation)")
    lines.append(f"Period: {start_year} → {end_year}")
    lines.append("-" * 80)

    reply_cagr = calculate_cagr(plot_df, REPLY_BASED, start_year, end_year)
    click_cagr_incl = calculate_cagr(plot_df, CLICK_BASED, start_year, end_year)
    click_cagr_excl = calculate_cagr(plot_df, CLICK_BASED_FOR_AVG_EXCL_POSTAL, start_year, end_year)

    def fmt(x):
        if np.isinf(x):
            return "inf%"
        if pd.isna(x):
            return "N/A"
        return f"{x:.2f}%"

    lines.append(f"Reply-Based CAGR: {fmt(reply_cagr)}")
    lines.append(f"Click-Based CAGR (Incl. Postal): {fmt(click_cagr_incl)}")
    lines.append(f"Click-Based CAGR (Excl. Postal): {fmt(click_cagr_excl)}")
    lines.append("")

    for col in plot_df.columns:
        cagr = calculate_cagr(plot_df, col, start_year, end_year)
        lines.append(f"{col} CAGR: {fmt(cagr)}")

    with open(output_file, "a") as f:
        f.write("\n".join(lines))

    print("Explicit CAGR section appended to output file.")

def compute_reply_vs_click_stats(df, output_file=None):
    total_msgs = len(df)

    reply_count = df[CATEGORY_COLUMN].isin(REPLY_BASED).sum()
    click_count = df[CATEGORY_COLUMN].isin(CLICK_BASED).sum()

    other_count = total_msgs - (reply_count + click_count)

    reply_pct = (reply_count / total_msgs) * 100 if total_msgs else 0
    click_pct = (click_count / total_msgs) * 100 if total_msgs else 0
    other_pct = (other_count / total_msgs) * 100 if total_msgs else 0

    print("\n--- Reply vs Click-Based (Percent of Whole Dataset) ---")
    print(f"Total messages: {total_msgs:,}")
    print(f"Reply-based: {reply_count:,} ({reply_pct:.2f}%)")
    print(f"Click-based: {click_count:,} ({click_pct:.2f}%)")
    print(f"Other/Uncategorized: {other_count:,} ({other_pct:.2f}%)")

    if output_file:
        with open(output_file, "a") as f:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("Reply vs Click-Based (Percent of Whole Dataset)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total messages: {total_msgs:,}\n")
            f.write(f"Reply-based: {reply_count:,} ({reply_pct:.2f}%)\n")
            f.write(f"Click-based: {click_count:,} ({click_pct:.2f}%)\n")
            f.write(f"Other/Uncategorized: {other_count:,} ({other_pct:.2f}%)\n")

    return {
        "total": total_msgs,
        "reply_count": reply_count,
        "reply_pct": reply_pct,
        "click_count": click_count,
        "click_pct": click_pct,
        "other_count": other_count,
        "other_pct": other_pct,
    }

def write_category_counts_by_year_to_file(plot_df, output_file):
    try:
        with open(output_file, "a") as f:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("Counts of Messages per Scam Category by Year\n")
            f.write("-" * 80 + "\n")
            f.write(plot_df.to_string())
        print("Category-by-year counts appended to output file.")
    except Exception as e:
        print(f"Error writing category-by-year counts: {e}")

def visualize_categories_by_year(csv_file_path):
    try:
        df = pd.read_csv(
            csv_file_path,
            engine="python",     
            on_bad_lines="skip",  
        )
        df = df.loc[:, ~df.columns.duplicated()].copy()

        print(f"Successfully loaded {csv_file_path} with {len(df)} rows.")
    except Exception as e:
        print(f"Error reading '{csv_file_path}': {e}")
        raise  

    required_cols = [TIMESTAMP_COLUMN, CATEGORY_COLUMN]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: The CSV must contain '{TIMESTAMP_COLUMN}' and '{CATEGORY_COLUMN}' columns.")
        return

    print(f"Converting '{TIMESTAMP_COLUMN}' to 'Year'...")

    try:
        df["datetime"] = df[TIMESTAMP_COLUMN].apply(_latest_valid_ts)

        df["Year"] = df["datetime"].dt.year
        df.dropna(subset=["Year"], inplace=True)
        df["Year"] = df["Year"].astype(int)
    except Exception as e:
        print(f"FATAL ERROR converting datetime strings: {e}.")
        return

    if NAME_MAP:
        df[CATEGORY_COLUMN] = df[CATEGORY_COLUMN].map(NAME_MAP).fillna(df[CATEGORY_COLUMN])

    print("Pivoting data to aggregate counts by Year and Category...")
    try:
        if df.empty:
            return

        plot_df = df.pivot_table(index='Year', columns=CATEGORY_COLUMN, aggfunc='size', fill_value=0)

        all_categories_present = plot_df.columns.tolist()
        reply_cols = sorted([c for c in all_categories_present if c in REPLY_BASED])
        click_cols = sorted([c for c in all_categories_present if c in CLICK_BASED])
        other_cols = sorted([c for c in all_categories_present if c not in REPLY_BASED and c not in CLICK_BASED])

        final_category_order = reply_cols + click_cols + other_cols
        plot_df = plot_df.reindex(columns=final_category_order, fill_value=0).sort_index()

        TOLL_COL = NAME_MAP.get("Toll/DMV", "Toll/DMV")
        POSTAL_COL = NAME_MAP.get("Postal", "Postal")

        if TOLL_COL in plot_df.columns:
            plot_df.loc[plot_df.index < 2024, TOLL_COL] = 0

        if POSTAL_COL in plot_df.columns:
            plot_df.loc[plot_df.index < 2023, POSTAL_COL] = 0
    except Exception as e:
        print(f"Error pivoting the table: {e}")
        return

    output_growth_file = "category_yearly_growth.txt"
    if not plot_df.empty:
        row_totals = plot_df.sum(axis=1)
        proportion_df = plot_df.div(row_totals, axis=0)
    else:
        proportion_df = pd.DataFrame()

    calculate_yearly_growth(plot_df, output_growth_file)
    calculate_and_save_yearly_group_growth(plot_df, output_growth_file)
    calculate_and_save_group_cagr(plot_df, output_growth_file)
    write_explicit_cagr_to_file(plot_df, output_growth_file)
    compute_reply_vs_click_stats(df, output_growth_file)
    write_category_counts_by_year_to_file(plot_df, output_growth_file)
    calculate_and_print_group_growth(plot_df)

    if proportion_df.empty:
        print("Cannot plot: Proportional DataFrame is empty.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    all_categories = proportion_df.columns.tolist()
    color_map = {}

    reply_cats_present = [c for c in all_categories if c in REPLY_BASED]
    click_cats_present = [c for c in all_categories if c in CLICK_BASED]
    other_cats_present = [c for c in all_categories if c not in REPLY_BASED and c not in CLICK_BASED]

    n_reply = len(reply_cats_present)
    if n_reply > 0:
        cmap_reply = cm.get_cmap("viridis")
        colors_reply_rgba = cmap_reply(np.linspace(0.2, 1, n_reply))
        colors_reply_hex = [mpl.colors.rgb2hex(c) for c in colors_reply_rgba]
        colors_reply_hex[0] = REPLY_COLOR_OVERRIDE
        for i, cat in enumerate(reply_cats_present):
            color_map[cat] = colors_reply_hex[i]

    n_click = len(click_cats_present)
    if n_click > 0:
        click_palette = (COLOR_PALETTE_CLICK * ((n_click + len(COLOR_PALETTE_CLICK) - 1) // len(COLOR_PALETTE_CLICK)))[:n_click]
        for i, cat in enumerate(click_cats_present):
            color_map[cat] = click_palette[i]

    n_other = len(other_cats_present)
    if n_other > 0:
        cmap_other = cm.get_cmap("Pastel1")
        colors_other = [mpl.colors.rgb2hex(cmap_other(i)) for i in np.linspace(0.1, 0.9, n_other)]
        for i, cat in enumerate(other_cats_present):
            color_map[cat] = colors_other[i]


    FIXED_COLOR_MAP = {
        "Postal": "#7fc97f",                       
        "Toll/DMV": "#2b4871",                      
        "Gift/\nPrize": "#f0027f",                 
        "E-Commerce": "#05badd",                   
        "Account Verification/\nPayment": "#ff3902", 

        "Wrong\nNumber *": "#f2d14b",               
        "Romance *": "#19a7a8",                     
        "Fake Job *": "#c96cc7",                    
    }
    for k, v in FIXED_COLOR_MAP.items():
        if k in proportion_df.columns:
            color_map[k] = v

    hatch_map = {}
    for cat in proportion_df.columns:
        hatch_map[cat] = 'xx' if cat in REPLY_BASED else None

    color_list = [color_map.get(cat, 'gray') for cat in proportion_df.columns]
    hatch_list = [hatch_map[cat] for cat in proportion_df.columns]

    proportion_df.plot(kind='bar', stacked=True, ax=ax, color=color_list, edgecolor='black', width=0.8, rot=0)

    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    for container, hatch in zip(ax.containers, hatch_list):
        if hatch:
            for bar in container.patches:
                bar.set_edgecolor('black')
                bar.set_linewidth(0.5)
                bar.set_hatch(hatch)

    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Proportion of Total Count (Per Year)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0))

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    TICK_FONT_SIZE = 12
    ax.tick_params(axis='x', labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

    fig.text(
        0.5,
        0.015,
        "* = Reply-based scam",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#333333"
    )

    handles, labels = ax.get_legend_handles_labels()

    seen = set()
    dedup_handles, dedup_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            dedup_handles.append(h)
            dedup_labels.append(l)
            seen.add(l)

    handles, labels = dedup_handles, dedup_labels

    reply_labels = [l for l in labels if l in REPLY_BASED]
    click_labels = [l for l in labels if l in CLICK_BASED]
    other_labels = [l for l in labels if (l not in REPLY_BASED and l not in CLICK_BASED)]
    desired_order = reply_labels + click_labels + other_labels

    handle_by_label = {l: h for h, l in zip(handles, labels)}
    handles_ordered = [handle_by_label[l] for l in desired_order]

    fig.legend(
        handles=handles_ordered,
        labels=desired_order,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4, 
        frameon=False,
        fontsize=11,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.4,
    )

    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.22)

    output_filename = "yearly_proportional_stacked_bar_chart_final.pdf"
    fig.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"\nChart saved to {output_filename}")

    plt.close(fig)

if __name__ == "__main__":
    input_file = 'all_clusters_relabelled_with_cluster_name.csv'
    print("\n--- Running the Final Script ---")
    visualize_categories_by_year(input_file)