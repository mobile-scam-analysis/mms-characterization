#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter  
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times", "DejaVu Serif", "Liberation Serif"]
rcParams["axes.unicode_minus"] = False

# Change path to data CSV
SCAM_PATH = "llm_data.csv"

EXTRA_GPT5_2025_SCAM_PATH = "merged_all_2025_combined_classified.csv"
EXTRA_LLAMA3_MISTRAL_2025_SCAM_PATH = "merged_all_2025_combined_llama3_mistral_classified.csv"
EXTRA_LLAMA4_SCOUT_2025_SCAM_PATH = "merged_all_2025_combined_scout_classified.csv"

CSV_PATH = "data.csv"


NAME_MAP = {
    "Wrong Number": "Wrong\nNumber",
    "Account Verification/Payment": "Account Verification/\nPayment",
    "Gift/Prize": "Gift/\nPrize",
    "Fake Job": "Fake\nJob",
    "Toll/DMV": "Toll/\nDMV",
}

REPLY_BASED = [
    "Fake\nJob",
    "Romance",
    "Wrong\nNumber",
]
CLICK_BASED = [
    "Account Verification/\nPayment",
    "E-Commerce",
    "Toll/\nDMV",
    "Postal",
    "Gift/\nPrize",
    "Investment/\nCryptocurrency",
]


CLICK_COLOR_BY_CAT = {
    "Postal": "#C6EBBE",                        
    "Toll/\nDMV": "#2b4871",                   
    "Gift/\nPrize": "#ff00a8",                 
    "E-Commerce": "#05badd",                    
    "Account Verification/\nPayment": "#ff3902",
}

REPLY_COLOR_BY_CAT = {
    "Wrong\nNumber": "#f2d14b",  
    "Romance": "#19a7a8",       
    "Fake\nJob": "#b455b6",     
}

REPLY_HATCH = "xx"

cluster_names_clean = {
    0: "Romance",
    1: "Account Verification/Payment",
    2: "Fake Job",
    4: "Account Verification/Payment",
    8: "Postal",
    9: "Wrong Number",
    12: "E-Commerce",
    14: "Gift/Prize",
    18: "Toll/DMV",
}

MODEL_CANON = {
    "mistral": "mistral_fraud_result",
    "llama3": "llama_3_3_70b_instruct_result",
    "llama4": "llama_4_scout_result",
    "gpt5": "gpt_5_result",
}

MODEL_ALIASES = {
    "mistral_fraud_result": ["mistral_fraud_result"],
    "llama_3_3_70b_instruct_result": ["llama_3_3_70b_instruct_result", "llama_3_result"],
    "llama_4_scout_result": ["llama_4_scout_result"],
    "gpt_5_result": ["gpt_5_result"],
}

DISPLAY_INDEX_RENAME = {
    "mistral_fraud_result": "Mistral Fraud",
    "llama_3_3_70b_instruct_result": "LLaMA 3.3",
    "llama_4_scout_result": "LLaMA 4\nScout",
    "gpt_5_result": "GPT-5",
}


def _first_existing_col(df: pd.DataFrame, options: list[str]) -> str | None:
    for c in options:
        if c in df.columns:
            return c
    return None


def _ensure_canonical_model_cols(df: pd.DataFrame) -> pd.DataFrame:
    for canon, aliases in MODEL_ALIASES.items():
        if canon in df.columns:
            continue
        alt = _first_existing_col(df, aliases)
        if alt and alt != canon:
            df[canon] = df[alt]
    return df


def append_extra_scam_csv(scam_df: pd.DataFrame, extra_path: str, label: str) -> pd.DataFrame:
    try:
        extra_df = pd.read_csv(extra_path, engine="python", on_bad_lines="skip")
        extra_df = _ensure_canonical_model_cols(extra_df)

        if "true_label" not in extra_df.columns:
            extra_df["true_label"] = "scam"

        if "cluster_name" not in extra_df.columns:
            extra_df["cluster_name"] = "All"

        for c in scam_df.columns:
            if c not in extra_df.columns:
                extra_df[c] = np.nan

        extra_df = extra_df[scam_df.columns]

        scam_df = pd.concat([scam_df, extra_df], ignore_index=True)
        print(f"[info] Appended extra {label} scam rows: {len(extra_df):,}")
        return scam_df

    except FileNotFoundError:
        print(f"[warn] Extra {label} CSV not found: {extra_path} (skipping)")
        return scam_df
    except Exception as e:
        print(f"[warn] Could not append extra {label} CSV ({extra_path}): {e} (skipping)")
        return scam_df


def merge_cluster_by_image(scam_df: pd.DataFrame) -> pd.DataFrame:
    rel = pd.read_csv(CSV_PATH, low_memory=False)

    rel = rel[["Image Filename", "cluster"]].dropna(subset=["cluster"])
    rel["cluster"] = pd.to_numeric(rel["cluster"], errors="coerce")

    scam_df = scam_df.drop(
        columns=[c for c in ["cluster", "cluster_name"] if c in scam_df.columns],
        errors="ignore"
    )

    merged = scam_df.merge(rel, on="Image Filename", how="left")
    merged["cluster_name"] = merged["cluster"].map(cluster_names_clean)
    return merged


def normalize_cluster_name_series(s: pd.Series) -> pd.Series:
    def normalize_one(c):
        if pd.isna(c):
            return c
        try:
            c_int = int(c)
            return cluster_names_clean.get(c_int, c)
        except (ValueError, TypeError):
            return c
    return s.apply(normalize_one)


def norm_result_label(v) -> str:
    s = str(v).strip().lower().replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    if s in {"not scam", "not a scam", "benign", "ham", "legit", "legitimate"}:
        return "NOT SCAM"
    if s in {"scam", "fraud"}:
        return "SCAM"
    return str(v)


def visualize_fn_by_model_proportions_extended_line():
    scam_df = pd.read_csv(SCAM_PATH)
    scam_df = _ensure_canonical_model_cols(scam_df)

    scam_df = append_extra_scam_csv(scam_df, EXTRA_GPT5_2025_SCAM_PATH, "GPT-5 (2025)")
    scam_df = append_extra_scam_csv(scam_df, EXTRA_LLAMA3_MISTRAL_2025_SCAM_PATH, "LLaMA 3.3 + Mistral Fraud (2025)")
    scam_df = append_extra_scam_csv(scam_df, EXTRA_LLAMA4_SCOUT_2025_SCAM_PATH, "LLaMA 4 Scout (2025)")

    scam_df = scam_df.drop(columns=[c for c in ["cluster", "cluster_name"] if c in scam_df.columns], errors="ignore")

    scam_df = merge_cluster_by_image(scam_df)
    scam_df = scam_df[scam_df["cluster"] != 4]

    scam_df["cluster_name"] = normalize_cluster_name_series(scam_df["cluster_name"])
    scam_df["cluster_name"] = scam_df["cluster_name"].map(NAME_MAP).fillna(scam_df["cluster_name"])

    expected_cols = [
        MODEL_CANON["mistral"],
        MODEL_CANON["llama3"],
        MODEL_CANON["llama4"],
        MODEL_CANON["gpt5"],
    ]
    model_cols_present = [c for c in expected_cols if c in scam_df.columns]
    if not model_cols_present:
        raise SystemExit("No model result columns found in the scam data after appends.")

    for c in model_cols_present:
        scam_df[c] = scam_df[c].apply(norm_result_label)

    long_df = scam_df.melt(
        id_vars=["cluster_name"],
        value_vars=model_cols_present,
        var_name="model_name",
        value_name="result",
    )

    fn_df = long_df[long_df["result"] == "NOT SCAM"]
    if fn_df.empty:
        print("No 'NOT SCAM' results found to plot.")
        return

    pivot_df = fn_df.pivot_table(
        index="model_name",
        columns="cluster_name",
        aggfunc="size",
        fill_value=0,
    )

    pivot_df = pivot_df.rename(index=DISPLAY_INDEX_RENAME)

    desired_first = "Mistral Fraud"

    current_order = pivot_df.index.tolist()

    if desired_first in current_order:
        new_order = (
            [desired_first]
            + [m for m in current_order if m != desired_first]
        )
        pivot_df = pivot_df.reindex(new_order)

    overall_total_counts = scam_df["cluster_name"].value_counts()
    sorted_categories = overall_total_counts.index.tolist()

    pivot_df.loc["Scam Data\nDistribution"] = overall_total_counts
    pivot_df = pivot_df.fillna(0)

    final_sorted_columns = [col for col in sorted_categories if col in pivot_df.columns]
    extra_cols = [col for col in pivot_df.columns if col not in final_sorted_columns]
    pivot_df = pivot_df[final_sorted_columns + extra_cols]

    prevalence_sorted_cols = pivot_df.columns.tolist()
    reply_cols_present = [c for c in prevalence_sorted_cols if c in REPLY_BASED]
    click_cols_present = [c for c in prevalence_sorted_cols if c in CLICK_BASED]
    other_cols_present = [c for c in prevalence_sorted_cols if c not in reply_cols_present and c not in click_cols_present]

    pivot_df = pivot_df[
        REPLY_BASED
        + click_cols_present
        + other_cols_present
    ]

    row_totals = pivot_df.sum(axis=1)
    prop_df = pivot_df.div(row_totals, axis=0).fillna(0)

    color_map = {}
    hatch_map = {}

    for cat in REPLY_BASED:
        if cat in prop_df.columns:
            color_map[cat] = REPLY_COLOR_BY_CAT[cat]
            hatch_map[cat] = REPLY_HATCH

    for cat in CLICK_BASED:
        if cat in prop_df.columns:
            color_map[cat] = CLICK_COLOR_BY_CAT.get(cat, "#999999")
            hatch_map[cat] = None

    other_cats_present = [c for c in prop_df.columns if c not in REPLY_BASED and c not in CLICK_BASED]
    if other_cats_present:
        cmap_other = plt.colormaps.get_cmap("tab20")
        colors_other = cmap_other(np.linspace(0, 1, min(len(other_cats_present), 20)))
        for i, cat in enumerate(other_cats_present):
            color_map[cat] = colors_other[i % 20]
            hatch_map[cat] = None

    categories_ordered = prop_df.columns.tolist()
    color_list = [color_map.get(cat, "gray") for cat in categories_ordered]
    hatch_list = [hatch_map.get(cat, None) for cat in categories_ordered]

    fig, ax = plt.subplots(figsize=(6, 4))

    prop_df.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=color_list,
        legend=False,
        alpha=0.95,
    )

    for patch in ax.patches:
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)

    for container, hatch in zip(ax.containers, hatch_list):
        if hatch:
            for bar in container.patches:
                bar.set_hatch(hatch)

    ax.set_ylabel(
        "Proportion",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(None)

    ax.set_xticks(range(len(prop_df.index)))
    ax.set_xticklabels([])

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_ylim(0, 1.0)

    handles, labels = ax.get_legend_handles_labels()
    modified_labels = [f"{label} *" if label in reply_cols_present else label for label in labels]

    legend_map = {
        original_label: (handle, modified_label)
        for original_label, handle, modified_label in zip(labels, handles, modified_labels)
    }

    custom_legend_order = (
        click_cols_present
        + REPLY_BASED
        + other_cats_present
    )

    final_handles = []
    final_labels = []
    for label in custom_legend_order:
        if label in legend_map:
            handle, mod_label = legend_map[label]
            final_handles.append(handle)
            final_labels.append(mod_label)

    num_click_items = len(click_cols_present)
    mid_point = (num_click_items + 1) // 2

    click_handles_1 = final_handles[:mid_point]
    click_labels_1 = final_labels[:mid_point]

    click_handles_2 = final_handles[mid_point:num_click_items]
    click_labels_2 = final_labels[mid_point:num_click_items]

    reply_other_handles = final_handles[num_click_items:]
    reply_other_labels = final_labels[num_click_items:]

    interleaved_handles = []
    interleaved_labels = []
    max_rows = max(len(click_handles_1), len(click_handles_2), len(reply_other_handles))

    for i in range(max_rows):
        if i < len(click_handles_1):
            interleaved_handles.append(click_handles_1[i])
            interleaved_labels.append(click_labels_1[i])
        if i < len(click_handles_2):
            interleaved_handles.append(click_handles_2[i])
            interleaved_labels.append(click_labels_2[i])
        if i < len(reply_other_handles):
            interleaved_handles.append(reply_other_handles[i])
            interleaved_labels.append(reply_other_labels[i])

    fig.legend(
        handles=interleaved_handles,
        labels=interleaved_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),  
        ncol=5,
        frameon=False,
        fontsize=9,
        handletextpad=0.35,
        columnspacing=0.9,
        labelspacing=0.4,
    )

    ax.axvline(
        x=3.5,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        ymax=1,
        ymin=-0,
        clip_on=False,
    )

    bar_labels = list(prop_df.index)
    individual_label_y_pos = -0.06  
    for i, label in enumerate(bar_labels):
        display_label = label
        if label == "Mistral Fraud":
            display_label = "Mistral\nFraud"
        elif label == "Scam Data\nDistribution":
            display_label = "Full Dataset\nDistribution"

        ax.text(
            i,
            individual_label_y_pos,
            display_label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#333333",
        )

    ax.text(
        1.5,
        -0.26,  
        "Model",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#333333",
    )

    fig.text(
        0.5,
        0.01,  
        "* = Reply-based scam",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )

    fig.subplots_adjust(
        left=0.14,   
        right=0.98,
        top=0.97,
        bottom=0.40
    )
    output_filename = "fn_distribution.pdf"
    fig.savefig(output_filename, bbox_inches="tight", pad_inches=0.18, dpi=300) 
    print(f"\nChart saved to {output_filename}")

    plt.close(fig)

    print("\n--- Summary Data (Proportions) ---")
    print(prop_df)

    print("\n--- Raw Counts (for verification) ---")
    print(pivot_df)


if __name__ == "__main__":
    visualize_fn_by_model_proportions_extended_line()