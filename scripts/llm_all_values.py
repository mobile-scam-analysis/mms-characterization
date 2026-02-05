#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times", "DejaVu Serif", "Liberation Serif"]
rcParams["axes.unicode_minus"] = False

SCAM_PATH   = "merged_full_scam_gpt5_output.csv"

# EXTRA_GPT5_2025_SCAM_PATH = "merged_all_2025_combined_classified.csv"

# EXTRA_LLAMA3_MISTRAL_2025_SCAM_PATH = "merged_all_2025_combined_llama3_mistral_classified.csv"

# EXTRA_LLAMA4_SCOUT_2025_SCAM_PATH = "merged_all_2025_combined_scout_classified.csv"

# EXTRA_LLAMA4_MAVERICK_2025_SCAM_PATH = "merged_all_2025_combined_maverick_classified.csv"

BENIGN_PATH = "merged_benign.csv"
SAVE_PATH   = "llm_overall_by_model.pdf"

EXPECTED_RESULT_COLS = [
    "mistral_fraud_result",
    "llama_3_3_70b_instruct_result",
    "llama_4_scout_result",
    "gpt_5_result",               
]

DISPLAY_NAME = {
    "mistral_fraud_result": "Mistral Fraud",
    "llama_3_3_70b_instruct_result": "LLaMA 3.3",
    "llama_4_scout_result": "LLaMA 4\nScout",
    "gpt_5_result": "GPT-5",
}

PALETTE = {
    "TP": "#05badd",
    "FN": "#ff3902",
    "TN": "#ffb404",
    "FP": "#2b4871",
}
EDGE = "#333333"

scam_df   = pd.read_csv(SCAM_PATH)
benign_df = pd.read_csv(BENIGN_PATH)

def append_extra_scam_csv(scam_df: pd.DataFrame, extra_path: str, label: str) -> pd.DataFrame:
    try:
        extra_df = pd.read_csv(
            extra_path,
            engine="python",
            on_bad_lines="skip"
        )
        
        if "llama_3_result" in extra_df.columns and "llama_3_3_70b_instruct_result" not in extra_df.columns:
            extra_df = extra_df.rename(columns={"llama_3_result": "llama_3_3_70b_instruct_result"})

        extra_df["true_label"] = "scam"
        if "cluster_name" not in extra_df.columns:
            extra_df["cluster_name"] = "All"

        for c in extra_df.columns:
            if c not in scam_df.columns:
                scam_df[c] = np.nan

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

## Append all extra CSVs
# scam_df = append_extra_scam_csv(scam_df, EXTRA_GPT5_2025_SCAM_PATH, "GPT-5 (2025)")
# scam_df = append_extra_scam_csv(scam_df, EXTRA_LLAMA3_MISTRAL_2025_SCAM_PATH, "LLaMA 3.3 + Mistral Fraud (2025)")
# scam_df = append_extra_scam_csv(scam_df, EXTRA_LLAMA4_SCOUT_2025_SCAM_PATH, "LLaMA 4 Scout (2025)")

scam_df["true_label"]   = "scam"
benign_df["true_label"] = "not_scam"

if "cluster_name" not in scam_df.columns:   scam_df["cluster_name"]   = "All"
if "cluster_name" not in benign_df.columns: benign_df["cluster_name"] = "All"

df = pd.concat([scam_df, benign_df], ignore_index=True)

model_cols = [c for c in EXPECTED_RESULT_COLS if c in df.columns]
if not model_cols:
    raise SystemExit("No model result columns found in the input files.")

def norm_pred(v: str) -> str:
    s = str(v).strip().lower().replace(" ", "_").replace(".", "")
    if s == "scam": return "scam"
    if s in {"not_scam", "notscam"}: return "not_scam"
    return "unknown"

def outcome(true_label, pred_label):
    if true_label == "scam" and pred_label == "scam": return "TP"
    if true_label == "scam" and pred_label == "not_scam": return "FN"
    if true_label == "not_scam" and pred_label == "not_scam": return "TN"
    if true_label == "not_scam" and pred_label == "scam": return "FP"
    return "Unknown"

long = df.melt(
    id_vars=["true_label"],
    value_vars=model_cols,
    var_name="model_col",
    value_name="pred_raw"
)
long["model"] = long["model_col"].map(DISPLAY_NAME).fillna(long["model_col"])
long["predicted_label"] = long["pred_raw"].apply(norm_pred)
long["Outcome"] = long.apply(lambda r: outcome(r["true_label"], r["predicted_label"]), axis=1)
long = long[long["Outcome"] != "Unknown"]

grp = long.groupby(["model", "true_label", "Outcome"]).size().reset_index(name="count")
grp["percentage"] = grp["count"] / grp.groupby(["model", "true_label"])["count"].transform("sum") * 100

models = sorted(grp["model"].unique().tolist(), key=lambda m: (m != "Mistral Fraud", m))
outcome_order = ["TP", "FN", "TN", "FP"]
idx = pd.MultiIndex.from_product([models, outcome_order], names=["model", "Outcome"])
grp_pivot = grp.groupby(["model", "Outcome"])["percentage"].sum().reindex(idx, fill_value=0).reset_index()

BAR_W   = 0.22    
INTRA   = 0.00    
INTER   = 0.2    

fig_w = max(5.5, 2.2 + 1.2 * len(models))
fig, ax = plt.subplots(figsize=(6, 4))

k = len(outcome_order)
group_span   = k * BAR_W + (k - 1) * INTRA
group_stride = group_span + INTER
x = np.arange(len(models)) * group_stride

offset_units = np.arange(k) - (k - 1) / 2.0
bar_offsets  = offset_units * (BAR_W + INTRA)

for j, oc in enumerate(outcome_order):
    vals = grp_pivot.loc[grp_pivot["Outcome"] == oc, "percentage"].values
    bars = ax.bar(
        x + bar_offsets[j],
        vals,
        width=BAR_W,
        color=PALETTE[oc],
        edgecolor=EDGE,
        linewidth=0.6,
        label=oc,
    )
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100
    for b, v in zip(bars, vals):
        if v > 0.01:
            pad = max(0.8, 0.008 * ymax) + 0.01 * v
            y = min(v + pad, ymax - 0.6)
            ax.text(b.get_x() + b.get_width()/2, y, f"{v:.1f}%",
                    ha="center", va="bottom", rotation=90, fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
ax.set_xlabel("Model", fontsize=12, fontweight="bold")
ax.set_ylim(0, 100)

ax.grid(True, axis="y", linestyle="--", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

legend_handles = [Line2D([0],[0], color=PALETTE[o], lw=6, label=o) for o in outcome_order]
fig.legend(legend_handles, outcome_order, title="Outcome",
           title_fontsize=8, fontsize=6, loc="center right",
           bbox_to_anchor=(0.95, 0.82), frameon=False)

fig.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.10)
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.show()