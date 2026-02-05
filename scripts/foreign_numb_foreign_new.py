import pandas as pd
import phonenumbers
import torch
from phonenumbers import region_code_for_number, is_possible_number, is_valid_number
import re
# from collections import defaultdict, Counter
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sentence_transformers import SentenceTransformer
from matplotlib import rcParams


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "Liberation Serif"]
})

if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon (MPS) backend.")
else:
    device = "cpu"
    print("MPS not available. Using CPU.")

# Add path to CSV
df = pd.read_csv("data.csv")
cluster_names_clean = {
    0: "Romance",
    1: "Account Verification/Payment",
    2: "Fake Job",
    4: "Account Verification/Payment",
    8: "Postal",
    9: "Wrong Number",
    12: "E-Commerce",
    14: "Gift/Prize",
    18: "Toll/DMV"
}
cluster_display_names = {
    "Wrong Number": "Wrong\nNumber *",
    "Romance": "Romance *",
    "Account Verification/Payment": "Account Verification/\nPayment",
    "Fake Job": "Fake Job *",
    "Postal": "Postal",
    "E-Commerce": "E-Commerce",
    "Toll/DMV": "Toll/\nDMV",
    "Gift/Prize": "Gift/\nPrize",
}

df["cluster_name"] = df["cluster"].map(cluster_names_clean)
df = df.dropna(subset=["cluster", "cleaned"])
df = df.dropna(subset=["Phone Number", "cluster_name"])


def get_country(row):
    numbers = set(re.findall(r'\+?\d[\d\s\-()]{7,}\d', str(row["Phone Number"])))
    for raw in numbers:
        try:
            num = phonenumbers.parse(raw, None) if raw.startswith("+") else phonenumbers.parse(raw, "US")
            country = region_code_for_number(num)
            if country == "US":
                return "US"
            elif country:
                return "Foreign"
        except:
            continue
    return "Unknown"


df["origin_type"] = df.apply(get_country, axis=1)
df = df[df["origin_type"] != "Unknown"]

model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

similarity_summary = []

for cluster, group in df.groupby("cluster_name"):
    nanpa_texts = group[group["origin_type"] == "US"]["Extracted Text"].dropna().astype(str).tolist()
    non_nanpa_texts = group[group["origin_type"] == "Foreign"]["Extracted Text"].dropna().astype(str).tolist()

    record = {"cluster_name": cluster}
    
    nanpa_embeds = None
    non_nanpa_embeds = None

    # Intra-NANPA
    if len(nanpa_texts) >= 2:
        nanpa_embeds = model.encode(nanpa_texts, batch_size=64, show_progress_bar=False)
        sim_matrix = cosine_similarity(nanpa_embeds)
        upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        record["intra_nanpa"] = upper.mean()
    else:
        record["intra_nanpa"] = np.nan

    if len(non_nanpa_texts) >= 2:
        non_nanpa_embeds = model.encode(non_nanpa_texts, batch_size=64, show_progress_bar=False)
        sim_matrix = cosine_similarity(non_nanpa_embeds)
        upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        record["intra_nonnanpa"] = upper.mean()
    else:
        record["intra_nonnanpa"] = np.nan

    if len(nanpa_texts) >= 1 and len(non_nanpa_texts) >= 1:
        if nanpa_embeds is None:
            nanpa_embeds = model.encode(nanpa_texts, batch_size=64, show_progress_bar=False)
        if non_nanpa_embeds is None:
            non_nanpa_embeds = model.encode(non_nanpa_texts, batch_size=64, show_progress_bar=False)

        sim_matrix = cosine_similarity(nanpa_embeds, non_nanpa_embeds)
        record["inter_nanpa_nonnanpa"] = sim_matrix.mean()
    else:
        record["inter_nanpa_nonnanpa"] = np.nan 

    similarity_summary.append(record)

summary_df = pd.DataFrame(similarity_summary).set_index("cluster_name")
summary_df = summary_df.dropna()

non_starred_order = [
    "Account Verification/Payment",
    "E-Commerce",
    "Gift/Prize",
    "Postal",
    "Toll/DMV"
]
starred_order = [
    "Fake Job",
    "Romance",
    "Wrong Number"
]

original_labels_present = summary_df.index.tolist()

final_order = [
    name for name in starred_order if name in original_labels_present
] + [
    name for name in non_starred_order if name in original_labels_present
]

summary_df = summary_df.reindex(final_order)

summary_df = summary_df.rename(index=cluster_display_names)

fig, ax = plt.subplots(figsize=(12, 8))
hm = sns.heatmap(
    summary_df,
    annot=True,
    cmap="Oranges",
    vmin=0, vmax=1,
    annot_kws={"size": 18},
    cbar_kws=dict(label="Mean Cosine Similarity", fraction=0.10, pad=0.02, shrink=1.0, aspect=90)
)

label_fs = 22
tick_fs  = 20
ax.set_xlabel("Origin Comparison Type", fontsize=label_fs, labelpad=10, fontweight='bold')
ax.set_ylabel("Scam Category", fontsize=label_fs, labelpad=10, fontweight='bold')
ax.tick_params(axis='x', labelsize=tick_fs)
ax.tick_params(axis='y', labelsize=tick_fs)

label_map = {
    "intra_nanpa": "NANP",
    "intra_nonnanpa": "non-NANP",
    "inter_nanpa_nonnanpa": "inter-category"
}
ax.set_xticklabels(
    [label_map.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()],
    rotation=0, fontsize=tick_fs
)

cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=tick_fs)
cbar.ax.yaxis.label.set_fontproperties(ax.xaxis.label.get_fontproperties())
cbar.ax.yaxis.label.set_size(ax.xaxis.label.get_size())
cbar.set_label("Mean Cosine Similarity", labelpad=18)

fig.text(0.1, 0.02, "* Denotes Reply-Based Scams", ha='left', va='center', fontsize=18)

fig.tight_layout()
fig.savefig("combined_similarity_heatmap.pdf", dpi=300, bbox_inches="tight")
plt.show()