import pandas as pd
import ast
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "Liberation Serif"]
})

df = pd.read_csv("combined_all_virustotal.csv")
df = df.drop_duplicates(subset='url', keep='first')

json_col = 'full_virustotal_json'

def parse_json(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return None

df[json_col] = df[json_col].apply(parse_json)

def extract_results(entry):
    try:
        results = entry['data']['attributes']['last_analysis_results']
        cat_counter = Counter()
        res_counter = Counter()
        for engine_result in results.values():
            cat = engine_result.get("category")
            res = engine_result.get("result")
            if cat:
                cat_counter[cat] += 1
            if res:
                res_counter[res] += 1
        return cat_counter, res_counter
    except:
        return Counter(), Counter()

category_total = Counter()
result_total = Counter()
for entry in df[json_col]:
    cat_counts, res_counts = extract_results(entry)
    category_total.update(cat_counts)
    result_total.update(res_counts)

def result_counts_per_row(entry):
    try:
        results = entry['data']['attributes']['last_analysis_results']
        counter = Counter()
        for engine_result in results.values():
            res = engine_result.get("result")
            if res:
                counter[res] += 1
        return pd.Series(counter)
    except:
        return pd.Series()

result_matrix = df[json_col].apply(result_counts_per_row).fillna(0).astype(int)

malicious_group = ["malicious", "phishing", "malware", "suspicious", "spam", "not recommended"]
clean_group = ["clean"]

malicious_raw = result_matrix.reindex(columns=malicious_group, fill_value=0).sum(axis=1)
clean_raw = result_matrix.reindex(columns=clean_group, fill_value=0).sum(axis=1)

verdicts_used = malicious_raw + clean_raw
verdicts_total = result_matrix.sum(axis=1)

valid_mask = verdicts_used > 0
malicious_prop = malicious_raw[valid_mask] / verdicts_total[valid_mask]
clean_prop = clean_raw[valid_mask] / verdicts_total[valid_mask]

fig, ax = plt.subplots(figsize=(12, 8))

orange_blue = LinearSegmentedColormap.from_list(
    "orange_blue", ["#0055cc", "#f7b267", "#ff6200"], N=256
)

import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, FuncFormatter

x = clean_prop.to_numpy()
y = malicious_prop.to_numpy()

bins = 350 
H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])

sigma = 5.0  # try 3–7
Z = gaussian_filter(H, sigma=sigma).T 

H_max = float(H.max())
Z_max = float(Z.max())

if Z_max > 0:
    Z = Z * (H_max / Z_max)

# grid centers
xc = (xedges[:-1] + xedges[1:]) / 2
yc = (yedges[:-1] + yedges[1:]) / 2
X, Y = np.meshgrid(xc, yc)

vmin = 1
vmax = max(vmin + 1, float(H.max())) 
true_max = int(H.max())
print("True max bin count:", true_max)

levels = np.geomspace(vmin, vmax, 80)

contour = ax.contourf(
    X, Y, Z,
    levels=levels,
    cmap=orange_blue,
    norm=LogNorm(vmin=vmin, vmax=vmax),
    antialiased=True
)


cbar = fig.colorbar(contour, ax=ax, aspect=30, pad=0.02)
cbar.set_label("Number of URLs", fontsize=28, fontweight="bold")

locator = LogLocator(base=10, subs=(1.0, 2.0, 5.0))
ticks = [t for t in locator.tick_values(vmin, vmax) if vmin <= t <= vmax]

if true_max not in ticks:
    ticks.append(true_max)

ticks = sorted(set(ticks))
cbar.set_ticks(ticks)

cbar.formatter = FuncFormatter(lambda v, pos: f"{int(v)}" if v >= 1 else "")
cbar.update_ticks()

ax.set_ylim(0, 1)
ax.set_xlim(0, 1)
ax.set_xlabel("Clean Proportion", fontsize=28, fontweight="bold")
ax.set_ylabel("Malicious Proportion", fontsize=28, fontweight="bold")
ax.tick_params(axis='both', labelsize=16)
ax.grid(True, linestyle="--", alpha=0.3)

X_SPLIT = 0.5  
Y_SPLIT = 0.5  

# Divider lines
ax.axvline(X_SPLIT, color="#333333", linestyle=":", linewidth=1.8, zorder=6)
ax.axhline(Y_SPLIT, color="#333333", linestyle=":", linewidth=1.8, zorder=6)

ax.add_patch(Rectangle((0, Y_SPLIT), X_SPLIT, 1 - Y_SPLIT,  # top-left
                       facecolor="#2ca02c", alpha=0.10, edgecolor="none", zorder=0))
ax.add_patch(Rectangle((X_SPLIT, Y_SPLIT), 1 - X_SPLIT, 1 - Y_SPLIT,  # top-right
                       facecolor="#9467bd", alpha=0.10, edgecolor="none", zorder=0))
ax.add_patch(Rectangle((0, 0), X_SPLIT, Y_SPLIT,  # bottom-left
                       facecolor="#1f77b4", alpha=0.10, edgecolor="none", zorder=0))
ax.add_patch(Rectangle((X_SPLIT, 0), 1 - X_SPLIT, Y_SPLIT,  # bottom-right
                       facecolor="#ff0000", alpha=0.10, edgecolor="none", zorder=0))

ax.text(0.25, 0.75, "Q1:\nLow Clean / High Malicious",  ha="center", va="center", fontsize=16, weight="bold")
ax.text(0.75, 0.75, "Q2:\nHigh Clean / High Malicious", ha="center", va="center", fontsize=16, weight="bold")
ax.text(0.25, 0.25, "Q3:\nLow Clean / Low Malicious",   ha="center", va="center", fontsize=16, weight="bold")
ax.text(0.75, 0.25, "Q4:\nHigh Clean / Low Malicious",  ha="center", va="center", fontsize=16, weight="bold")


plt.tight_layout()
plt.savefig("/home/luall10/virustotal_figure.pdf", dpi=300, bbox_inches='tight')
plt.show()