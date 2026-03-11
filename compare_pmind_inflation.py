import glob
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from evaluate_pmind_clusters import all_domains

# initial config
INFLATION_VALUES = [14, 20, 40, 60]
#INFLATION_VALUES = ["baseline - NLE", "NLE with threshold (1e-10)", "bitscore"]
#INFLATION_VALUES = ["baseline", 5, 10, 20]
CLUSTER_FILES = {I: f"PMIND2026_cluster/diamond_thresh20_clusters_I{I}.txt" for I in INFLATION_VALUES}
#CLUSTER_FILES = {"baseline":"PMIND2026_cluster/diamond_clusters_I20.txt",5:"PMIND2026_cluster/diamond_thresh5_clusters_I20.txt", 10:"PMIND2026_cluster/diamond_thresh10_clusters_I20.txt", 20:"PMIND2026_cluster/diamond_thresh20_clusters_I20.txt"}
#CLUSTER_FILES = {"baseline - NLE":"PMIND2026_cluster/diamond_clusters_I20.txt","NLE with threshold (1e-10)":"PMIND2026_cluster/diamond_thresh10_clusters_I20.txt", "bitscore":"PMIND2026_cluster/diamond_bitscore_clusters_I20.txt"}
ANNOT_DIR = "PMIND2026_cluster/annotations/darkdino_annotation"
OUT_DIR = "PMIND2026_cluster/figures"
COLORS = {14: "#2196F3", 20: "#4CAF50", 40: "#FF9800", 60: "#E91E63"}
#COLORS = {"baseline - NLE": "#2196F3", "NLE with threshold (1e-10)": "#4CAF50", "bitscore": "#FF9800"}
#COLORS = {"baseline": "#2196F3", 5: "#4CAF50", 10: "#FF9800", 20: "#E91E63"}

#load annotations
print("Loading the annotations")
annot = defaultdict(set) # all annotations
pfam_annot = defaultdict(set) # pfam annotations only

for f in glob.glob(f"{ANNOT_DIR}/*.tsv"):
    for line in open(f, encoding="utf-8-sig", errors="ignore"):
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            annot[parts[0]].add(parts[2])
            if parts[1] == "Pfam":
                pfam_annot[parts[0]].add(parts[2])

print(f"Proteins with any annotation : {len(annot):,}")
print(f"Proteins with Pfam annotation : {len(pfam_annot):,}")

# computing stats per inflation value
def compute_homogeneity(members, annotation_dict):
    """homogeneity score if more than two members of a cluster are annotated, None if not"""
    annotated = [m for m in members if m in annotation_dict]
    if len(annotated) < 2:
        return None
    all_domains = []
    for a in annotated:
        all_domains.extend(annotation_dict[a])
    if not all_domains:
        return None
    top_count = Counter(all_domains).most_common(1)[0][1]
    return top_count / len(annotated)

results = {}

for i in INFLATION_VALUES:
    path = CLUSTER_FILES[i]
    try:
        clusters = []
        with open(path) as fh:
            for line in fh:
                members = line.strip().split("\t")
                if members:
                    clusters.append(members)
    except FileNotFoundError:
        print(f"Cluster file {path} not found, skipping I = {i}")
        continue

    sizes = [len(c) for c in clusters]
    homog_all = [] # homogeneity score using all databases
    homog_pfam = [] # homogeneity score using pfam
    annot_rates = [] # fraction of annotated members per cluster
    fully_dark = 0
    fully_annotated = 0
    mixed = 0

    for c in clusters:
        n = len(c)
        na = sum(1 for m in c if m in annot)
        annot_rates.append(na / n)
        if na == 0:
            fully_dark += 1
        elif na == n:
            fully_annotated += 1
        else:
            mixed += 1

        h_all = compute_homogeneity(c, annot)
        h_pfam = compute_homogeneity(c, pfam_annot)
        if h_all is not None: homog_all.append(h_all)
        if h_pfam is not None: homog_pfam.append(h_pfam)


    results[i] = {
        "n_clusters"       : len(clusters),
        "sizes"            : sizes,
        "singletons"       : sum(1 for s in sizes if s == 1),
        "max_size"         : max(sizes),
        "mean_size"        : np.mean(sizes),
        "median_size"      : np.median(sizes),
        "fully_dark"       : fully_dark,
        "fully_annotated"  : fully_annotated,
        "mixed"            : mixed,
        "annot_rates"      : annot_rates,
        "mean_annot_rate"  : np.mean(annot_rates),
        "homog_all"        : homog_all,
        "homog_pfam"       : homog_pfam,
        "mean_homog_all"   : np.mean(homog_all)   if homog_all  else 0,
        "mean_homog_pfam"  : np.mean(homog_pfam)  if homog_pfam else 0,
        "n_evaluable_all"  : len(homog_all),
        "n_evaluable_pfam" : len(homog_pfam),
        "pct_high_homog"   : sum(1 for h in homog_all if h >= 0.8) / len(homog_all) if homog_all else 0,
        "pct_low_homog"    : sum(1 for h in homog_all if h < 0.5)  / len(homog_all) if homog_all else 0,
    }

    print(f"Inflation {i}")
    print(f"  Total clusters        : {results[i]['n_clusters']}")
    print(f"  Singletons            : {results[i]['singletons']}")
    print(f"  Max cluster size      : {results[i]['max_size']}")
    print(f"  Mean cluster size     : {results[i]['mean_size']:.1f}")
    print(f"  Fully dark clusters   : {fully_dark}")
    print(f"  Fully annotated       : {fully_annotated}")
    print(f"  Mixed                 : {mixed}")
    print(f"  Mean annotation rate  : {results[i]['mean_annot_rate']:.3f}")
    print(f"  Evaluable clusters    : {results[i]['n_evaluable_all']} (all db)  |  {results[i]['n_evaluable_pfam']} (Pfam)")
    print(f"  Mean homogeneity (all): {results[i]['mean_homog_all']:.3f}")
    print(f"  Mean homogeneity (Pfam): {results[i]['mean_homog_pfam']:.3f}")
    print(f"  % clusters homog≥0.8  : {results[i]['pct_high_homog'] * 100:.1f}%")
    print(f"  % clusters homog<0.5  : {results[i]['pct_low_homog'] * 100:.1f}%")

valid_I = [i for i in INFLATION_VALUES if i in results]

# figures
plt.rcParams.update({
    "font.family"  : "monospace",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"    : True,
    "grid.alpha"   : 0.3,
    "grid.linestyle": "--",
})

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("MCL Clustering — darkdino — Inflation Parameter", fontsize=14, fontweight="bold", y=1.01) # changed from Inflation Parameter
labels = [f"I={I}" for I in valid_I]
colors = [COLORS[I] for I in valid_I]

# no of clusters
ax = axes[0, 0]
vals = [results[I]["n_clusters"] for I in valid_I]
bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.5)
ax.set_title("Number of Clusters", fontweight="bold")
ax.set_ylabel("Count")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")

# mean homogeneity (all db vs pfam)
ax = axes[0, 1]
x = np.arange(len(valid_I))
w = 0.35
b1 = ax.bar(x - w/2, [results[I]["mean_homog_all"]  for I in valid_I], w, label="All databases", color=colors, edgecolor="white", linewidth=1.5)
b2 = ax.bar(x + w/2, [results[I]["mean_homog_pfam"] for I in valid_I], w, label="Pfam only", color=[c + "88" for c in colors], edgecolor=colors, linewidth=1.5, linestyle="--")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.05)
ax.set_title("Mean Homogeneity Score", fontweight="bold")
ax.set_ylabel("Homogeneity (0–1)")
ax.legend(fontsize=8)
for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
            f"{h:.2f}", ha="center", va="bottom", fontsize=9)

# high homogeneity (>=0.8) vs low (<0.5)
ax = axes[0, 2]
x = np.arange(len(valid_I))
w = 0.35
ax.bar(x - w/2, [results[I]["pct_high_homog"]*100 for I in valid_I], w, label="≥0.8 (good)", color="#4CAF50", edgecolor="white", linewidth=1.5)
ax.bar(x + w/2, [results[I]["pct_low_homog"]*100  for I in valid_I], w, label="<0.5 (poor)", color="#F44336", edgecolor="white", linewidth=1.5)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_title("Cluster Quality Distribution", fontweight="bold")
ax.set_ylabel("% of evaluable clusters")
ax.set_ylim(0, 110)
ax.legend(fontsize=8)

# cluster size distribution

ax = axes[1, 0]
data = [results[I]["sizes"] for I in valid_I]
bp = ax.boxplot(data, labels=labels, patch_artist=True, medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_title("Cluster Size Distribution", fontweight="bold")
ax.set_ylabel("Cluster size (proteins)")
ax.set_yscale("log")

# homogeneity distribution
ax = axes[1, 1]
for I in valid_I:
    h = results[I]["homog_all"]
    if h:
        ax.hist(h, bins=20, range=(0, 1), alpha=0.5, color=COLORS[I], label=f"I={I}", edgecolor="none")
ax.set_title("Homogeneity Score Distribution\n(all databases)", fontweight="bold")
ax.set_xlabel("Homogeneity score")
ax.set_ylabel("Number of clusters")
ax.legend(fontsize=8)

# cluster composition (dark, mixed, fully annotated)
ax = axes[1, 2]
x = np.arange(len(valid_I))
w = 0.6
dark_vals  = [results[I]["fully_dark"]      for I in valid_I]
mixed_vals = [results[I]["mixed"]           for I in valid_I]
full_vals  = [results[I]["fully_annotated"] for I in valid_I]

ax.bar(x, dark_vals,  w, label="Fully dark",       color="#607D8B", edgecolor="white")
ax.bar(x, mixed_vals, w, bottom=dark_vals,          label="Mixed",  color="#FFC107", edgecolor="white")
ax.bar(x, full_vals,  w,
       bottom=[d+m for d,m in zip(dark_vals, mixed_vals)],
       label="Fully annotated", color="#8BC34A", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_title("Cluster Annotation Composition", fontweight="bold")
ax.set_ylabel("Number of clusters")
ax.legend(fontsize=8)

plt.tight_layout()
outpath = f"{OUT_DIR}/mcl_inflation_comparison_darkdino_thresh20.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"\nFigure saved -> {outpath}")
plt.close()

# summary table
print(f"\n{'='*75}")
print(f"{'SUMMARY TABLE':^75}")
print(f"{'='*75}")
print(f"{'Metric':<35} {'I=1.4':>8} {'I=2.0':>8} {'I=4.0':>8} {'I=6.0':>8}")
print(f"{'-'*75}")
metrics = [
    ("N clusters",           "n_clusters",        "{:.0f}"),
    ("Singletons",           "singletons",         "{:.0f}"),
    ("Max cluster size",     "max_size",            "{:.0f}"),
    ("Mean cluster size",    "mean_size",           "{:.1f}"),
    ("Fully dark",           "fully_dark",          "{:.0f}"),
    ("Fully annotated",      "fully_annotated",     "{:.0f}"),
    ("Mean homog (all db)",  "mean_homog_all",      "{:.3f}"),
    ("Mean homog (Pfam)",    "mean_homog_pfam",     "{:.3f}"),
    ("% homog ≥ 0.8",        "pct_high_homog",      "{:.1%}"),
    ("% homog < 0.5",        "pct_low_homog",       "{:.1%}"),
]
for label, key, fmt in metrics:
    row = f"{label:<35}"
    for I in INFLATION_VALUES:
        if I in results:
            row += f" {fmt.format(results[I][key]):>8}"
        else:
            row += f" {'N/A':>8}"
    print(row)
print(f"{'='*75}")