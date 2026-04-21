import glob
import pickle
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

INFLATION_VALUES = [14, 20, 40, 60]
CLUSTER_FILES = {I: f"PMIND2026_cluster/diamond_thresh20_clusters_I{I}.txt" for I in INFLATION_VALUES}
ANNOT_DIR = "PMIND2026_cluster/annotations/darkdino_annotation"
MAPPING_DIR = "PMIND2026_cluster/go_mappings"
OUT_DIR = "PMIND2026_cluster/figures"
COLORS = {14: "#2196F3", 20: "#4CAF50", 40: "#FF9800", 60: "#E91E63"}
INSPECT_I = 14

print("Loading GO annotations (annot_go_clean)")
with open(f"{MAPPING_DIR}/annot_go_mf_pruned.pkl", "rb") as f:
    annot_go: dict = pickle.load(f)
print(f" Proteins with GO-MF annotation (only from the clean dbs) : {len(annot_go):,}")

print("Loading GO term names")
go_names: dict[str, str] = {}
with open(f"{MAPPING_DIR}/go-basic.obo","r", encoding="utf-8") as f:
    current_id = None
    for line in f:
        line = line.strip()
        if line.startswith("id: GO:"):
            current_id = line.split("id: ")[1].strip()
        elif line.startswith("name:") and current_id:
            go_names[current_id] = line.split("name: ")[1].strip()
            current_id = None
print(f"  Loaded {len(go_names):,} GO term names")

# homogeneity functions
def compute_homogeneity_jaccard(members, annotation_dict, min_annot_fraction=0.0):
    """mean pairwise jaccard similarity between GO term sets of annotated members"""
    annotated = [m for m in members if m in annotation_dict]
    if len(annotated) < 2:
        return None
    if min_annot_fraction > 0.0 and len(annotated) / len(members) < min_annot_fraction:
        return None
    sets = [annotation_dict[m] for m in annotated]
    scores = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            if u:
                scores.append(len(sets[i] & sets[j]) / len(u))
    return float(np.mean(scores)) if scores else None

def compute_homogeneity_v1(members, annotation_dict):
    """most frequent annotation per cluster"""
    annotated = [m for m in members if m in annotation_dict]
    if len(annotated) < 2:
        return None
    all_terms = []
    for a in annotated:
        all_terms.extend(annotation_dict[a])
    if not all_terms:
        return None
    top_count = Counter(all_terms).most_common(1)[0][1]
    return top_count / len(annotated)

results = {}
clusters_by_I = {} # keeping clusters in memory for later analyses

for i in INFLATION_VALUES:
    path = CLUSTER_FILES[i]
    try:
        clusters = []
        with open(path) as f:
            for line in f:
                members = line.strip().split("\t")
                if members:
                    clusters.append(members)
    except FileNotFoundError:
        print(f"File {path} not found, skipping {i}")
        continue

    clusters_by_I[i] = clusters
    sizes = [len(c) for c in clusters]

    fully_dark = 0
    fully_annotated = 0
    mixed = 0
    annot_rates = []
    homog_jaccard = []
    homog_jaccard_f = []
    homog_v1 = []

    for c in clusters:
        n = len(c)
        na = sum(1 for m in c if m in annot_go)
        annot_rates.append(na / n)
        if na == 0 : fully_dark += 1
        elif na == n : fully_annotated += 1
        else: mixed += 1

        hj = compute_homogeneity_jaccard(c, annot_go)
        hjf = compute_homogeneity_jaccard(c, annot_go, min_annot_fraction=0.3)
        hjv = compute_homogeneity_v1(c, annot_go)

        if hj is not None: homog_jaccard.append(hj)
        if hjf is not None: homog_jaccard_f.append(hjf)
        if hjv is not None: homog_v1.append(hjv)

    # annotation transfer potential
    transferable = sum(1 for c in clusters if any(m in annot_go for m in c) and any(m not in annot_go for m in c))
    dark_transferable = sum(1 for c in clusters for m in c if m not in annot_go and any(x in annot_go for x in c if x!=m))

    results[i] = {
        "n_clusters": len(clusters),
        "sizes": sizes,
        "singletons": sum(1 for s in sizes if s == 1),
        "max_size": max(sizes),
        "mean_size": float(np.mean(sizes)),
        "median_size": float(np.median(sizes)),
        "fully_dark": fully_dark,
        "fully_annotated": fully_annotated,
        "mixed": mixed,
        "annot_rates": annot_rates,
        "mean_annot_rate": float(np.mean(annot_rates)),
        "homog_jaccard": homog_jaccard,
        "homog_jaccard_f": homog_jaccard_f,
        "homog_v1": homog_v1,
        "mean_homog_jaccard": float(np.mean(homog_jaccard)) if homog_jaccard else 0.0,
        "mean_homog_jaccard_f": float(np.mean(homog_jaccard_f)) if homog_jaccard_f else 0.0,
        "mean_homog_v1": float(np.mean(homog_v1)) if homog_v1 else 0.0,
        "n_eval_jaccard": len(homog_jaccard),
        "n_eval_jaccard_f": len(homog_jaccard_f),
        "pct_high_homog": sum(1 for h in homog_jaccard if h >= 0.8) / len(homog_jaccard) if homog_jaccard else 0.0,
        "pct_low_homog": sum(1 for h in homog_jaccard if h < 0.5) / len(homog_jaccard) if homog_jaccard else 0.0,
        "transferable": transferable,
        "dark_transferable": dark_transferable,
    }

    print(f"\nInflation I={i}")
    print(f"  Total clusters          : {results[i]['n_clusters']:,}")
    print(f"  Singletons              : {results[i]['singletons']:,}")
    print(f"  Max cluster size        : {results[i]['max_size']:,}")
    print(f"  Mean cluster size       : {results[i]['mean_size']:.1f}")
    print(f"  Fully dark  (GO)        : {fully_dark:,}")
    print(f"  Mixed       (GO)        : {mixed:,}")
    print(f"  Fully annotated (GO)    : {fully_annotated:,}")
    print(f"  Mean GO annotation rate : {results[i]['mean_annot_rate']:.3f}")
    print(f"  Evaluable (Jaccard)     : {results[i]['n_eval_jaccard']:,}")
    print(f"  Mean homog (Jaccard)    : {results[i]['mean_homog_jaccard']:.3f}")
    print(f"  Mean homog (Jac+filter) : {results[i]['mean_homog_jaccard_f']:.3f}")
    print(f"  % clusters homog ≥ 0.8  : {results[i]['pct_high_homog'] * 100:.1f}%")
    print(f"  % clusters homog < 0.5  : {results[i]['pct_low_homog'] * 100:.1f}%")
    print(f"  Clusters with transfer potential : {transferable:,}")
    print(f"  Dark proteins receivable         : {dark_transferable:,}")

valid_I = [i for i in INFLATION_VALUES if i in results]

plt.rcParams.update({
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

labels = [f"I={i}" for i in valid_I]
colors = [COLORS[i] for i in valid_I]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("MCL Clustering — GO-MF Annotation — Inflation Comparison", fontsize=14, fontweight="bold", y=1.01)

# number of clusters
ax = axes[0, 0]
vals = [results[I]["n_clusters"] for I in valid_I]
bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.5)
ax.set_title("Number of Clusters", fontweight="bold")
ax.set_ylabel("Count")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# mean homogeneity (Jaccard vs jaccard filter)
ax = axes[0, 1]
x, w = np.arange(len(valid_I)), 0.35
b1 = ax.bar(x - w / 2, [results[I]["mean_homog_jaccard"] for I in valid_I], w, label="Jaccard", color=colors, edgecolor="white", linewidth=1.5)
b2 = ax.bar(x + w / 2, [results[I]["mean_homog_jaccard_f"] for I in valid_I], w,
            label="Jaccard+filter",
            color=[c + "88" for c in colors], edgecolor=colors, linewidth=1.5)
ax.set_xticks(x);
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.05)
ax.set_title("Mean GO Homogeneity (Jaccard)", fontweight="bold")
ax.set_ylabel("Homogeneity (0–1)")
ax.legend(fontsize=8)
for bar in b1:
    h = bar.get_height()
ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

# high / low homogeneity
ax = axes[0, 2]
x, w = np.arange(len(valid_I)), 0.35
ax.bar(x - w/2, [results[I]["pct_high_homog"]*100 for I in valid_I], w,
       label="≥0.8 (good)", color="#4CAF50", edgecolor="white", linewidth=1.5)
ax.bar(x + w/2, [results[I]["pct_low_homog"]*100  for I in valid_I], w,
       label="<0.5 (poor)", color="#F44336", edgecolor="white", linewidth=1.5)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_title("Cluster Quality Distribution (GO)", fontweight="bold")
ax.set_ylabel("% of evaluable clusters")
ax.set_ylim(0, 110)
ax.legend(fontsize=8)

# cluster size distribution
ax = axes[1, 0]
data = [results[I]["sizes"] for I in valid_I]
bp   = ax.boxplot(data, labels=labels, patch_artist=True,
                  medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax.set_title("Cluster Size Distribution", fontweight="bold")
ax.set_ylabel("Cluster size (proteins)")
ax.set_yscale("log")

# homogeneity distribution
ax = axes[1, 1]
for I in valid_I:
    h = results[I]["homog_jaccard"]
    if h:
        ax.hist(h, bins=20, range=(0, 1), alpha=0.5,
                color=COLORS[I], label=f"I={I}", edgecolor="none")
ax.set_title("GO Homogeneity Distribution\n(Jaccard)", fontweight="bold")
ax.set_xlabel("Homogeneity score")
ax.set_ylabel("Number of clusters")
ax.legend(fontsize=8)

# cluster composition (dark / mixed / annotated) — GO-based
ax = axes[1, 2]
x, w = np.arange(len(valid_I)), 0.6
dark_vals = [results[I]["fully_dark"] for I in valid_I]
mixed_vals = [results[I]["mixed"] for I in valid_I]
full_vals = [results[I]["fully_annotated"] for I in valid_I]
ax.bar(x, dark_vals, w, label="Fully dark (GO)", color="#607D8B", edgecolor="white")
ax.bar(x, mixed_vals, w, bottom=dark_vals,
       label="Mixed (GO)", color="#FFC107", edgecolor="white")
ax.bar(x, full_vals, w,
       bottom=[d + m for d, m in zip(dark_vals, mixed_vals)],
       label="Fully annotated (GO)", color="#8BC34A", edgecolor="white")
ax.set_xticks(x);
ax.set_xticklabels(labels)
ax.set_title("Cluster Composition (GO-MF)", fontweight="bold")
ax.set_ylabel("Number of clusters")
ax.legend(fontsize=8)

plt.tight_layout()
outpath = f"{OUT_DIR}/mcl_inflation_comparison_go.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {outpath}")
plt.close()

# summary table

print(f"\n{'='*75}")
print(f"{'SUMMARY TABLE (GO-MF)':^75}")
print(f"{'='*75}")
print(f"{'Metric':<38} {'I=1.4':>8} {'I=2.0':>8} {'I=4.0':>8} {'I=6.0':>8}")
print(f"{'-'*75}")
metrics = [
    ("N clusters",              "n_clusters",           "{:.0f}"),
    ("Singletons",              "singletons",           "{:.0f}"),
    ("Max cluster size",        "max_size",             "{:.0f}"),
    ("Mean cluster size",       "mean_size",            "{:.1f}"),
    ("Fully dark (GO)",         "fully_dark",           "{:.0f}"),
    ("Mixed (GO)",              "mixed",                "{:.0f}"),
    ("Fully annotated (GO)",    "fully_annotated",      "{:.0f}"),
    ("Mean GO annot rate",      "mean_annot_rate",      "{:.3f}"),
    ("N eval (Jaccard)",        "n_eval_jaccard",       "{:.0f}"),
    ("Mean homog (Jaccard)",    "mean_homog_jaccard",   "{:.3f}"),
    ("Mean homog (Jac+filter)", "mean_homog_jaccard_f", "{:.3f}"),
    ("% homog ≥ 0.8",           "pct_high_homog",       "{:.1%}"),
    ("% homog < 0.5",           "pct_low_homog",        "{:.1%}"),
    ("Transferable clusters",   "transferable",         "{:.0f}"),
    ("Dark proteins receivable","dark_transferable",    "{:.0f}"),
]
for label, key, fmt in metrics:
    row = f"{label:<38}"
    for I in INFLATION_VALUES:
        row += f" {fmt.format(results[I][key]):>8}" if I in results else f" {'N/A':>8}"
    print(row)
print(f"{'='*75}")

# heterogenous cluster analysis

print(f"\n{'=' * 70}")
print(f"Heterogeneous cluster analysis — GO Jaccard  (I={INSPECT_I})")
print(f"{'=' * 70}")

clusters_inspect = clusters_by_I.get(INSPECT_I, [])

het_clusters_go = []
for idx, cluster in enumerate(clusters_inspect):
    annotated = [m for m in cluster if m in annot_go]
    if len(annotated) < 2:
        continue
    h = compute_homogeneity_jaccard(cluster, annot_go)
    if h is not None and h < 1.0:
        het_clusters_go.append((h, idx, cluster, annotated))

het_clusters_go.sort(key=lambda x: x[0])
print(f"Total heterogeneous clusters (Jaccard < 1.0) : {len(het_clusters_go):,}")

buckets_het = {"0.0–0.5": 0, "0.5–0.7": 0, "0.7–0.9": 0, "0.9–1.0": 0}
for h, *_ in het_clusters_go:
    if h < 0.5:
        buckets_het["0.0–0.5"] += 1
    elif h < 0.7:
        buckets_het["0.5–0.7"] += 1
    elif h < 0.9:
        buckets_het["0.7–0.9"] += 1
    else:
        buckets_het["0.9–1.0"] += 1

print("\nHomogeneity bucket breakdown:")
for bucket, count in buckets_het.items():
    bar = "█" * (count * 40 // max(buckets_het.values()) if max(buckets_het.values()) else 0)
    print(f"  {bucket} : {count:>8,}  {bar}")

# most conflicting GO terms across all heterogeneous clusters
go_conflict: Counter = Counter()
for h, idx, cluster, annotated in het_clusters_go:
    all_go = [annot_go[m] for m in annotated]
    union = set().union(*all_go)
    for term in union:
        has = sum(1 for s in all_go if term in s)
        hasnt = len(all_go) - has
        if has > 0 and hasnt > 0:
            go_conflict[term] += 1

print(f"\nTop 20 most conflicting GO-MF terms:")
print(f"{'Term':<12} {'Name':<45} {'Clusters':>8}")
print(f"{'-' * 68}")
for term, count in go_conflict.most_common(20):
    print(f"  {term}  {go_names.get(term, '?'):<45}  {count:>6,}")

# near-perfect clusters (1–2 GO outliers)
print(f"\n{'=' * 70}")
print(f"Near-perfect GO clusters with 1–2 outliers  (I={INSPECT_I})")
print(f"{'=' * 70}")

# load full annotations for outlier detail
annot_full: dict = defaultdict(list)
for f in glob.glob(f"{ANNOT_DIR}/*.tsv"):
    for line in open(f, encoding="utf-8-sig", errors="ignore"):
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            annot_full[parts[0]].append((parts[1], parts[2], parts[3]))
        elif len(parts) == 3:
            annot_full[parts[0]].append((parts[1], parts[2], ""))

near_perfect = []
for idx, cluster in enumerate(clusters_inspect):
    annotated = [m for m in cluster if m in annot_go]
    if len(annotated) < 3:
        continue
    # find the GO term shared by the most annotated members (majority term)
    term_counts: Counter = Counter()
    for m in annotated:
        term_counts.update(annot_go[m])
    if not term_counts:
        continue
    top_term, top_count = term_counts.most_common(1)[0]
    score = top_count / len(annotated)

    if score < 1.0:
        majority = [m for m in annotated if top_term in annot_go[m]]
        outliers = [m for m in annotated if top_term not in annot_go[m]]
        if 1 <= len(outliers) <= 2:
            near_perfect.append({
                "idx": idx,
                "score": score,
                "size": len(cluster),
                "n_annotated": len(annotated),
                "n_majority": len(majority),
                "n_outliers": len(outliers),
                "top_term": top_term,
                "top_name": go_names.get(top_term, "?"),
                "majority": majority,
                "outliers": outliers,
                "dark": [m for m in cluster if m not in annot_go],
            })

near_perfect.sort(key=lambda x: x["size"], reverse=True)
n1 = sum(1 for c in near_perfect if c["n_outliers"] == 1)
n2 = sum(1 for c in near_perfect if c["n_outliers"] == 2)
print(f"Found {len(near_perfect):,} near-perfect clusters")
print(f"  1 outlier  : {n1:,}")
print(f"  2 outliers : {n2:,}\n")

print("Top 20 largest near-perfect clusters:\n")
for rank, c in enumerate(near_perfect[:20]):
    print(f"{'─' * 70}")
    print(f"Rank {rank + 1:>3} | Cluster {c['idx']} | Size {c['size']} "
          f"| Annotated {c['n_annotated']} | Homog {c['score']:.3f}")
    print(f"  Majority GO term : {c['top_term']}  {c['top_name']}")
    print(f"  Majority members : {c['n_majority']}  |  Dark : {len(c['dark'])}")
    print()

    if len(c["majority"]) <= 5:
        for m in c["majority"]:
            go_str = ", ".join(sorted(annot_go.get(m, set()))[:4])
            print(f"  [majority] {m}")
            print(f"             GO: {go_str[:100]}")
    else:
        top_go: Counter = Counter()
        for m in c["majority"]:
            top_go.update(annot_go[m])
        summary = ", ".join(
            f"{go_names.get(t, '?')}(x{n})" for t, n in top_go.most_common(5))
        print(f"  [majority] {c['n_majority']} proteins, top GO: {summary[:100]}")
    print()

    for m in c["outliers"]:
        print(f"  [OUTLIER]  {m}")
        go_str = ", ".join(sorted(annot_go.get(m, set()))[:6])
        print(f"    GO terms : {go_str[:100]}")
        by_db: dict = defaultdict(list)
        for db, dom_id, desc in annot_full.get(m, []):
            by_db[db].append(f"{dom_id}" + (f" ({desc})" if desc else ""))
        for db, entries in sorted(by_db.items()):
            unique = list(dict.fromkeys(entries))
            print(f"    {db:20s}: {', '.join(unique[:3])}"
                  + (" ..." if len(unique) > 3 else ""))
        shared = annot_go.get(m, set()) & annot_go.get(c["majority"][0], set()) \
            if c["majority"] else set()
        if shared:
            shared_names = ", ".join(go_names.get(t, t) for t in list(shared)[:3])
            print(f"    Shares GO terms with majority: {shared_names}")
        else:
            print(f"    No shared GO terms with majority")
    print()

# outlier GO term summary

print(f"\n{'=' * 70}")
print("GO terms carried by outlier proteins")
print(f"{'=' * 70}\n")

outlier_go: Counter = Counter()
for c in near_perfect:
    for m in c["outliers"]:
        outlier_go.update(annot_go.get(m, set()))

print(f"{'GO term':<12} {'Name':<45} {'Count':>8}")
print(f"{'-' * 68}")
for term, count in outlier_go.most_common(20):
    print(f"  {term}  {go_names.get(term, '?'):<45}  {count:>6,}")

# outlier overlap analysis

print(f"\n{'=' * 70}")
print("Outlier GO overlap with majority")
print(f"{'=' * 70}\n")

fully_different = 0
partial_overlap = 0
has_top_term = 0

for c in near_perfect:
    maj_go = annot_go.get(c["majority"][0], set()) if c["majority"] else set()
    for m in c["outliers"]:
        out_go = annot_go.get(m, set())
        if c["top_term"] in out_go:
            has_top_term += 1
        elif out_go & maj_go:
            partial_overlap += 1
        else:
            fully_different += 1

total_outliers = sum(c["n_outliers"] for c in near_perfect)
print(f"Total outlier proteins : {total_outliers:,}")
print(f"  Fully different GO   : {fully_different:,} "
      f"({100 * fully_different / total_outliers:.1f}%)  <- likely mis-clustered")
print(f"  Partial GO overlap   : {partial_overlap:,} "
      f"({100 * partial_overlap / total_outliers:.1f}%)  <- multi-function proteins")
print(f"  Has majority GO term : {has_top_term:,}  "
      f"({100 * has_top_term / total_outliers:.1f}%)  <- annotation noise")

# annotation transfer potential

print(f"\n{'='*70}")
print("GO Annotation transfer potential")
print(f"{'='*70}\n")
for I in valid_I:
    print(f"  I={I}  transferable clusters : {results[I]['transferable']:>8,}  "
          f"dark proteins receivable : {results[I]['dark_transferable']:>8,}")