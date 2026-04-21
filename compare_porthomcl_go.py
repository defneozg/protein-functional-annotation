import glob
import pickle
import re
from collections import defaultdict, Counter
import matplotlib
matplotlib.use("Agg") # non-interactive backend for cluster use
import matplotlib.pyplot as plt
import numpy as np
import os

CLUSTER_FILE    = "porthomcl_backup/8.all.ort.group"
MAPPING_FILE    = "porthomcl_backup/taxon_code_mapping.txt"
ANNOT_DIR       = "PMIND2026_cluster/annotations/darkdino_annotation"
MAPPING_DIR     = "PMIND2026_cluster/go_mappings"
OUT_DIR         = "PMIND2026_cluster/figures"
FIGURE_PREFIX   = "porthomcl"

os.makedirs(OUT_DIR, exist_ok=True)

# build taxon code -> original species prefix mapping
print("Loading taxon code mapping")
code_to_species = {} # ex "AAE1" -> "Abedinium_dasypus_EP00812"  (no .fasta, no |)
with open(MAPPING_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        code, fname = line.split("\t")
        base = fname.replace(".fasta", "")
        code_to_species[code] = base

print(f"  Loaded {len(code_to_species)} taxon codes")

def decode_id(coded_id):
    """Convert  AAE1|sequence00001 back to  Abedinium_dasypus_EP00812|sequence00001"""
    if "|" not in coded_id:
        return coded_id   # fallback, return as-is
    code, seqnum = coded_id.split("|", 1)
    species = code_to_species.get(code, code) # fallback to code if unknown
    return f"{species}|{seqnum}"

# load GO annotations (keyed by original IDs)
print("Loading GO annotations (annot_go_mf.pkl)")
with open(f"{MAPPING_DIR}/annot_go_mf.pkl", "rb") as f:
    annot_go: dict = pickle.load(f)
print(f"  Proteins with GO-MF annotation: {len(annot_go):,}")

# load GO term names
print("Loading GO term names...")
go_names: dict[str, str] = {}
with open(f"{MAPPING_DIR}/go-basic.obo", "r", encoding="utf-8") as f:
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

# load and decode PorthoMCL clusters
print(f"\nLoading PorthoMCL clusters from {CLUSTER_FILE}")
clusters = []
n_failed = 0
with open(CLUSTER_FILE) as f:
    for line in f:
        # PorthoMCL format: first field is cluster size (integer), rest are members
        parts = line.strip().split("\t")
        if not parts:
            continue
        # skip the size field (first column is an integer)
        try:
            int(parts[0])
            members_coded = parts[1:]
        except ValueError:
            members_coded = parts # no size field, all are members

        # decode taxon codes back to original IDs
        members = [decode_id(m) for m in members_coded if m]
        if members:
            clusters.append(members)

print(f"  Loaded {len(clusters):,} clusters")
print(f"  Total sequences: {sum(len(c) for c in clusters):,}")

# sanity check, see if IDs match annotation keys
sample_ids = clusters[0][:3] if clusters else []
print(f"\n  Sample decoded IDs: {sample_ids}")
print(f"  First ID in annot_go? {sample_ids[0] in annot_go if sample_ids else 'N/A'}")

# compute statistics (same logic as compare_with_annot_go.py)
print("\nComputing cluster statistics")

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
    if na == 0:
        fully_dark += 1
    elif na == n:
        fully_annotated += 1
    else:
        mixed += 1

    hj  = compute_homogeneity_jaccard(c, annot_go)
    hjf = compute_homogeneity_jaccard(c, annot_go, min_annot_fraction=0.3)
    hjv = compute_homogeneity_v1(c, annot_go)

    if hj  is not None: homog_jaccard.append(hj)
    if hjf is not None: homog_jaccard_f.append(hjf)
    if hjv is not None: homog_v1.append(hjv)

transferable = sum(
    1 for c in clusters
    if any(m in annot_go for m in c) and any(m not in annot_go for m in c)
)
dark_transferable = sum(
    1 for c in clusters
    for m in c
    if m not in annot_go and any(x in annot_go for x in c if x != m)
)

results = {
    "n_clusters":         len(clusters),
    "sizes":              sizes,
    "singletons":         sum(1 for s in sizes if s == 1),
    "max_size":           max(sizes),
    "mean_size":          float(np.mean(sizes)),
    "median_size":        float(np.median(sizes)),
    "fully_dark":         fully_dark,
    "fully_annotated":    fully_annotated,
    "mixed":              mixed,
    "annot_rates":        annot_rates,
    "mean_annot_rate":    float(np.mean(annot_rates)),
    "homog_jaccard":      homog_jaccard,
    "homog_jaccard_f":    homog_jaccard_f,
    "homog_v1":           homog_v1,
    "mean_homog_jaccard": float(np.mean(homog_jaccard))   if homog_jaccard   else 0.0,
    "mean_homog_jaccard_f": float(np.mean(homog_jaccard_f)) if homog_jaccard_f else 0.0,
    "mean_homog_v1":      float(np.mean(homog_v1))        if homog_v1        else 0.0,
    "n_eval_jaccard":     len(homog_jaccard),
    "n_eval_jaccard_f":   len(homog_jaccard_f),
    "pct_high_homog":     sum(1 for h in homog_jaccard if h >= 0.8) / len(homog_jaccard) if homog_jaccard else 0.0,
    "pct_low_homog":      sum(1 for h in homog_jaccard if h <  0.5) / len(homog_jaccard) if homog_jaccard else 0.0,
    "transferable":       transferable,
    "dark_transferable":  dark_transferable,
}

print(f"\n{'='*60}")
print(f"PorthoMCL Clustering — GO-MF Annotation Analysis")
print(f"{'='*60}")
print(f"  Total clusters            : {results['n_clusters']:,}")
print(f"  Singletons                : {results['singletons']:,}")
print(f"  Max cluster size          : {results['max_size']:,}")
print(f"  Mean cluster size         : {results['mean_size']:.1f}")
print(f"  Median cluster size       : {results['median_size']:.1f}")
print(f"  Fully dark  (no GO)       : {fully_dark:,}")
print(f"  Mixed                     : {mixed:,}")
print(f"  Fully annotated (GO)      : {fully_annotated:,}")
print(f"  Mean GO annotation rate   : {results['mean_annot_rate']:.3f}")
print(f"  Evaluable (Jaccard)       : {results['n_eval_jaccard']:,}")
print(f"  Mean homog (Jaccard)      : {results['mean_homog_jaccard']:.3f}")
print(f"  Mean homog (Jac+filter)   : {results['mean_homog_jaccard_f']:.3f}")
print(f"  % clusters homog ≥ 0.8   : {results['pct_high_homog']*100:.1f}%")
print(f"  % clusters homog < 0.5   : {results['pct_low_homog']*100:.1f}%")
print(f"  Clusters with transfer    : {transferable:,}")
print(f"  Dark proteins receivable  : {dark_transferable:,}")

print("\nGenerating plots")

COLOR = "#2196F3"

plt.rcParams.update({
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("PorthoMCL Clustering — GO-MF Annotation Analysis",
             fontsize=14, fontweight="bold", y=1.01)

# cluster size distribution
ax = axes[0, 0]
size_counts = Counter(sizes)
xs = sorted(size_counts.keys())
ys = [size_counts[x] for x in xs]
ax.bar([str(x) if x <= 10 else ">10" for x in xs[:11]],
       ys[:11], color=COLOR, edgecolor="white", linewidth=1.5)
ax.set_title("Cluster Size Distribution (size ≤ 10)", fontweight="bold")
ax.set_xlabel("Cluster size")
ax.set_ylabel("Count")

# annotation rate distribution
ax = axes[0, 1]
ax.hist(annot_rates, bins=20, color=COLOR, edgecolor="white", linewidth=1.2)
ax.set_title("GO Annotation Rate per Cluster", fontweight="bold")
ax.set_xlabel("Fraction of members with GO annotation")
ax.set_ylabel("Number of clusters")

# homogeneity distribution
ax = axes[0, 2]
if homog_jaccard:
    ax.hist(homog_jaccard, bins=20, color=COLOR, edgecolor="white", linewidth=1.2)
ax.set_title("GO Homogeneity (Jaccard) Distribution", fontweight="bold")
ax.set_xlabel("Jaccard homogeneity")
ax.set_ylabel("Number of clusters")
ax.set_xlim(0, 1)

# dark / mixed / annotated breakdown
ax = axes[1, 0]
labels = ["Fully dark", "Mixed", "Fully annotated"]
vals   = [fully_dark, mixed, fully_annotated]
colors = ["#607D8B", "#FF9800", "#4CAF50"]
bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.5)
ax.set_title("Cluster Annotation Breakdown", fontweight="bold")
ax.set_ylabel("Number of clusters")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# transfer potential
ax = axes[1, 1]
transfer_vals = [transferable, dark_transferable]
transfer_labels = ["Clusters with\ntransfer potential", "Dark proteins\nreceivable"]
bars = ax.bar(transfer_labels, transfer_vals, color=["#9C27B0", "#3F51B5"],
              edgecolor="white", linewidth=1.5)
ax.set_title("Annotation Transfer Potential", fontweight="bold")
ax.set_ylabel("Count")
for bar, v in zip(bars, transfer_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# high/low homogeneity
ax = axes[1, 2]
hom_labels = ["≥0.8\n(good)", "0.5–0.8\n(moderate)", "<0.5\n(poor)"]
n_high = sum(1 for h in homog_jaccard if h >= 0.8)
n_mid  = sum(1 for h in homog_jaccard if 0.5 <= h < 0.8)
n_low  = sum(1 for h in homog_jaccard if h < 0.5)
hom_vals = [n_high, n_mid, n_low]
bars = ax.bar(hom_labels, hom_vals,
              color=["#4CAF50", "#FF9800", "#F44336"],
              edgecolor="white", linewidth=1.5)
ax.set_title("Homogeneity Buckets", fontweight="bold")
ax.set_ylabel("Number of clusters")
for bar, v in zip(bars, hom_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
fig_path = f"{OUT_DIR}/{FIGURE_PREFIX}_go_analysis.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {fig_path}")

# heterogeneous cluster analysis
print(f"\n{'='*70}")
print(f"Heterogeneous cluster analysis — GO Jaccard")
print(f"{'='*70}")

het_clusters_go = []
for idx, cluster in enumerate(clusters):
    annotated = [m for m in cluster if m in annot_go]
    if len(annotated) < 2:
        continue
    h = compute_homogeneity_jaccard(cluster, annot_go)
    if h is not None and h < 1.0:
        het_clusters_go.append((h, idx, cluster, annotated))

het_clusters_go.sort(key=lambda x: x[0])
print(f"Total heterogeneous clusters (Jaccard < 1.0): {len(het_clusters_go):,}")

buckets_het = {"0.0–0.5": 0, "0.5–0.7": 0, "0.7–0.9": 0, "0.9–1.0": 0}
for h, *_ in het_clusters_go:
    if h < 0.5:   buckets_het["0.0–0.5"] += 1
    elif h < 0.7: buckets_het["0.5–0.7"] += 1
    elif h < 0.9: buckets_het["0.7–0.9"] += 1
    else:         buckets_het["0.9–1.0"] += 1

print("\nHomogeneity bucket breakdown:")
for bucket, count in buckets_het.items():
    bar = "█" * (count * 40 // max(buckets_het.values()) if max(buckets_het.values()) else 0)
    print(f"  {bucket} : {count:>8,}  {bar}")

# most conflicting GO terms
go_conflict: Counter = Counter()
for h, idx, cluster, annotated in het_clusters_go:
    all_go = [annot_go[m] for m in annotated]
    union = set().union(*all_go)
    for term in union:
        has   = sum(1 for s in all_go if term in s)
        hasnt = len(all_go) - has
        if has > 0 and hasnt > 0:
            go_conflict[term] += 1

print(f"\nTop 20 most conflicting GO-MF terms:")
print(f"{'Term':<12} {'Name':<45} {'Clusters':>8}")
print(f"{'-'*68}")
for term, count in go_conflict.most_common(20):
    print(f"  {term}  {go_names.get(term, '?'):<45}  {count:>6,}")

# near-perfect clusters
print(f"\n{'='*70}")
print(f"Near-perfect GO clusters with 1–2 outliers")
print(f"{'='*70}")

# load full annotations for outlier detail
annot_full: dict = defaultdict(list)
for fp in glob.glob(f"{ANNOT_DIR}/*.tsv"):
    for line in open(fp, encoding="utf-8-sig", errors="ignore"):
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            annot_full[parts[0]].append((parts[1], parts[2], parts[3]))
        elif len(parts) == 3:
            annot_full[parts[0]].append((parts[1], parts[2], ""))

near_perfect = []
for idx, cluster in enumerate(clusters):
    annotated = [m for m in cluster if m in annot_go]
    if len(annotated) < 3:
        continue
    term_counts: Counter = Counter()
    for m in annotated:
        term_counts.update(annot_go[m])
    if not term_counts:
        continue
    top_term, top_count = term_counts.most_common(1)[0]
    score = top_count / len(annotated)
    if score < 1.0:
        majority = [m for m in annotated if top_term in     annot_go[m]]
        outliers = [m for m in annotated if top_term not in annot_go[m]]
        if 1 <= len(outliers) <= 2:
            near_perfect.append({
                "idx": idx, "score": score, "size": len(cluster),
                "n_annotated": len(annotated),
                "n_majority": len(majority), "n_outliers": len(outliers),
                "top_term": top_term, "top_name": go_names.get(top_term, "?"),
                "majority": majority, "outliers": outliers,
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
    print(f"{'─'*70}")
    print(f"Rank {rank+1:>3} | Cluster {c['idx']} | Size {c['size']} "
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
            f"{go_names.get(t,'?')}(x{n})" for t, n in top_go.most_common(5))
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

# outlier GO summary
print(f"\n{'='*70}")
print("GO terms carried by outlier proteins")
print(f"{'='*70}\n")

outlier_go: Counter = Counter()
for c in near_perfect:
    for m in c["outliers"]:
        outlier_go.update(annot_go.get(m, set()))

print(f"{'GO term':<12} {'Name':<45} {'Count':>8}")
print(f"{'-'*68}")
for term, count in outlier_go.most_common(20):
    print(f"  {term}  {go_names.get(term, '?'):<45}  {count:>6,}")

# outlier overlap analysis
print(f"\n{'='*70}")
print("Outlier GO overlap with majority")
print(f"{'='*70}\n")

fully_different = 0
partial_overlap = 0
has_top_term    = 0

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
      f"({100*fully_different/total_outliers:.1f}%)  <- likely mis-clustered")
print(f"  Partial GO overlap   : {partial_overlap:,} "
      f"({100*partial_overlap/total_outliers:.1f}%)  <- multi-function proteins")
print(f"  Has majority GO term : {has_top_term:,}  "
      f"({100*has_top_term/total_outliers:.1f}%)  <- annotation noise")

# annotation transfer summary
print(f"\n{'='*70}")
print("GO Annotation transfer potential")
print(f"{'='*70}\n")
print(f"  Transferable clusters    : {transferable:>8,}")
print(f"  Dark proteins receivable : {dark_transferable:>8,}")

print(f"\nDone. Figure saved to {OUT_DIR}/{FIGURE_PREFIX}_go_analysis.png")