"""
Compare clustering results (for now plain MCL, PorthoMCL, OrthoFinder)

analysis:

- Coverage comparison : how many proteins each method assigns
- Pairwise ARI / NMI : global similarity on shared proteins
- Split / merge analysis : Jaccard between best-matching cluster pairs
- Containment heatmap : subset relationships
"""

import os
import pickle
from collections import defaultdict
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import random

# config
PATHS = {
    "orthofinder": "orthofinder_backup/Orthogroups/Orthogroups.txt",
    "porthomcl": "porthomcl_backup/8.all.ort.group",
    "mcl": "PMIND2026_cluster/diamond_thresh20_clusters_I20.txt"
}

TAXON_MAPPING = "porthomcl_backup/taxon_code_mapping.txt"
OUT_DIR = "PMIND2026_cluster/figures"
MIN_SIZE = 2 # for now we only ignore singletons
MAX_ARI_PROTEINS = 6_000_000 # subsample for ARI if the dataset is too large
TOP_N_SPLITMERGE = 500_000 # we only analyse top N largest clusters for split/merge

os.makedirs(OUT_DIR, exist_ok=True)

# taxon code mapping for PorthoMCL
print("loading taxon code mapping")
code_to_species = {}
with open(TAXON_MAPPING) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        code, fname = line.split("\t")
        code_to_species[code] = fname.replace(".fasta", "")

def decode_porthomcl_id(coded_id):
    if "|" not in coded_id:
        return coded_id
    code, seqnum = coded_id.split("|", 1)
    return f"{code_to_species.get(code, code)}|{seqnum}"

# cluster file parsers
def load_orthofinder(path, min_size=2):
    clusters = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            og_id, rest = line.split(":", 1)
            members = frozenset(m for m in rest.strip().split() if m)
            if len(members) >= min_size:
                clusters[og_id] = members
    return clusters

def load_porthomcl(path, min_size=2):
    clusters = {}
    idx = 0
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            try:
                int(parts[0])
                members_coded = parts[1:]
            except ValueError:
                members_coded = parts
            members = frozenset(decode_porthomcl_id(m) for m in members_coded if m)
            if len(members) >= min_size:
                clusters[f"PMC{idx:07d}"] = members
                idx += 1
    return clusters

def load_mcl(path, min_size=2):
    clusters = {}
    idx = 0
    with open(path) as f:
        for line in f:
            members = frozenset(m for m in line.strip().split() if m)
            if len(members) >= min_size:
                clusters[f"MCL{idx:07d}"] = members
                idx += 1
    return clusters

# load all clusterings
print("loading all clusterings")
loaders = {
    "orthofinder": load_orthofinder,
    "porthomcl": load_porthomcl,
    "mcl": load_mcl
}

clusterings = {}
for name, path in PATHS.items():
    if not os.path.exists(path):
        print(f"WARNING: file {path} does not exist, skipping {name}")
        continue
    loader = loaders[name]
    clusters = loader(path, min_size=MIN_SIZE)
    clusterings[name] = clusters
    total_seqs = sum(len(v) for v in clusters.values())
    print(f"{name:15s}: {len(clusters):>8,} clusters -- {total_seqs:>9,} sequences")

method_names = list(clusterings.keys())

# coverage analysis
print("\n" + "="*70)
print("1. COVERAGE ANALYSIS")
print("="*70)

all_proteins = set()
method_proteins = {}
for name, clusters in clusterings.items():
    proteins = set().union(*clusters.values())
    method_proteins[name] = proteins
    all_proteins |= proteins

total_universe = len(all_proteins)
print(f"\nTotal unique proteins across all methods : {total_universe:,}")
print(f"\n{'Method':<15} {'Clustered':>10} {'% of union':>12} {'Clusters':>10}")
print("-"*50)
for name in method_names:
    n = len(method_proteins[name])
    n_clusters = len(clusterings[name])
    print(f"  {name:<13} {n:>10,} {n/total_universe*100:>11.1f}%  {n_clusters:>10,}")

# pairwise overlaps
print(f"\nPairwise protein overlap :")
print(f"{'Methods':<30} {'Shared':>10} {'% of smaller':>14}")
print("-"*56)
for a, b in combinations(method_names, 2):
    shared = method_proteins[a] & method_proteins[b]
    smaller = min(len(method_proteins[a]), len(method_proteins[b]))
    print(f"  {a} ∩ {b:<15} {len(shared):>10,} {len(shared)/smaller*100:>13.1f}%")

# coverage bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Clustering Coverage Comparison", fontsize=13, fontweight="bold")
colors = {"orthofinder": "#9C27B0", "porthomcl": "#2196F3", "mcl": "#4CAF50"}
color_list = [colors.get(n, "#607D8B") for n in method_names]

# proteins assigned
ax = axes[0]
vals = [len(method_proteins[n]) for n in method_names]
bars = ax.bar(method_names, vals, color=color_list, edgecolor="white", linewidth=1.5)
ax.set_title("Proteins Assigned to Clusters", fontweight="bold")
ax.set_ylabel("Count")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
            f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(total_universe, color="red", linestyle="--", linewidth=1, label=f"Total universe ({total_universe:,})")
ax.legend(fontsize=8)

# number of clusters
ax = axes[1]
vals2 = [len(clusterings[n]) for n in method_names]
bars2 = ax.bar(method_names, vals2, color=color_list, edgecolor="white", linewidth=1.5)
ax.set_title("Number of Clusters (size ≥ 2)", fontweight="bold")
ax.set_ylabel("Count")
for bar, v in zip(bars2, vals2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparison_coverage.png", dpi=150, bbox_inches="tight")
print(f"\n  Saved: {OUT_DIR}/comparison_coverage.png")

# pairwise ARI / NMI on shared proteins
print("\n" + "="*70)
print("2. PAIRWISE ARI / NMI on shared proteins")
print("="*70)

def make_label_array(proteins_list, clusters):
    """assign integer cluster label to each protein, -1 if unclustered"""
    protein_to_cluster = {}
    for cid, (og_id, members) in enumerate(clusters.items()):
        for p in members:
            protein_to_cluster[p] = cid
    return np.array([protein_to_cluster.get(p, -1) for p in proteins_list])

ari_matrix = np.full((len(method_names), len(method_names)), np.nan)
nmi_matrix = np.full((len(method_names), len(method_names)), np.nan)

for i, name_a in enumerate(method_names):
    ari_matrix[i, i] = 1.0
    nmi_matrix[i, i] = 1.0
    for j, name_b in enumerate(method_names):
        if j <= i:
            continue

        # we only compare proteins present in BOTH methods
        shared = sorted(method_proteins[name_a] & method_proteins[name_b])
        if not shared:
            print(f"  {name_a} vs {name_b}: no shared proteins, skipping")
            continue

        # subsample if too large
        if len(shared) > MAX_ARI_PROTEINS:
            random.seed(42)
            shared = random.sample(shared, MAX_ARI_PROTEINS)
            print(f"  {name_a} vs {name_b}: subsampled to {MAX_ARI_PROTEINS:,} proteins")

        labels_a = make_label_array(shared, clusterings[name_a])
        labels_b = make_label_array(shared, clusterings[name_b])

        # removing proteins that are not assigned in either (label = -1)
        mask = (labels_a >= 0) & (labels_b >= 0)
        la, lb = labels_a[mask], labels_b[mask]

        if len(la) < 2:
            continue

        ari = adjusted_rand_score(la, lb)
        nmi = normalized_mutual_info_score(la, lb, average_method='arithmetic')

        ari_matrix[i, j] = ari_matrix[j, i] = ari
        nmi_matrix[i, j] = nmi_matrix[j, i] = nmi

        print(f"  {name_a} vs {name_b}:")
        print(f"  Shared proteins (both assigned) : {mask.sum():,}")
        print(f"  ARI = {ari:.4f} (1.0 = identical, 0.0 = random)")
        print(f"  NMI = {nmi:.4f} (1.0 = identical, 0.0 = independent)")

# plot ARI / NMI heatmaps
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Pairwise Clustering Similarity", fontsize=13, fontweight="bold")

for ax, matrix, title in zip(axes,
                             [ari_matrix, nmi_matrix],
                             ["Adjusted Rand Index (ARI)", "Normalized Mutual Info (NMI)"]):
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd")
    ax.set_xticks(range(len(method_names)))
    ax.set_yticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=30, ha="right")
    ax.set_yticklabels(method_names)
    ax.set_title(title, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(method_names)):
        for j in range(len(method_names)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if v > 0.6 else "black")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparison_ari_nmi.png", dpi=150, bbox_inches="tight")
print(f"\n  Saved: {OUT_DIR}/comparison_ari_nmi.png")

# split / merge analysis : for each cluster in method A, find the best-matching cluster in B
# low Jaccard means the cluster is split across many clusters in other methods
print("\n" + "="*70)
print("3. SPLIT / MERGE ANALYSIS")
print("="*70)

def split_merge_analysis(clusters_a, name_a, clusters_b, name_b, top_n=TOP_N_SPLITMERGE):
    # build reverse index : protein -> cluster_id
    protein_to_b = {}
    for og_id, members in clusters_b.items():
        for p in members:
            protein_to_b[p] = og_id

    results = []
    # sort by size descending, take top_n
    sorted_a = sorted(clusters_a.items(), key=lambda x: len(x[1]), reverse=True)[:top_n]

    for og_id_a, members_a in sorted_a:
        # find all B clusters that share at least one protein with this A cluster
        b_overlaps = defaultdict(int)
        for p in members_a:
            if p in protein_to_b:
                b_overlaps[protein_to_b[p]] += 1

        if not b_overlaps:
            results.append((og_id_a, len(members_a), 0.0, None, 0))
            continue

        # Jaccard for each overlapping cluster
        best_jaccard = 0.0
        best_b_id = None
        for og_id_b, overlap in b_overlaps.items():
            members_b = clusters_b[og_id_b]
            union = len(members_a) + len(members_b) - overlap
            jaccard = overlap / union if union > 0 else 0.0
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_b_id = og_id_b

        results.append((og_id_a, len(members_a), best_jaccard, best_b_id, len(b_overlaps)))

    return results


# run split/merge for each pair
splitmerge_results = {}
for name_a, name_b in combinations(method_names, 2):
    print(f"  Analysing {name_a} -> {name_b}")
    res = split_merge_analysis(clusterings[name_a], name_a, clusterings[name_b], name_b)
    splitmerge_results[(name_a, name_b)] = res

    jaccards = [r[2] for r in res]
    n_perfect = sum(1 for j in jaccards if j >= 0.9)
    n_good = sum(1 for j in jaccards if 0.5 <= j < 0.9)
    n_split = sum(1 for j in jaccards if 0.1 <= j < 0.5)
    n_lost = sum(1 for j in jaccards if j < 0.1)

    print(f"    Clusters analysed : {len(res):,}")
    print(f"    Best-match Jaccard — mean={np.mean(jaccards):.3f}  median={np.median(jaccards):.3f}")
    print(f"    Jaccard >= 0.9 (near-identical match) : {n_perfect:>8,} ({n_perfect / len(res) * 100:.1f}%)")
    print(f"    Jaccard  0.5-0.9 (partial match)      : {n_good:>8,} ({n_good / len(res) * 100:.1f}%)")
    print(f"    Jaccard  0.1-0.5 (split)              : {n_split:>8,} ({n_split / len(res) * 100:.1f}%)")
    print(f"    Jaccard < 0.1    (lost/no match)      : {n_lost:>8,} ({n_lost / len(res) * 100:.1f}%)")
    print()

    # run reverse direction
    print(f"  Analysing {name_b} → {name_a}...")
    res_rev = split_merge_analysis(clusterings[name_b], name_b, clusterings[name_a], name_a)
    splitmerge_results[(name_b, name_a)] = res_rev

    jaccards_rev = [r[2] for r in res_rev]
    n_perfect = sum(1 for j in jaccards_rev if j >= 0.9)
    n_good = sum(1 for j in jaccards_rev if 0.5 <= j < 0.9)
    n_split = sum(1 for j in jaccards_rev if 0.1 <= j < 0.5)
    n_lost = sum(1 for j in jaccards_rev if j < 0.1)

    print(f"    Clusters analysed : {len(res_rev):,}")
    print(f"    Best-match Jaccard — mean={np.mean(jaccards_rev):.3f}  median={np.median(jaccards_rev):.3f}")
    print(f"    Jaccard >= 0.9 (near-identical match) : {n_perfect:>8,} ({n_perfect / len(res_rev) * 100:.1f}%)")
    print(f"    Jaccard  0.5-0.9 (partial match)      : {n_good:>8,} ({n_good / len(res_rev) * 100:.1f}%)")
    print(f"    Jaccard  0.1-0.5 (split)              : {n_split:>8,} ({n_split / len(res_rev) * 100:.1f}%)")
    print(f"    Jaccard < 0.1    (lost/no match)      : {n_lost:>8,} ({n_lost / len(res_rev) * 100:.1f}%)")
    print()

n_pairs = len(method_names) * (len(method_names) - 1)
fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 5))
if n_pairs == 1:
    axes = [axes]

fig.suptitle("Split/Merge Analysis — Best-Match Jaccard Distribution",
             fontsize=13, fontweight="bold")

ax_idx = 0
for name_a, name_b in [(a, b) for a in method_names for b in method_names if a != b]:
    if (name_a, name_b) not in splitmerge_results:
        continue
    res = splitmerge_results[(name_a, name_b)]
    jaccards = [r[2] for r in res]

    ax = axes[ax_idx]
    ax_idx += 1

    buckets = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
    bucket_labels = ["<0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "≥0.9"]
    counts = []
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        counts.append(sum(1 for j in jaccards if lo <= j < hi))

    bucket_colors = ["#F44336", "#FF7043", "#FF9800", "#FFC107", "#8BC34A", "#4CAF50"]
    bars = ax.bar(bucket_labels, counts, color=bucket_colors, edgecolor="white", linewidth=1.5)
    ax.set_title(f"{name_a} → {name_b}\n(best-match Jaccard)", fontweight="bold", fontsize=10)
    ax.set_xlabel("Jaccard similarity")
    ax.set_ylabel("Number of clusters")
    ax.tick_params(axis='x', rotation=30)
    for bar, v in zip(bars, counts):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    f"{v:,}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparison_splitmerge.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {OUT_DIR}/comparison_splitmerge.png")

# containment analysis : what fraction of clusters in A are subsets of clusters in B
print("\n" + "="*70)
print("4. CONTAINMENT ANALYSIS")
print("="*70)

def containment_analysis(clusters_a, name_a, clusters_b, name_b):
    """for each cluster in A, we check whether it is mostly contained in a single cluster in B
       |A_i ∩ B_j| / |A_i|
    """
    protein_to_b = {}
    for og_id, members in clusters_b.items():
        for p in members:
            protein_to_b[p] = og_id

    containments = []
    for og_id_a, members_a in clusters_a.items():
        b_overlaps = defaultdict(int)
        for p in members_a:
            if p in protein_to_b:
                b_overlaps[protein_to_b[p]] += 1

        if not b_overlaps:
            containments.append(0.0)
            continue

        best_overlap = max(b_overlaps.values())
        containments.append(best_overlap / len(members_a))

    return containments

print(f"{'Direction':<35} {'Mean contain.':>14} {'% fully contained':>18} {'% >50% contained':>18}")
print("-" * 88)

containment_data = {}
for name_a, name_b in [(a, b) for a in method_names for b in method_names if a != b]:
    conts = containment_analysis(clusterings[name_a], name_a, clusterings[name_b], name_b)
    containment_data[(name_a, name_b)] = conts
    fully = sum(1 for c in conts if c >= 0.95) / len(conts) * 100
    majority = sum(1 for c in conts if c >= 0.5) / len(conts) * 100
    print(f"  {name_a} clusters in {name_b:<15} {np.mean(conts):>13.3f}  {fully:>17.1f}%  {majority:>17.1f}%")


n_directions = len(method_names) * (len(method_names) - 1)
fig, axes = plt.subplots(1, n_directions, figsize=(6 * n_directions, 5))
if n_directions == 1:
    axes = [axes]

fig.suptitle("Containment Analysis — How much of each cluster fits inside another?", fontsize=12, fontweight="bold")
ax_idx = 0
for name_a, name_b in [(a, b) for a in method_names for b in method_names if a != b]:
    conts = containment_data[(name_a, name_b)]
    ax = axes[ax_idx]
    ax_idx += 1

    ax.hist(conts, bins=20, color=colors.get(name_a, "#607D8B"),
            edgecolor="white", linewidth=1.2)
    ax.set_title(f"{name_a} → {name_b}\ncontainment", fontweight="bold", fontsize=10)
    ax.set_xlabel("Fraction of cluster contained in best-matching cluster")
    ax.set_ylabel("Number of clusters")
    ax.set_xlim(0, 1)
    ax.axvline(np.mean(conts), color="red", linestyle="--",
               linewidth=1.5, label=f"mean={np.mean(conts):.2f}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparison_containment.png", dpi=150, bbox_inches="tight")
print(f"\n  Saved: {OUT_DIR}/comparison_containment.png")

# detailed examples of split/merge large clusters
print("\n" + "="*70)
print("5. DETAILED EXAMPLES : largest clusters with interesting split/merge")
print("="*70)

if len(method_names) >= 2:
    name_a, name_b = method_names[0], method_names[1]
    res = splitmerge_results.get((name_a, name_b), [])

    # find largest clusters with Jaccard in 0.1-0.7 (interesting split cases)
    interesting = [(og_id, size, jacc, match_b, n_partial)
                   for og_id, size, jacc, match_b, n_partial in res
                   if 0.1 <= jacc <= 0.7 and size >= 10]
    interesting.sort(key=lambda x: -x[1])

    print(f"\nTop 10 largest {name_a} clusters that are split in {name_b}:")
    print(f"  (Jaccard 0.1-0.7 = partial match = likely split across multiple clusters)\n")
    print(f"  {'Cluster':<15} {'Size':>6} {'Best Jaccard':>13} {'# partial matches in ' + name_b:>25}")
    print(f"  {'-' * 65}")
    for og_id, size, jacc, match_b, n_partial in interesting[:10]:
        print(f"  {og_id:<15} {size:>6,} {jacc:>13.3f} {n_partial:>25,}")

    # find clusters with very high Jaccard (near-identical across methods)
    perfect = [(og_id, size, jacc, match_b)
               for og_id, size, jacc, match_b, _ in res
               if jacc >= 0.9 and size >= 5]
    perfect.sort(key=lambda x: -x[1])

    print(f"\nTop 10 largest {name_a} clusters with near-identical match in {name_b}:")
    print(f"  (Jaccard >= 0.9 = both methods agree on this cluster)\n")
    print(f"  {'Cluster A':<15} {'Size':>6} {'Jaccard':>8} {'Best match in ' + name_b:>20}")
    print(f"  {'-' * 55}")
    for og_id, size, jacc, match_b in perfect[:10]:
        print(f"  {og_id:<15} {size:>6,} {jacc:>8.3f} {str(match_b):<20}")

# summary table
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"\n{'Metric':<40}", end="")
for name in method_names:
    print(f"  {name:>15}", end="")
print()
print("-" * (40 + 17 * len(method_names)))

metrics = {
    "Clusters (size >= 2)": lambda n: f"{len(clusterings[n]):,}",
    "Proteins assigned": lambda n: f"{len(method_proteins[n]):,}",
    "% of total universe": lambda n: f"{len(method_proteins[n]) / total_universe * 100:.1f}%",
    "Mean cluster size": lambda n: f"{np.mean([len(v) for v in clusterings[n].values()]):.1f}",
    "Median cluster size": lambda n: f"{np.median([len(v) for v in clusterings[n].values()]):.1f}",
    "Largest cluster": lambda n: f"{max(len(v) for v in clusterings[n].values()):,}",
}

for metric, fn in metrics.items():
    print(f"  {metric:<38}", end="")
    for name in method_names:
        print(f"  {fn(name):>15}", end="")
    print()

print(f"\nAll figures saved to: {OUT_DIR}/")
print("  comparison_coverage.png")
print("  comparison_ari_nmi.png")
print("  comparison_splitmerge.png")
print("  comparison_containment.png")
print("\nDone.")











