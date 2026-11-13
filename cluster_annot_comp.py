import os
import pickle
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PATHS = {
    "OrthoFinder": "orthofinder_backup/Orthogroups/Orthogroups.txt",
    "PorthoMCL":   "porthomcl_backup/8.all.ort.group",
    "MCL":         "PMIND2026_cluster/diamond_clusters_I20.txt",
}
TAXON_MAPPING = "porthomcl_backup/taxon_code_mapping.txt"
MAPPING_DIR   = "PMIND2026_cluster/go_mappings"
OUT_DIR       = "PMIND2026_cluster/figures/report_figs"
MIN_SIZE      = 2

os.makedirs(OUT_DIR, exist_ok=True)

METHOD_COLORS = {
    "OrthoFinder": "#FEC89A",
    "PorthoMCL":   "#FCD5CE",
    "MCL":         "#F49595",
}
# taxon mapping
print("Loading taxon code mapping")
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

# GO annotations
print("Loading GO annotations")
with open(f"{MAPPING_DIR}/annot_go_mf_pruned.pkl", "rb") as f:
    annot_go: dict = pickle.load(f)
print(f"  Proteins with GO-MF annotation: {len(annot_go):,}")

def jaccard_homogeneity(members, annotation_dict):
    annotated = [m for m in members if m in annotation_dict]
    if len(annotated) < 2:
        return None
    sets = [annotation_dict[m] for m in annotated]
    scores = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            if u:
                scores.append(len(sets[i] & sets[j]) / len(u))
    return float(np.mean(scores)) if scores else None

def homog_summary(homog_list):
    if not homog_list:
        return {}
    return {
        "n":        len(homog_list),
        "mean":     np.mean(homog_list),
        "median":   np.median(homog_list),
        "pct_high": sum(1 for h in homog_list if h >= 0.8) / len(homog_list) * 100,
        "pct_mid":  sum(1 for h in homog_list if 0.5 <= h < 0.8) / len(homog_list) * 100,
        "pct_low":  sum(1 for h in homog_list if h < 0.5) / len(homog_list) * 100,
    }

# cluster parsers
def load_orthofinder(path, min_size=2):
    clusters = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, rest = line.split(":", 1)
            members = [m for m in rest.strip().split() if m]
            if len(members) >= min_size:
                clusters.append(members)
    return clusters

def load_porthomcl(path, min_size=2):
    clusters = []
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
            members = [decode_porthomcl_id(m) for m in members_coded if m]
            if len(members) >= min_size:
                clusters.append(members)
    return clusters

def load_mcl(path, min_size=2):
    clusters = []
    with open(path) as f:
        for line in f:
            members = [m for m in line.strip().split() if m]
            if len(members) >= min_size:
                clusters.append(members)
    return clusters

loaders = {
    "OrthoFinder": load_orthofinder,
    "PorthoMCL":   load_porthomcl,
    "MCL":         load_mcl,
}

all_results = {}

for method, path in PATHS.items():
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping {method}")
        continue

    print(f"\nLoading {method}...")
    clusters = loaders[method](path, min_size=MIN_SIZE)
    print(f"  {len(clusters):,} clusters, computing homogeneity")

    dark_sizes        = []
    annotated_sizes   = []
    mixed_sizes       = []
    mixed_annot_frac  = []
    mixed_annot_n     = []
    mixed_dark_n      = []
    annot_homog       = []
    mixed_homog       = []

    for i, members in enumerate(clusters):
        if i % 50000 == 0:
            print(f"    {i:,} / {len(clusters):,}")

        n  = len(members)
        na = sum(1 for m in members if m in annot_go)
        nd = n - na

        if na == 0:
            dark_sizes.append(n)
        elif na == n:
            annotated_sizes.append(n)
            h = jaccard_homogeneity(members, annot_go)
            if h is not None:
                annot_homog.append(h)
        else:
            mixed_sizes.append(n)
            mixed_annot_frac.append(na / n)
            mixed_annot_n.append(na)
            mixed_dark_n.append(nd)
            h = jaccard_homogeneity(members, annot_go)
            if h is not None:
                mixed_homog.append(h)

    total = len(clusters)
    all_results[method] = {
        "dark_sizes":       dark_sizes,
        "annotated_sizes":  annotated_sizes,
        "mixed_sizes":      mixed_sizes,
        "mixed_annot_frac": mixed_annot_frac,
        "mixed_annot_n":    mixed_annot_n,
        "mixed_dark_n":     mixed_dark_n,
        "annot_homog":      annot_homog,
        "mixed_homog":      mixed_homog,
        "total":            total,
    }

    print(f"\n{'='*60}")
    print(f"  {method}")
    print(f"{'='*60}")
    print(f"  Total clusters          : {total:,}")
    print(f"  Fully dark              : {len(dark_sizes):,} ({len(dark_sizes)/total*100:.1f}%)")
    print(f"  Fully annotated         : {len(annotated_sizes):,} ({len(annotated_sizes)/total*100:.1f}%)")
    print(f"  Mixed                   : {len(mixed_sizes):,} ({len(mixed_sizes)/total*100:.1f}%)")

    print(f"\n  -- Fully dark clusters --")
    if dark_sizes:
        print(f"    Size range   : {min(dark_sizes)} - {max(dark_sizes)}")
        print(f"    Mean size    : {np.mean(dark_sizes):.1f}")
        print(f"    Median size  : {np.median(dark_sizes):.1f}")
        print(f"    Size 2       : {sum(1 for s in dark_sizes if s == 2):,}")
        print(f"    Size 3-5     : {sum(1 for s in dark_sizes if 3 <= s <= 5):,}")
        print(f"    Size 6-10    : {sum(1 for s in dark_sizes if 6 <= s <= 10):,}")
        print(f"    Size 11-50   : {sum(1 for s in dark_sizes if 11 <= s <= 50):,}")
        print(f"    Size >50     : {sum(1 for s in dark_sizes if s > 50):,}")

    print(f"\n  -- Fully annotated clusters --")
    if annotated_sizes:
        print(f"    Size range   : {min(annotated_sizes)} - {max(annotated_sizes)}")
        print(f"    Mean size    : {np.mean(annotated_sizes):.1f}")
        print(f"    Median size  : {np.median(annotated_sizes):.1f}")
        print(f"    Size 2       : {sum(1 for s in annotated_sizes if s == 2):,}")
        print(f"    Size 3-5     : {sum(1 for s in annotated_sizes if 3 <= s <= 5):,}")
        print(f"    Size 6-10    : {sum(1 for s in annotated_sizes if 6 <= s <= 10):,}")
        print(f"    Size 11-50   : {sum(1 for s in annotated_sizes if 11 <= s <= 50):,}")
        print(f"    Size >50     : {sum(1 for s in annotated_sizes if s > 50):,}")
        hs = homog_summary(annot_homog)
        if hs:
            print(f"    -- Jaccard homogeneity (n={hs['n']:,} evaluable) --")
            print(f"    Mean Jaccard : {hs['mean']:.3f}")
            print(f"    Median       : {hs['median']:.3f}")
            print(f"    >= 0.8 (high): {hs['pct_high']:.1f}%")
            print(f"    0.5-0.8 (mid): {hs['pct_mid']:.1f}%")
            print(f"    < 0.5 (low)  : {hs['pct_low']:.1f}%")

    print(f"\n  -- Mixed clusters --")
    if mixed_sizes:
        print(f"    Size range           : {min(mixed_sizes)} - {max(mixed_sizes)}")
        print(f"    Mean size            : {np.mean(mixed_sizes):.1f}")
        print(f"    Median size          : {np.median(mixed_sizes):.1f}")
        print(f"    Mean annotated frac  : {np.mean(mixed_annot_frac):.3f}")
        print(f"    Median annot. frac   : {np.median(mixed_annot_frac):.3f}")
        print(f"    Frac annot < 0.25    : {sum(1 for f in mixed_annot_frac if f < 0.25):,} "
              f"({sum(1 for f in mixed_annot_frac if f < 0.25)/len(mixed_annot_frac)*100:.1f}%)")
        print(f"    Frac annot 0.25-0.5  : {sum(1 for f in mixed_annot_frac if 0.25 <= f < 0.5):,} "
              f"({sum(1 for f in mixed_annot_frac if 0.25 <= f < 0.5)/len(mixed_annot_frac)*100:.1f}%)")
        print(f"    Frac annot 0.5-0.75  : {sum(1 for f in mixed_annot_frac if 0.5 <= f < 0.75):,} "
              f"({sum(1 for f in mixed_annot_frac if 0.5 <= f < 0.75)/len(mixed_annot_frac)*100:.1f}%)")
        print(f"    Frac annot >= 0.75   : {sum(1 for f in mixed_annot_frac if f >= 0.75):,} "
              f"({sum(1 for f in mixed_annot_frac if f >= 0.75)/len(mixed_annot_frac)*100:.1f}%)")
        total_mixed_proteins = sum(mixed_sizes)
        total_mixed_annot    = sum(mixed_annot_n)
        total_mixed_dark     = sum(mixed_dark_n)
        print(f"    Total proteins in mixed  : {total_mixed_proteins:,}")
        print(f"      annotated              : {total_mixed_annot:,} ({total_mixed_annot/total_mixed_proteins*100:.1f}%)")
        print(f"      dark                   : {total_mixed_dark:,} ({total_mixed_dark/total_mixed_proteins*100:.1f}%)")
        print(f"    Dark proteins receivable : {total_mixed_dark:,}")
        hs = homog_summary(mixed_homog)
        if hs:
            print(f"    -- Jaccard homogeneity on annotated members (n={hs['n']:,} evaluable) --")
            print(f"    Mean Jaccard : {hs['mean']:.3f}")
            print(f"    Median       : {hs['median']:.3f}")
            print(f"    >= 0.8 (high): {hs['pct_high']:.1f}%")
            print(f"    0.5-0.8 (mid): {hs['pct_mid']:.1f}%")
            print(f"    < 0.5 (low)  : {hs['pct_low']:.1f}%")

size_bins   = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100, 200, 500, 1e9]
size_labels = ["2","3","4","5","6","7","8","9","10","11-15","16-20",
               "21-30","31-50","51-100","101-200","201-500",">500"]

def bin_sizes(sizes):
    counts = [0] * (len(size_bins) - 1)
    for s in sizes:
        for i, (lo, hi) in enumerate(zip(size_bins[:-1], size_bins[1:])):
            if lo <= s < hi:
                counts[i] += 1
                break
    return counts

for method, res in all_results.items():
    dark_sizes       = res["dark_sizes"]
    annotated_sizes  = res["annotated_sizes"]
    mixed_sizes      = res["mixed_sizes"]
    mixed_annot_frac = res["mixed_annot_frac"]
    mixed_annot_n    = res["mixed_annot_n"]
    mixed_dark_n     = res["mixed_dark_n"]
    annot_homog      = res["annot_homog"]
    mixed_homog      = res["mixed_homog"]
    total            = res["total"]

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f"{method} — Cluster Annotation & Homogeneity Analysis", fontsize=14, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.38)

    # Row 0: size distributions
    for col, (label, sizes, cat_color) in enumerate([
        ("Fully Dark",      dark_sizes,      "#607D8B"),
        ("Fully Annotated", annotated_sizes, "#4CAF50"),
        ("Mixed",           mixed_sizes,     "#FF9800"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        counts = bin_sizes(sizes)
        nonzero = [(l, c) for l, c in zip(size_labels, counts) if c > 0]
        if nonzero:
            lbls, cnts = zip(*nonzero)
            ax.bar(range(len(lbls)), cnts, color=cat_color, edgecolor="white", linewidth=1.2)
            ax.set_xticks(range(len(lbls)))
            ax.set_xticklabels(lbls, rotation=45, ha="right", fontsize=7)
            ax.set_yscale("log")
        ax.set_title(f"{label}\n({len(sizes):,} clusters)", fontweight="bold", fontsize=10)
        ax.set_ylabel("Count (log scale)")
        ax.set_xlabel("Cluster size")
        if sizes:
            ax.text(0.97, 0.95, f"mean={np.mean(sizes):.1f}\nmed={np.median(sizes):.0f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # Row 0 col 3: breakdown bar
    ax = fig.add_subplot(gs[0, 3])
    cats       = ["Fully dark", "Mixed", "Fully annotated"]
    vals       = [len(dark_sizes), len(mixed_sizes), len(annotated_sizes)]
    cats_colors = ["#607D8B", "#FF9800", "#4CAF50"]
    bars = ax.bar(cats, vals, color=cats_colors, edgecolor="white", linewidth=1.5)
    ax.set_title("Cluster breakdown", fontweight="bold", fontsize=10)
    ax.set_ylabel("Count")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.002,
                f"{v:,}\n({v/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)

    # Row 1: homogeneity panels
    ax = fig.add_subplot(gs[1, 0])
    if annot_homog:
        ax.hist(annot_homog, bins=25, color="#4CAF50", edgecolor="white", linewidth=1.1)
        ax.axvline(np.mean(annot_homog), color="red", linestyle="--", linewidth=1.5, label=f"mean={np.mean(annot_homog):.3f}")
        ax.axvline(np.median(annot_homog), color="navy", linestyle=":", linewidth=1.5, label=f"median={np.median(annot_homog):.3f}")
        ax.legend(fontsize=8)
    ax.set_title("Fully annotated\nJaccard homogeneity", fontweight="bold", fontsize=10)
    ax.set_xlabel("Jaccard homogeneity")
    ax.set_ylabel("Number of clusters")
    ax.set_xlim(0, 1)

    # fully annotated buckets
    ax = fig.add_subplot(gs[1, 1])
    if annot_homog:
        bucket_labels_h = ["<0.5\n(poor)", "0.5-0.8\n(mid)", ">=0.8\n(high)"]
        counts_h = [
            sum(1 for h in annot_homog if h < 0.5),
            sum(1 for h in annot_homog if 0.5 <= h < 0.8),
            sum(1 for h in annot_homog if h >= 0.8),
        ]
        colors_h = ["#F44336", "#FF9800", "#4CAF50"]
        bars = ax.bar(bucket_labels_h, counts_h, color=colors_h, edgecolor="white", linewidth=1.5)
        for bar, v in zip(bars, counts_h):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f"{v:,}\n({v/len(annot_homog)*100:.1f}%)",
                    ha="center", va="bottom", fontsize=8)
    ax.set_title("Fully annotated\nhomogeneity buckets", fontweight="bold", fontsize=10)
    ax.set_ylabel("Number of clusters")

    # mixed histogram
    ax = fig.add_subplot(gs[1, 2])
    if mixed_homog:
        ax.hist(mixed_homog, bins=25, color="#FF9800", edgecolor="white", linewidth=1.1)
        ax.axvline(np.mean(mixed_homog), color="red", linestyle="--", linewidth=1.5, label=f"mean={np.mean(mixed_homog):.3f}")
        ax.axvline(np.median(mixed_homog), color="navy", linestyle=":", linewidth=1.5, label=f"median={np.median(mixed_homog):.3f}")
        ax.legend(fontsize=8)
    ax.set_title("Mixed clusters\nJaccard homogeneity (annotated members)", fontweight="bold", fontsize=10)
    ax.set_xlabel("Jaccard homogeneity")
    ax.set_ylabel("Number of clusters")
    ax.set_xlim(0, 1)

    # mixed buckets
    ax = fig.add_subplot(gs[1, 3])
    if mixed_homog:
        bucket_labels_h = ["<0.5\n(poor)", "0.5-0.8\n(mid)", ">=0.8\n(high)"]
        counts_h = [
            sum(1 for h in mixed_homog if h < 0.5),
            sum(1 for h in mixed_homog if 0.5 <= h < 0.8),
            sum(1 for h in mixed_homog if h >= 0.8),
        ]
        colors_h = ["#F44336", "#FF9800", "#4CAF50"]
        bars = ax.bar(bucket_labels_h, counts_h, color=colors_h, edgecolor="white", linewidth=1.5)
        for bar, v in zip(bars, counts_h):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f"{v:,}\n({v/len(mixed_homog)*100:.1f}%)",
                    ha="center", va="bottom", fontsize=8)
    ax.set_title("Mixed clusters\nhomogeneity buckets", fontweight="bold", fontsize=10)
    ax.set_ylabel("Number of clusters")

    # Row 2: mixed cluster detail panels
    if mixed_sizes:
        # annotated fraction distribution
        ax = fig.add_subplot(gs[2, 0])
        ax.hist(mixed_annot_frac, bins=20, color="#FF9800", edgecolor="white", linewidth=1.2)
        ax.set_title("Mixed clusters\nAnnotated fraction distribution", fontweight="bold", fontsize=10)
        ax.set_xlabel("Fraction of annotated proteins")
        ax.set_ylabel("Number of clusters")
        ax.set_xlim(0, 1)
        ax.axvline(np.mean(mixed_annot_frac), color="red", linestyle="--", linewidth=1.5, label=f"mean={np.mean(mixed_annot_frac):.2f}")
        ax.axvline(np.median(mixed_annot_frac), color="navy", linestyle=":", linewidth=1.5, label=f"median={np.median(mixed_annot_frac):.2f}")
        ax.legend(fontsize=8)

        # annotated fraction vs size scatter
        ax = fig.add_subplot(gs[2, 1])
        sample_n = min(5000, len(mixed_sizes))
        idx = np.random.choice(len(mixed_sizes), sample_n, replace=False)
        xs = [mixed_sizes[i] for i in idx]
        ys = [mixed_annot_frac[i] for i in idx]
        ax.scatter(xs, ys, alpha=0.3, s=6, color="#FF9800")
        ax.set_xscale("log")
        ax.set_title(f"Mixed: annotated fraction vs size\n(sample n={sample_n:,})", fontweight="bold", fontsize=10)
        ax.set_xlabel("Cluster size (log scale)")
        ax.set_ylabel("Annotated fraction")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.5)

        # homogeneity vs annotated fraction scatter
        ax = fig.add_subplot(gs[2, 2])
        if mixed_homog:
            # rebuild evaluable fracs in order
            evaluable_fracs = []
            evaluable_homog = list(mixed_homog)
            h_idx = 0
            for members in loaders[method](PATHS[method], min_size=MIN_SIZE):
                if h_idx >= len(evaluable_homog):
                    break
                n  = len(members)
                na = sum(1 for m in members if m in annot_go)
                if 0 < na < n:
                    annotated = [m for m in members if m in annot_go]
                    if len(annotated) >= 2:
                        evaluable_fracs.append(na / n)
                        h_idx += 1
            sample_n2 = min(3000, len(evaluable_homog))
            idx2 = np.random.choice(len(evaluable_homog), sample_n2, replace=False)
            xs2 = [evaluable_fracs[i] for i in idx2]
            ys2 = [evaluable_homog[i] for i in idx2]
            sc = ax.scatter(xs2, ys2, alpha=0.3, s=6, c=xs2, cmap="RdYlGn", vmin=0, vmax=1)
            plt.colorbar(sc, ax=ax, label="Annotated fraction")
        ax.set_title("Mixed: homogeneity vs annotated fraction\n(sample)", fontweight="bold", fontsize=10)
        ax.set_xlabel("Annotated fraction")
        ax.set_ylabel("Jaccard homogeneity")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # stacked bar: annotated vs dark by size bucket
        ax = fig.add_subplot(gs[2, 3])
        bucket_edges = [2, 3, 5, 10, 20, 50, 200, 1e9]
        bucket_labels_mix = ["2", "3-4", "5-9", "10-19", "20-49", "50-199", ">=200"]
        annot_by_bucket = [0.0] * len(bucket_labels_mix)
        dark_by_bucket  = [0.0] * len(bucket_labels_mix)
        for size, na, nd in zip(mixed_sizes, mixed_annot_n, mixed_dark_n):
            for i, (lo, hi) in enumerate(zip(bucket_edges[:-1], bucket_edges[1:])):
                if lo <= size < hi:
                    annot_by_bucket[i] += na
                    dark_by_bucket[i]  += nd
                    break
        x = np.arange(len(bucket_labels_mix))
        ax.bar(x, annot_by_bucket, label="Annotated", color="#4CAF50", edgecolor="white", linewidth=1.2)
        ax.bar(x, dark_by_bucket, bottom=annot_by_bucket, label="Dark", color="#607D8B", edgecolor="white", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_labels_mix, rotation=30, ha="right", fontsize=8)
        ax.set_title("Mixed: annotated vs dark\nproteins by size bucket", fontweight="bold", fontsize=10)
        ax.set_xlabel("Cluster size bucket")
        ax.set_ylabel("Number of proteins")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/annot_dist_{method.lower()}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {out_path}")
    plt.close()

print("\nGenerating combined comparison plot")

methods = list(all_results.keys())

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("Cluster Annotation & Homogeneity — All Methods Compared", fontsize=14, fontweight="bold")

# Row 0: annotated fraction in mixed clusters
for col, method in enumerate(methods):
    ax = axes[0, col]
    res = all_results[method]
    color = METHOD_COLORS[method]
    mixed_annot_frac = res["mixed_annot_frac"]
    if not mixed_annot_frac:
        ax.set_visible(False)
        continue
    ax.hist(mixed_annot_frac, bins=25, color=color, edgecolor="white", linewidth=1.1, density=True)
    ax.set_title(f"{method}\nAnnotated fraction in mixed clusters", fontweight="bold", fontsize=10)
    ax.set_xlabel("Fraction annotated")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.axvline(np.mean(mixed_annot_frac), color="red", linestyle="--", linewidth=1.5, label=f"mean={np.mean(mixed_annot_frac):.2f}")
    ax.legend(fontsize=8)

# Row 1: overlapping homogeneity distributions per method
for col, method in enumerate(methods):
    ax = axes[1, col]
    res = all_results[method]
    annot_h = res["annot_homog"]
    mixed_h = res["mixed_homog"]
    if annot_h:
        ax.hist(annot_h, bins=25, color="#4CAF50", alpha=0.6, edgecolor="white", linewidth=1.0, density=True, label="Fully annotated")
    if mixed_h:
        ax.hist(mixed_h, bins=25, color="#FF9800", alpha=0.6, edgecolor="white", linewidth=1.0, density=True, label="Mixed")
    ax.set_title(f"{method}\nHomogeneity distribution", fontweight="bold", fontsize=10)
    ax.set_xlabel("Jaccard homogeneity")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)

# Row 2 col 0: cluster type breakdown
ax = axes[2, 0]
x = np.arange(3)
w = 0.25
cat_labels = ["Fully dark", "Mixed", "Fully annotated"]
for i, method in enumerate(methods):
    res = all_results[method]
    total = res["total"]
    vals = [
        len(res["dark_sizes"]) / total * 100,
        len(res["mixed_sizes"]) / total * 100,
        len(res["annotated_sizes"]) / total * 100,
    ]
    ax.bar(x + i * w, vals, w, label=method, color=METHOD_COLORS[method], edgecolor="white", linewidth=1.2)
ax.set_xticks(x + w)
ax.set_xticklabels(cat_labels)
ax.set_title("Cluster type breakdown (%)", fontweight="bold", fontsize=10)
ax.set_ylabel("% of all clusters")
ax.legend(fontsize=8)

# Row 2 col 1: mean homogeneity comparison
ax = axes[2, 1]
x = np.arange(len(methods))
w = 0.35
mean_annot_h = [np.mean(all_results[m]["annot_homog"]) if all_results[m]["annot_homog"] else 0 for m in methods]
mean_mixed_h = [np.mean(all_results[m]["mixed_homog"])  if all_results[m]["mixed_homog"]  else 0 for m in methods]
ax.bar(x - w/2, mean_annot_h, w, label="Fully annotated", color="#4CAF50", edgecolor="white", linewidth=1.2)
ax.bar(x + w/2, mean_mixed_h, w, label="Mixed", color="#FF9800", edgecolor="white", linewidth=1.2)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, 1)
ax.set_title("Mean Jaccard homogeneity\nby cluster type", fontweight="bold", fontsize=10)
ax.set_ylabel("Mean Jaccard")
ax.legend(fontsize=8)
for i, (va, vm) in enumerate(zip(mean_annot_h, mean_mixed_h)):
    ax.text(i - w/2, va + 0.01, f"{va:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + w/2, vm + 0.01, f"{vm:.3f}", ha="center", va="bottom", fontsize=8)

# Row 2 col 2: dark proteins receivable
ax = axes[2, 2]
dark_receivable = [sum(all_results[m]["mixed_dark_n"]) for m in methods]
bars = ax.bar(methods, dark_receivable, color=[METHOD_COLORS[m] for m in methods], edgecolor="white", linewidth=1.5)
ax.set_title("Dark proteins receivable\nvia annotation transfer", fontweight="bold", fontsize=10)
ax.set_ylabel("Number of dark proteins")
for bar, v in zip(bars, dark_receivable):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
            f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
out_path = f"{OUT_DIR}/annot_dist_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {out_path}")
plt.close()

print("\nDone. All figures saved to:", OUT_DIR)