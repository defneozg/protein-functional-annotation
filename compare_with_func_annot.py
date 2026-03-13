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

# exclude dbs for structural/biophysical predictors
EXCLUDE_DBS = {
    "MobiDBLite", "Phobius", "TMHMM", "Coils", "ProSitePatterns"
}
EXCLUDE_DOMAIN_IDS = {
    "mobidb-lite", "TMhelix", "TRANSMEMBRANE",
    "NON_CYTOPLASMIC_DOMAIN", "CYTOPLASMIC_DOMAIN",
    "SIGNAL_PEPTIDE", "SIGNAL_PEPTIDE_C_REGION",
    "SIGNAL_PEPTIDE_H_REGION", "SIGNAL_PEPTIDE_N_REGION",
    "Coil"
}

#load annotations
print("Loading the annotations")
annot_func = defaultdict(set) # all annotations
pfam_annot = defaultdict(set) # pfam annotations only

for f in glob.glob(f"{ANNOT_DIR}/*.tsv"):
    for line in open(f, encoding="utf-8-sig", errors="ignore"):
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        prot, db, dom_id = parts[0], parts[1], parts[2]
        if db not in EXCLUDE_DBS and dom_id not in EXCLUDE_DOMAIN_IDS:
            annot_func[prot].add(dom_id)
        if db == "Pfam":
            pfam_annot[prot].add(dom_id)

print(f"Proteins with functional annotation : {len(annot_func):,}")
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
        na = sum(1 for m in c if m in annot_func)
        annot_rates.append(na / n)
        if na == 0:
            fully_dark += 1
        elif na == n:
            fully_annotated += 1
        else:
            mixed += 1

        h_all = compute_homogeneity(c, annot_func)
        h_pfam = compute_homogeneity(c, pfam_annot)
        if h_all is not None: homog_all.append(h_all)
        if h_pfam is not None: homog_pfam.append(h_pfam)

    bad_clusters = [(i, c) for i, c in enumerate(clusters)
                    if compute_homogeneity(c, annot_func) is not None
                    and compute_homogeneity(c, annot_func) < 0.5]

    # tells us why low-homogeneity clusters are bad, checks the domains
    for e, cluster in bad_clusters[:20]:
        print(f"\nCluster {e}, size = {len(cluster)}")
        domains = Counter()
        for m in cluster:
            domains.update(annot_func.get(m, set()))
        print("  Top domains :", domains.most_common(5))

    # annotation transfer potential, check how many dark proteins co-cluster with annotated
    transferable = sum(1 for c in clusters if any(m in annot_func for m in c) and any(m not in annot_func for m in c)) # has annotated AND dark member
    print(f"Clusters where annotation transfer is possible : {transferable}")
    dark_transferable = sum(1 for c in clusters for m in c if m not in annot_func and any(x in annot_func for x in c if x!=m))
    print(f"Dark proteins that can receive annotation : {dark_transferable}")

    # homogeneity vs cluster size, to check if large clusters are less homogenous
    sizes = [len(c) for c in clusters]
    homogs_each = [compute_homogeneity(c, annot_func) or 0 for c in clusters]
    plt.scatter(sizes, homogs_each, alpha=0.1, s=1)
    plt.xscale("log")
    plt.xlabel('Cluster size')
    plt.ylabel('Homogeneity')
    plt.savefig(f"PMIND2026_cluster/figures/size_vs_homogeneity_thresh20_func_{i}.png", dpi=150)

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

# singleton length analysis
print("\n Loading FASTA lengths")
fasta_lengths = {}
try:
    with open("/tmp/fasta_lengths_sorted.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                fasta_lengths[parts[0]] = int(parts[1])
    print(f"  Loaded {len(fasta_lengths)} FASTA lengths")
except FileNotFoundError:
    print(f"  No FASTA lengths found in /tmp/fasta_lengths_sorted.txt")
    fasta_lengths = {}

if fasta_lengths:
    singleton_lens = {} # collecting singleton lens per inflation value
    non_singleton_lens = {}

    for i in valid_I:
        clusters = []
        with open(CLUSTER_FILES[i]) as f:
            for line in f:
                members = line.strip().split("\t")
                if members:
                    clusters.append(members)

        s_lens, ns_lens = [], []
        for cluster in clusters:
            for m in cluster:
                l = fasta_lengths.get(m)
                if l is None:
                    continue
                if len(cluster) == 1:
                    s_lens.append(l)
                else:
                    ns_lens.append(l)
        singleton_lens[i] = s_lens
        non_singleton_lens[i] = ns_lens

        print(f"\n  I={i} singletons ({len(s_lens):,} proteins):")
        print(f"    Min : {min(s_lens) if s_lens else 'N/A'} aa")
        print(f"    Max : {max(s_lens) if s_lens else 'N/A'} aa")
        print(f"    Mean : {np.mean(s_lens):.1f} aa" if s_lens else "    Mean : N/A")
        print(f"    Median : {np.median(s_lens):.0f} aa" if s_lens else "    Median : N/A")
        pct_short = 100 * sum(1 for l in s_lens if l < 100) / len(s_lens) if s_lens else 0
        print(f"    % <100 aa : {pct_short:.1f}%")

    # figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Singleton vs Non-Singleton Sequence Length Distribution", fontsize=13, fontweight="bold")
    axes = axes.flatten()
    bins = [0, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 5000]

    for ax, i in zip(axes, valid_I):
        s = singleton_lens[i]
        ns = non_singleton_lens[i]
        # clip for display
        s_clip = [min(l, 5000) for l in s]
        ns_clip = [min(l, 5000) for l in ns]

        ax.hist(ns_clip, bins=bins, color=COLORS[i], alpha=0.6, label=f"In cluster (n={len(ns):,})", density=True)
        ax.hist(s_clip, bins=bins, color="#E53935", alpha=0.7, label=f"Singleton (n={len(s):,})", density=True)
        ax.set_title(f"I = {i}", fontweight="bold")
        ax.set_xlabel("Sequence length (aa, capped at 5000)")
        ax.set_ylabel("Density")
        ax.axvline(100, color="black", linestyle="--", linewidth=1,alpha=0.5, label="100 aa threshold")
        ax.legend(fontsize=8)

        # annotate % for singletons under 100aa
        pct = 100 * sum(1 for l in s if l < 100) / len(s) if s else 0
        ax.text(0.97, 0.95, f"{pct:.1f}% singletons\nunder 100 aa",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#E53935",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E53935", alpha=0.8))
    plt.tight_layout()
    outpath = f"{OUT_DIR}/singleton_length_distribution.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {outpath}")
    plt.close()

    # figure 2
    buckets = {"<50": (0, 50), "50-99": (50, 100), "100-199": (100, 200), "200-499": (200, 500), "≥500": (500, 999999)}
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(buckets))
    width = 0.18
    offsets = np.linspace(-0.27, 0.27, len(valid_I))
    for offset, i in zip(offsets, valid_I):
        s = singleton_lens[i]
        ns = non_singleton_lens[i]
        pcts = []
        for lo, hi in buckets.values():
            n_s = sum(1 for l in s if lo <= l < hi)
            n_ns = sum(1 for l in ns if lo <= l < hi)
            total = n_s + n_ns
            pcts.append(100 * n_s / total if total > 0 else 0)
        ax.bar(x + offset, pcts, width, label=f"I={i}", color=COLORS[I], edgecolor="white", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(list(buckets.keys()))
    ax.set_xlabel("Sequence length bucket")
    ax.set_ylabel("% of proteins that are singletons")
    ax.set_title("Singleton Rate by Sequence Length and Inflation Value",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    outpath2 = f"{OUT_DIR}/singleton_rate_by_length.png"
    plt.savefig(outpath2, dpi=150, bbox_inches="tight")
    print(f"Figure saved -> {outpath2}")
    plt.close()

# analysing heterogeneous clusters (homology<1.0)
INSPECT_I = 14 # inflation value / clustering param we want to inspect
print(f"\n{'='*70}")
print(f"Heterogenous cluster analysis (Condition = {INSPECT_I})")
print(f"\n{'='*70}")

clusters_inspect = []
with open(CLUSTER_FILES[INSPECT_I]) as f:
    for line in f:
        members = line.strip().split("\t")
        if members:
            clusters_inspect.append(members)

# loading full annotations
print("Loading full annotation details")
annot_full = defaultdict(list)   # protein -> [(db, domain_id, description), ...]
for f in glob.glob(f"{ANNOT_DIR}/*.tsv"):
    for line in open(f, encoding="utf-8-sig", errors="ignore"):
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            annot_full[parts[0]].append((parts[1], parts[2], parts[3]))
        elif len(parts) == 3:
            annot_full[parts[0]].append((parts[1], parts[2], ""))

het_clusters = []
for i, cluster in enumerate(clusters_inspect):
    annotated = [m for m in cluster if m in annot_func]
    if len(annotated) < 2:
        continue

    # compute homogeneity on all db domain ids
    all_domains = []
    for m in annotated:
        all_domains.extend(annot_func[m])
    if not all_domains:
        continue
    top_domain, top_count = Counter(all_domains).most_common(1)[0]
    score = top_count / len(annotated)

    if score < 1.0:
        het_clusters.append((score, i, cluster, annotated, top_domain))

# sort by worst homogeneity
het_clusters.sort(key=lambda x: x[0])
print(f"Total clusters with homogeneity < 1.0 : {len(het_clusters):,}")
print("Showing worst 30 (lowest homogeneity first)\n")

buckets_het = {"0.0-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
for score, *_ in het_clusters:
    if   score < 0.5: buckets_het["0.0-0.5"] += 1
    elif score < 0.7: buckets_het["0.5-0.7"] += 1
    elif score < 0.9: buckets_het["0.7-0.9"] += 1
    else: buckets_het["0.9-1.0"] += 1

print("Homogeneity bucket breakdown:")
for bucket, count in buckets_het.items():
    bar = "█" * (count * 40 // max(buckets_het.values()) if buckets_het.values() else 0)
    print(f"  {bucket} : {count:>8,}  {bar}")
print()

# per cluster detail for worst 30
"""for rank, (score, i, cluster, annotated, top_domain) in enumerate(het_clusters[:30]):
    print(f"{'─'*70}")
    print(f"Rank {rank+1:>3} | Cluster index {i} | Size {len(cluster)} "
          f"| Annotated {len(annotated)} | Homogeneity {score:.3f}")
    print(f"  Dominant domain: {top_domain}")
    print()

    # per protein annotation breakdown
    for m in cluster:
        in_annot = m in annot_func
        marker = "annotated" if in_annot else "x"
        print(f"  [{marker}] {m}")
        if in_annot:
            # group by db, show domain + description
            by_db = defaultdict(list)
            for db, dom_id, desc in annot_full.get(m, []):
                by_db[db].append(f"{dom_id} ({desc})" if desc else dom_id)
            for db, entries in sorted(by_db.items()):
                unique_entries = list(dict.fromkeys(entries))
                print(f"       {db:20s}: {', '.join(unique_entries[:3])}"
                      + (" ..." if len(unique_entries) > 3 else ""))
        else:
            print(f"       (no annotation)")
    print()"""

# domain clash summary across all heterogenous clusters
print(f"\n{'='*70}")
print("Domain clash summary (most frequent conflicting domains)")
print(f"\n{'='*70}")

clash_pairs = Counter()
for score, i, cluster, annotated, top_domain in het_clusters:
    # for each annotated protein not carrying the top domain, record the clash
    for m in annotated:
        protein_domains = annot_func[m]
        if top_domain in protein_domains:
            for d in protein_domains:
                if d != top_domain:
                    pair = tuple(sorted([top_domain, d]))
                    clash_pairs[pair] += 1

print(f"Top 30 most frequent clashing domain pairs:")
print(f"{'Domain A':<35} {'Domain B':<35} {'Count':>8}")
print(f"{'-'*78}")
for (da, db), count in clash_pairs.most_common(30):
    print(f"{da:<35} {db:<35} {count:>8,}")

# check which Pfam domain appears most in heterogenous clusters
print(f"\n{'='*70}")
print("Pfam domains most associated with heterogenous clusters")
print(f"\n{'='*70}")

pfam_in_het = Counter()
for score, i, cluster, annotated, top_domain in het_clusters:
    for m in annotated:
        for db, dom_id, desc in annot_full.get(m, []):
            if db == "Pfam":
                pfam_in_het[(dom_id, desc)] += 1

print(f"{'Pfam ID':<15} {'Description':<45} {'Appearances':>10}")
print(f"{'-'*72}")
for (dom_id, desc), count in pfam_in_het.most_common(20):
    print(f"{dom_id:<15} {desc[:45]:<45} {count:>10,}")

# near-perfect clusters with 1-2 outliers, good candidates for embedding validation
# to check whether they are correctly clustered with distance
print(f"\n{'='*70}")
print(f" Near perfect clusters with few outliers (Condition = {INSPECT_I})")
print(f"{'='*70}")

near_perfect = []
for i, cluster in enumerate(clusters_inspect):
    annotated = [m for m in cluster if m in annot_func]
    if len(annotated) < 3:
        continue

    all_domains = []
    for m in annotated:
        all_domains.extend(annot_func[m])
    if not all_domains:
        continue

    top_domain, top_count = Counter(all_domains).most_common(1)[0]
    score = top_count / len(annotated)

    if score < 1.0:
        majority = [m for m in annotated if top_domain in annot_func[m]]
        outliers = [m for m in annotated if top_domain not in annot_func[m]]

        # for the moment, we keep only those having 1-2 outliers, can change !!
        if 1 <= len(outliers) <= 2:
            near_perfect.append({
                "idx": i,
                "score": score,
                "size": len(cluster),
                "n_annotated": len(annotated),
                "n_majority": len(majority),
                "n_outliers": len(outliers),
                "top_domain": top_domain,
                "majority": majority,
                "outliers": outliers,
                "dark": [m for m in cluster if m not in annot_func],
            })

# descending cluster size
near_perfect.sort(key=lambda x: x["size"], reverse=True)
print(f"Found {len(near_perfect):,} near-perfect clusters with 1-2 outliers\n")

n1 = sum(1 for c in near_perfect if c["n_outliers"] == 1)
n2 = sum(1 for c in near_perfect if c["n_outliers"] == 2)
print(f"  1 outlier  : {n1:,} clusters")
print(f"  2 outliers : {n2:,} clusters\n")

print(f"Top 20 largest near-perfect clusters:\n")
for rank, c in enumerate(near_perfect[:20]):
    print(f"{'─'*70}")
    print(f"Rank {rank+1:>3} | Cluster {c['idx']} | "
          f"Size {c['size']} | Annotated {c['n_annotated']} | "
          f"Homogeneity {c['score']:.3f}")
    print(f"  Majority domain : {c['top_domain']}  "
          f"({c['n_majority']} proteins)")
    print(f"  Dark members    : {len(c['dark'])}")
    print()

    # majority proteins summarized
    if len(c["majority"]) <= 5:
        for m in c["majority"]:
            by_db = defaultdict(list)
            for db, dom_id, desc in annot_full.get(m, []):
                by_db[db].append(f"{dom_id}" + (f" ({desc})" if desc else ""))
            annot_str = " | ".join(
                f"{db}: {', '.join(list(dict.fromkeys(v))[:2])}"
                for db, v in sorted(by_db.items())
            )
            print(f"  [majority] {m}")
            print(f"             {annot_str[:120]}")

    else: # domain summary for large groups
        dom_summary = Counter()
        for m in c["majority"]:
            dom_summary.update(annot_func[m])
        print(f"  [majority] {c['n_majority']} proteins, "
              f"top domains: "
              + ", ".join(f"{d}(x{n})" for d, n in dom_summary.most_common(5)))
    print()

    # outlier proteins in full detail
    for m in c["outliers"]:
        print(f"  [OUTLIER]  {m}")
        by_db = defaultdict(list)
        for db, dom_id, desc in annot_full.get(m, []):
            by_db[db].append(f"{dom_id}" + (f" ({desc})" if desc else ""))
        for db, entries in sorted(by_db.items()):
            unique = list(dict.fromkeys(entries))
            print(f"    {db:20s}: {', '.join(unique[:4])}"
                  + (" ..." if len(unique) > 4 else ""))
        # Flag if outlier shares ANY domain with majority
        shared = annot_func[m] & annot_func[c["majority"][0]] if c["majority"] else set()
        if shared:
            print(f"    Shares domains with majority: {', '.join(list(shared)[:3])}")
        else:
            print(f"    No shared domains with majority")
    print()

# outlier domain summary
print(f"\n{'='*70}")
print("Outlier domains:")
print(f"{'='*70}\n")

outlier_domains = Counter()
outlier_pfam    = Counter()

for c in near_perfect:
    for m in c["outliers"]:
        outlier_domains.update(annot_func[m])
        for db, dom_id, desc in annot_full.get(m, []):
            if db == "Pfam":
                outlier_pfam[(dom_id, desc)] += 1

print(f"Top 20 domain IDs carried by outlier proteins:")
print(f"{'Domain ID':<35} {'Count':>8}")
print(f"{'-'*45}")
for dom, count in outlier_domains.most_common(20):
    print(f"{dom:<35} {count:>8,}")

print(f"\nTop 15 Pfam domains carried by outliers:")
print(f"{'Pfam ID':<15} {'Description':<45} {'Count':>8}")
print(f"{'-'*70}")
for (dom_id, desc), count in outlier_pfam.most_common(15):
    print(f"{dom_id:<15} {desc[:45]:<45} {count:>8,}")

# shared domain check
print(f"\n{'='*70}")
print("Outlier overlap with majority domain")
print(f"{'='*70}\n")

fully_different = 0   # outlier shares zero domains with majority
partial_overlap = 0   # outlier shares some but not the top domain
multi_domain    = 0   # outlier has top domain PLUS extra domains

for c in near_perfect:
    maj_domains = annot_func.get(c["majority"][0], set()) if c["majority"] else set()
    for m in c["outliers"]:
        out_domains = annot_func[m]
        if c["top_domain"] in out_domains:
            multi_domain += 1 # has the top domain, just also has others
        elif out_domains & maj_domains:
            partial_overlap += 1 # shares some domains but not the key one
        else:
            fully_different += 1 # completely different annotation

total_outliers = sum(c["n_outliers"] for c in near_perfect)
print(f"Total outlier proteins analysed : {total_outliers:,}")
print(f"  Fully different annotation    : {fully_different:,} "
      f"({100*fully_different/total_outliers:.1f}%)  <- likely mis-clustered")
print(f"  Partial domain overlap        : {partial_overlap:,} "
      f"({100*partial_overlap/total_outliers:.1f}%)  <- multi-domain proteins")
print(f"  Has top domain + extras       : {multi_domain:,}  "
      f"({100*multi_domain/total_outliers:.1f}%)  <- annotation noise")
