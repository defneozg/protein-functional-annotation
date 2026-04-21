import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict, Counter

BASE     = "orthofinder_backup"
STATS    = f"{BASE}/Comparative_Genomics_Statistics"
DUPS     = f"{BASE}/Gene_Duplication_Events"
TREE_DIR = f"{BASE}/Species_Tree"
OUT_DIR  = "PMIND2026_cluster/figures/orthofinder_viz"

os.makedirs(OUT_DIR, exist_ok=True)

def short_name(s):
    parts = s.split("_")
    if len(parts) >= 3:
        return f"{parts[0][:3]}.{parts[1][:3]} {parts[-1]}"
    return s[:20]

# 1. Statistics_Overall.tsv — key \t value, no header
print("Reading Statistics_Overall.tsv")
overall = {}
with open(f"{STATS}/Statistics_Overall.tsv") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            overall[parts[0].strip()] = parts[1].strip()
print(f"  {len(overall)} stats: {list(overall.keys())[:5]}...")

# 2. Statistics_PerSpecies.tsv
print("Reading Statistics_PerSpecies.tsv")
per_species = {}
per_species_species = []

with open(f"{STATS}/Statistics_PerSpecies.tsv") as f:
    for i, line in enumerate(f):
        parts = line.rstrip("\n").split("\t")
        if i == 0:
            # parts[0] is empty, rest are species names
            per_species_species = [p.strip() for p in parts[1:] if p.strip()]
        else:
            metric = parts[0].strip()
            if not metric:
                continue
            per_species[metric] = {}
            for sp, val in zip(per_species_species, parts[1:]):
                try:
                    per_species[metric][sp] = float(val.strip())
                except ValueError:
                    per_species[metric][sp] = 0.0

print(f"  {len(per_species_species)} species")
print(f"  Metrics found: {list(per_species.keys())}")

species = per_species_species

# extract key vectors
n_genes   = [per_species.get("Number of genes", {}).get(sp, 0.0) for sp in species]
n_in_og   = [per_species.get("Number of genes in orthogroups", {}).get(sp, 0.0) for sp in species]
n_unassign = [g - o for g, o in zip(n_genes, n_in_og)]
# compute percentage ourselves
pct_in_og = [o / g * 100 if g > 0 else 0.0 for g, o in zip(n_genes, n_in_og)]

print(f"  Sample pct_in_og: {[round(p,1) for p in pct_in_og[:5]]}")
print(f"  Sample n_genes:   {n_genes[:5]}")

# 3. Species × species ortholog matrices
def load_species_matrix(path):
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return [], {}
    sp_cols = []
    data = {}
    with open(path) as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if i == 0:
                sp_cols = [p.strip() for p in parts[1:] if p.strip()]
            else:
                row_sp = parts[0].strip()
                if not row_sp:
                    continue
                data[row_sp] = {}
                for col_sp, val in zip(sp_cols, parts[1:]):
                    try:
                        data[row_sp][col_sp] = float(val.strip())
                    except ValueError:
                        data[row_sp][col_sp] = 0.0
    return sp_cols, data

print("Reading ortholog stats matrices...")
_, mat_121 = load_species_matrix(f"{STATS}/OrthologuesStats_one-to-one.tsv")
_, mat_12m = load_species_matrix(f"{STATS}/OrthologuesStats_one-to-many.tsv")
_, mat_m21 = load_species_matrix(f"{STATS}/OrthologuesStats_many-to-one.tsv")
_, mat_m2m = load_species_matrix(f"{STATS}/OrthologuesStats_many-to-many.tsv")
print(f"  Loaded matrices for {len(mat_121)} species")

# 4. Duplications per node and per orthogroup
print("Reading duplication stats")
dup_per_node = {}
path_dpn = f"{STATS}/Duplications_per_Species_Tree_Node.tsv"
if os.path.exists(path_dpn):
    with open(path_dpn) as f:
        for i, line in enumerate(f):
            if i == 0: continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try: dup_per_node[parts[0]] = int(parts[1])
                except ValueError: pass

dup_per_og = []
path_dpo = f"{STATS}/Duplications_per_Orthogroup.tsv"
if os.path.exists(path_dpo):
    with open(path_dpo) as f:
        for i, line in enumerate(f):
            if i == 0: continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try: dup_per_og.append(int(parts[1]))
                except ValueError: pass

print(f"  {len(dup_per_node)} tree nodes, {len(dup_per_og)} orthogroups")

# PLOT 1: Overview — stats table + per-species bars
print("\nPlotting Figure 1: Overview")

fig = plt.figure(figsize=(20, 14))
fig.suptitle("OrthoFinder — Statistics Overview", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.42)

# Panel 1: Overall stats table
ax = fig.add_subplot(gs[0, 0])
ax.axis("off")
display_keys = [
    "Number of species", "Number of genes",
    "Number of genes in orthogroups", "Number of unassigned genes",
    "Percentage of genes in orthogroups", "Number of orthogroups",
    "Number of species-specific orthogroups",
    "Mean orthogroup size", "Median orthogroup size",
    "G50 (assigned genes)", "O50 (assigned genes)",
    "Number of orthogroups with all species present",
    "Number of single-copy orthogroups",
]
rows = [[k, overall.get(k, "N/A")] for k in display_keys if k in overall]
if rows:
    tbl = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.auto_set_column_width([0, 1])
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#9C27B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F3E5F5")
ax.set_title("Overall Statistics", fontweight="bold", fontsize=10)

# Panel 2: % genes in orthogroups per species (sorted)
ax = fig.add_subplot(gs[0, 1])
pairs = sorted(zip(pct_in_og, [short_name(s) for s in species]), reverse=True)
pcts_s, names_s = zip(*pairs)
y = np.arange(len(names_s))
cols = ["#4CAF50" if p >= 80 else "#FF9800" if p >= 60 else "#F44336" for p in pcts_s]
ax.barh(list(y), list(pcts_s), color=cols, edgecolor="white", linewidth=0.4)
ax.set_yticks(list(y))
ax.set_yticklabels(list(names_s), fontsize=4.5)
ax.set_xlabel("% genes in orthogroups")
ax.axvline(80, color="green", linestyle="--", linewidth=1, alpha=0.6, label="80%")
ax.legend(fontsize=7)
ax.set_xlim(0, 108)
for i, p in enumerate(pcts_s):
    ax.text(p + 0.3, i, f"{p:.0f}%", va="center", fontsize=3.5)
ax.set_title("% Genes in Orthogroups per Species\n(sorted)", fontweight="bold", fontsize=10)

# Panel 3: Number of genes (sorted)
ax = fig.add_subplot(gs[0, 2])
pairs2 = sorted(zip(n_genes, [short_name(s) for s in species]), reverse=True)
ng_s, names_s2 = zip(*pairs2)
y2 = np.arange(len(names_s2))
ax.barh(list(y2), list(ng_s), color="#2196F3", edgecolor="white", linewidth=0.4)
ax.set_yticks(list(y2))
ax.set_yticklabels(list(names_s2), fontsize=4.5)
ax.set_xlabel("Number of genes")
ax.set_title("Number of Genes per Species\n(sorted)", fontweight="bold", fontsize=10)

# Panel 4: Duplications per orthogroup
ax = fig.add_subplot(gs[1, 0])
if dup_per_og:
    c = Counter(dup_per_og)
    xs = list(range(min(16, max(c.keys()) + 1)))
    ys = [c.get(x, 0) for x in xs]
    ax.bar(xs, ys, color="#E91E63", edgecolor="white", linewidth=1.1)
    ax.set_yscale("log")
    ax.set_xlabel("Duplications per orthogroup")
    ax.set_ylabel("Number of orthogroups (log)")
    n0 = c.get(0, 0)
    ax.text(0.97, 0.97, f"{n0/len(dup_per_og)*100:.1f}% with 0 dups",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.set_title("Duplications per Orthogroup", fontweight="bold", fontsize=10)

# Panel 5: Top nodes by duplication count
ax = fig.add_subplot(gs[1, 1])
if dup_per_node:
    top = sorted(dup_per_node.items(), key=lambda x: -x[1])[:30]
    nn, nc = zip(*top)
    y3 = np.arange(len(nn))
    ax.barh(list(y3), list(nc), color="#FF5722", edgecolor="white", linewidth=0.7)
    ax.set_yticks(list(y3))
    ax.set_yticklabels([n[:22] for n in nn], fontsize=5.5)
    ax.set_xlabel("Gene duplications")
ax.set_title("Gene Duplications per\nSpecies Tree Node (top 30)", fontweight="bold", fontsize=10)

# Panel 6: Assigned vs unassigned stacked, top 30
ax = fig.add_subplot(gs[1, 2])
triples = sorted(zip(n_genes, n_in_og, n_unassign, [short_name(s) for s in species]),
                 reverse=True)[:30]
ng_t, nin_t, nun_t, names_t = zip(*triples)
y4 = np.arange(len(names_t))
ax.barh(list(y4), list(nin_t), color="#4CAF50", edgecolor="white",
        linewidth=0.4, label="In orthogroups")
ax.barh(list(y4), list(nun_t), left=list(nin_t), color="#F44336",
        edgecolor="white", linewidth=0.4, label="Unassigned")
ax.set_yticks(list(y4))
ax.set_yticklabels(list(names_t), fontsize=5)
ax.set_xlabel("Number of genes")
ax.legend(fontsize=7, loc="lower right")
ax.set_title("Assigned vs Unassigned Genes\n(top 30 by gene count)", fontweight="bold", fontsize=10)

plt.tight_layout()
out1 = f"{OUT_DIR}/orthofinder_overview.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"  Saved: {out1}")
plt.close()

# PLOT 2: Ortholog relationship types
print("Plotting Figure 2: Ortholog types")

all_sp = list(mat_121.keys())
if all_sp:
    def row_sum(mat, sp): return sum(mat.get(sp, {}).values())

    sum_121 = [row_sum(mat_121, s) for s in all_sp]
    sum_12m = [row_sum(mat_12m, s) for s in all_sp]
    sum_m21 = [row_sum(mat_m21, s) for s in all_sp]
    sum_m2m = [row_sum(mat_m2m, s) for s in all_sp]
    sum_tot = [a + b + c + d for a, b, c, d in zip(sum_121, sum_12m, sum_m21, sum_m2m)]

    order = sorted(range(len(all_sp)), key=lambda i: -sum_tot[i])
    names_o   = [short_name(all_sp[i]) for i in order]
    s121_o    = [sum_121[i] for i in order]
    s12m_o    = [sum_12m[i] for i in order]
    sm21_o    = [sum_m21[i] for i in order]
    sm2m_o    = [sum_m2m[i] for i in order]
    stot_o    = [sum_tot[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(10, len(all_sp) * 0.2)),
                              gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("OrthoFinder — Ortholog Relationship Types per Species",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    y = np.arange(len(names_o))
    bottoms = np.zeros(len(names_o))
    for vals, color, label in [
        (s121_o, "#4CAF50", "one-to-one"),
        (s12m_o, "#2196F3", "one-to-many"),
        (sm21_o, "#FF9800", "many-to-one"),
        (sm2m_o, "#E91E63", "many-to-many"),
    ]:
        ax.barh(list(y), vals, 0.7, left=list(bottoms),
                color=color, edgecolor="white", linewidth=0.3, label=label)
        bottoms += np.array(vals)
    ax.set_yticks(list(y))
    ax.set_yticklabels(list(names_o), fontsize=5)
    ax.set_xlabel("Total ortholog relationships (sum across all species pairs)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Ortholog pairs by relationship type", fontweight="bold", fontsize=10)

    ax = axes[1]
    pct_121 = [s / t * 100 if t > 0 else 0 for s, t in zip(s121_o, stot_o)]
    cols121 = ["#4CAF50" if p >= 60 else "#FF9800" if p >= 40 else "#F44336" for p in pct_121]
    ax.barh(list(y), pct_121, color=cols121, edgecolor="white", linewidth=0.3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([], fontsize=5)
    ax.set_xlabel("% one-to-one")
    ax.set_xlim(0, 100)
    ax.axvline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("% one-to-one", fontweight="bold", fontsize=10)

    plt.tight_layout()
    out2 = f"{OUT_DIR}/orthofinder_ortholog_types.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out2}")
    plt.close()

# PLOT 3: One-to-one ortholog heatmap (top 40)
print("Plotting Figure 3: 1-1 ortholog heatmap")

sp_121 = list(mat_121.keys())
if sp_121:
    totals_121 = {sp: sum(mat_121[sp].values()) for sp in sp_121}
    top40 = sorted(totals_121, key=lambda s: -totals_121[s])[:40]
    top40_short = [short_name(s) for s in top40]

    mat_arr = np.array([[mat_121.get(si, {}).get(sj, 0.0)
                         for sj in top40] for si in top40])

    fig, ax = plt.subplots(figsize=(16, 14))
    fig.suptitle("OrthoFinder — One-to-One Ortholog Pairs Heatmap\n(top 40 species)", fontsize=13, fontweight="bold")
    im = ax.imshow(mat_arr, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(top40)))
    ax.set_yticks(range(len(top40)))
    ax.set_xticklabels(top40_short, rotation=90, fontsize=6)
    ax.set_yticklabels(top40_short, fontsize=6)
    plt.colorbar(im, ax=ax, label="One-to-one ortholog pairs", shrink=0.7)

    plt.tight_layout()
    out3 = f"{OUT_DIR}/orthofinder_121_heatmap.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out3}")
    plt.close()

# PLOT 4: Gene duplication events (Duplications.tsv)
print("Plotting Figure 4: Duplication detail")

dup_file = f"{DUPS}/Duplications.tsv"
if os.path.exists(dup_file):
    dups_detail = []
    headers_d = []
    with open(dup_file) as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if i == 0:
                headers_d = parts
            else:
                dups_detail.append(dict(zip(headers_d, parts)))
    print(f"  {len(dups_detail)} events, columns: {headers_d}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("OrthoFinder — Gene Duplication Events Detail",
                 fontsize=13, fontweight="bold")

    # support distribution
    ax = axes[0]
    support_col = next((h for h in headers_d if "support" in h.lower()), None)
    if support_col:
        supports = []
        for d in dups_detail:
            try: supports.append(float(d[support_col]))
            except (ValueError, KeyError): pass
        if supports:
            ax.hist(supports, bins=25, color="#E91E63", edgecolor="white", linewidth=1.0)
            ax.axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="0.5 threshold")
            ax.set_xlabel("Support score")
            ax.set_ylabel("Number of events")
            ax.legend(fontsize=8)
            high = sum(1 for s in supports if s >= 0.5)
            ax.text(0.97, 0.97, f"{high:,} events\n>= 0.5 ({high/len(supports)*100:.1f}%)",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_title("Duplication Support Score", fontweight="bold", fontsize=10)

    # per node
    ax = axes[1]
    node_col = next((h for h in headers_d if "node" in h.lower()), None)
    if node_col:
        nc = Counter(d[node_col] for d in dups_detail if node_col in d)
        top = nc.most_common(30)
        if top:
            nn, nv = zip(*top)
            y = np.arange(len(nn))
            ax.barh(list(y), list(nv), color="#FF5722", edgecolor="white", linewidth=0.7)
            ax.set_yticks(list(y))
            ax.set_yticklabels([n[:22] for n in nn], fontsize=5.5)
            ax.set_xlabel("Number of events")
    ax.set_title("Events per Species Tree Node\n(top 30)", fontweight="bold", fontsize=10)

    # per OG
    ax = axes[2]
    og_col = next((h for h in headers_d
                   if "orthogroup" in h.lower() or h.strip() == "OG"), None)
    if og_col:
        oc = Counter(d[og_col] for d in dups_detail if og_col in d)
        counts = list(oc.values())
        cc = Counter(counts)
        xs = sorted(cc.keys())[:15]
        ys = [cc[x] for x in xs]
        ax.bar(xs, ys, color="#9C27B0", edgecolor="white", linewidth=1.1)
        ax.set_yscale("log")
        ax.set_xlabel("Events per orthogroup")
        ax.set_ylabel("Number of orthogroups (log)")
    ax.set_title("Events per Orthogroup", fontweight="bold", fontsize=10)

    plt.tight_layout()
    out4 = f"{OUT_DIR}/orthofinder_duplications.png"
    plt.savefig(out4, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out4}")
    plt.close()

# PLOT 5: Species tree order with stats
print("Plotting Figure 5: Species tree order")

tree_file = f"{TREE_DIR}/SpeciesTree_rooted.txt"
newick_str = ""
if os.path.exists(tree_file):
    with open(tree_file) as f:
        newick_str = f.read().strip()

# try ete3 just for tree parsing (no TreeStyle needed)
leaves_u = []
try:
    from ete3 import Tree
    t = Tree(newick_str, format=1)
    for leaf in t.get_leaves():
        leaves_u.append(leaf.name)
    print(f"  ete3 Tree parsing: {len(leaves_u)} leaves")
except Exception as e:
    print(f"  ete3 not available ({e}), using regex leaf extraction")
    # Fallback: regex extraction from newick
    raw_leaves = re.findall(r'([A-Za-z][A-Za-z0-9_]+)(?::\d*\.?\d+(?:e[+-]?\d+)?)?[,)]',
                            newick_str)
    seen = set()
    for l in raw_leaves:
        if l not in seen and len(l) > 5 and not l.startswith("N"):
            seen.add(l)
            leaves_u.append(l)
    print(f"  Regex extraction: {len(leaves_u)} leaves")

if leaves_u:
    # build per-leaf data
    pct_leaf = [pct_in_og[species.index(l)] if l in species else 0.0 for l in leaves_u]
    ng_leaf  = [n_genes[species.index(l)]   if l in species else 0.0 for l in leaves_u]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(14, len(leaves_u) * 0.24)),
                              gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("OrthoFinder — Species in Tree Order",
                 fontsize=13, fontweight="bold")

    y = np.arange(len(leaves_u))
    leaf_short = [short_name(l) for l in leaves_u]

    ax = axes[0]
    cols = ["#4CAF50" if p >= 80 else "#FF9800" if p >= 60 else "#F44336"
            for p in pct_leaf]
    ax.barh(list(y), pct_leaf, color=cols, edgecolor="white", linewidth=0.4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(list(leaf_short), fontsize=5.5)
    ax.set_xlabel("% genes in orthogroups")
    ax.axvline(80, color="grey", linestyle="--", linewidth=0.8, alpha=0.5, label="80%")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 108)
    for i, p in enumerate(pct_leaf):
        if p > 0:
            ax.text(p + 0.3, i, f"{p:.0f}%", va="center", fontsize=4)
    ax.set_title("% Genes in Orthogroups\n(species in tree order)",
                 fontweight="bold", fontsize=10)

    ax = axes[1]
    ax.barh(list(y), ng_leaf, color="#2196F3", edgecolor="white", linewidth=0.4)
    ax.set_yticks(list(y))
    ax.set_yticklabels([], fontsize=5.5)
    ax.set_xlabel("Number of genes")
    ax.set_title("Gene count\n(tree order)", fontweight="bold", fontsize=10)

    plt.tight_layout()
    out5 = f"{OUT_DIR}/species_tree_stats.png"
    plt.savefig(out5, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out5}")
    plt.close()

print("\nDone. Figures saved to:", OUT_DIR)
print("  orthofinder_overview.png")
print("  orthofinder_ortholog_types.png")
print("  orthofinder_121_heatmap.png")
print("  orthofinder_duplications.png")
print("  species_tree_stats.png")