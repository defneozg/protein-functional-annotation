"""
Characterize clusters by GO homogeneity tiers and analyse dark protein gains.

Strong : Jaccard ≥ 0.8
Mediumn : 0.5 ≤ Jaccard < 0.8
Weak : Jaccard < 0.5

For each configuration (baseline + each tau), the script reports:
  1. Number of clusters in each tier and how it changes
  2. Number of dark proteins receivable per tier
  3. Where the NEW dark proteins (gained vs baseline) actually land
  4. Figures: tier breakdown, protein flow, per-tier size distributions
"""

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
SEP = "=" * 75

# tier definitions
TIER_FORT = "Strong (≥0.8)"
TIER_MOYENNE = "Medium (0.5–0.8)"
TIER_FAIBLE = "Weak (<0.5)"
TIER_UNEVALUABLE = "Not evaluated"  # <2 annotated members

TIER_ORDER = [TIER_FORT, TIER_MOYENNE, TIER_FAIBLE]
TIER_COLORS = {
    TIER_FORT:    "#4CAF50",
    TIER_MOYENNE: "#FF9800",
    TIER_FAIBLE:  "#F44336",
    TIER_UNEVALUABLE: "#9E9E9E",
}

def classify_tier(jaccard):
    if jaccard is None:
        return TIER_UNEVALUABLE
    if jaccard >= 0.8:
        return TIER_FORT
    if jaccard >= 0.5:
        return TIER_MOYENNE
    return TIER_FAIBLE

# i/o helpers
def load_clusters(path: Path) -> list[list[str]]:
    clusters = []
    with open(path) as fh:
        for line in fh:
            members = line.strip().split("\t")
            if members and members[0]:
                clusters.append(members)
    return clusters

def load_go_annotations(pkl_path: Path) -> dict[str, set]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def load_go_names(obo_path: Path) -> dict[str, str]:
    names = {}
    current_id = None
    with open(obo_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("id: GO:"):
                current_id = line.split("id: ")[1].strip()
            elif line.startswith("name:") and current_id:
                names[current_id] = line.split("name: ")[1].strip()
                current_id = None
    return names

# core analysis
def compute_homogeneity_jaccard(members, annot):
    annotated = [m for m in members if m in annot]
    if len(annotated) < 2:
        return None
    sets = [annot[m] for m in annotated]
    scores = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            if u:
                scores.append(len(sets[i] & sets[j]) / len(u))
    return float(np.mean(scores)) if scores else None

def analyse_clustering(clusters, annot, label=""):
    """per-cluster stats and return a dict with tier-level summaries.
    returns:
        result dict with keys:
          - per_cluster: list of dicts (one per cluster)
          - tier_counts: {tier: n_clusters}
          - tier_dark:   {tier: n_dark_proteins}
          - dark_receivable_set: set of dark protein IDs that are in mixed clusters
          - dark_per_tier: {tier: set(dark_protein_ids)}
    """
    per_cluster = []
    tier_counts = Counter()
    tier_dark = Counter() # dark proteins per tier
    tier_mixed_clusters = Counter() # mixed clusters per tier
    dark_receivable_set = set()
    dark_per_tier = defaultdict(set)
    tier_sizes = defaultdict(list) # cluster sizes per tier
    tier_homogs = defaultdict(list) # homogeneity values per tier
    fully_dark = 0
    fully_annotated = 0
    mixed = 0

    for idx, members in enumerate(clusters):
        n = len(members)
        annotated = [m for m in members if m in annot]
        dark = [m for m in members if m not in annot]
        na = len(annotated)
        nd = len(dark)

        if na == 0:
            fully_dark += 1 # no tier assignment for fully dark clusters
            continue
        elif nd == 0:
            fully_annotated += 1 # fully annotated: compute homog but no dark to receive
            h = compute_homogeneity_jaccard(members, annot)
            tier = classify_tier(h)
            tier_counts[tier] += 1
            if h is not None:
                tier_homogs[tier].append(h)
            tier_sizes[tier].append(n)
            per_cluster.append({
                "idx": idx, "size": n, "n_annot": na, "n_dark": nd,
                "homog": h, "tier": tier, "type": "fully_annotated",
            })
            continue
        else:
            mixed += 1

        h = compute_homogeneity_jaccard(members, annot)
        tier = classify_tier(h)
        tier_counts[tier] += 1
        tier_mixed_clusters[tier] += 1
        tier_dark[tier] += nd
        tier_sizes[tier].append(n)
        if h is not None:
            tier_homogs[tier].append(h)

        for m in dark:
            dark_receivable_set.add(m)
            dark_per_tier[tier].add(m)

        per_cluster.append({
            "idx": idx, "size": n, "n_annot": na, "n_dark": nd,
            "homog": h, "tier": tier, "type": "mixed",
        })

    return {
        "label": label,
        "n_clusters": len(clusters),
        "fully_dark": fully_dark,
        "fully_annotated": fully_annotated,
        "mixed": mixed,
        "per_cluster": per_cluster,
        "tier_counts": dict(tier_counts),
        "tier_mixed_clusters": dict(tier_mixed_clusters),
        "tier_dark": dict(tier_dark),
        "tier_sizes": dict(tier_sizes),
        "tier_homogs": dict(tier_homogs),
        "dark_receivable_set": dark_receivable_set,
        "dark_per_tier": dict(dark_per_tier),
    }

def print_tier_summary(results_list, annot):
    labels = [r["label"] for r in results_list]
    col_w = max(12, max(len(l) for l in labels) + 2)
    label_w = 36

    print(f"\n{SEP}")
    print(f"{'TIER CHARACTERIZATION — MIXED CLUSTERS':^75}")
    print(f"{SEP}")
    header = f"{'Metric':<{label_w}}"
    for lab in labels:
        header += f" {lab:>{col_w}}"
    print(header)
    print(f"{'-' * (label_w + (col_w + 1) * len(labels))}")
    # basic stats
    rows = [
        ("Total clusters",        lambda r: f"{r['n_clusters']:,}"),
        ("Fully dark",            lambda r: f"{r['fully_dark']:,}"),
        ("Mixed",                 lambda r: f"{r['mixed']:,}"),
        ("Fully annotated",       lambda r: f"{r['fully_annotated']:,}"),
        ("Dark receivable (all)", lambda r: f"{len(r['dark_receivable_set']):,}"),
    ]
    for row_label, fn in rows:
        line = f"{row_label:<{label_w}}"
        for r in results_list:
            line += f" {fn(r):>{col_w}}"
        print(line)
    print()

    # per-tier breakdown: clusters
    print(f"{'--- Mixed clusters per tier ---':<{label_w}}")
    for tier in TIER_ORDER + [TIER_UNEVALUABLE]:
        line = f"  {tier:<{label_w - 2}}"
        for r in results_list:
            n = r["tier_mixed_clusters"].get(tier, 0)
            total_mixed = r["mixed"]
            pct = 100 * n / total_mixed if total_mixed else 0
            line += f" {f'{n:,} ({pct:.1f}%)':>{col_w}}"
        print(line)

    print()

    # per-tier breakdown: dark proteins
    print(f"{'--- Dark proteins per tier ---':<{label_w}}")
    for tier in TIER_ORDER + [TIER_UNEVALUABLE]:
        line = f"  {tier:<{label_w - 2}}"
        for r in results_list:
            n = r["tier_dark"].get(tier, 0)
            total_dark = len(r["dark_receivable_set"])
            pct = 100 * n / total_dark if total_dark else 0
            line += f" {f'{n:,} ({pct:.1f}%)':>{col_w}}"
        print(line)
    print()

    # mean homogeneity per tier
    print(f"{'--- Mean Jaccard per tier ---':<{label_w}}")
    for tier in TIER_ORDER:
        line = f"  {tier:<{label_w - 2}}"
        for r in results_list:
            h_list = r["tier_homogs"].get(tier, [])
            mean_h = float(np.mean(h_list)) if h_list else 0.0
            line += f" {mean_h:>{col_w}.3f}"
        print(line)
    print(f"\n{SEP}")

def print_new_dark_analysis(baseline_result, reweighted_results):
    """For each reweighted config, identify dark proteins that are NEW (not receivable in baseline) and show which tier they land in."""
    base_dark = baseline_result["dark_receivable_set"]
    print(f"\n{SEP}")
    print(f"{'NEW DARK PROTEINS : WHERE DO THEY LAND?':^75}")
    print(f"{SEP}")
    print(f"  Dark proteins receivable in baseline: {len(base_dark):,}")
    print()

    for r in reweighted_results:
        new_dark_set = r["dark_receivable_set"] - base_dark
        lost_dark_set = base_dark - r["dark_receivable_set"]
        n_new = len(new_dark_set)
        n_lost = len(lost_dark_set)
        n_total = len(r["dark_receivable_set"])

        print(f"  {r['label']}")
        print(f"    Total receivable      : {n_total:,}  "
              f"(Δ = +{n_total - len(base_dark):,})")
        print(f"    NEW (gained)          : {n_new:,}")
        print(f"    Lost vs baseline      : {n_lost:,}")
        print()

        if n_new == 0:
            print(f"    No new dark proteins.\n")
            continue

        # which tier do the new dark proteins land in?
        new_per_tier = Counter()
        for tier_name, tier_set in r["dark_per_tier"].items():
            overlap = tier_set & new_dark_set
            new_per_tier[tier_name] = len(overlap)

        print(f"    New dark proteins by tier:")
        for tier in TIER_ORDER + [TIER_UNEVALUABLE]:
            n = new_per_tier.get(tier, 0)
            pct = 100 * n / n_new if n_new else 0
            bar = "█" * int(pct / 2)
            print(f"      {tier:<24} {n:>8,}  ({pct:5.1f}%)  {bar}")

        # also: what about lost proteins — which tier were they in?
        if n_lost > 0:
            lost_per_tier = Counter()
            for tier_name, tier_set in baseline_result["dark_per_tier"].items():
                overlap = tier_set & lost_dark_set
                lost_per_tier[tier_name] = len(overlap)
            print(f"\n    Lost dark proteins by tier (in baseline):")
            for tier in TIER_ORDER + [TIER_UNEVALUABLE]:
                n = lost_per_tier.get(tier, 0)
                pct = 100 * n / n_lost if n_lost else 0
                print(f"      {tier:<24} {n:>8,}  ({pct:5.1f}%)")
        print()
    print(SEP)


def print_tier_transition(baseline_result, reweighted_results, annot):
    """for proteins that were already receivable in baseline, check whether their host cluster's tier changed (e.g., moved from faible to fort)"""
    # build baseline protein->tier mapping
    base_prot_tier = {}
    for tier_name, tier_set in baseline_result["dark_per_tier"].items():
        for pid in tier_set:
            base_prot_tier[pid] = tier_name

    print(f"\n{SEP}")
    print(f"{'TIER TRANSITIONS FOR EXISTING DARK PROTEINS':^75}")
    print(f"{SEP}")

    for r in reweighted_results:
        shared = baseline_result["dark_receivable_set"] & r["dark_receivable_set"]
        # build reweighted protein->tier
        rew_prot_tier = {}
        for tier_name, tier_set in r["dark_per_tier"].items():
            for pid in tier_set:
                rew_prot_tier[pid] = tier_name
        # count transitions
        transitions = Counter()
        for pid in shared:
            from_tier = base_prot_tier.get(pid, "?")
            to_tier = rew_prot_tier.get(pid, "?")
            transitions[(from_tier, to_tier)] += 1

        print(f"\n  {r['label']}  ({len(shared):,} shared dark proteins)")
        print(f"  {'From (baseline)':<24} → {'To (reweighted)':<24}  {'Count':>8}  {'%':>6}")
        print(f"  {'-' * 70}")

        for (ft, tt), n in sorted(transitions.items(), key=lambda x: -x[1]):
            pct = 100 * n / len(shared) if shared else 0
            changed = "  ←" if ft != tt else ""
            print(f"  {ft:<24} → {tt:<24}  {n:>8,}  {pct:>5.1f}%{changed}")
    print(f"\n{SEP}")

def plot_tier_breakdown(results_list, out_dir):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    labels = [r["label"] for r in results_list]
    x = np.arange(len(labels))
    w = 0.55

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cluster & Dark Protein Distribution by GO Homogeneity Tier", fontsize=13, fontweight="bold")

    for ax_idx, (ax, data_key, title, ylabel) in enumerate([
        (axes[0], "tier_mixed_clusters", "Mixed Clusters per Tier", "Number of clusters"),
        (axes[1], "tier_dark", "Dark Proteins Receivable per Tier", "Number of dark proteins"),
    ]):
        bottoms = np.zeros(len(labels))
        for tier in TIER_ORDER:
            vals = [r[data_key].get(tier, 0) for r in results_list]
            ax.bar(x, vals, w, bottom=bottoms, label=tier, color=TIER_COLORS[tier], edgecolor="white", linewidth=1.2)
            # label inside each segment
            for i, v in enumerate(vals):
                if v > 0:
                    total = sum(r[data_key].get(t, 0) for t in TIER_ORDER for r in [results_list[i]])
                    pct = 100 * v / total if total else 0
                    if pct > 5:  # only label if segment is big enough
                        ax.text(x[i], bottoms[i] + v / 2, f"{pct:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold", color="white")
            bottoms += vals

        # unevaluable on top (grey)
        vals = [r[data_key].get(TIER_UNEVALUABLE, 0) for r in results_list]
        if any(v > 0 for v in vals):
            ax.bar(x, vals, w, bottom=bottoms, label=TIER_UNEVALUABLE, color=TIER_COLORS[TIER_UNEVALUABLE], edgecolor="white", linewidth=1.2, alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylabel(ylabel)
        if ax_idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    path = out_dir / "tier_breakdown_stacked.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved → {path}")


def plot_new_dark_landing(baseline_result, reweighted_results, out_dir):
    """Bar chart: where do the new dark proteins land (by tier)?"""
    base_dark = baseline_result["dark_receivable_set"]

    configs = []
    for r in reweighted_results:
        new_dark = r["dark_receivable_set"] - base_dark
        n_new = len(new_dark)
        if n_new == 0:
            continue
        per_tier = {}
        for tier_name, tier_set in r["dark_per_tier"].items():
            per_tier[tier_name] = len(tier_set & new_dark)
        configs.append((r["label"], n_new, per_tier))

    if not configs:
        return

    labels = [c[0] for c in configs]
    x = np.arange(len(labels))
    w = 0.5

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.5), 5))
    fig.suptitle("New Dark Proteins : Which Homogeneity Tier Do They Land In?", fontsize=13, fontweight="bold")
    bottoms = np.zeros(len(labels))
    for tier in TIER_ORDER + [TIER_UNEVALUABLE]:
        vals = [c[2].get(tier, 0) for c in configs]
        bars = ax.bar(x, vals, w, bottom=bottoms, label=tier,
                      color=TIER_COLORS[tier], edgecolor="white", linewidth=1.2)
        for i, v in enumerate(vals):
            if v > 0:
                pct = 100 * v / configs[i][1]
                if pct > 4:
                    ax.text(x[i], bottoms[i] + v / 2, f"{v:,}\n({pct:.0f}%)", ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Number of new dark proteins")
    ax.legend(fontsize=8, loc="upper right")

    # total new on top of each bar
    for i, c in enumerate(configs):
        ax.text(x[i], bottoms[i] + bottoms.max() * 0.02, f"n={c[1]:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = out_dir / "new_dark_by_tier.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved → {path}")


def plot_homogeneity_distributions(results_list, out_dir):
    """Overlaid histograms of Jaccard homogeneity per config (mixed clusters only)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("GO Jaccard Homogeneity Distribution (Mixed Clusters)", fontsize=13, fontweight="bold")
    config_colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

    for r, color in zip(results_list, config_colors):
        # collect all homogeneity scores from mixed clusters
        all_h = []
        for tier in TIER_ORDER:
            all_h.extend(r["tier_homogs"].get(tier, []))
        if all_h:
            ax.hist(all_h, bins=40, range=(0, 1), alpha=0.4, color=color, label=r["label"], edgecolor="none")

    # tier boundaries
    ax.axvline(0.5, color="#F44336", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.axvline(0.8, color="#4CAF50", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.text(0.25, ax.get_ylim()[1] * 0.9, "Weak", ha="center", color="#F44336", fontsize=10, fontweight="bold")
    ax.text(0.65, ax.get_ylim()[1] * 0.9, "Medium", ha="center", color="#FF9800", fontsize=10, fontweight="bold")
    ax.text(0.9, ax.get_ylim()[1] * 0.9, "Strong", ha="center", color="#4CAF50", fontsize=10, fontweight="bold")

    ax.set_xlabel("GO Jaccard Homogeneity")
    ax.set_ylabel("Number of clusters")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = out_dir / "homogeneity_distribution_overlay.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved -> {path}")

def main():
    p = argparse.ArgumentParser(description="Characterize clusters by GO homogeneity tiers (faible/moyenne/fort)", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--baseline", required=True, type=Path, help="Baseline MCL cluster file (one cluster per line, tab-separated)")
    p.add_argument("--reweighted", required=True, nargs="+", type=Path, help="Reweighted cluster files (same format)")
    p.add_argument("--labels", required=True, nargs="+", help="Labels for each file (baseline first, then reweighted). Must have 1 + len(reweighted) labels.")
    p.add_argument("--go_pkl", required=True, type=Path, help="Path to annot_go_mf.pkl")
    p.add_argument("--go_obo", default=None, type=Path, help="Path to go-basic.obo (optional, for GO term names)")
    p.add_argument("--out_dir", required=True, type=Path, help="Output directory for figures and tables")
    args = p.parse_args()

    if len(args.labels) != 1 + len(args.reweighted):
        sys.exit(f"[ERROR] Expected {1 + len(args.reweighted)} labels "
                 f"(1 baseline + {len(args.reweighted)} reweighted), "
                 f"got {len(args.labels)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # load data
    print("Loading GO annotations")
    annot = load_go_annotations(args.go_pkl)
    print(f"  Proteins with GO-MF annotation: {len(annot):,}")

    go_names = {}
    if args.go_obo and args.go_obo.exists():
        go_names = load_go_names(args.go_obo)
        print(f"  GO term names: {len(go_names):,}")

    # load and analyse all clusterings
    all_paths = [args.baseline] + list(args.reweighted)
    all_labels = args.labels
    all_results = []

    for path, label in zip(all_paths, all_labels):
        print(f"\nAnalysing: {label}  ({path.name})")
        clusters = load_clusters(path)
        print(f"  Loaded {len(clusters):,} clusters")
        result = analyse_clustering(clusters, annot, label=label)
        all_results.append(result)

    baseline_result = all_results[0]
    reweighted_results = all_results[1:]

    # print reports
    print_tier_summary(all_results, annot)
    print_new_dark_analysis(baseline_result, reweighted_results)
    print_tier_transition(baseline_result, reweighted_results, annot)

    # figures
    print("\nGenerating figures ...")
    plot_tier_breakdown(all_results, args.out_dir)
    plot_new_dark_landing(baseline_result, reweighted_results, args.out_dir)
    plot_homogeneity_distributions(all_results, args.out_dir)

    # save summary to file
    summary_path = args.out_dir / "tier_characterization_summary.txt"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_tier_summary(all_results, annot)
        print_new_dark_analysis(baseline_result, reweighted_results)
        print_tier_transition(baseline_result, reweighted_results, annot)
    summary_path.write_text(buf.getvalue())
    print(f"\nSummary saved → {summary_path}")

    print(f"\n{SEP}")
    print(f"  Done. Output → {args.out_dir}")
    print(SEP)


if __name__ == "__main__":
    main()