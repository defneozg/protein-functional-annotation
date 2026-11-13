"""
Metrics computed for each comparison (baseline vs reweighted):
  - Coverage: proteins assigned, cluster count
  - ARI / NMI: global similarity on shared proteins (subsampled)
  - Homogeneity / Completeness / V-measure (sklearn)
  - Best-match Jaccard: split/merge analysis
  - Containment: how much each baseline cluster is preserved
  - GO-MF Jaccard homogeneity: functional coherence
"""
import argparse
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)

SEP = "=" * 60

def load_clustering(path: Path, min_size: int = 2) -> dict[str, frozenset]:
    clusters = {}
    with open(path) as fh:
        for i, line in enumerate(fh):
            members = frozenset(m for m in line.strip().split("\t") if m)
            if len(members) >= min_size:
                clusters[f"C{i:08d}"] = members
    return clusters

def load_go(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return {k: set(v) if not isinstance(v, set) else v for k, v in data.items()}

def go_jaccard_homogeneity(members, protein2go, max_sample=50, rng=None):
    if rng is None:
        rng = random.Random(0)
    ann = [m for m in members if protein2go.get(m)]
    if len(ann) < 2:
        return None
    if len(ann) > max_sample:
        ann = rng.sample(ann, max_sample)
    vals = []
    for i in range(len(ann)):
        gi = protein2go[ann[i]]
        for j in range(i + 1, len(ann)):
            gj = protein2go[ann[j]]
            u  = len(gi | gj)
            vals.append(len(gi & gj) / u if u > 0 else 0.0)
    return float(np.mean(vals)) if vals else None

def discover_cluster_files(static_dir, dynamic_dir) -> dict[str, Path]:
    files = {}
    if static_dir:
        for fp in sorted(Path(static_dir).glob("clusters_reweighted_alpha*_I*.txt")):
            alpha = fp.stem.split("alpha")[1].split("_")[0]
            files[f"static_α={alpha}"] = fp
    if dynamic_dir:
        for fp in sorted(Path(dynamic_dir).glob("clusters_reweighted_dynamic_*_I*.txt")):
            stem = fp.stem
            try:
                amax = stem.split("amax")[1].split("_")[0]
                tau  = stem.split("tau")[1].split("_")[0]
                files[f"dyn_α_max={amax}_τ={tau}"] = fp
            except IndexError:
                files[fp.stem] = fp
    return files

def ari_nmi_comparison(clusters_a: dict, clusters_b: dict, max_proteins: int) -> dict:
    """ARI, NMI, Homogeneity, Completeness, V-measure on shared proteins"""
    prot_a = set().union(*clusters_a.values())
    prot_b = set().union(*clusters_b.values())
    shared = sorted(prot_a & prot_b)

    if not shared:
        return {}

    if len(shared) > max_proteins:
        shared = random.sample(shared, max_proteins)

    p2a = {p: cid for cid, members in enumerate(clusters_a.values())
           for p in members}
    p2b = {p: cid for cid, members in enumerate(clusters_b.values())
           for p in members}

    labels_a = np.array([p2a.get(p, -1) for p in shared])
    labels_b = np.array([p2b.get(p, -1) for p in shared])

    # only keep proteins clustered by both
    mask = (labels_a >= 0) & (labels_b >= 0)
    if mask.sum() < 100:
        return {}

    la, lb = labels_a[mask], labels_b[mask]
    ari  = adjusted_rand_score(la, lb)
    nmi  = normalized_mutual_info_score(la, lb)
    hom, comp, vmeas = homogeneity_completeness_v_measure(la, lb)

    return {
        "ari": ari, "nmi": nmi,
        "homogeneity": hom, "completeness": comp, "v_measure": vmeas,
        "n_shared": int(mask.sum()),
    }


def best_match_jaccard(clusters_a: dict, clusters_b: dict,
                        top_n: int = 100_000) -> list[float]:
    """for each cluster in A, find best-matching cluster in B (by Jaccard)"""
    p2b = {}
    for cid, members in clusters_b.items():
        for p in members:
            p2b[p] = cid

    # build B cluster lookup
    b_clusters = {cid: set(members) for cid, members in clusters_b.items()}

    jaccards = []
    items = list(clusters_a.items())
    if len(items) > top_n:
        items = random.sample(items, top_n)

    for cid_a, members_a in items:
        members_a = set(members_a)
        # count overlaps with each B cluster
        b_overlaps = defaultdict(int)
        for p in members_a:
            if p in p2b:
                b_overlaps[p2b[p]] += 1

        if not b_overlaps:
            jaccards.append(0.0)
            continue

        best_b = max(b_overlaps, key=b_overlaps.get)
        inter  = b_overlaps[best_b]
        union  = len(members_a) + len(b_clusters[best_b]) - inter
        jaccards.append(inter / union if union > 0 else 0.0)

    return jaccards

def go_homogeneity_comparison(clusters: dict, protein2go: dict, n_sample: int = 500_000) -> dict:
    rng = random.Random(2)
    items = list(clusters.values())
    if len(items) > n_sample:
        items = random.sample(items, n_sample)

    scores = []
    for members in items:
        s = go_jaccard_homogeneity(members, protein2go, rng=rng)
        if s is not None:
            scores.append(s)

    if not scores:
        return {}
    arr = np.array(scores)
    return {
        "mean_homog"  : float(arr.mean()),
        "pct_high"    : float((arr >= 0.8).mean()),
        "pct_low"     : float((arr < 0.5).mean()),
        "n_eval"      : len(scores),
    }

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline", required=True)
    p.add_argument("--static_dir", default=None)
    p.add_argument("--dynamic_dir", default=None)
    p.add_argument("--go_pkl", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_ari_proteins", type=int, default=3_000_000)
    p.add_argument("--top_n_jaccard", type=int, default=100_000)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(SEP)
    print("  Reweighted Clustering Comparison")
    print(SEP)

    # load baseline
    print(f"\n  Loading baseline: {args.baseline}")
    baseline = load_clustering(Path(args.baseline))
    print(f"  Baseline: {len(baseline):,} clusters")

    # discover reweighted files
    cluster_files = discover_cluster_files(args.static_dir, args.dynamic_dir)
    print(f"\n  Found {len(cluster_files)} reweighted clustering files:")
    for label, fp in cluster_files.items():
        print(f"    {label:<35} {fp.name}")

    # load GO
    print(f"\n  Loading GO annotations")
    protein2go = load_go(Path(args.go_pkl))

    # baseline GO homogeneity
    print(f"\n  Computing baseline GO homogeneity ...")
    base_go = go_homogeneity_comparison(baseline, protein2go)
    print(f"  Baseline: mean_homog={base_go['mean_homog']:.4f}  "
          f"pct_high={base_go['pct_high']*100:.1f}%  "
          f"pct_low={base_go['pct_low']*100:.1f}%")

    # compare each reweighted clustering to baseline
    results = {}

    for label, fp in cluster_files.items():
        print(f"\n{SEP}")
        print(f"  {label}")
        print(SEP)

        if not fp.exists():
            print(f"  [SKIP] File not found: {fp}")
            continue

        t0 = time.time()
        clust = load_clustering(fp)
        print(f"  Loaded {len(clust):,} clusters in {time.time()-t0:.1f}s")

        # coverage
        n_prot = sum(len(v) for v in clust.values())
        n_base = sum(len(v) for v in baseline.values())
        print(f"  Proteins: {n_prot:,} (baseline: {n_base:,})")

        # ARI / NMI / H / C / V
        print(f"  Computing ARI/NMI")
        ari_res = ari_nmi_comparison(baseline, clust, args.max_ari_proteins)
        if ari_res:
            print(f"    ARI={ari_res['ari']:.4f}  NMI={ari_res['nmi']:.4f}  "
                  f"H={ari_res['homogeneity']:.4f}  C={ari_res['completeness']:.4f}  "
                  f"V={ari_res['v_measure']:.4f}  "
                  f"(n_shared={ari_res['n_shared']:,})")

        # best-match Jaccard (baseline → reweighted)
        print(f"  Computing best-match Jaccard")
        jaccards_fwd = best_match_jaccard(baseline, clust, args.top_n_jaccard)
        jaccards_rev = best_match_jaccard(clust, baseline, args.top_n_jaccard)
        j_fwd = np.array(jaccards_fwd)
        j_rev = np.array(jaccards_rev)
        print(f"    Baseline→Reweighted: mean={j_fwd.mean():.4f}  "
              f">=0.9: {(j_fwd>=0.9).mean()*100:.1f}%  "
              f"<0.1: {(j_fwd<0.1).mean()*100:.1f}%")
        print(f"    Reweighted→Baseline: mean={j_rev.mean():.4f}  "
              f">=0.9: {(j_rev>=0.9).mean()*100:.1f}%  "
              f"<0.1: {(j_rev<0.1).mean()*100:.1f}%")

        # GO homogeneity
        print(f"  Computing GO homogeneity")
        go_res = go_homogeneity_comparison(clust, protein2go)
        if go_res:
            delta_mean = go_res['mean_homog'] - base_go['mean_homog']
            delta_high = go_res['pct_high']   - base_go['pct_high']
            print(f"    mean_homog={go_res['mean_homog']:.4f} ({delta_mean:+.4f})  "
                  f"pct_high={go_res['pct_high']*100:.1f}% ({delta_high*100:+.1f}%)  "
                  f"pct_low={go_res['pct_low']*100:.1f}%")

        results[label] = {
            "n_clusters" : len(clust),
            "n_proteins" : n_prot,
            **ari_res,
            "j_fwd_mean" : float(j_fwd.mean()),
            "j_rev_mean" : float(j_rev.mean()),
            "j_fwd_high" : float((j_fwd >= 0.9).mean()),
            "j_rev_high" : float((j_rev >= 0.9).mean()),
            **{f"go_{k}": v for k, v in go_res.items()},
        }

    print(f"\n{SEP}")
    print("  SUMMARY TABLE")
    print(SEP)

    cols = ["n_clusters", "ari", "nmi", "v_measure",
            "j_fwd_mean", "j_rev_mean", "go_mean_homog",
            "go_pct_high", "go_pct_low"]
    col_w = 12

    print(f"\n  {'Configuration':<35}", end="")
    for c in cols:
        print(f"  {c:>{col_w}}", end="")
    print()
    print(f"  {'-'*35}", end="")
    for _ in cols:
        print(f"  {'-'*col_w}", end="")
    print()

    # baseline row
    print(f"  {'baseline':<35}", end="")
    baseline_row = {
        "n_clusters"   : len(baseline),
        "ari"          : 1.0,
        "nmi"          : 1.0,
        "v_measure"    : 1.0,
        "j_fwd_mean"   : 1.0,
        "j_rev_mean"   : 1.0,
        "go_mean_homog": base_go.get("mean_homog", float("nan")),
        "go_pct_high"  : base_go.get("pct_high", float("nan")),
        "go_pct_low"   : base_go.get("pct_low", float("nan")),
    }
    for c in cols:
        v = baseline_row.get(c, float("nan"))
        if isinstance(v, int):
            print(f"  {v:>{col_w},}", end="")
        else:
            print(f"  {v:>{col_w}.4f}", end="")
    print()

    for label, res in results.items():
        print(f"  {label:<35}", end="")
        for c in cols:
            v = res.get(c, float("nan"))
            if isinstance(v, int):
                print(f"  {v:>{col_w},}", end="")
            elif isinstance(v, float):
                print(f"  {v:>{col_w}.4f}", end="")
            else:
                print(f"  {'N/A':>{col_w}}", end="")
        print()

    if results:
        labels = list(results.keys())
        metrics_to_plot = [
            ("ARI vs baseline", "ari", "#4C72B0"),
            ("NMI vs baseline", "nmi", "#DD8452"),
            ("V-measure vs baseline", "v_measure", "#55A868"),
            ("GO mean homogeneity", "go_mean_homog", "#C44E52"),
            ("Fwd Jaccard (base→rewt)", "j_fwd_mean", "#8172B2"),
            ("Rev Jaccard (rewt→base)", "j_rev_mean", "#CCB974"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Reweighted Clustering vs Baseline — Key Metrics", fontsize=13, fontweight="bold")
        axes = axes.flatten()

        # colour code: static = blue shades, dynamic = orange shades
        n_static  = sum(1 for l in labels if l.startswith("static"))
        n_dynamic = sum(1 for l in labels if l.startswith("dyn"))
        static_colors  = plt.cm.Blues(np.linspace(0.4, 0.9, max(n_static, 1)))
        dynamic_colors = plt.cm.Oranges(np.linspace(0.4, 0.9, max(n_dynamic, 1)))

        def get_color(label, si=[0], di=[0]):
            if label.startswith("static"):
                c = static_colors[si[0] % len(static_colors)]
                si[0] += 1
            else:
                c = dynamic_colors[di[0] % len(dynamic_colors)]
                di[0] += 1
            return c

        colors = [get_color(l) for l in labels]

        for ax, (title, key, _) in zip(axes, metrics_to_plot):
            vals = [results[l].get(key, float("nan")) for l in labels]
            base_val = baseline_row.get(key, float("nan"))

            bars = ax.barh(labels, vals, color=colors, edgecolor="white", linewidth=0.5)
            ax.axvline(base_val, color="red", linestyle="--", linewidth=1.5, label=f"baseline={base_val:.4f}")
            ax.set_title(title, fontweight="bold", fontsize=9)
            ax.legend(fontsize=7)
            # add value labels
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(v, bar.get_y() + bar.get_height()/2, f" {v:.4f}", va="center", fontsize=6)

        fig.tight_layout()
        fig.savefig(out_dir / "reweighted_comparison_metrics.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  -> reweighted_comparison_metrics.png")

        # GO homogeneity comparison plot
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(labels) + 1)
        all_labels_w_base = ["baseline"] + labels
        all_mean_homog = [base_go.get("mean_homog", float("nan"))] + \
                            [results[l].get("go_mean_homog", float("nan")) for l in labels]
        all_pct_high = [base_go.get("pct_high", float("nan"))] + \
                            [results[l].get("go_pct_high", float("nan")) for l in labels]
        all_colors_go = ["#4CAF50"] + colors

        bars1 = ax.bar(x - 0.2, all_mean_homog, 0.4, color=all_colors_go, alpha=0.8, label="Mean GO homogeneity", edgecolor="white")
        bars2 = ax.bar(x + 0.2, all_pct_high,   0.4, color=all_colors_go, alpha=0.4, label="% clusters homog ≥ 0.8", edgecolor="white", hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels_w_base, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Score")
        ax.set_title("GO-MF Homogeneity — Baseline vs All Reweighted Variants", fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(out_dir / "go_homogeneity_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> go_homogeneity_comparison.png")

    # save TSV summary
    tsv_path = out_dir / "comparison_summary.tsv"
    with open(tsv_path, "w") as fh:
        fh.write("configuration\t" + "\t".join(cols) + "\n")
        fh.write("baseline\t" + "\t".join(
            str(baseline_row.get(c, "")) for c in cols) + "\n")
        for label, res in results.items():
            fh.write(label + "\t" + "\t".join(
                str(res.get(c, "")) for c in cols) + "\n")
    print(f"  -> comparison_summary.tsv")

    print(f"\n{SEP}")
    print("  Done.")
    print(SEP)

if __name__ == "__main__":
    main()