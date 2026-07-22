"""
Visualize the distributions of:
1- raw -log(evalue)
2- cosine similarity
3- combined weight (after static or dynamic reweighting)
"""

import argparse
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL_EVALUE = 12
COL_COSINE = 14
EVALUE_CLAMP = 1e-300


def sample_values(path: Path, col_idx: int, n_sample: int, stride: int = None) -> np.ndarray:
    vals = []
    with open(path) as fh:
        next(fh)
        for i, line in enumerate(fh):
            if stride and i % stride != 0:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= col_idx:
                continue
            v = parts[col_idx]
            if v in ("NA", "nan", ""):
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
            if len(vals) >= n_sample:
                break
    return np.array(vals, dtype=np.float32)

def read_abc_weights(path: Path, n_sample: int = None) -> np.ndarray:
    vals = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                vals.append(float(parts[2]))
            except ValueError:
                continue
    return np.array(vals, dtype=np.float32)

def plot_single_distribution(ax, arr: np.ndarray, label: str, color: str, bins: np.ndarray, density: bool = True):
    if len(arr) == 0:
        return
    ax.hist(arr, bins=bins, density=density, color=color, alpha=0.7, edgecolor="white", linewidth=0.2, label=f"{label}\n(μ={arr.mean():.3f}, σ={arr.std():.3f})")
    ax.axvline(arr.mean(), color=color, linestyle="--", linewidth=1.5, alpha=0.9)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cosine_file", required=True, help="diamond_with_cosine.tsv")
    p.add_argument("--static_dir", default=None, help="Directory with reweighted_alpha*.abc files")
    p.add_argument("--dynamic_dir", default=None, help="Directory with reweighted_dynamic_*.abc files")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    SEP = "=" * 60
    print(SEP)
    print("  Edge Weight Distribution Visualization")
    print(SEP)
    # load raw cosine and evalue from cosine file
    print(f"\n  Reading cosine + evalue from {args.cosine_file}")
    t0 = time.time()
    cosines, neg_log_e = [], []
    with open(args.cosine_file) as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15 or parts[COL_COSINE] == "NA":
                continue
            try:
                cos = float(parts[COL_COSINE])
                evalue = float(parts[COL_EVALUE])
                nle = min(-np.log10(max(evalue, EVALUE_CLAMP)), 300.0)
                cosines.append(cos)
                neg_log_e.append(nle)
            except ValueError:
                continue

    cosines = np.array(cosines, dtype=np.float32)
    neg_log_e = np.array(neg_log_e, dtype=np.float32)
    print(f"  Read {len(cosines):,} edges in {time.time() - t0:.1f}s")

    # load static reweighted weights
    static_weights = {}
    if args.static_dir:
        static_dir = Path(args.static_dir)
        abc_files = sorted(static_dir.glob("reweighted_alpha*.abc"))
        print(f"\n  Loading {len(abc_files)} static ABC files")
        for fp in abc_files:
            # extract alpha from filename
            stem = fp.stem
            alpha = stem.replace("reweighted_alpha", "")
            w = read_abc_weights(fp, n_sample=None)
            static_weights[f"α={alpha}"] = w
            print(f"    {fp.name}: {len(w):,} edges  mean={w.mean():.3f}")

    # load dynamic reweighted weights
    dynamic_weights = {}
    if args.dynamic_dir:
        dyn_dir = Path(args.dynamic_dir)
        dyn_files = sorted(dyn_dir.glob("reweighted_dynamic_*.abc"))
        print(f"\n  Loading {len(dyn_files)} dynamic ABC files")
        for fp in dyn_files:
            # extract amax and tau from filename
            stem = fp.stem  # ex. reweighted_dynamic_amax0.40_tau0.050
            try:
                amax = stem.split("amax")[1].split("_")[0]
                tau = stem.split("tau")[1]
                key = f"amax={amax} τ={tau}"
            except IndexError:
                key = fp.stem
            w = read_abc_weights(fp, n_sample=None)
            dynamic_weights[key] = w
            print(f"    {fp.name}: {len(w):,} edges  mean={w.mean():.3f}")

    # panel 1: raw signal distributions
    print(f"\n  Plotting raw signal distributions")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Raw Edge Signal Distributions", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.hist(cosines, bins=80, density=True, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(cosines.mean(), color="red", linestyle="--", linewidth=1.5, label=f"mean={cosines.mean():.3f}")
    ax.axvline(np.median(cosines), color="orange", linestyle=":", linewidth=1.5, label=f"median={np.median(cosines):.3f}")
    ax.set_title("Embedding cosine similarity", fontweight="bold")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax = axes[1]
    bins_nle = np.linspace(0, 300, 80)
    ax.hist(neg_log_e, bins=bins_nle, density=True, color="#DD8452", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(neg_log_e.mean(), color="red", linestyle="--", linewidth=1.5, label=f"mean={neg_log_e.mean():.1f}")
    ax.set_title("-log₁₀(evalue) distribution", fontweight="bold")
    ax.set_xlabel("-log₁₀(evalue)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--"})
    fig.tight_layout()
    fig.savefig(out_dir / "01_raw_signals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> 01_raw_signals.png")

    # panel 1b: cosine similarity vs -log10(evalue) joint density
    print(f"  Plotting cosine similarity vs -log10(evalue)")
    pearson_r = np.corrcoef(neg_log_e, cosines)[0, 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(neg_log_e, cosines, gridsize=90, bins="log", cmap="viridis", mincnt=1, extent=(0, 300, 0, 1))
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log10(count)")
    ax.axvline(20, color="red", linestyle="--", linewidth=1.3, label="-log10(evalue)=20\n(baseline threshold)")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1)
    ax.set_xlabel("-log₁₀(evalue)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Cosine similarity vs -log₁₀(evalue)  (Pearson r={pearson_r:.3f}, n={len(cosines):,})", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "01b_cosine_vs_logevalue.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> 01b_cosine_vs_logevalue.png  (Pearson r={pearson_r:.3f})")

    # panel 2: static reweighted distributions
    if static_weights:
        print(f"  Plotting static reweighted distributions")
        n_static = len(static_weights)
        fig, axes = plt.subplots(1, n_static + 1, figsize=(5 * (n_static + 1), 5))
        fig.suptitle("Static Reweighting — Combined Edge Weight Distributions", fontsize=12, fontweight="bold")
        # baseline = normalised -log(evalue) × 200 (what pure seq would give)
        baseline_w = np.clip(neg_log_e, 0, 200)
        ax = axes[0]
        ax.hist(baseline_w, bins=80, density=True, color="#4CAF50", alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(baseline_w.mean(), color="red", linestyle="--", linewidth=1.5, label=f"mean={baseline_w.mean():.2f}")
        ax.set_title("Baseline\n(α=0, pure -log evalue)", fontweight="bold")
        ax.set_xlabel("Edge weight [0, 200]")
        ax.legend(fontsize=8)

        COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00"]
        for i, (label, w) in enumerate(static_weights.items()):
            ax = axes[i + 1]
            ax.hist(w, bins=80, density=True, color=COLORS[i % len(COLORS)], alpha=0.85, edgecolor="white", linewidth=0.3)
            ax.axvline(w.mean(), color="red", linestyle="--", linewidth=1.5, label=f"mean={w.mean():.2f}")
            ax.set_title(f"Static {label}", fontweight="bold")
            ax.set_xlabel("Edge weight [0, 200]")
            ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / "02_static_weights.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> 02_static_weights.png")

        # overlay all static on one plot for comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(baseline_w, bins=80, density=True, color="#4CAF50", alpha=0.4, label="Baseline (α=0)", edgecolor="white", linewidth=0.2)
        for i, (label, w) in enumerate(static_weights.items()):
            ax.hist(w, bins=80, density=True, color=COLORS[i % len(COLORS)], alpha=0.4, label=f"Static {label}", edgecolor="white", linewidth=0.2)
            ax.axvline(w.mean(), color=COLORS[i % len(COLORS)], linestyle="--", linewidth=1.2)
        ax.set_title("Static Reweighting — All Alpha Values Overlaid", fontweight="bold")
        ax.set_xlabel("Edge weight [0, 200]")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "03_static_overlay.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> 03_static_overlay.png")

    # panel 3: dynamic reweighted distributions
    if dynamic_weights:
        print(f"  Plotting dynamic reweighted distributions")
        # group by amax for cleaner plots
        by_amax = {}
        for label, w in dynamic_weights.items():
            amax = label.split(" ")[0]
            by_amax.setdefault(amax, {})[label] = w

        for amax, variants in by_amax.items():
            n_v = len(variants)
            fig, axes = plt.subplots(1, n_v, figsize=(5 * n_v, 5))
            if n_v == 1:
                axes = [axes]
            fig.suptitle(f"Dynamic Reweighting — {amax} (varying τ)", fontsize=12, fontweight="bold")
            COLORS_TAU = ["#56B4E9", "#009E73", "#E69F00"]
            for i, (label, w) in enumerate(variants.items()):
                ax = axes[i]
                tau = label.split("τ=")[1] if "τ=" in label else str(i)
                ax.hist(w, bins=80, density=True, color=COLORS_TAU[i % len(COLORS_TAU)], alpha=0.85, edgecolor="white", linewidth=0.3)
                ax.axvline(w.mean(), color="red", linestyle="--", linewidth=1.5, label=f"mean={w.mean():.2f}")
                ax.set_title(f"τ={tau}", fontweight="bold")
                ax.set_xlabel("Edge weight [0, 200]")
                ax.legend(fontsize=8)

            fig.tight_layout()
            safe_amax = amax.replace("=", "").replace(".", "")
            fig.savefig(out_dir / f"04_dynamic_{safe_amax}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  -> 04_dynamic_{safe_amax}.png")

        # overlay all dynamic on one plot
        fig, ax = plt.subplots(figsize=(12, 5))
        if static_weights:
            ax.hist(baseline_w, bins=80, density=True, color="#4CAF50", alpha=0.3, label="Baseline (α=0)", edgecolor="white", linewidth=0.2)
        all_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(dynamic_weights)))
        for i, (label, w) in enumerate(dynamic_weights.items()):
            ax.hist(w, bins=80, density=True, color=all_colors[i], alpha=0.35, edgecolor="white", linewidth=0.2, label=f"Dyn {label} (μ={w.mean():.1f})")
            ax.axvline(w.mean(), color=all_colors[i], linestyle="--", linewidth=1.0)
        ax.set_title("Dynamic Reweighting — All Configurations Overlaid", fontweight="bold")
        ax.set_xlabel("Edge weight [0, 200]")
        ax.set_ylabel("Density")
        ax.legend(fontsize=6.5, ncol=3)
        fig.tight_layout()
        fig.savefig(out_dir / "05_dynamic_overlay.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> 05_dynamic_overlay.png")

    print(f"\n{SEP}")
    print("  Summary statistics")
    print(SEP)
    print(f"  {'Configuration':<35} {'mean':>8}  {'std':>8}  {'p5':>8}  {'p95':>8}  {'n':>10}")
    print(f"  {'-' * 80}")

    all_configs = {
        "Cosine similarity": cosines,
        "-log10(evalue)": neg_log_e,
        "Baseline weight (α=0)": np.clip(neg_log_e, 0, 200),
    }
    all_configs.update({f"Static {k}": v for k, v in static_weights.items()})
    all_configs.update({f"Dynamic {k}": v for k, v in dynamic_weights.items()})

    for label, arr in all_configs.items():
        if len(arr) == 0:
            continue
        print(f"  {label:<35} {arr.mean():>8.3f}  {arr.std():>8.3f}  "
              f"{np.percentile(arr, 5):>8.3f}  {np.percentile(arr, 95):>8.3f}  "
              f"{len(arr):>10,}")

    print(f"\n  Figures saved to: {out_dir}")
    print(SEP)

if __name__ == "__main__":
    main()

