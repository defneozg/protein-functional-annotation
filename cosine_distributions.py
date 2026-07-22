"""
three analysis modes:
1- Random sample: pairwise comparison on stratified random samples of 200K proteins across species
2- All alignments: 89M BLAST alignments
3- Low-PID alignments: cosine distribution restricted to alignment edges with pident < threshold (ex 40%).
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# column indices in diamond_with_cosine.tsv
COL_Q      = 0
COL_S      = 4
COL_PID    = 9    # pident
COL_EVALUE = 12
COL_COSINE = 14

EVALUE_CLAMP = 1e-300


def species_from_id(pid: str) -> str:
    return pid.split("|")[0]

def load_stratified_sample(
    h5_dir: Path,
    n_sample: int,
    rng: np.random.Generator,
) -> tuple[list[str], list[str], np.ndarray]:
    """stratified sample: each species contributes proportionally."""
    h5_files = sorted(h5_dir.glob("*.h5"))
    sizes = {}
    dim = None
    for fp in h5_files:
        with h5py.File(fp, "r") as hf:
            sizes[fp] = len(hf["ids"])
            if dim is None:
                dim = hf["embeddings"].shape[1]
    total = sum(sizes.values())
    print(f"  Total proteins: {total:,} across {len(h5_files)} species")
    print(f"  Sampling {n_sample:,} stratified by species ...")

    all_ids, all_species, all_embs = [], [], []
    for fp in tqdm(h5_files, desc="  Loading"):
        n_take = max(1, round(n_sample * sizes[fp] / total))
        with h5py.File(fp, "r") as hf:
            ids  = [s.decode() if isinstance(s, bytes) else s for s in hf["ids"][:]]
            embs = hf["embeddings"][:].astype(np.float32)
        n_take = min(n_take, len(ids))
        idx = rng.choice(len(ids), size=n_take, replace=False)
        all_ids.extend([ids[i] for i in idx])
        all_species.extend([fp.stem] * n_take)
        all_embs.append(embs[idx])

    emb_matrix = np.concatenate(all_embs, axis=0)
    print(f"  Sampled {len(all_ids):,} proteins | dim={emb_matrix.shape[1]}")
    return all_ids, all_species, emb_matrix

def l2_norm(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1.0, norms)

def random_pairwise_cosine(
    embs: np.ndarray,
    species: list[str],
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sample n_pairs random pairs from the embedding matrix"""
    N = len(embs)
    embs_norm = l2_norm(embs)
    sp_arr = np.array(species)

    # sample pairs
    idx_a = rng.integers(0, N, size=n_pairs)
    idx_b = rng.integers(0, N, size=n_pairs)
    # avoid self-pairs
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % N

    # batch cosine computation
    cosines = (embs_norm[idx_a] * embs_norm[idx_b]).sum(axis=1)

    within_mask = sp_arr[idx_a] == sp_arr[idx_b]
    cross_mask  = ~within_mask

    return cosines, cosines[within_mask], cosines[cross_mask]


def read_alignment_cosines(
    cosine_file: Path,
    pid_threshold: float = None,
    max_edges: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read cosine similarities from diamond_with_cosine.tsv. If pid_threshold is set, only keep edges with pident < threshold"""
    cosines   = []
    neg_log_e = []
    n_read = 0
    n_kept = 0

    with open(cosine_file) as fh:
        next(fh)   # skip header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15 or parts[COL_COSINE] == "NA":
                continue
            try:
                pid    = float(parts[COL_PID])
                cosine = float(parts[COL_COSINE])
                evalue = float(parts[COL_EVALUE])
            except ValueError:
                continue

            n_read += 1

            if pid_threshold is not None and pid >= pid_threshold:
                continue   # filter out high-PID edges

            cosines.append(cosine)
            neg_log_e.append(min(-np.log10(max(evalue, EVALUE_CLAMP)), 300.0))
            n_kept += 1

            if n_read % 10_000_000 == 0:
                label = f"PID<{pid_threshold}" if pid_threshold else "all"
                print(f"    {n_read:>12,} read | {n_kept:,} kept ({label})",
                      end="\r")

            if max_edges and n_kept >= max_edges:
                break

    label = f"PID<{pid_threshold}" if pid_threshold else "all"
    print(f"    {n_read:,} edges read | {n_kept:,} kept ({label})")
    return np.array(cosines, dtype=np.float32), np.array(neg_log_e, dtype=np.float32)


def print_stats(label: str, arr: np.ndarray):
    if len(arr) == 0:
        print(f"  {label}: no data")
        return
    print(f"  {label}:")
    print(f"    n      = {len(arr):,}")
    print(f"    mean   = {arr.mean():.4f}")
    print(f"    std    = {arr.std():.4f}")
    print(f"    median = {np.median(arr):.4f}")
    print(f"    p5     = {np.percentile(arr, 5):.4f}")
    print(f"    p95    = {np.percentile(arr, 95):.4f}")
    print(f"    < 0.5  = {100*(arr < 0.5).mean():.1f}%")
    print(f"    > 0.9  = {100*(arr > 0.9).mean():.1f}%")


def plot_distributions(data: dict[str, np.ndarray], out_path: Path, title: str = "Cosine similarity distributions",):
    COLORS = {
        "Random (within-species)"  : "#55A868",
        "Random (cross-species)"   : "#C44E52",
        "Random (all pairs)"       : "#4C72B0",
        f"BLAST all edges"         : "#DD8452",
        "BLAST low PID"            : "#8172B2",
    }

    plt.rcParams.update({
        "font.family"         : "sans-serif",
        "axes.spines.top"     : False,
        "axes.spines.right"   : False,
        "axes.grid"           : True,
        "grid.alpha"          : 0.25,
        "grid.linestyle"      : "--",
    })

    fig, ax = plt.subplots(figsize=(13, 6))
    BINS = np.linspace(0, 1, 100)

    for label, arr in data.items():
        if len(arr) == 0:
            continue
        color = COLORS.get(label, "#999999")
        # normalise histogram to density so different sample sizes are comparable
        ax.hist(arr, bins=BINS, density=True, alpha=0.55,
                color=color, label=f"{label} (n={len(arr):,}, μ={arr.mean():.3f})",
                edgecolor="white", linewidth=0.2)
        ax.axvline(arr.mean(), color=color, linestyle="--",
                   linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Cosine similarity", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.name} saved")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--h5_dir", required=True, help="Directory with .h5 embedding files")
    p.add_argument("--cosine_file", required=True, help="diamond_with_cosine.tsv")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_sample", type=int, default=200_000, help="Proteins to sample for random pairwise cosine (default: 200000)")
    p.add_argument("--n_pairs", type=int, default=2_000_000, help="Random pairs to compute (default: 2M)")
    p.add_argument("--pid_threshold", type=float, nargs="+", default=[40.0], help="PID threshold(s) for low-similarity BLAST edges. Multiple values processed in one pass (default: 40.0)")
    p.add_argument("--skip_random",  action="store_true", help="Skip the random sample computation (use if h5 files not available)")
    p.add_argument("--skip_blast",   action="store_true", help="Skip reading the BLAST cosine file")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    SEP = "=" * 60
    print(SEP)
    print("  Cosine Similarity Distribution Analysis")
    print(SEP)

    data_for_plot = {}

    # random pairwise cosine
    if not args.skip_random:
        print(f"\n[1] Random pairwise cosine (n_sample={args.n_sample:,})")
        t0 = time.time()
        ids, species, embs = load_stratified_sample(
            Path(args.h5_dir), args.n_sample, rng
        )
        cos_all, cos_within, cos_cross = random_pairwise_cosine(
            embs, species, args.n_pairs, rng
        )
        print(f"  Computed {args.n_pairs:,} random pairs in {time.time()-t0:.1f}s")
        print_stats("All random pairs",           cos_all)
        print_stats("Within-species random pairs", cos_within)
        print_stats("Cross-species random pairs",  cos_cross)

        np.save(out_dir / "cosine_random_all.npy",    cos_all)
        np.save(out_dir / "cosine_random_within.npy", cos_within)
        np.save(out_dir / "cosine_random_cross.npy",  cos_cross)

        data_for_plot["Random (all pairs)"]          = cos_all
        data_for_plot["Random (within-species)"]     = cos_within
        data_for_plot["Random (cross-species)"]      = cos_cross

    # BLAST alignment cosines
    if not args.skip_blast:
        print(f"\n[2] All BLAST alignment edges")
        t0 = time.time()
        cos_blast_all, nle_all = read_alignment_cosines(Path(args.cosine_file))
        print(f"  Read in {time.time()-t0:.1f}s")
        print_stats("All BLAST edges", cos_blast_all)
        np.save(out_dir / "cosine_blast_all.npy", cos_blast_all)
        data_for_plot["BLAST all edges"] = cos_blast_all

        thresholds = sorted(args.pid_threshold)
        print(f"\n[3] Low-PID BLAST edges — thresholds: {thresholds}")
        print(f"  Reading all edges once and binning by PID ...")
        t0 = time.time()

        # collect cosines and neg_log_e per threshold in one pass
        pid_cosines = {thr: [] for thr in thresholds}
        pid_nle     = {thr: [] for thr in thresholds}
        n_read_pid  = 0

        with open(Path(args.cosine_file)) as fh:
            next(fh)
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 15 or parts[COL_COSINE] == "NA":
                    continue
                try:
                    pid    = float(parts[COL_PID])
                    cosine = float(parts[COL_COSINE])
                    evalue = float(parts[COL_EVALUE])
                except ValueError:
                    continue
                nle = min(-np.log10(max(evalue, EVALUE_CLAMP)), 300.0)
                for thr in thresholds:
                    if pid < thr:
                        pid_cosines[thr].append(cosine)
                        pid_nle[thr].append(nle)
                n_read_pid += 1
                if n_read_pid % 10_000_000 == 0:
                    print(f"    {n_read_pid:>12,} read", end="\r")

        print(f"    {n_read_pid:,} edges read in {time.time()-t0:.1f}s")

        from scipy.stats import spearmanr, pearsonr
        for thr in thresholds:
            cos_arr = np.array(pid_cosines[thr], dtype=np.float32)
            nle_arr = np.array(pid_nle[thr],     dtype=np.float32)
            label   = f"BLAST PID<{int(thr)}%"
            print_stats(label, cos_arr)
            np.save(out_dir / f"cosine_blast_pid{int(thr)}.npy", cos_arr)
            data_for_plot[label] = cos_arr
            if len(nle_arr) > 1000:
                r_p, _ = pearsonr(nle_arr,  cos_arr)
                r_s, _ = spearmanr(nle_arr, cos_arr)
                print(f"    Pearson r={r_p:.4f}  Spearman rho={r_s:.4f}")

    if data_for_plot:
        print(f"\n  Generating combined distribution plot ...")
        plot_distributions(
            data_for_plot,
            out_path=out_dir / "cosine_distributions_combined.png",
            title=(
                "Cosine similarity distributions\n"
                "Random pairs vs BLAST edges vs Low-PID BLAST edges"
            ),
        )

        # also save a cleaner comparison: random all vs BLAST all vs low-PID
        if "Random (all pairs)" in data_for_plot and "BLAST all edges" in data_for_plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.rcParams.update({
                "font.family": "sans-serif",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "grid.linestyle": "--",
            })

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(
                "Embedding cosine similarity: is the high mean a model artefact?",
                fontsize=12, fontweight="bold"
            )
            BINS = np.linspace(0, 1, 80)

            panels = [
                ("Random pairs\n(no sequence bias)",
                 data_for_plot.get("Random (all pairs)", np.array([])),
                 "#4C72B0"),
                ("All BLAST edges\n(sequence-similar pairs)",
                 data_for_plot.get("BLAST all edges", np.array([])),
                 "#DD8452"),
                (f"Low-PID BLAST edges\n(PID < {args.pid_threshold}%)",
                 data_for_plot.get("BLAST low PID", np.array([])),
                 "#8172B2"),
            ]
            for ax, (label, arr, color) in zip(axes, panels):
                if len(arr) == 0:
                    ax.set_visible(False)
                    continue
                ax.hist(arr, bins=BINS, density=True, color=color,
                        alpha=0.85, edgecolor="white", linewidth=0.3)
                ax.axvline(arr.mean(), color="red", linestyle="--",
                           linewidth=1.5, label=f"mean={arr.mean():.3f}")
                ax.axvline(np.median(arr), color="orange", linestyle=":",
                           linewidth=1.5, label=f"median={np.median(arr):.3f}")
                ax.set_title(
                    f"{label}\nn={len(arr):,}  std={arr.std():.3f}",
                    fontweight="bold", fontsize=10
                )
                ax.set_xlabel("Cosine similarity")
                ax.set_ylabel("Density")
                ax.legend(fontsize=8)

            fig.tight_layout()
            fig.savefig(out_dir / "cosine_three_panel.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"  -> cosine_three_panel.png saved")

    print(f"\n{SEP}")
    print(f"  Done. Output in {out_dir}")
    print(SEP)


if __name__ == "__main__":
    main()