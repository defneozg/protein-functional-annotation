import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm

# logging helper
_t0_global = time.time()

def log(msg:str):
    elapsed = time.time() - _t0_global
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"[{h:02d}:{m:02d}:{s:02d}] {msg}", flush=True)

def log_mem():
    """print current ram usage"""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info().rss / 1e9
        log(f"  RAM usage : {mem:.1f} GB")
    except ImportError:
        pass

# load all embeddings
def load_all_embeddings(h5_dir: Path) -> tuple[list[str], list[str], np.ndarray]:
    h5_files = sorted(h5_dir.glob("*.h5"))
    if not h5_files:
        sys.exit(f"[ERROR] No .h5 files in {h5_dir}")
    log(f"Found {len(h5_files)} .h5 files in {h5_dir}")

    # pre-allocate
    total = 0
    dim = None
    for fp in h5_files:
        with h5py.File(fp, "r") as hf:
            total += len(hf["ids"])
            if dim is None:
                dim = hf["embeddings"].shape[1]

    log(f"Total proteins: {total:,} | Dimension: {dim}")
    log(f"Pre-allocating embeddings matrix ({total:,} x {dim} float32 = {total*dim*4 / 1e9:.1f} GB)")

    emb_matrix = np.empty((total, dim), dtype=np.float32)
    all_ids = []
    all_species = []
    row = 0

    for fp in tqdm(h5_files, desc="Loading .h5 files"):
        with h5py.File(fp, "r") as hf:
            ids = [s.decode() if isinstance(s, bytes) else s for s in hf["ids"][:]]
            embs = hf["embeddings"][:].astype(np.float32)
        n = len(ids)
        emb_matrix[row:row+n] = embs
        all_ids.extend(ids)
        all_species.extend([fp.stem] * n)
        row += n
    log(f"Loaded {row:,} proteins")
    log_mem()
    return all_ids, all_species, emb_matrix

# UMAP
def run_umap(embs: np.ndarray, n_components: int, n_neighbors: int, min_dist: float, n_jobs: int, random_state: int = 42) -> np.ndarray:
    try:
        import umap
    except ImportError:
        sys.exit(f"[ERROR] umap-learn not installed")

    log(f"Running UMAP: {embs.shape[0]:,} x {embs.shape[1]} -> {embs.shape[0]:,} x {n_components}  (n_neighbors={n_neighbors}, min_dist={min_dist}, n_jobs={n_jobs})")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        low_memory=False,
        n_jobs=n_jobs,
        verbose=True,
    )
    t0 = time.time()
    out = reducer.fit_transform(embs)
    log(f"UMAP {n_components}D done in {(time.time() - t0)/60:.1f} min")
    log_mem()
    return out.astype(np.float32)

# HDBSCAN
def run_hdbscan(data:np.ndarray, min_cluster_size:int, min_samples:int, n_jobs:int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import hdbscan
    except ImportError:
        sys.exit(f"[ERROR] hdbscan not installed")

    log(f"Running HDBSCAN: {data.shape[0]:,} x {data.shape[1]} (min_cluster_size={min_cluster_size}, min_samples={min_samples})")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=n_jobs,
        prediction_data=False,
    )
    t0 = time.time()
    labels = clusterer.fit_predict(data)
    probs = clusterer.probabilities_.astype(np.float32)
    log(f"HDBSCAN done in {(time.time() - t0)/60:.1f} min")
    log_mem()
    return labels.astype(np.int32), probs

# main
def main():
    p = argparse.ArgumentParser(
        description="Full-scale UMAP + HDBSCAN on all protein embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--h5_dir", required=True, help="Path to .h5 files")
    p.add_argument("--out_dir", required=True, help="Path to output directory")
    # umap params
    p.add_argument("--n_neighbors", type=int, default=30, help="UMAP n_neighbors (default: 30)")
    p.add_argument("--min_dist_2d", type=float, default=0.1, help="Minimum distance for 2D viz (default: 0.1)")
    p.add_argument("--min_dist_hd", type=float, default=0.0, help="Minimum distance for HD viz (default: 0.0)")
    p.add_argument("--umap_dim", type=int, default=10, help="High-dim UMAP dimensions for HDBSCAN input (default: 10)")

    # hdbscan params
    p.add_argument("--min_cluster_size", type=int, default=10, help="Minimum number of clusters (default: 10)")
    p.add_argument("--min_samples", type=int, default=5, help="Minimum number of samples (default: 5)")

    p.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs (default: -1 -> all available)")
    # resume flags
    p.add_argument("--skip_umap_2d", action="store_true", help="Skip UMAP 2D")
    p.add_argument("--skip_umap_hd", action="store_true", help="Skip high-dim HDBSCAN")
    p.add_argument("--skip_hdbscan", action="store_true", help="Skip HDBSCAN")

    args = p.parse_args()
    h5_dir = Path(args.h5_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    SEP = "=" * 60
    log(SEP)
    log("  Full-scale UMAP + HDBSCAN on tall8")
    log(SEP)
    log(f"  h5_dir: {h5_dir}")
    log(f"  out_dir: {out_dir}")
    log(f"  n_jobs: {args.n_jobs}")

    # step 1: load all embeddings
    ids_path = out_dir / "all_ids.txt"
    sp_path = out_dir / "all_species.txt"

    if ids_path.exists() and sp_path.exists():
        log("Loading saved IDs and species labels")
        all_ids = ids_path.read_text().splitlines()
        all_species = sp_path.read_text().splitlines()
        # still need emb for UMAP if not skipping both
        need_embs = not (
            (args.skip_umap_2d or (out_dir / "umap_2d.npy").exists()) and
            (args.skip_umap_hd or (out_dir / f"umap_{args.umap_dim}d.npy").exists())
        )
        if need_embs:
            log("Loading embeddings")
            _, _, embs = load_all_embeddings(h5_dir)
        else:
            embs = None
            log("Both UMAPs already exist, skipping emb loading")
    else:
        all_ids, all_species, embs = load_all_embeddings(h5_dir)
        log("Saving IDs and species")
        ids_path.write_text("\n".join(all_ids))
        sp_path.write_text("\n".join(all_species))

    N = len(all_ids)
    log(f"Working with {N:,} proteins")

    # step 2: UMAP 2d
    umap_2d_path = out_dir / "umap_2d.npy"
    if args.skip_umap_2d and umap_2d_path.exists():
        log("Skipping UMAP 2D (already exists)")
    else:
        log(SEP)
        log("step 2: UMAP 2D (vis)")
        xy_2d = run_umap(
            embs,
            n_components=2,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist_2d,
            n_jobs=args.n_jobs,
        )
        np.save(umap_2d_path, xy_2d)
        log(f"Saved UMAP 2D to: {umap_2d_path}")
        del xy_2d # free ram

    # step 3: umap high dim
    umap_hd_path = out_dir / f"umap_{args.umap_dim}d.npy"
    if args.skip_umap_hd and umap_hd_path.exists():
        log("Skipping high-dim UMAP (already exists)")
        xy_hd = np.load(umap_hd_path)
    else:
        log(SEP)
        log(f"step 3: UMAP {args.umap_dim}D (HDBSCAN input)")
        xy_hd = run_umap(
            embs,
            n_components=args.umap_dim,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist_hd,
            n_jobs=args.n_jobs,
        )
        np.save(umap_hd_path, xy_hd)
        log(f"Saved UMAP HD to: {umap_hd_path}")

    del embs

    # step 4: hdbscan
    labels_path = out_dir / "hdbscan_labels.npy"
    probs_path = out_dir / "hdbscan_probs.npy"
    if args.skip_hdbscan and labels_path.exists():
        log("Skipping HDBSCAN (already exists)")
        labels = np.load(labels_path)
        probs = np.load(probs_path)
    else:
        log(SEP)
        log("step 4: HDBSCAN")
        labels, probs = run_hdbscan(
            xy_hd,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            n_jobs=args.n_jobs,
        )
        np.save(labels_path, labels)
        np.save(probs_path, probs)
        log(f"Saved HDBSCAN to: {labels_path}")

    # step 5: stats
    log(SEP)
    log("step 5: cluster statistics")

    n_clusters = int((labels >= 0).sum() > 0) and len(set(labels[labels >= 0]))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    noise_pct = 100 * n_noise / N

    cluster_sizes = Counter(labels[labels >= 0].tolist())
    sizes = np.array(list(cluster_sizes.values()))

    stats_lines = [
        f"Total proteins      : {N:,}",
        f"Clusters found      : {n_clusters:,}",
        f"Noise points        : {n_noise:,} ({noise_pct:.2f}%)",
        f"Clustered proteins  : {N - n_noise:,} ({100 - noise_pct:.2f}%)",
        f"",
        f"Cluster size stats:",
        f"  min    : {sizes.min() if len(sizes) else 0}",
        f"  median : {np.median(sizes):.0f}" if len(sizes) else "  median : 0",
        f"  mean   : {sizes.mean():.1f}" if len(sizes) else "  mean   : 0",
        f"  max    : {sizes.max() if len(sizes) else 0}",
        f"  > 100  : {(sizes > 100).sum():,}",
        f"  > 1000 : {(sizes > 1000).sum():,}",
        f"",
        f"UMAP params:",
        f"  n_neighbors : {args.n_neighbors}",
        f"  min_dist 2D : {args.min_dist_2d}",
        f"  min_dist {args.umap_dim}D : {args.min_dist_hd}",
        f"",
        f"HDBSCAN params:",
        f"  min_cluster_size : {args.min_cluster_size}",
        f"  min_samples      : {args.min_samples}",
        f"",
        f"Total wall time : {(time.time() - _t0_global) / 3600:.2f}h",
    ]

    for line in stats_lines:
        log(line)

    stats_path = out_dir / "run_stats.txt"
    stats_path.write_text("\n".join(stats_lines))
    log(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()