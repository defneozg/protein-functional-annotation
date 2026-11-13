"""
Pipeline:
1. Load ProtT5 embeddings from HDF5 files
2. Build a kNN graph using cosine similarity with FAISS for scalability
3. Run Leiden community detection (for one or more resolution values)
4. Export output in the sale format as MCL
"""
import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm

_t0_global = time.time()

def log(msg:str):
    elapsed = time.time() - _t0_global
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"[{h:02d}:{m:02d}:{s:02d}] {msg}", flush=True)

def log_mem():
    try:
        import psutil
        mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
        log(f"  RAM usage : {mem:.1f} GB")
    except ImportError:
        pass

SEP = "=" * 60

# load embeddings
def load_all_embeddings(h5_dir: Path) -> tuple[list[str], list[str], np.ndarray]:
    h5_files = sorted(h5_dir.glob("*.h5"))
    if not h5_files:
        sys.exit(f"[ERROR] No HDF5 files found in {h5_dir}")
    log(f"Found {len(h5_files)} HDF5 files in {h5_dir}")
    # first pass: count total proteins and get dim
    total = 0
    dim = None
    for fp in h5_files:
        with h5py.File(fp, "r") as hf:
            total += len(hf["ids"])
            if dim is None:
                dim = hf["embeddings"].shape[1]
    log(f"Total proteins : {total:,} | Dimension: {dim}")
    log(f"Pre-allocating matrix ({total:,} x {dim} float 32 = {total * dim * 4 / 1e9:.1f} GB)")

    emb_matrix = np.empty((total, dim), dtype=np.float32)
    all_ids = []
    all_species = []
    row = 0

    for fp in tqdm(h5_files, desc="Loading embeddings"):
        with h5py.File(fp, "r") as hf:
            ids = [s.decode() if isinstance(s, bytes) else s for s in hf["ids"][:]]
            embs = hf["embeddings"][:].astype(np.float32)
        n = len(ids)
        emb_matrix[row:row + n] = embs
        all_ids.extend(ids)
        all_species.extend([fp.stem] * n)
        row += n

    log(f"Loaded {row:,} proteins")
    log_mem()
    return all_ids, all_species, emb_matrix

# build kNN graph from cos sim
def build_knn_graph_faiss(embs:np.ndarray, k: int = 20, min_cosine: float = 0.0, n_jobs : int = -1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """build symetric kNN graph using FAISS IndexFlatIP (inner product on L2-normalised vectors = cosine similarity)"""
    try:
        import faiss
    except ImportError:
        sys.exit("[ERROR] faiss is not installed")

    N, D = embs.shape
    log(f"Building kNN graph: N={N:,}  D={D}  k={k}  min_cosine={min_cosine}")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs_norm = (embs / norms).astype(np.float32)

    if n_jobs > 0:
        faiss.omp_set_num_threads(n_jobs)
    else:
        faiss.omp_set_num_threads(os.cpu_count() or 4)

    # for very large datasets (>500k) use IVF index for speed
    if N > 500_000:
        n_cells = min(int(np.sqrt(N)) * 4, N // 10)
        n_probe = min(max(n_cells // 4, 10), 256)
        log(f"  Using IVF index: n_cells={n_cells} n_probe={n_probe}")
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFFlat(quantizer, D, n_cells, faiss.METRIC_INNER_PRODUCT)
        index.train(embs_norm)
        index.add(embs_norm)
        index.nprobe = n_probe
    else:
        log(f"  Using brute-force IndexFlatIP")
        index = faiss.IndexFlatIP(D)
        index.add(embs_norm)

    # search k+1 neighbours (first hit is self)
    t0 = time.time()
    log(f"  Searching {k+1} nearest neighbors")
    similarities, indices = index.search(embs_norm, k + 1)
    log(f"  kNN search done in {time.time() - t0:.1f}s")
    # build edge list (undirected, skip self-loops)
    log(f"  Assembling edge list")
    src_list, tgt_list, wt_list = [], [], []
    for i in range(N):
        for j_idx in range(k + 1):
            j = int(indices[i, j_idx])
            sim = float(similarities[i, j_idx])
            if j == i:
                continue
            if sim < min_cosine:
                continue
            # only keep i < j to avoid double counting
            if i < j:
                src_list.append(i)
                tgt_list.append(j)
                wt_list.append(sim)

    sources = np.array(src_list, dtype=np.int64)
    targets = np.array(tgt_list, dtype=np.int64)
    weights = np.array(wt_list, dtype=np.float32)
    # symetrise to make it undirected
    sources_sym = np.concatenate([sources, targets])
    targets_sym = np.concatenate([targets, sources])
    weights_sym = np.concatenate([weights, weights])

    n_edges_unique = len(sources)
    log(f"  Edges (unique, undirected): {n_edges_unique:,}")
    log(f"  Mean cosine similarity: {weights.mean():.4f}")
    log(f"  Cosine range: [{weights.min():.4f}, {weights.max():.4f}]")
    log_mem()
    return sources_sym, targets_sym, weights_sym

def build_knn_graph_sklearn(embs: np.ndarray, k: int = 20, min_cosine: float = 0.0, n_jobs:int = -1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fallback using sklearn (slower but no FAISS)"""
    from sklearn.neighbors import NearestNeighbors

    N, D = embs.shape
    log(f"Building kNN graph (sklearn fallback): N={N:,}  D={D}  k={k}")
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine", n_jobs=n_jobs, algorithm="brute")
    t0 = time.time()
    nn.fit(embs)
    distances, indices = nn.kneighbors(embs)
    log(f"  kNN search done in {time.time() - t0:.1f}s")
    # sklearn cosine returns distance
    similarities = 1.0 - distances
    src_list, tgt_list, wt_list = [], [], []
    for i in range(N):
        for j_idx in range(k + 1):
            j = int(indices[i, j_idx])
            sim = float(similarities[i, j_idx])
            if j == i:
                continue
            if sim < min_cosine:
                continue
            if i < j:
                src_list.append(i)
                tgt_list.append(j)
                wt_list.append(sim)

    sources = np.array(src_list, dtype=np.int64)
    targets = np.array(tgt_list, dtype=np.int64)
    weights = np.array(wt_list, dtype=np.float32)
    sources_sym = np.concatenate([sources, targets])
    targets_sym = np.concatenate([targets, sources])
    weights_sym = np.concatenate([weights, weights])
    log(f"  Edges (unique, undirected): {len(sources):,}")
    log_mem()
    return sources_sym, targets_sym, weights_sym

# run leiden
def build_igraph(sources: np.ndarray, targets: np.ndarray, weights: np.ndarray, n_nodes: int, graph_path: Path = None) -> np.ndarray:
    import igraph as ig
    log(f"  Building igraph object {n_nodes:,} nodes, {len(sources):,} directed edges")
    t0 = time.time()
    g = ig.Graph(n=n_nodes, edges=list(zip(sources.tolist(), targets.tolist())), directed=False)
    g.es["weight"] = weights.tolist()
    # collapse parallel edges by taking matrix weight
    g.simplify(combine_edges="max")
    log(f"  igraph built in {time.time() - t0:.1f}s")
    log(f"  Simplified graph: {g.vcount():,} nodes, {g.ecount():,} edges")

    if graph_path is not None:
        t0 = time.time()
        g.write_pickle(str(graph_path))
        log(f"  Saved igraph to {graph_path.name} ({graph_path.stat().st_size / 1e9:.2f} GB) in {time.time() - t0:.1f}s")

    return g

def run_leiden_on_graph(g, resolution: float = 1.0, partition_type: str = "l", n_iterations: int = -1, seed: int = 42) -> np.ndarray:
    import leidenalg
    # choose partition type
    if partition_type.upper() == "CPM":
        pt = leidenalg.CPMVertexPartition
        log(f"  Partition type: CPMVertexPartition (resolution={resolution})")
    elif partition_type.lower() == "modularity":
        pt = leidenalg.RBConfigurationVertexPartition
        log(f"  Partition type: RBConfigurationVertexPartition (resolution={resolution})")
    else:
        sys.exit("[ERROR] partition_type must be 'CPM' or 'modularity'")

    log(f"  Running Leiden (res={resolution}, type={partition_type}, n_iterations={n_iterations}, seed={seed})")
    t0 = time.time()
    partition = leidenalg.find_partition(g, pt, weights="weight", resolution_parameter=resolution, n_iterations=n_iterations, seed=seed)
    elapsed = time.time() - t0
    log(f"  Leiden done in {elapsed:.1f}s")
    log(f"  Quality: {partition.quality():.4f}")
    labels = np.array(partition.membership, dtype=np.int32)
    return labels

def run_leiden(sources: np.ndarray, targets: np.ndarray, weights: np.ndarray, n_nodes: int, resolution: float = 1.0, partition_type: str = "CPM", n_iterations: int = -1, seed: int = 42) -> np.ndarray:
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        sys.exit("[ERROR] igraph and/or leidenalg not installed")
    g = build_igraph(sources, targets, weights, n_nodes)
    return run_leiden_on_graph(g, resolution, partition_type, n_iterations, seed)

# parallel worker (runs in subprocess)
def _leiden_worker(args_tuple):
    """Standalone worker for ProcessPoolExecutor. Each worker loads the pre-built igraph from disk, runs Leiden at its assigned resolution, and saves labels + cluster file"""
    (
        graph_path,  # Path to saved igraph pickle
        resolution,  # resolution parameter
        partition_type,  # "CPM" or "modularity"
        n_iterations,  # -1 = until convergence
        seed,  # random seed
        ids_path,  # Path to all_ids.txt
        out_dir,  # output directory
        k,  # k from kNN (for filename)
        min_cluster_size,
    ) = args_tuple

    import igraph as ig
    import leidenalg

    out_dir = Path(out_dir)
    pid = os.getpid()
    t0 = time.time()
    print(f"  [PID {pid}] res={resolution}  loading graph", flush=True)
    g = ig.Graph.Read_Pickle(str(graph_path))

    if partition_type.upper() == "CPM":
        pt = leidenalg.CPMVertexPartition
    else:
        pt = leidenalg.RBConfigurationVertexPartition

    print(f"  [PID {pid}] res={resolution}  running Leiden", flush=True)
    t1 = time.time()
    partition = leidenalg.find_partition(g, pt, weights="weight", resolution_parameter=resolution, n_iterations=n_iterations, seed=seed)
    leiden_time = time.time() - t1
    labels = np.array(partition.membership, dtype=np.int32)
    quality = partition.quality()
    del g, partition  # free RAM in this subprocess

    # save labels
    labels_path = out_dir / f"leiden_res{resolution:.2f}_labels.npy"
    np.save(labels_path, labels)

    # load IDs and export clusters
    all_ids = Path(ids_path).read_text().splitlines()
    pt_tag = "cpm" if partition_type.upper() == "CPM" else "mod"
    cluster_path = out_dir / f"leiden_{pt_tag}_k{k}_res{resolution:.2f}.txt"

    # inline export (avoid importing the full module again)
    cluster_map: dict[int, list[str]] = {}
    for i, lab in enumerate(labels):
        lab = int(lab)
        if lab not in cluster_map:
            cluster_map[lab] = []
        cluster_map[lab].append(all_ids[i])
    sorted_clusters = sorted(cluster_map.values(), key=len, reverse=True)
    sorted_clusters = [c for c in sorted_clusters if len(c) >= min_cluster_size]

    with open(cluster_path, "w") as fh:
        for members in sorted_clusters:
            fh.write("\t".join(members) + "\n")

    n_clustered = sum(len(c) for c in sorted_clusters)
    total_time = time.time() - t0

    # compute stats
    sizes = Counter(labels.tolist())
    big = {k_: v for k_, v in sizes.items() if v >= min_cluster_size}
    sz = np.array(list(big.values())) if big else np.array([0])

    result = {
        "resolution": resolution,
        "n_clusters": len(big),
        "n_clustered": n_clustered,
        "n_total": len(labels),
        "quality": quality,
        "leiden_time": leiden_time,
        "total_time": total_time,
        "size_min": int(sz.min()),
        "size_median": float(np.median(sz)),
        "size_mean": float(sz.mean()),
        "size_max": int(sz.max()),
        "cluster_path": str(cluster_path),
    }

    print(
        f"  [PID {pid}] res={resolution}  done in {total_time:.1f}s  "
        f"({len(big):,} clusters, {n_clustered:,} proteins, "
        f"quality={quality:.4f})",
        flush=True,
    )
    return result

# export clusters
def export_clusters(labels: np.ndarray, all_ids: list[str], out_path: Path, min_size: int = 2):
    cluster_map: dict[int, list[str]] = {}
    for i, label in enumerate(labels):
        lab = int(label)
        if lab not in cluster_map:
            cluster_map[lab] = []
        cluster_map[lab].append(all_ids[i])

    # sort clusters by size (largest first)
    sorted_clusters = sorted(cluster_map.values(), key=len, reverse=True)
    sorted_clusters = [c for c in sorted_clusters if len(c) >= min_size]
    with open(out_path, "w") as fh:
        for members in sorted_clusters:
            fh.write("\t".join(members) + "\n")

    n_clustered = sum(len(c) for c in sorted_clusters)
    log(f"  Exported {len(sorted_clusters):,} clusters ({n_clustered:,} proteins, min_size={min_size}) -> {out_path.name}")

    return sorted_clusters

def print_cluster_stats(labels: np.ndarray, resolution: float, min_size: int = 2):
    unique_labels = set(labels.tolist())
    sizes = Counter(labels.tolist())
    # filter singletons / small
    big_clusters = {k: v for k, v in sizes.items() if v >= min_size}
    n_clusters = len(big_clusters)
    if n_clusters == 0:
        log(f"  [res={resolution}] No clusters with >= {min_size} members")
        return
    sz = np.array(list(big_clusters.values()))
    n_clustered = int(sz.sum())
    n_total = len(labels)
    n_singletons = sum(1 for v in sizes.values() if v == 1)

    log(f"  [res={resolution}]  clusters={n_clusters:,}  clustered={n_clustered:,}/{n_total:,} ({100 * n_clustered / n_total:.1f}%)  singletons={n_singletons:,}")
    log(f"    sizes: min={sz.min()}  median={np.median(sz):.0f}  mean={sz.mean():.1f}  max={sz.max()}")
    log(f"    >100: {(sz > 100).sum():,}  >1000: {(sz > 1000).sum():,}")

# optional : also build graph from existing alignment cosines
def build_hybrid_graph(sources_knn: np.ndarray, targets_knn: np.ndarray, weights_knn: np.ndarray, cosine_file: Path, alpha: float = 0.5, min_cosine: float = 0.0, id_to_idx: dict = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """build hybrid graph combining kNN from embeddings with sequence-alignemnt edges that also have cosine similarities"""
    COL_Q, COL_S, COL_EVALUE, COL_COSINE = 0, 4, 12, 14
    EVALUE_CLAMP = 1e-300
    NEG_LOG_MAX = 200.0
    log(f"  Building hybrid graph (alpha={alpha})")
    log(f"  Reading alignment edges from {cosine_file}")

    aln_src, aln_tgt, aln_wt = [], [], []
    n_read = 0
    n_added = 0

    with open(cosine_file) as fh:
        next(fh) # skip header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                continue
            n_read += 1
            q, s = parts[COL_Q], parts[COL_S]
            cos_str = parts[COL_COSINE]
            if cos_str == "NA":
                continue
            qi = id_to_idx.get(q)
            si = id_to_idx.get(s)
            if qi is None or si is None:
                continue
            evalue = float(parts[COL_EVALUE])
            w_seq = min(-np.log10(max(evalue, EVALUE_CLAMP)), NEG_LOG_MAX) / NEG_LOG_MAX
            cosine = float(cos_str)

            w_combined = (1.0 - alpha) * w_seq + alpha * max(cosine, 0.0)

            if w_combined < min_cosine:
                continue

            aln_src.append(qi)
            aln_tgt.append(si)
            aln_wt.append(w_combined)
            n_added += 1
    log(f"  Read {n_read:,} alignment edges, kept {n_added:,}")
    # combine with kNN edges
    all_src = np.concatenate([sources_knn, np.array(aln_src, dtype=np.int64), np.array(aln_tgt, dtype=np.int64)])
    all_tgt = np.concatenate([targets_knn, np.array(aln_tgt, dtype=np.int64), np.array(aln_src, dtype=np.int64)])
    all_wt = np.concatenate([weights_knn, np.array(aln_wt, dtype=np.float32), np.array(aln_wt, dtype=np.float32)])
    log(f"  Hybrid graph: {len(all_src):,} total directed edges")
    return all_src, all_tgt, all_wt

def main():
    p = argparse.ArgumentParser(
        description="Leiden clustering on ProtT5 protein embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--h5_dir", required=True, type=Path, help="Directory containing per-species .h5 embedding files")
    p.add_argument("--out_dir", required=True, type=Path, help="Output directory for cluster files and stats")
    p.add_argument("--k", type=int, default=20, help="Number of nearest neighbours for kNN graph (default: 20)")
    p.add_argument("--min_cosine", type=float, default=0.0, help="Minimum cosine similarity to keep an edge (default: 0.0)")
    p.add_argument("--use_sklearn", action="store_true", help="Use sklearn instead of FAISS for kNN (slower, no FAISS needed)")
    p.add_argument("--resolutions", type=float, nargs="+", default=[1.0], help="Resolution parameter(s) for Leiden (default: 1.0). Higher = more/smaller clusters")
    p.add_argument("--partition_type", default="CPM", choices=["CPM", "modularity"], help="Leiden partition type (default: CPM).")
    p.add_argument("--n_iterations", type=int, default=-1, help="Max Leiden iterations (-1 = until convergence, default: -1)")
    p.add_argument("--seed", type=int, default=42)
    # Hybrid graph (optional)
    p.add_argument("--cosine_file", type=Path, default=None, help="diamond_with_cosine.tsv file for hybrid graph (optional)")
    p.add_argument("--alpha", type=float, default=0.5, help="Mixing weight for hybrid graph")

    p.add_argument("--min_cluster_size", type=int, default=2, help="Minimum cluster size to export (default: 2)")
    p.add_argument("--n_jobs", type=int, default=-1, help="Number of CPU threads (-1 = all, default: -1)")
    p.add_argument("--parallel", type=int, default=1, metavar="WORKERS", help="Number of parallel workers for resolution sweep.")
    p.add_argument("--skip_knn", action="store_true", help="Skip kNN graph building (load from saved files)")

    args = p.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    n_workers = min(args.parallel, len(args.resolutions))

    log(SEP)
    log("  Leiden Clustering on ProtT5 Embeddings")
    log(SEP)
    log(f"  h5_dir         : {args.h5_dir}")
    log(f"  out_dir        : {out_dir}")
    log(f"  k              : {args.k}")
    log(f"  min_cosine     : {args.min_cosine}")
    log(f"  resolutions    : {args.resolutions}")
    log(f"  partition_type : {args.partition_type}")
    log(f"  min_cluster_sz : {args.min_cluster_size}")
    log(f"  parallel       : {n_workers} worker(s)")
    if args.cosine_file:
        log(f"  cosine_file    : {args.cosine_file}")
        log(f"  alpha          : {args.alpha}")
    log(SEP)

    # load emb
    ids_path = out_dir / "all_ids.txt"
    sp_path = out_dir / "all_species.txt"

    if ids_path.exists() and sp_path.exists() and args.skip_knn:
        log("Loading saved IDs and species")
        all_ids = ids_path.read_text().splitlines()
        all_species = sp_path.read_text().splitlines()
        embs = None
    else:
        all_ids, all_species, embs = load_all_embeddings(args.h5_dir)
        ids_path.write_text("\n".join(all_ids))
        sp_path.write_text("\n".join(all_species))

    N = len(all_ids)
    log(f"Working with {N:,} proteins")

    # build kNN
    knn_src_path = out_dir / "knn_sources.npy"
    knn_tgt_path = out_dir / "knn_targets.npy"
    knn_wt_path = out_dir / "knn_weights.npy"

    if args.skip_knn and knn_src_path.exists():
        log("Loading saved kNN graph")
        sources = np.load(knn_src_path)
        targets = np.load(knn_tgt_path)
        weights = np.load(knn_wt_path)
        log(f"  Loaded {len(sources):,} edges")
    else:
        log(SEP)
        log("Step 2: Building kNN graph from embeddings")
        if embs is None:
            log("  Loading embeddings")
            all_ids, all_species, embs = load_all_embeddings(args.h5_dir)

        if args.use_sklearn:
            sources, targets, weights = build_knn_graph_sklearn(embs, k=args.k, min_cosine=args.min_cosine, n_jobs=args.n_jobs)
        else:
            sources, targets, weights = build_knn_graph_faiss(embs, k=args.k, min_cosine=args.min_cosine, n_jobs=args.n_jobs)

        np.save(knn_src_path, sources)
        np.save(knn_tgt_path, targets)
        np.save(knn_wt_path, weights)
        log(f"  Saved kNN graph to {out_dir}")
        del embs  # free RAM

    # hybrid graph
    if args.cosine_file:
        log(SEP)
        log("Building hybrid graph (kNN + alignment edges)")
        id_to_idx = {pid: i for i, pid in enumerate(all_ids)}
        sources, targets, weights = build_hybrid_graph(sources, targets, weights, args.cosine_file, args.alpha, args.min_cosine, id_to_idx)

    # build iGraph once
    log(SEP)
    log("Step 3: Building igraph (once)")
    graph_path = out_dir / "knn_graph.igraph.pickle"
    if graph_path.exists() and args.skip_knn:
        log(f"  Using saved igraph from {graph_path.name}")
    else:
        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            sys.exit("[ERROR] igraph and/or leidenalg not installed")

        g = build_igraph(sources, targets, weights, n_nodes=N, graph_path=graph_path)
        del sources, targets, weights # free RAM

    # run Leiden at each resolution
    log(SEP)
    log(f"Step 4: Leiden clustering ({len(args.resolutions)} resolutions, {n_workers} worker(s))")

    if n_workers <= 1:
        # sequential mode
        if "g" not in dir():
            import igraph as ig
            log(f"  Loading igraph from {graph_path.name}")
            g = ig.Graph.Read_Pickle(str(graph_path))

        for res in args.resolutions:
            log(f"\n{'─' * 40}")
            log(f"Resolution = {res}")

            labels = run_leiden_on_graph(g, resolution=res, partition_type=args.partition_type, n_iterations=args.n_iterations, seed=args.seed)
            # stats
            print_cluster_stats(labels, res, min_size=args.min_cluster_size)
            # save labels
            labels_path = out_dir / f"leiden_res{res:.2f}_labels.npy"
            np.save(labels_path, labels)
            # export in MCL-compatible format
            pt_tag = "cpm" if args.partition_type.upper() == "CPM" else "mod"
            cluster_path = out_dir / f"leiden_{pt_tag}_k{args.k}_res{res:.2f}.txt"
            export_clusters(labels, all_ids, cluster_path, min_size=args.min_cluster_size)

    else:
        # parallel mode
        if "g" in dir():
            del g
        import gc; gc.collect()
        from concurrent.futures import ProcessPoolExecutor, as_completed
        worker_args = [
            (
                str(graph_path),
                res,
                args.partition_type,
                args.n_iterations,
                args.seed,
                str(ids_path),
                str(out_dir),
                args.k,
                args.min_cluster_size,
            )
            for res in args.resolutions
        ]
        log(f"  Launching {n_workers} parallel workers for  {len(args.resolutions)} resolutions")
        log(f"  NOTE: each worker loads its own copy of the graph  ({graph_path.stat().st_size / 1e9:.2f} GB on disk). Ensure sufficient RAM for {n_workers} copies.")
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_leiden_worker, wa): wa[1] for wa in worker_args}
            for future in as_completed(futures):
                res_val = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    log(f"  [ERROR] Resolution {res_val} failed: {exc}")
                    import traceback;
                    traceback.print_exc()

        # print summary table sorted by resolution
        results.sort(key=lambda r: r["resolution"])
        log(f"\n{'─' * 70}")
        log(f"  {'res':>6s}  {'clusters':>9s}  {'clustered':>10s}  {'quality':>10s}  {'time':>7s}  {'median_sz':>9s}")
        log(f"  {'─' * 6}  {'─' * 9}  {'─' * 10}  {'─' * 10}  {'─' * 7}  {'─' * 9}")
        for r in results:
            pct = 100 * r["n_clustered"] / r["n_total"]
            log(f"  {r['resolution']:6.2f}  {r['n_clusters']:>9,}  "
                f"{r['n_clustered']:>8,} ({pct:4.1f}%)  "
                f"{r['quality']:>10.2f}  {r['leiden_time']:>6.1f}s  "
                f"{r['size_median']:>9.0f}")

    wall_time = time.time() - _t0_global
    log(f"\n{SEP}")
    log(f"Done. Total wall time: {wall_time / 60:.1f} min")
    log(f"Output: {out_dir}")
    log(SEP)

if __name__ == "__main__":
    main()











