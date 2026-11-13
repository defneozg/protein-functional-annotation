import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np

_t0 = time.time()

def log(msg: str):
    elapsed = time.time() - _t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"[{h:02d}:{m:02d}:{s:02d}] {msg}", flush=True)

SEP = "=" * 60

# load abc file -> igraph
def load_abc_to_igraph(abc_path: Path, normalize: bool = True, graph_path: Path = None):
    import igraph as ig
    log(f"Reading ABC file: {abc_path}")
    # first pass: collect unique node labels and edges
    label2idx: dict[str, int] = {}
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    n_lines = 0

    t0 = time.time()
    with open(abc_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                # try space-separated (some ABC files use spaces)
                parts = line.rstrip("\n").split()
            if len(parts) < 3:
                continue

            a, b = parts[0], parts[1]
            try:
                w = float(parts[2])
            except ValueError:
                continue

            if a == b:
                continue  # skip self-loops

            if a not in label2idx:
                label2idx[a] = len(label2idx)
            if b not in label2idx:
                label2idx[b] = len(label2idx)

            ai, bi = label2idx[a], label2idx[b]
            edges.append((ai, bi))
            weights.append(w)

            n_lines += 1
            if n_lines % 10_000_000 == 0:
                elapsed = time.time() - t0
                log(f"  {n_lines / 1e6:.0f}M edges read  ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    n_nodes = len(label2idx)
    log(f"  Read {n_lines:,} edges, {n_nodes:,} nodes in {elapsed:.1f}s")
    # build label list (idx -> label)
    labels = [""] * n_nodes
    for label, idx in label2idx.items():
        labels[idx] = label
    del label2idx

    # normalise weights
    w_arr = np.array(weights, dtype=np.float32)
    w_max = float(w_arr.max())
    w_min = float(w_arr.min())
    log(f"  Weight range: [{w_min:.4f}, {w_max:.4f}]")

    if normalize and w_max > 0:
        w_arr /= w_max
        log(f"  Normalised to [0, 1] (divided by {w_max:.4f})")

    # build igraph (undirected, mirrors + max-simplify)
    log(f"  Building igraph: {n_nodes:,} nodes, {len(edges):,} edges")
    t0 = time.time()

    g = ig.Graph(n=n_nodes, edges=edges, directed=False)
    g.es["weight"] = w_arr.tolist()
    del edges, weights, w_arr

    # simplify: parallel edges → keep max weight
    n_before = g.ecount()
    g.simplify(combine_edges="max")
    n_after = g.ecount()
    log(f"  Simplified: {n_before:,} → {n_after:,} edges "
        f"(removed {n_before - n_after:,} duplicates)")
    log(f"  igraph built in {time.time() - t0:.1f}s")

    # connected components info
    components = g.connected_components()
    sizes = [len(c) for c in components]
    log(f"  Connected components: {len(sizes):,}")
    log(f"    Giant component: {max(sizes):,} nodes ({100 * max(sizes) / n_nodes:.1f}%)")
    log(f"    Singletons: {sum(1 for s in sizes if s == 1):,}")

    # save pickle
    if graph_path is not None:
        t0 = time.time()
        g.write_pickle(str(graph_path))
        sz_gb = graph_path.stat().st_size / 1e9
        log(f"  Saved igraph pickle: {graph_path.name} ({sz_gb:.2f} GB) "
            f"in {time.time() - t0:.1f}s")

    return g, labels

def run_leiden_on_graph(g, resolution, partition_type, n_iterations, seed):
    import leidenalg
    if partition_type.upper() == "CPM":
        pt = leidenalg.CPMVertexPartition
    else:
        pt = leidenalg.RBConfigurationVertexPartition

    t0 = time.time()
    partition = leidenalg.find_partition(
        g, pt,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=n_iterations,
        seed=seed,
    )
    elapsed = time.time() - t0
    quality = partition.quality()
    log(f"  Leiden done in {elapsed:.1f}s  |  quality={quality:.4f}")
    return np.array(partition.membership, dtype=np.int32)

def _leiden_worker(args_tuple):
    """worker for parallel resolution sweep"""
    (graph_path, resolution, partition_type, n_iterations, seed, labels_file, out_dir, min_cluster_size, abc_stem) = args_tuple

    import igraph as ig
    import leidenalg

    out_dir = Path(out_dir)
    pid = os.getpid()
    t0 = time.time()

    print(f"  [PID {pid}] res={resolution}  loading graph ...", flush=True)
    g = ig.Graph.Read_Pickle(str(graph_path))

    if partition_type.upper() == "CPM":
        pt = leidenalg.CPMVertexPartition
    else:
        pt = leidenalg.RBConfigurationVertexPartition

    print(f"  [PID {pid}] res={resolution}  running Leiden "
          f"(n_iter={n_iterations}) ...", flush=True)
    t1 = time.time()
    partition = leidenalg.find_partition(
        g, pt,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=n_iterations,
        seed=seed,
    )
    leiden_time = time.time() - t1
    membership = np.array(partition.membership, dtype=np.int32)
    quality = partition.quality()
    del g, partition

    # load labels and export clusters
    all_labels = Path(labels_file).read_text().splitlines()

    cluster_map: dict[int, list[str]] = {}
    for i, lab in enumerate(membership):
        lab = int(lab)
        if lab not in cluster_map:
            cluster_map[lab] = []
        cluster_map[lab].append(all_labels[i])

    sorted_clusters = sorted(cluster_map.values(), key=len, reverse=True)
    sorted_clusters = [c for c in sorted_clusters if len(c) >= min_cluster_size]

    pt_tag = "cpm" if partition_type.upper() == "CPM" else "mod"
    cluster_path = out_dir / f"leiden_{pt_tag}_{abc_stem}_res{resolution}.txt"
    with open(cluster_path, "w") as fh:
        for members in sorted_clusters:
            fh.write("\t".join(members) + "\n")

    n_clustered = sum(len(c) for c in sorted_clusters)
    total_time = time.time() - t0

    sizes = Counter(membership.tolist())
    big = {k: v for k, v in sizes.items() if v >= min_cluster_size}
    sz = np.array(list(big.values())) if big else np.array([0])

    result = {
        "resolution": resolution,
        "n_clusters": len(big),
        "n_clustered": n_clustered,
        "n_total": len(membership),
        "n_singletons": sum(1 for v in sizes.values() if v == 1),
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
        f"singletons={result['n_singletons']:,}, quality={quality:.4f})",
        flush=True,
    )
    return result

def export_clusters(membership, labels, out_path, min_size=2):
    cluster_map: dict[int, list[str]] = {}
    for i, lab in enumerate(membership):
        lab = int(lab)
        if lab not in cluster_map:
            cluster_map[lab] = []
        cluster_map[lab].append(labels[i])

    sorted_clusters = sorted(cluster_map.values(), key=len, reverse=True)
    sorted_clusters = [c for c in sorted_clusters if len(c) >= min_size]

    with open(out_path, "w") as fh:
        for members in sorted_clusters:
            fh.write("\t".join(members) + "\n")

    n_clustered = sum(len(c) for c in sorted_clusters)
    log(f"  Exported {len(sorted_clusters):,} clusters "
        f"({n_clustered:,} proteins) -> {out_path.name}")
    return sorted_clusters

def print_stats(membership, resolution, min_size=2):
    sizes = Counter(membership.tolist())
    big = {k: v for k, v in sizes.items() if v >= min_size}
    n_clusters = len(big)
    if n_clusters == 0:
        log(f"  [res={resolution}] No clusters with >= {min_size} members")
        return
    sz = np.array(list(big.values()))
    n_clustered = int(sz.sum())
    n_total = len(membership)
    n_singletons = sum(1 for v in sizes.values() if v == 1)
    log(f"  [res={resolution}]  clusters={n_clusters:,}  "
        f"clustered={n_clustered:,}/{n_total:,} ({100 * n_clustered / n_total:.1f}%)  "
        f"singletons={n_singletons:,}")
    log(f"    sizes: min={sz.min()}  median={np.median(sz):.0f}  "
        f"mean={sz.mean():.1f}  max={sz.max()}")


def main():
    p = argparse.ArgumentParser(
        description="Leiden clustering on BLAST/DIAMOND ABC-format alignment graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--abc", required=True, type=Path, help="ABC-format alignment file (protein1 TAB protein2 TAB weight)")
    p.add_argument("--out_dir", required=True, type=Path, help="Output directory")
    # graph construction
    p.add_argument("--no_normalize", action="store_true", help="Do NOT normalise weights to [0,1]. Use if weights are already in a meaningful scale for the resolution parameter.")
    p.add_argument("--skip_graph", action="store_true", help="Skip ABC loading")
    p.add_argument("--resolutions", type=float, nargs="+", default=[0.01], help="Resolution parameter(s) for Leiden")
    p.add_argument("--partition_type", default="CPM", choices=["CPM", "modularity"], help="Leiden partition type (default: CPM)")
    p.add_argument("--n_iterations", type=int, default=2, help="Leiden iterations (default: 2, -1 = full convergence)")
    p.add_argument("--seed", type=int, default=42)
    # output
    p.add_argument("--min_cluster_size", type=int, default=2)
    # parallelism
    p.add_argument("--parallel", type=int, default=1, metavar="WORKERS", help="Parallel workers for resolution sweep (default: 1)")

    args = p.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_workers = min(args.parallel, len(args.resolutions))
    abc_stem = args.abc.stem  # for output filenames

    graph_path = out_dir / "alignment_graph.igraph.pickle"
    labels_path = out_dir / "node_labels.txt"

    log(SEP)
    log("  Leiden Clustering on Alignment Graph (ABC)")
    log(SEP)
    log(f"  ABC file       : {args.abc}")
    log(f"  out_dir        : {out_dir}")
    log(f"  normalise      : {not args.no_normalize}")
    log(f"  resolutions    : {args.resolutions}")
    log(f"  partition_type : {args.partition_type}")
    log(f"  n_iterations   : {args.n_iterations}")
    log(f"  parallel       : {n_workers} worker(s)")
    log(SEP)

    if args.skip_graph and graph_path.exists() and labels_path.exists():
        log(f"Reusing saved graph from {graph_path.name}")
        node_labels = labels_path.read_text().splitlines()
        log(f"  {len(node_labels):,} nodes")
        g = None  # will load from pickle when needed
    else:
        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            sys.exit("[ERROR] pip install leidenalg igraph")

        g, node_labels = load_abc_to_igraph(
            args.abc,
            normalize=not args.no_normalize,
            graph_path=graph_path,
        )
        labels_path.write_text("\n".join(node_labels))
        log(f"  Saved node labels → {labels_path.name}")

    log(SEP)
    log(f"Leiden clustering ({len(args.resolutions)} resolutions, "
        f"{n_workers} worker(s), n_iter={args.n_iterations})")

    if n_workers <= 1:
        # sequential
        if g is None:
            import igraph as ig
            log(f"  Loading igraph from {graph_path.name}")
            g = ig.Graph.Read_Pickle(str(graph_path))

        for res in args.resolutions:
            log(f"\n{'─' * 40}")
            log(f"Resolution = {res}")
            membership = run_leiden_on_graph(g, res, args.partition_type, args.n_iterations, args.seed)
            print_stats(membership, res, args.min_cluster_size)

            pt_tag = "cpm" if args.partition_type.upper() == "CPM" else "mod"
            cluster_path = out_dir / f"leiden_{pt_tag}_{abc_stem}_res{res}.txt"
            export_clusters(membership, node_labels, cluster_path,
                            min_size=args.min_cluster_size)
    else:
        # parallel
        if g is not None:
            del g
        import gc; gc.collect()
        from concurrent.futures import ProcessPoolExecutor, as_completed

        worker_args = [
            (str(graph_path), res, args.partition_type, args.n_iterations,
             args.seed, str(labels_path), str(out_dir),
             args.min_cluster_size, abc_stem)
            for res in args.resolutions
        ]

        log(f"  Launching {n_workers} parallel workers ...")
        log(f"  Graph on disk: {graph_path.stat().st_size / 1e9:.2f} GB  "
            f"(each worker loads its own copy)")

        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_leiden_worker, wa): wa[1]
                for wa in worker_args
            }
            for future in as_completed(futures):
                res_val = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    log(f"  [ERROR] Resolution {res_val} failed: {exc}")
                    import traceback; traceback.print_exc()

        # summary table
        results.sort(key=lambda r: r["resolution"])
        log(f"\n{'─' * 85}")
        log(f"  {'res':>8s}  {'clusters':>9s}  {'clustered':>12s}  "
            f"{'singletons':>10s}  {'quality':>10s}  {'time':>7s}  "
            f"{'median':>7s}  {'max':>7s}")
        log(f"  {'─' * 8}  {'─' * 9}  {'─' * 12}  {'─' * 10}  "
            f"{'─' * 10}  {'─' * 7}  {'─' * 7}  {'─' * 7}")
        for r in results:
            pct = 100 * r["n_clustered"] / r["n_total"]
            log(f"  {r['resolution']:>8.4f}  {r['n_clusters']:>9,}  "
                f"{r['n_clustered']:>8,} ({pct:4.1f}%)  "
                f"{r['n_singletons']:>10,}  "
                f"{r['quality']:>10.2f}  {r['leiden_time']:>6.1f}s  "
                f"{r['size_median']:>7.0f}  {r['size_max']:>7,}")

    wall_time = time.time() - _t0
    log(f"\n{SEP}")
    log(f"Done. Total wall time: {wall_time / 60:.1f} min")
    log(f"Output: {out_dir}")
    log(SEP)

if __name__ == "__main__":
    main()