"""
Two modes:
  viz      — UMAP visualization of protein embeddings, colored by:
               (a) GO molecular function (gray for unannotated)
               (b) species of origin

  cluster  — HDBSCAN clustering on UMAP-reduced embeddings, then
             compare resulting clusters to MCL clusters.

Install
-------
    pip install umap-learn hdbscan matplotlib numpy h5py scikit-learn tqdm
"""

import argparse
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure

SEP = "-"*60

# I/O helpers
def load_go(pkl_path: Path) -> dict[str, list]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return {k: set(v) if not isinstance(v, set) else v for k, v in data.items()}

def load_h5_dir_sampled(h5_dir: Path, n_sample: int, rng: np.random.Generator, protein2go:dict = None, n_annotated_oversample: int = 10_000) -> tuple[list[str], list[str], np.ndarray]:
    """stratified sample across all h5 files, each species contributes proportionally to its size"""
    h5_files = sorted(h5_dir.glob("*.h5"))
    if not h5_files:
        sys.exit(f"[ERROR] no h5 files in {h5_dir}")

    # first pass; get sizes
    sizes = {}
    for fp in h5_files:
        with h5py.File(fp, "r") as hf:
            sizes[fp] = len(hf["ids"])
    total = sum(sizes.values())
    print(f"  Total proteins across {len(h5_files)} species: {total}")
    print(f"  Background sampling {n_sample:,} stratified across species")
    if protein2go:
        print(f"  Annotated oversample : up to {n_annotated_oversample:,}")

    all_ids, all_species, all_embs = [], [], []
    seen_ids = set()
    # pass 1 : stratified background sample
    for fp in tqdm(h5_files, desc="  Loading (background)"):
        n_take = max(1, round(n_sample * sizes[fp] / total))
        with h5py.File(fp, "r") as hf:
            ids_all = [s.decode() if isinstance(s, bytes) else s for s in hf["ids"][:]]
            embs_all = hf["embeddings"][:].astype(np.float32)

        # random sample within the species
        idx = rng.choice(len(ids_all), size=min(n_take, len(ids_all)), replace=False)
        species_name = fp.stem # e.g. "Amoebophrya_sp_EP00392"
        for i in idx:
            all_ids.append(ids_all[i])
            all_species.append(species_name)
            seen_ids.add(ids_all[i])
        all_embs.append(embs_all[idx])

    # pass 2 : oversample annotated proteins not already included
    if protein2go:
        n_added = 0
        for fp in tqdm(h5_files, desc="  Loading (annotated)"):
            if n_added >= n_annotated_oversample:
                break
            with h5py.File(fp, "r") as hf:
                ids_all = [s.decode() if isinstance(s, bytes) else s for s in hf["ids"][:]]
                embs_all = hf["embeddings"][:].astype(np.float32)
            species_name = fp.stem
            for i, pid in enumerate(ids_all):
                if n_added >= n_annotated_oversample:
                    break
                if pid in seen_ids or pid not in protein2go:
                    continue
                all_ids.append(pid)
                all_species.append(species_name)
                all_embs.append(embs_all[i:i+1])
                seen_ids.add(pid)
                n_added += 1
        print(f"  Added {n_added:,} annotated proteins (oversampled)")

    emb_matrix = np.concatenate(all_embs, axis=0)
    print(f"  Sampled {len(all_ids):,} proteins | dim={emb_matrix.shape[1]}")
    return all_ids, all_species, emb_matrix

def load_mcl_clusters(mcl_file: Path) -> dict[str, int]:
    protein2cluster = {}
    with open(mcl_file) as fh:
        for cid, line in enumerate(fh):
            for pid in line.strip().split("\t"):
                if pid:
                    protein2cluster[pid] = cid
    return protein2cluster

# UMAP
def run_umap(embs: np.ndarray, n_components: int = 2, **kwargs) -> np.ndarray:
    try:
        import umap
    except ImportError:
        sys.exit("[ERROR] cannot import umap, run pip install umap-learn")

    default_kwargs = dict(
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        low_memory=True,
        verbose=True,
    )
    default_kwargs.update(kwargs)
    reducer = umap.UMAP(n_components=n_components, **default_kwargs)
    return reducer.fit_transform(embs)

# plotting
CB_PALETTE = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
    "#D55E00", "#CC79A7", "#000000", "#999999", "#44AA99",
    "#332288", "#88CCEE", "#DDCC77", "#117733", "#882255",
    "#AA4499", "#661100", "#6699CC", "#AA4466", "#4477AA",
]

def plot_umap_go(xy: np.ndarray, ids: list[str], protein2go: dict[str, set], out_path: Path, n_top_terms: int = 10, point_size: float = 2.0, alpha_annotated: float = 0.8, alpha_unannotated: float = 0.08):
    """unannotated proteins in gray, rest color-coded"""
    print("\n  Building GO color map")
    # assign each protein a representative GO term (most frequent terms across the sample)
    term_counter = Counter()
    for pid in ids:
        for t in protein2go.get(pid, set()):
            term_counter[t] += 1
    top_terms = [t for t, _ in term_counter.most_common(n_top_terms)]
    term2color = {t: CB_PALETTE[i % len(CB_PALETTE)] for i, t in enumerate(top_terms)}
    print(f"  Top {n_top_terms} GO terms : {top_terms}")

    # classify each protein
    categories = []  # "unannotated" | "other_annotated" | GO-term string
    for pid in ids:
        go_terms = protein2go.get(pid, set())
        if not go_terms:
            categories.append("unannotated")
        else:
            best = max(
                (t for t in go_terms if t in term2color),
                key=lambda t: term_counter[t],
                default=None,
            )
            categories.append(best if best else "other_annotated")
    categories = np.array(categories)

    n_annot = (categories != "unannotated").sum()
    n_unannot = (categories == "unannotated").sum()
    n_in_top = sum((categories == t).sum() for t in top_terms)
    print(f"  Unannotated     : {n_unannot:,}")
    print(f"  Other annotated : {(categories == 'other_annotated').sum():,}")
    print(f"  In top-{n_top_terms} terms  : {n_in_top:,}")

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor("#F5F5F5")
    ax.set_facecolor("#F5F5F5")
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Layer 1: unannotated -> light gray, tiny, very transparent
    mask = categories == "unannotated"
    ax.scatter(xy[mask, 0], xy[mask, 1], c="#CCCCCC", s=1, alpha=0.15, linewidths=0, rasterized=True, zorder=1)

    # Layer 2: annotated but not in top-N —> mid gray, small
    mask = categories == "other_annotated"
    ax.scatter(xy[mask, 0], xy[mask, 1], c="#999999", s=3, alpha=0.4, linewidths=0, rasterized=True, zorder=2)

    # Layer 3: top-N GO terms —> bright colors, larger, on top
    for z, term in enumerate(top_terms):
        mask = categories == term
        n = mask.sum()
        if n == 0:
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1], c=term2color[term], s=8, alpha=0.85, linewidths=0, rasterized=True, zorder=3 + z, label=f"{term}  (n={n:,})")

    handles = [
        mpatches.Patch(color=term2color[t], label=f"{t}  (n={(categories == t).sum():,})") for t in top_terms if (categories == t).sum() > 0
    ]
    handles += [
        mpatches.Patch(color="#999999", label=f"Other annotated  (n={(categories == 'other_annotated').sum():,})"),
        mpatches.Patch(color="#CCCCCC", label=f"Unannotated  (n={n_unannot:,})"),
    ]
    ax.legend(handles=handles, fontsize=7.5, framealpha=0.9, facecolor="white", loc="lower right", markerscale=2, title=f"MF GO term (top {n_top_terms})", title_fontsize=8)
    ax.set_title(
        f"UMAP — Protein Embeddings colored by MF GO term\n"
        f"n={len(ids):,}  |  annotated={n_annot:,} ({100 * n_annot / len(ids):.1f}%)"
        f"  |  top-{n_top_terms} GO terms highlighted",
        fontsize=11, pad=10,
    )
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> {out_path.name} saved")

def plot_umap_species(xy: np.ndarray, species: list[str], out_path: Path, n_top_species: int = 20, point_size: float = 2.0, alpha: float = 0.3):
    print("\n  Building species color map")
    species_arr = np.array(species)
    species_counts = Counter(species)
    top_species = [s for s, _ in species_counts.most_common(n_top_species)]
    sp2color = {s: CB_PALETTE[i % len(CB_PALETTE)] for i, s in enumerate(top_species)}
    colors = np.array([sp2color.get(s, "#CCCCCC") for s in species])
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")

    # shuffle so no species systematically occludes another
    shuffle_idx = np.random.default_rng(0).permutation(len(species))
    ax.scatter(xy[shuffle_idx, 0], xy[shuffle_idx, 1], c=colors[shuffle_idx], s=point_size, alpha=alpha, linewidths=0, rasterized=True)

    patches = [mpatches.Patch(color=sp2color[s], label=f"{s.replace('_', ' ')} ({species_counts[s]:,})") for s in top_species]
    if len(species_counts) > n_top_species:
        patches.append(mpatches.Patch(color="#CCCCCC", label="Other species"))

    ax.legend(handles=patches, fontsize=6, markerscale=3,
              loc="upper right", framealpha=0.7,
              labelcolor="white", facecolor="#222222",
              title="Species", title_fontsize=7)
    ax.set_title(
        f"UMAP of protein embeddings — colored by species\n"
        f"n={len(species):,}  across {len(species_counts)} species",
        fontsize=11, color="white",
    )
    ax.set_xlabel("UMAP 1", color="white");
    ax.set_ylabel("UMAP 2", color="white")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> {out_path.name} saved")

# viz mode
def run_viz(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    print(SEP)
    print(f"  Embedding Visualization")
    print(SEP)
    # load GO
    print("Loading GO annotations")
    protein2go = load_go(Path(args.go_pkl))
    print(f"  GO-annotated proteins: {len(protein2go):,}")
    # load sample
    ids, species, embs = load_h5_dir_sampled(Path(args.h5_dir), args.n_sample, rng, protein2go=protein2go, n_annotated_oversample=args.n_annotated_oversample)
    # UMAP 2D
    print(f"\n  Running UMAP (n={len(ids):,}, dim={embs.shape[1]})")
    t0 = time.time()
    xy = run_umap(embs, n_components = 2)
    print(f"  UMAP done in {time.time() - t0:.1f}s")
    # save coords so we can replot without re-running UMAP
    np.save(out_dir / "umap_xy.npy", xy)
    np.save(out_dir / "umap_embs.npy", embs)
    with open(out_dir / "umap_ids.txt", "w") as f:
        f.write("\n".join(ids))
    with open(out_dir / "umap_species.txt", "w") as f:
        f.write("\n".join(species))
    print(f"  UMAP coordinates saved to {out_dir}/umap_xy.npy")

    # plot 1: go func
    print(f"\n  Plot 1: GO molecular function coloring")
    plot_umap_go(xy, ids, protein2go, out_path=out_dir / "umap_go_function.png", n_top_terms=args.n_top_terms)

    # plot 2: species
    print(f"\n  Plot 2: Species coloring")
    plot_umap_species(xy, species, out_path=out_dir / "umap_species.png")

    print(f"\n{SEP}")
    print(f" Visualization done, figs in {out_dir}")
    print(SEP)

# cluster mode
def run_cluster(args):
    try:
        import hdbscan
    except ImportError:
        sys.exit("[ERROR] hdbscan not installed")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    print(SEP)
    print(f"  Embedding Clustering (HDBSCAN)")
    print(SEP)

    # load UMAP coords if already computed
    xy_path = out_dir / "umap_xy.npy"
    ids_path = out_dir / "umap_ids.txt"
    sp_path = out_dir / "umap_species.txt"
    emb_path = out_dir / "umap_embs.npy"

    if xy_path.exists() and ids_path.exists():
        print("\n  Loading UMAP coordinates")
        xy = np.load(xy_path)
        ids = open(ids_path).read().splitlines()
        species = open(sp_path).read().splitlines() if sp_path.exists() else ["?"] * len(ids)
        embs = np.load(emb_path) if emb_path.exists() else None
        print(f"  Loaded {len(ids):,} proteins")
    else:
        print("\n  No saved UMAP found, loading embeddings and running UMAP")
        ids, species, embs = load_h5_dir_sampled(Path(args.h5_dir), args.n_sample, rng)
        xy = run_umap(embs, n_components = 2)
        np.save(xy_path, xy)
        np.save(emb_path, embs)
        with open(ids_path, "w") as f: f.write("\n".join(ids))
        with open(sp_path, "w") as f: f.write("\n".join(species))

    # HDBSCAN, running on higher dim 10D if embs available, fall back to 2D
    if embs is not None and args.umap_dim_cluster > 2:
        print(f"\n  Running UMAP to {args.umap_dim_cluster}D for clustering")
        xy_hd = run_umap(embs, n_components = args.umap_dim_cluster, n_neighbors = 30, min_dist = 0.0)
        cluster_input = xy_hd
        print(f"  Using {args.umap_dim_cluster}D UMAP for HDBSCAN")
    else:
        cluster_input = xy
        print("  Using 2D UMAP for HDBSCAN")

    print(f"\n  Running HDBSCAN")
    print(f"  min_cluster_size={args.min_cluster_size}  min_samples={args.min_samples}  metric=euclidean")
    t0 = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom", # excess of mass (fewer but larger clusters)
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(cluster_input)
    elapsed = time.time() - t0

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Clusters found : {n_clusters:,}")
    print(f"  Noise points   : {n_noise:,} ({100 * n_noise / len(labels):.1f}%)")
    # cluster size distribution
    cluster_sizes = Counter(labels[labels >= 0])
    sizes = np.array(list(cluster_sizes.values()))
    print(f"  Cluster sizes : min = {sizes.min()}  median={np.median(sizes):.0f} mean={sizes.mean():.0f}  max={sizes.max()}")

    # plot HDBSCAN clusters on 2D UMAP
    print(f"\n  Plotting HDBSCAN clusters")
    top_cluster_ids = [c for c, _ in Counter({k: v for k, v in cluster_sizes.items()}).most_common(20)]
    c2color = {c: CB_PALETTE[i % len(CB_PALETTE)] for i, c in enumerate(top_cluster_ids)}
    colors = np.array([c2color.get(l, "#CCCCCC") if l >= 0 else "#222222" for l in labels])
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#111111"); fig.patch.set_facecolor("#111111")
    noise_mask = labels == -1
    ax.scatter(xy[noise_mask, 0], xy[noise_mask, 1], c="#222222", s=1, alpha=0.3, linewidths=0, rasterized=True)
    ax.scatter(xy[~noise_mask, 0], xy[~noise_mask, 1], c=colors[~noise_mask], s=2, alpha=0.5, linewidths=0, rasterized=True)
    ax.set_title(
        f"HDBSCAN clusters on UMAP\n"
        f"{n_clusters} clusters  |  {n_noise:,} noise points ({100 * n_noise / len(labels):.1f}%)",
        color="white", fontsize=11,
    )
    ax.set_xlabel("UMAP 1", color="white");
    ax.set_ylabel("UMAP 2", color="white")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.tight_layout()
    fig.savefig(out_dir / "umap_hdbscan.png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> umap_hdbscan.png saved")

    # compare to mcl clusters
    if args.mcl_file:
        print(f"\n  Comparing to MCL clusters ({args.mcl_file})")
        protein2mcl = load_mcl_clusters(Path(args.mcl_file))
        # build parallel arrays for proteins that appear in both
        hdb_labels_matched, mcl_labels_matched = [], []
        n_missing_mcl = 0
        for pid, hdb_label in zip(ids, labels):
            if hdb_label == -1:
                continue # skip noise
            mcl_label = protein2mcl.get(pid)
            if mcl_label is None:
                n_missing_mcl += 1
                continue
            hdb_labels_matched.append(hdb_label)
            mcl_labels_matched.append(mcl_label)

        hdb_arr = np.array(hdb_labels_matched)
        mcl_arr = np.array(mcl_labels_matched)
        n_matched = len(hdb_arr)
        print(f"  Proteins in both clusterings: {n_matched:,}")
        print(f"  Not in MCL file: {n_missing_mcl:,}")

        ari = adjusted_rand_score(mcl_arr, hdb_arr)
        nmi = normalized_mutual_info_score(mcl_arr, hdb_arr)
        hom, comp, vmeas = homogeneity_completeness_v_measure(mcl_arr, hdb_arr)
        print(f"\n  Clustering comparison (HDBSCAN vs MCL):")
        print(f"  Adjusted Rand Index: {ari:.4f}  (1=identical, 0=random, can be negative)")
        print(f"  Normalized Mutual Info: {nmi:.4f}  (1=identical, 0=independent)")
        print(f"  Homogeneity: {hom:.4f}  (each HDBSCAN cluster contains only one MCL cluster)")
        print(f"  Completeness: {comp:.4f} (each MCL cluster maps to only one HDBSCAN cluster)")
        print(f"  V-measure: {vmeas:.4f} (harmonic mean of H and C)")

        # GO homogeneity of HDBSCAN clusters vs MCL clusters
        if args.go_pkl:
            print(f"\n  GO homogeneity comparison")
            protein2go = load_go(Path(args.go_pkl))

            def go_homogeneity(cluster_labels, protein_ids):
                """for each cluster, compute pairwise Jaccard similarity"""
                id2label = dict(zip(protein_ids, cluster_labels))
                clusters = defaultdict(list)
                for pid, lbl in id2label.items():
                    if lbl >= 0:
                        clusters[lbl].append(pid)

                homog_scores = []
                for lbl, members in clusters.items():
                    ann = [p for p in members if protein2go.get(p)]
                    if len(ann) < 2:
                        continue
                    pair_jaccards = []
                    for i in range(len(ann)):
                        go_i = protein2go[ann[i]]
                        for j in range(i + 1, len(ann)):
                            go_j = protein2go[ann[j]]
                            inter = len(go_i & go_j)
                            union = len(go_i | go_j)
                            pair_jaccards.append(inter / union if union > 0 else 0.0)
                    homog_scores.append(np.mean(pair_jaccards))

                return np.mean(homog_scores) if homog_scores else 0.0, len(homog_scores)

            hdb_homog, hdb_n = go_homogeneity(labels, ids)
            mcl_homog, mcl_n = go_homogeneity([protein2mcl.get(pid, -1) for pid in ids], ids)

            print(f"  HDBSCAN GO homogeneity : {hdb_homog:.4f}  (over {hdb_n} clusters)")
            print(f"  MCL GO homogeneity : {mcl_homog:.4f}  (over {mcl_n} clusters)")
            diff = hdb_homog - mcl_homog
            print(f"  ∆ (HDBSCAN - MCL) : {diff:.4f}  {'HDBSCAN better' if diff > 0 else 'MCL better'}")

    print(f'\n{SEP}')
    print(f"  Clustering complete. Output : {out_dir}")
    print(SEP)

# CLI
def build_parser():
    p = argparse.ArgumentParser(
        description="UMAP visualization and HDBSCAN clustering of protein embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # viz
    v = sub.add_parser("viz", help="UMAP visualization colored by GO / species")
    v.add_argument("--h5_dir", required=True)
    v.add_argument("--go_pkl", required=True)
    v.add_argument("--out_dir", required=True)
    v.add_argument("--n_sample", type=int, default=50_000, help="Total proteins to sample across all species (default = 50000)")
    v.add_argument("--n_top_terms", type=int, default=10, help="Number of top GO terms in legend (default = 10)")
    v.add_argument("--n_annotated_oversample", type=int, default=10_000, help="Extra annotated proteins to add on top of background sample (default = 10000)")

    # cluster
    c = sub.add_parser("cluster", help="HDBSCAN clustering and comparison to MCL")
    c.add_argument("--h5_dir", required=True)
    c.add_argument("--go_pkl", default=None)
    c.add_argument("--mcl_file", default=None)
    c.add_argument("--out_dir", required=True, help="Same output dir as viz, reuses saved UMAP coords")
    c.add_argument("--n_sample", type=int, default=50_000)
    c.add_argument("--min_cluster_size", type=int, default=10, help="Minimum cluster size for HDBSCAN (default = 10)")
    c.add_argument("--min_samples", type=int, default=5, help="Minimum number of samples for HDBSCAN (default = 5)")
    c.add_argument("--umap_dim_cluster", type=int, default=10, help="UMAP dims for clustering input (default: 10, use 2 to reuse viz coords)")

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "viz":
        run_viz(args)
    elif args.mode == "cluster":
        run_cluster(args)

if __name__ == "__main__":
    main()