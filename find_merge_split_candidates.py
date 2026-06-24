"""
Comparison of HDBSCAN and MCL to find :
    MERGE candidates : pairs of MCL clusters that fall into the same HDBSCAN cluster and have high GO similarity
    SPLIT candidates : large MCL clusters whose members fall into distinct HDBSCAN clusters, where splitting according to latter improves GO homogeneity
"""
import argparse
import h5py
import time
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
import numpy as np
import pickle

SEP = "=" * 60

# IO helpers
def load_go(pkl_path: Path) -> dict[str, int]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return {k: set(v) if not isinstance(v, set) else v for k, v in data.items()}

def load_mcl_clusters(mcl_file: Path) -> dict[str, int]:
    protein2cluster = {}
    with open(mcl_file) as fh:
        for cid, line in enumerate(fh):
            for pid in line.strip().split("\t"):
                if pid:
                    protein2cluster[pid] = cid
    return protein2cluster

def _find(d: Path, *candidates: str) -> Path:
    for name in candidates:
        p = d / name
        if p.exists():
            return p
    return d / candidates[0]

# GO similarity helpers
def mean_jaccard_within(go_sets: list[set]) -> float:
    """mean pairwise jaccard similarity among a list of GO term sets"""
    n = len(go_sets)
    if n < 2:
        return float("nan")
    vals = []
    for i in range(n):
        gi = go_sets[i]
        for j in range(i + 1, n):
            gj = go_sets[j]
            union = len(gi | gj)
            vals.append(len(gi & gj) / union if union > 0 else 0.0)
    return float(np.mean(vals)) if vals else float("nan")

def mean_jaccard_cross(go_sets_a: list[set], go_sets_b: list[set]) -> float:
    """mean pairwise jaccard similarity between two lists of GO terms"""
    if not go_sets_a or not go_sets_b:
        return float("nan")
    vals = []
    for gi in go_sets_a:
        for gj in go_sets_b:
            union = len(gi | gj)
            vals.append(len(gi & gj) / union if union > 0 else 0.0)
    return float(np.mean(vals)) if vals else float("nan")

# MERGE candidates
def find_merge_candidates(ids: list[str], hdb_labels: np.ndarray, mcl_labels: list[int], protein2go: dict[str, set], umap10d: np.ndarray = None, max_annot_per_cluster: int = 100, min_mcl_size: int = 3, min_annot_per_side: int = 5, max_per_hdb_cluster: int = 1, top_n: int = 200):
    """for each HDBSCAN cluster, find the MCL clusters whose members fall inside it.
    For every pair of MCL clusters co-occuring in the same HDBSCAN cluster, compute cross-cluster jaccard"""
    print("\n  Building HDBSCAN -> MCL co-occurence map")
    # group protein indices by (hdb_label, mcl_label)
    hdb2mcl_members = defaultdict(lambda: defaultdict(list))
    mcl_sizes_global = Counter(mcl_labels)
    mcl_all_members = defaultdict(list)
    for idx, (hdb, mcl) in enumerate(zip(hdb_labels, mcl_labels)):
        if mcl < 0:
            continue
        mcl_all_members[mcl].append(idx)
        if hdb >= 0:
            hdb2mcl_members[hdb][mcl].append(idx)

    # precompute GO sets from full MCL membership per cluster
    print(f"  Precomputing GO sets from full MCL membership")
    mcl_go_cache = {}
    for mcl, all_idxs in mcl_all_members.items():
        ann = [ids[i] for i in all_idxs if protein2go.get(ids[i])]
        if len(ann) > max_annot_per_cluster:
            ann = list(np.random.default_rng(mcl % 1000).choice(ann, max_annot_per_cluster, replace=False))
        mcl_go_cache[mcl] = [protein2go[p] for p in ann]

    print(f"  Scanning {len(hdb2mcl_members)} HDBSCAN clusters for co-occuring MCL clusters")
    candidates = []
    rng = np.random.default_rng(0)

    for hdb, mcl_dict in hdb2mcl_members.items():
        mcl_ids_here = [m for m, members in mcl_dict.items() if mcl_sizes_global.get(m, 0) >= min_mcl_size]
        if len(mcl_ids_here) < 2:
            continue

        # cache centroids per MCL cluster (within HDBSCAN cluster)
        centroid_cache = {}
        for m in mcl_ids_here:
            idxs = mcl_dict[m]
            if umap10d is not None:
                centroid_cache[m] = umap10d[idxs].mean(axis=0)
        hdb_best = None
        for m1, m2 in combinations(mcl_ids_here, 2):
            go1 = mcl_go_cache.get(m1, [])
            go2 = mcl_go_cache.get(m2, [])
            # require a meaningful nb of annotations on both sides
            if len(go1) < min_annot_per_side or len(go2) < min_annot_per_side:
                continue
            jacc = mean_jaccard_cross(go1, go2)
            if np.isnan(jacc):
                continue
            dist = None
            if umap10d is not None and m1 in centroid_cache and m2 in centroid_cache:
                dist = float(np.linalg.norm(centroid_cache[m1] - centroid_cache[m2]))

            cand = {
                "hdbscan_cluster": int(hdb),
                "mcl_cluster_1": int(m1),
                "mcl_cluster_2": int(m2),
                "mcl1_size_total": mcl_sizes_global[m1],
                "mcl2_size_total": mcl_sizes_global[m2],
                "mcl1_size_here": len(mcl_dict[m1]), # hdbscan core only
                "mcl2_size_here": len(mcl_dict[m2]), # hdbscan core only
                "n_annot_1": len(go1), # full MCL
                "n_annot_2": len(go2), # full MCL
                "go_jaccard_cross": jacc,
                "umap10d_dist": dist,
            }

            if max_per_hdb_cluster != 0:
                # keep only the single best pair per HDBSCAN cluster
                if hdb_best is None or jacc > hdb_best["go_jaccard_cross"]:
                    hdb_best = cand
            else:
                candidates.append(cand)

        if max_per_hdb_cluster != 0 and hdb_best is not None:
            candidates.append(hdb_best)

    candidates.sort(key=lambda c: c["go_jaccard_cross"], reverse=True)
    return candidates if top_n == 0 else candidates[:top_n]

# SPLIT candidates
def find_split_candidates(ids: list[str], hdb_labels: np.ndarray, mcl_labels: list[int], protein2go: dict[str, set], min_mcl_size: int = 50, min_sub_size: int = 5, max_annot_per_cluster: int = 100, top_n: int = 100):
    """for each large MCL cluster, check how its members distribute across HDBSCAN clusters.
    Check if splitting along HDBSCAN boundaries improves the homogeneity, flag if True"""
    print("\n  Grouping proteins by MCL cluster")
    mcl2idx = defaultdict(list)
    for idx, m in enumerate(mcl_labels):
        if m < 0:
            continue
        mcl2idx[m].append(idx)

    large_clusters = {m: idxs for m, idxs in mcl2idx.items() if len(idxs) >= min_mcl_size}
    print(f"  MCL clusters with >= {min_mcl_size} proteins: {len(large_clusters):,}")
    candidates = []
    rng = np.random.default_rng(1)

    for m, idxs in large_clusters.items():
        # whole-cluster GO homogeneity
        ann_idx = [i for i in idxs if protein2go.get(ids[i])]
        if len(ann_idx) < 4:
            continue
        if len(ann_idx) > max_annot_per_cluster:
            ann_idx_sample = list(rng.choice(ann_idx, max_annot_per_cluster, replace=False))
        else:
            ann_idx_sample = ann_idx
        whole_go = [protein2go[ids[i]] for i in ann_idx_sample]
        whole_homog = mean_jaccard_within(whole_go)
        if np.isnan(whole_homog):
            continue

        # group by HDBSCAN sub-level (exclude noise)
        sub_groups = defaultdict(list)
        for i in idxs:
            h = hdb_labels[i]
            sub_groups[h if h >= 0 else -1].append(i)

        # need at least 2 substantial sub-groups
        non_noise = {h: g for h, g in sub_groups.items() if h>=0 and len(g) >= min_sub_size}
        if len(non_noise) < 2:
            continue

        all_sub_groups = dict(non_noise)
        if -1 in sub_groups and len(sub_groups[-1]) > 0:
            all_sub_groups[-1] = sub_groups[-1] # noise residual

        # weighted sub-cluster homogeneity
        sub_homogs = []
        sub_weights = []
        sub_details = []
        for h, g in all_sub_groups.items():
            ann_g = [i for i in g if protein2go.get(ids[i])]
            if len(ann_g) < 2:
                sub_homogs.append(0.0)
                sub_weights.append(len(g))
                sub_details.append((int(h), len(g), len(ann_g), 0.0))
                continue
            if len(ann_g) > max_annot_per_cluster:
                ann_g = list(rng.choice(ann_g, max_annot_per_cluster, replace=False))
            go_g = [protein2go[ids[i]] for i in ann_g]
            hom = mean_jaccard_within(go_g)
            if np.isnan(hom):
                continue
            sub_homogs.append(hom)
            sub_weights.append(len(g))
            sub_details.append((int(h), len(g), len(ann_g), hom))
        if len(sub_homogs) < 2:
            continue

        weighted_sub = float(np.average(sub_homogs, weights=sub_weights))
        improvement = weighted_sub - whole_homog

        candidates.append({
            "mcl_cluster": int(m),
            "mcl_size": len(idxs),
            "n_annot_whole": len(ann_idx),
            "whole_homogeneity": whole_homog,
            "n_hdb_subgroups": len(sub_groups),
            "weighted_sub_homog": weighted_sub,
            "improvement": improvement,
            "sub_details": sub_details,
        })

    candidates.sort(key=lambda c: c["improvement"], reverse=True)
    return candidates if top_n == 0 else candidates[:top_n]

def main():
    p = argparse.ArgumentParser(
        description="Find merge and split candidates for HDBSCAN vs MCL clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--umap_dir", required=True, help="Directory with all_ids.txt / umap_ids.txt, hdbscan_labels.npy, umap_10d.npy")
    p.add_argument("--mcl_file", required=True, help="MCL output")
    p.add_argument("--go_pkl", required=True, help="GO annotation pickle file")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--min_mcl_size_merge", type=int, default=3, help="Minimum MCL cluster size (default: 3)")
    p.add_argument("--min_annot_per_side", type=int, default=5, help="Min GO-annotated proteins required in EACH MCL cluster for a merge candidate (default: 5)")
    p.add_argument("--max_per_hdb_cluster", type=int, default=1, help="Max merge candidates kept per HDBSCAN cluster (default: 1), set to 0 for unlimited (keeps all pairs)")
    p.add_argument("--min_mcl_size_split", type=int, default=50, help="Min size of MCL cluster to consider for splitting (default: 50)")
    p.add_argument("--min_sub_size", type=int, default=5, help="Min size of HDBSCAN sub-group for split candidates (default: 5)")
    p.add_argument("--top_n", type=int, default=5000, help="Number of top candidates to output per category (default: 200)")
    args = p.parse_args()

    umap_dir = Path(args.umap_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(SEP)
    print("  Merge / Split Candidate Finder")
    print(SEP)

    # load data
    ids_path = _find(umap_dir, "umap_ids.txt", "all_ids.txt")
    hdb_path = _find(umap_dir, "hdbscan_labels.npy")
    umap_path = _find(umap_dir, "umap_10d.npy", "umap_10d.npy")

    print(f"\n  Loading ids from {ids_path.name}")
    ids = ids_path.read_text().splitlines()
    print(f"  {len(ids):,} proteins")

    print(f"  Loading HDBSCAN labels from {hdb_path.name}")
    hdb_labels = np.load(hdb_path)

    umap10d = None
    if umap_path.exists():
        print(f"  Loading UMAP-10D from {umap_path.name}")
        umap10d = np.load(umap_path)
    else:
        print(f"  [WARNING] No UMAP-10D file found, centroid distances will be omitted")

    print(f"\n  Loading MCL clusters from {args.mcl_file}")
    protein2mcl = load_mcl_clusters(Path(args.mcl_file))
    mcl_labels = [protein2mcl.get(pid, -1) for pid in ids]
    n_in_mcl = sum(1 for m in mcl_labels if m >= 0)
    print(f"  {n_in_mcl:,} / {len(ids):,} proteins found in MCL clusters")

    print(f"\n  Loading GO annotations from {args.go_pkl}")
    protein2go = load_go(Path(args.go_pkl))
    n_annot = sum(1 for pid in ids if protein2go.get(pid))
    print(f"  {n_annot:,} / {len(ids):,} proteins GO-annotated")

    # MERGE candidates
    print(f"\n{SEP}")
    print("  Finding MERGE candidates")
    print(SEP)
    t0 = time.time()
    merge_cands = find_merge_candidates(ids, hdb_labels, mcl_labels, protein2go, umap10d=umap10d, min_mcl_size=args.min_mcl_size_merge, min_annot_per_side=args.min_annot_per_side, max_per_hdb_cluster=args.max_per_hdb_cluster, top_n=args.top_n,)
    print(f"  Found {len(merge_cands):,} candidates in {time.time() - t0:.1f}s")

    merge_path = out_dir / "merge_candidates.tsv"
    with open(merge_path, "w") as fh:
        fh.write("hdbscan_cluster\tmcl_cluster_1\tmcl_cluster_2\t"
                 "mcl1_size_total\tmcl2_size_total\t"
                 "mcl1_size_in_hdb\tmcl2_size_in_hdb\t"
                 "n_annot_1\tn_annot_2\tgo_jaccard_cross\tumap10d_dist\n")
        for c in merge_cands:
            dist_str = f"{c['umap10d_dist']:.4f}" if c['umap10d_dist'] is not None else "NA"
            fh.write(f"{c['hdbscan_cluster']}\t{c['mcl_cluster_1']}\t{c['mcl_cluster_2']}\t"
                     f"{c['mcl1_size_total']}\t{c['mcl2_size_total']}\t"
                     f"{c['mcl1_size_here']}\t{c['mcl2_size_here']}\t"
                     f"{c['n_annot_1']}\t{c['n_annot_2']}\t"
                     f"{c['go_jaccard_cross']:.4f}\t{dist_str}\n")
    print(f"  -> {merge_path}")

    if merge_cands:
        print(f"\n  Top 5 merge candidates:")
        for c in merge_cands[:5]:
            print(f"    HDB={c['hdbscan_cluster']:>6}  "
                  f"MCL {c['mcl_cluster_1']:>6} (n={c['mcl1_size_total']:>4}, "
                  f"ann={c['n_annot_1']:>3}) <-> "
                  f"MCL {c['mcl_cluster_2']:>6} (n={c['mcl2_size_total']:>4}, "
                  f"ann={c['n_annot_2']:>3})  "
                  f"GO_jaccard={c['go_jaccard_cross']:.3f}")

    # SPLIT candidates
    print(f"\n{SEP}")
    print("  Finding SPLIT candidates")
    print(SEP)
    t0 = time.time()
    split_cands = find_split_candidates(ids, hdb_labels, mcl_labels, protein2go, min_mcl_size=args.min_mcl_size_split, min_sub_size=args.min_sub_size, top_n=args.top_n,)
    print(f"  Found {len(split_cands):,} candidates in {time.time() - t0:.1f}s")

    split_path = out_dir / "split_candidates.tsv"
    with open(split_path, "w") as fh:
        fh.write("mcl_cluster\tmcl_size\tn_annot_whole\twhole_homogeneity\t"
                 "n_hdb_subgroups\tweighted_sub_homogeneity\timprovement\t"
                 "sub_details(hdb_label:size:n_annot:homog)\n")
        for c in split_cands:
            sub_str = ";".join(f"{h}:{sz}:{na}:{ho:.3f}"
                               for h, sz, na, ho in c["sub_details"])
            fh.write(f"{c['mcl_cluster']}\t{c['mcl_size']}\t{c['n_annot_whole']}\t"
                     f"{c['whole_homogeneity']:.4f}\t{c['n_hdb_subgroups']}\t"
                     f"{c['weighted_sub_homog']:.4f}\t{c['improvement']:+.4f}\t"
                     f"{sub_str}\n")
    print(f"  -> {split_path}")

    if split_cands:
        print(f"\n  Top 5 split candidates:")
        for c in split_cands[:5]:
            print(f"    MCL {c['mcl_cluster']:>6} (n={c['mcl_size']:>5}, "
                  f"ann={c['n_annot_whole']:>3})  "
                  f"whole_homog={c['whole_homogeneity']:.3f}  ->  "
                  f"{c['n_hdb_subgroups']} sub-groups, "
                  f"weighted_homog={c['weighted_sub_homog']:.3f}  "
                  f"improvement={c['improvement']:+.3f}")

    print(f"\n{SEP}")
    print(f"  Done. Output in {out_dir}")
    print(SEP)


if __name__ == "__main__":
    main()

