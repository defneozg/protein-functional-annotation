"""
Structural characterization of MCL clusters that are merge/split candidates using their 10D UMAP
embedding coordinates.

For each candidate cluster, we compute:
1 - Compactness (globular vs dispersed):
    - mean_dist_centroid : mean distance of proteins from cluster centroid
    - radius_90 : 90th percentile distance of cluster centroid
    - aspect_ratio : ratio of first to last PCA eigenvalue (1=spherical, high=elongated)
    - mean_pairwise_distance : mean pairwise L2 distance

2 - Annotation Spatial Distribution:
    - annot_nn_homog : for each annotated protein, fraction of it's k-nearest neighbors sharing GO term
                        high = annotations form dense sub-groups
    - same_go_dist : mean distance between proteins sharing GO term
    - diff_go_dist : mean distance between proteins with different GO terms
    - go_spatial_ratio : same_go_dist / diff_go_dist , <1 means same-GO proteins are closer (spatially coherent annotations)

3 - Species Mixing:
    - n_species : no of species represented
    - species_entropy : Shannon entropy of species distrubution

4 - Dark Protein Distribution:
    - dark_nn_annot_frac : for each unannotated protein, fraction of its k-nearest neighbors that are annotated (how well-positioned are dark proteins for transfer)

5 - Sequence Identity:
    - mean_pident : average seq id among aligned pairs within cluster

6 - Annotation Confidence:
    - dark_conf_mean_nle :  for each unannotated protein in the cluster, the best -log10(evalue) to any annotated protein in the same cluster
    - annot_conf_mean_nle : mean -log10(evalue) between pairs of annotated proteins sharing a GO term within the cluster
"""
import argparse
import pickle
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

SEP = "="*60

def _find(d: Path, *candidates: str) -> Path:
    for name in candidates:
        p = d / name
        if p.exists():
            return p
    return d / candidates[0]

def load_go(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return {k: set(v) if not isinstance(v, set) else v for k, v in data.items()}

def load_mcl_clusters(mcl_file: Path) -> dict:
    clusters = {}
    with open(mcl_file) as fh:
        for cid, line in enumerate(fh):
            members = [p for p in line.strip().split("\t") if p]
            if members:
                clusters[cid] = members
    return clusters

def species_from_id(pid: str) -> str:
    return pid.split("|")[0]

# core characterization function
def characterize_cluster(members: list[str], id2row: dict, umap10d: np.ndarray, protein2go: dict, k: int = 10, max_sample: int = 200, rng: np.random.Generator = None, label: str = "") -> dict:
    if rng is None:
        rng = np.random.default_rng(0)
    # get embedding coords for members in umap
    rows = np.array([id2row[pid] for pid in members if pid in id2row])
    valid_pids = [pid for pid in members if pid in id2row]
    n_total = len(members)
    n_in_umap = len(rows)
    if n_in_umap < 3:
        return {"label": label, "n_total": n_total, "n_in_umap": n_in_umap, "error": "too few proteins in UMAP"}
    coords = umap10d[rows] # (n_in_umap, 10)

    # sample if too large
    if n_in_umap > max_sample:
        sample_idx = rng.choice(n_in_umap, size=max_sample, replace=False)
        coords_s = coords[sample_idx]
        pids_s = [valid_pids[i] for i in sample_idx]
    else:
        coords_s = coords
        pids_s = valid_pids

    n = len(coords_s)

    # 1- Compactness
    centroid = coords_s.mean(axis=0)
    distance_from_centroid = np.linalg.norm(coords_s - centroid, axis=1)
    mean_dist_centroid = float(distance_from_centroid.mean())
    radius_90 = float(np.percentile(distance_from_centroid, 90))
    # pairwise distances (upper triangle)
    diff = coords_s[:, None, :] - coords_s[None, :, :] # (n, n, 10)
    pdist = np.sqrt((diff**2).sum(axis=2))
    triu = pdist[np.triu_indices(n, k=1)]
    mean_pairwise_dist = float(triu.mean())
    # aspect ratio via PCA (ratio from largest to smallest eigenvalue)
    centered = coords_s - centroid
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        aspect_ratio = float(s[0] / (s[-1] + 1e-9))
    except Exception:
        aspect_ratio = float("nan")

    # 2- Annotation spatial distribution
    ann_mask = np.array([bool(protein2go.get(p)) for p in pids_s])
    dark_mask = ~ann_mask
    n_ann = int(ann_mask.sum())
    n_dark = int(dark_mask.sum())
    # k-nn within cluster (using precomputed pairwise distances)
    k_eff = min(k, n - 1)
    # annotation neighborhood homogeneity for each annotated protein (fraction of its k-nns sharing a GO term)
    annot_nn_homog_vals = []
    if n_ann >= 2:
        ann_idx = np.where(ann_mask)[0]
        for ai in ann_idx:
            go_i = protein2go[pids_s[ai]]
            # distance to all other proteins
            row_dists = pdist[ai].copy()
            row_dists[ai] = np.inf # exclude self
            nn_idx = np.argsort(row_dists)[:k_eff]
            shared = sum(1 for ni in nn_idx if protein2go.get(pids_s[ni]) and go_i & protein2go[pids_s[ni]])
            annot_nn_homog_vals.append(shared / k_eff)
    annot_nn_homog = float(np.mean(annot_nn_homog_vals)) if annot_nn_homog_vals else float("nan")

    # same GO vs different GO distance
    same_go_dists, different_go_dists = [], []
    if n_ann >= 4:
        ann_idx = np.where(ann_mask)[0]
        for i_pos, ai in enumerate(ann_idx):
            go_i = protein2go[pids_s[ai]]
            for aj in ann_idx[i_pos+1:]:
                go_j = protein2go[pids_s[aj]]
                d = float(pdist[ai, aj])
                if go_i & go_j:
                    same_go_dists.append(d)
                else:
                    different_go_dists.append(d)

    same_go_dist = float(np.mean(same_go_dists)) if same_go_dists else float("nan")
    diff_go_dist = float(np.mean(different_go_dists)) if different_go_dists else float("nan")
    go_spatial_ratio = (same_go_dist / diff_go_dist) if same_go_dist and diff_go_dist else float("nan")

    # 3- Species mixing
    sp_counts = Counter(species_from_id(p) for p in members)
    n_species = len(sp_counts)
    total_prot = sum(sp_counts.values())
    probs = np.array([c / total_prot for c in sp_counts.values()])
    sp_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    # 4- Dark protein distribution
    dark_nn_annot_fracs = []
    if n_dark >= 1 and n_ann >= 1:
        dark_idx = np.where(dark_mask)[0]
        for di in dark_idx:
            row_dists = pdist[di].copy()
            row_dists[di] = np.inf
            nn_idx = np.argsort(row_dists)[:k_eff]
            frac = sum(1 for ni in nn_idx if ann_mask[ni]) / k_eff
            dark_nn_annot_fracs.append(frac)
    dark_nn_annot_frac = float(np.mean(dark_nn_annot_fracs)) if dark_nn_annot_fracs else float("nan")

    return {
        "label": label,
        "n_total": n_total,
        "n_in_umap": n_in_umap,
        "n_annotated": n_ann,
        "n_dark": n_dark,
        # compactness
        "mean_dist_centroid": mean_dist_centroid,
        "radius_90": radius_90,
        "mean_pairwise_dist": mean_pairwise_dist,
        "aspect_ratio": aspect_ratio,
        # annotation spatial
        "annot_nn_homog": annot_nn_homog,
        "same_go_dist": same_go_dist,
        "diff_go_dist": diff_go_dist,
        "go_spatial_ratio": go_spatial_ratio,
        # species
        "n_species": n_species,
        "species_entropy": sp_entropy,
        # dark
        "dark_nn_annot_frac": dark_nn_annot_frac,
        # placeholders, filled in after return by caller using alignment data
        "mean_pident": float("nan"),
        "std_pident": float("nan"),
        "annot_conf_mean_nle": float("nan"),
        "dark_conf_mean_nle": float("nan"),
    }

def compute_alignment_metrics(members: list[str], protein2go: dict, pid_lookup: dict, nle_lookup: dict, max_pairs: int = 5000, rng: np.random.Generator = None) -> dict:
    """seq id within cluster and annotation confidence
        seq id: mean pairwise PID among all member pairs
        annotation confidence: for each unannotated protein, find its best alignment to an annotated protein within cluster and record the -log(eval)
            high values -> dark proteins are close in sequence to annotated ones
    """
    if rng is None:
        rng = np.random.default_rng(0)
    member_set = set(members)
    # seq id: sample pairs from the alignment lookup
    pid_vals = []
    pairs_checked = 0
    for (a, b), pid in pid_lookup.items():
        if a in member_set and b in member_set:
            pid_vals.append(pid)
            pairs_checked += 1
            if pairs_checked >= max_pairs:
                break

    mean_pident = float(np.mean(pid_vals)) if pid_vals else float("nan")
    std_pident = float(np.std(pid_vals)) if pid_vals else float("nan")

    # annotation confidence
    ann_set = {p for p in members if protein2go.get(p)}
    dark_set = {p for p in members if not protein2go.get(p)}
    # best NLE per dark protein to any annotated member
    dark_best_nle = []
    for dark_pid in dark_set:
        best = float("nan")
        for ann_pid in ann_set:
            key = (dark_pid, ann_pid) if dark_pid < ann_pid else (ann_pid, dark_pid)
            if key in nle_lookup:
                v = nle_lookup[key]
                if np.isnan(best) or v > best:
                    best = v
        if not np.isnan(best):
            dark_best_nle.append(best)

    dark_conf_mean_nle = float(np.mean(dark_best_nle)) if dark_best_nle else float("nan")

    # annotation confidence for annotated proteins: mean best NLE between annotated proteins sharing a GO term
    annot_conf_vals = []
    ann_list = list(ann_set)
    for i, a in enumerate(ann_list):
        go_a = protein2go[a]
        for b in ann_list[i+1:]:
            if protein2go[b] & go_a: # share GO term
                key = (a, b) if a < b else (b, a)
                if key in nle_lookup:
                    annot_conf_vals.append(nle_lookup[key])

    annot_conf_mean_nle = float(np.mean(annot_conf_vals)) if annot_conf_vals else float("nan")
    return {
        "mean_pident": mean_pident,
        "std_pident": std_pident,
        "n_pid_pairs": len(pid_vals),
        "annot_conf_mean_nle": annot_conf_mean_nle,
        "dark_conf_mean_nle": dark_conf_mean_nle,
        "n_dark_with_aln": len(dark_best_nle),
    }

def print_metrics(m: dict, indent: str = "  "):
    if "error" in m:
        print(f"{indent}[SKIP] {m['error']}")
        return
    print(f"{indent}Size              : {m['n_total']:,} total |  {m['n_in_umap']:,} in UMAP | {m['n_annotated']} annotated | {m['n_dark']} dark")
    print(f"{indent}Compactness:")
    print(f"{indent}  mean dist centroid : {m['mean_dist_centroid']:.4f}")
    print(f"{indent}  radius (p90)       : {m['radius_90']:.4f}")
    print(f"{indent}  mean pairwise dist : {m['mean_pairwise_dist']:.4f}")
    print(f"{indent}  aspect ratio       : {m['aspect_ratio']:.2f}  ({'elongated' if m['aspect_ratio'] > 5 else 'globular'})")
    print(f"{indent}Annotation spatial distribution:")
    print(f"{indent}  kNN GO homogeneity : {m['annot_nn_homog']:.4f}  (fraction of neighbours sharing GO term)")
    print(f"{indent}  same-GO dist       : {m['same_go_dist']:.4f}")
    print(f"{indent}  diff-GO dist       : {m['diff_go_dist']:.4f}")
    ratio = m['go_spatial_ratio']
    if not np.isnan(ratio):
        interp = ("same-GO proteins closer → annotations spatially coherent" if ratio < 0.95 else "annotations randomly distributed")
        print(f"{indent}  spatial ratio      : {ratio:.4f}  ({interp})")
    print(f"{indent}Species mixing:")
    print(f"{indent}  n_species          : {m['n_species']}")
    print(f"{indent}  species entropy    : {m['species_entropy']:.3f}")
    print(f"{indent}Dark protein distribution:")
    print(f"{indent}  dark kNN annot frac: {m['dark_nn_annot_frac']:.4f}  (fraction of dark protein neighbours that are annotated)")
    if not np.isnan(m.get("mean_pident", float("nan"))):
        print(f"{indent}Sequence identity (within cluster, from alignment):")
        print(f"{indent}  mean pident        : {m['mean_pident']:.2f}%  std={m['std_pident']:.2f}%  (n_pairs={m.get('n_pid_pairs', '?')})")
        print(f"{indent}Annotation confidence (-log10 evalue):")
        print(f"{indent}  dark -> annot      : {m['dark_conf_mean_nle']:.2f}  (n_dark_with_aln={m.get('n_dark_with_aln', '?')})")
        print(f"{indent}  annot -> same-GO   : {m['annot_conf_mean_nle']:.2f}")

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--merge_candidates", required=True)
    p.add_argument("--split_candidates", required=True)
    p.add_argument("--mcl_file", required=True)
    p.add_argument("--umap_dir", required=True)
    p.add_argument("--go_pkl", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--top_n", type=int, default=20, help="Number of top candidates to characterize (default: 20)")
    p.add_argument("--k", type=int, default=10, help="k for kNN within cluster (default: 10)")
    p.add_argument("--max_sample", type=int, default=200, help="max proteins sampled per cluster for pairwise metrics (default: 200)")
    p.add_argument("--align", default=None, help="diamon_alignments.filter.tsv (optional)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    umap_dir = Path(args.umap_dir)
    rng = np.random.default_rng(42)

    print(SEP)
    print("  Cluster Structural Characterization")
    print(SEP)
    print("\n  Loading UMAP 10D coordinates")
    umap_path = _find(umap_dir, "umap_10d.npy", "umap_10d.npy")
    ids_path = _find(umap_dir, "umap_ids.txt", "all_ids.txt")
    hdb_path = _find(umap_dir, "hdbscan_labels.npy")
    umap10d = np.load(umap_path)
    all_ids = ids_path.read_text().splitlines()
    hdb_labels = np.load(hdb_path)
    id2row = {pid: i for i, pid in enumerate(all_ids)}
    id2hdb = dict(zip(all_ids, hdb_labels))
    print(f"  {len(all_ids):,} proteins | dim={umap10d.shape[1]}")
    print("  Loading MCL clusters")
    clusters = load_mcl_clusters(Path(args.mcl_file))
    print("  Loading GO annotations")
    protein2go = load_go(Path(args.go_pkl))

    # pid_lookup[(a,b)] = pident  and  nle_lookup[(a,b)] = -log10(evalue)
    pid_lookup = {}
    nle_lookup = {}
    if args.align:
        print("  Loading alignment file for sequence identity / annotation confidence")
        EVALUE_CLAMP = 1e-300
        with open(args.align) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 13:
                    continue
                try:
                    q, s = parts[0], parts[4]
                    pid = float(parts[9])
                    evalue = float(parts[12])
                    nle = min(-np.log10(max(evalue, EVALUE_CLAMP)), 300.0)
                    key = (q, s) if q < s else (s, q)
                    pid_lookup[key] = pid
                    nle_lookup[key] = nle
                except (ValueError, IndexError):
                    continue
        print(f"  Loaded {len(pid_lookup):,} alignment edges")
    else:
        print("  No --align file provided")

    # load candidates
    merge_rows, split_rows = [], []
    with open(args.merge_candidates) as fh:
        header = fh.readline().strip().split("\t")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            merge_rows.append(dict(zip(header, parts)))

    print(f"  Merge candidates: {len(merge_rows):,}")
    print(f"  Split candidates: {len(split_rows):,}")

    # merge candidate charac
    print(f"\n{SEP}")
    print(f"  MERGE candidates (top {args.top_n}):")
    print(SEP)
    merge_results = []
    seen_pairs = set()

    for row in merge_rows[:args.top_n * 2]:
        c1 = int(row["mcl_cluster_1"])
        c2 = int(row["mcl_cluster_2"])
        jacc = float(row["go_jaccard_cross"])
        pair = (min(c1, c2), max(c1, c2))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        m1 = clusters.get(c1, [])
        m2 = clusters.get(c2, [])
        if not m1 or not m2:
            continue

        print(f"\n  HDB={row['hdbscan_cluster']}  MCL {c1} (n={len(m1)}) <-> MCL {c2} (n={len(m2)})  GO_jaccard={jacc:.3f}")
        # characterize each cluster individually
        r1 = characterize_cluster(m1, id2row, umap10d, protein2go, k=args.k, max_sample=args.max_sample, rng=rng, label=f"MCL_{c1}")
        r2 = characterize_cluster(m2, id2row, umap10d, protein2go, k=args.k, max_sample=args.max_sample, rng=rng, label=f"MCL_{c2}")
        if pid_lookup:
            aln1 = compute_alignment_metrics(m1, protein2go, pid_lookup, nle_lookup, rng=rng)
            aln2 = compute_alignment_metrics(m2, protein2go, pid_lookup, nle_lookup, rng=rng)
            r1.update(aln1)
            r2.update(aln2)
        print(f"  Cluster {c1}:")
        print_metrics(r1, indent="   ")
        print(f"  Cluster {c2}:")
        print_metrics(r2, indent="   ")

        # cross cluster distance: are the two clusters close in UMAP space?
        rows1 = np.array([id2row[p] for p in m1 if p in id2row])
        rows2 = np.array([id2row[p] for p in m2 if p in id2row])
        if len(rows1) > 0 and len(rows2) > 0:
            c1_coords = umap10d[rows1].mean(axis=0)
            c2_coords = umap10d[rows2].mean(axis=0)
            centroid_dist = float(np.linalg.norm(c1_coords - c2_coords))
            print(f"  Centroid distance between clusters: {centroid_dist:.4f}")
            # compare to within-cluster radius
            r1_rad = r1.get("radius_90", float("nan"))
            r2_rad = r2.get("radius_90", float("nan"))
            if not np.isnan(r1_rad) and not np.isnan(r2_rad):
                overlap = centroid_dist < max(r1_rad, r2_rad)
                print(f"  Clusters overlap in UMAP space : {overlap} (centroid dist < max radius)")
        merge_results.append({"c1": c1, "c2": c2, "jacc": jacc, "r1": r1, "r2": r2})
        if len(merge_results) >= args.top_n:
            break

    # SPLIT candidate characterization
    print(f"\n{SEP}")
    print(f"  SPLIT candidates (top {args.top_n}):")
    print(SEP)
    split_results = []
    for row in split_rows[:args.top_n]:
        cid = int(row["mcl_cluster"])
        imp = float(row["improvement"])
        whole = float(row["whole_homogeneity"])
        wsubh = float(row["weighted_sub_homog"])
        members = clusters.get(cid, [])
        if not members:
            continue
        print(f"\n  MCL {cid} (n={len(members)}) whole_homog={whole:.3f} -> weighted_sub={wsubh:.3f} improvement={imp:+.3f}")
        # characterize the whole cluster
        r_whole = characterize_cluster(members, id2row, umap10d, protein2go, k=args.k, max_sample=args.max_sample, rng=rng, label=f"MCL_{cid}_whole")
        if pid_lookup:
            aln_whole = compute_alignment_metrics(members, protein2go, pid_lookup, nle_lookup, rng=rng)
            r_whole.update(aln_whole)
        print("  Whole cluster:")
        print_metrics(r_whole, indent="   ")

        # characterize each HDBSCAN sub-group
        sub_groups = defaultdict(list)
        for pid in members:
            h = id2hdb.get(pid, -2)
            sub_groups[int(h)].append(pid)

        print(f"  Sub-groups ({len(sub_groups)} total):")
        sub_metrics = []
        for h, sub_members in sorted(sub_groups.items(), key=lambda x: -len(x[1])):
            if len(sub_members) < 3:
                continue
            label = f"noise_residual" if h < 0 else f"HDB_{h}"
            r_sub = characterize_cluster(sub_members, id2row, umap10d, protein2go, k=args.k, max_sample=args.max_sample, rng=rng, label=label)
            if pid_lookup:
                aln_sub = compute_alignment_metrics(sub_members, protein2go, pid_lookup, nle_lookup, rng=rng)
                r_sub.update(aln_sub)
            print(f"  Sub-group {label} (n={len(sub_members)}):")
            print_metrics(r_sub, indent="   ")
            sub_metrics.append(r_sub)

        # summary: do subgroups have lower spatial ratio than whole cluster?
        whole_ratio = r_whole.get("go_spatial_ratio", float("nan"))
        sub_ratios = [s.get("go_spatial_ratio", float("nan")) for s in sub_metrics if not np.isnan(s.get("go_spatial_ratio", float("nan")))]
        if sub_ratios:
            mean_sub_ratio = float(np.mean(sub_ratios))
            print(f"  GO spatial ratio  : whole={whole_ratio:.4f} mean sub-group={mean_sub_ratio:.4f}  ({'split improves spatial coherence' if mean_sub_ratio < whole_ratio else 'no improvement'})")
            split_results.append({"cid": cid, "r_whole": r_whole, "sub_metrics": sub_metrics})

    merge_tsv = out_dir / "merge_characterization.tsv"
    with open(merge_tsv, "w") as fh:
        cols = ["c1", "c2", "go_jaccard",
                "c1_n", "c1_mean_dist", "c1_aspect", "c1_annot_nn", "c1_spatial_ratio", "c1_species",
                "c2_n", "c2_mean_dist", "c2_aspect", "c2_annot_nn", "c2_spatial_ratio", "c2_species"]
        fh.write("\t".join(cols) + "\n")
        for r in merge_results:
            r1, r2 = r["r1"], r["r2"]
            fh.write("\t".join(str(x) for x in [
                r["c1"], r["c2"], f"{r['jacc']:.4f}",
                r1.get("n_total", ""), f"{r1.get('mean_dist_centroid', float('nan')):.4f}",
                f"{r1.get('aspect_ratio', float('nan')):.2f}",
                f"{r1.get('annot_nn_homog', float('nan')):.4f}",
                f"{r1.get('go_spatial_ratio', float('nan')):.4f}",
                r1.get("n_species", ""),
                r2.get("n_total", ""), f"{r2.get('mean_dist_centroid', float('nan')):.4f}",
                f"{r2.get('aspect_ratio', float('nan')):.2f}",
                f"{r2.get('annot_nn_homog', float('nan')):.4f}",
                f"{r2.get('go_spatial_ratio', float('nan')):.4f}",
                r2.get("n_species", ""),
            ]) + "\n")
    print(f"\n  -> {merge_tsv}")

    split_tsv = out_dir / "split_characterization.tsv"
    with open(split_tsv, "w") as fh:
        cols = ["mcl_cluster", "n_total", "n_subgroups",
                "whole_mean_dist", "whole_aspect", "whole_annot_nn",
                "whole_spatial_ratio", "whole_species",
                "mean_sub_mean_dist", "mean_sub_annot_nn", "mean_sub_spatial_ratio"]
        fh.write("\t".join(cols) + "\n")
        for r in split_results:
            rw = r["r_whole"]
            subs = [s for s in r["sub_metrics"] if not np.isnan(s.get("mean_dist_centroid", float("nan")))]
            mean_sub_dist = float(np.mean([s["mean_dist_centroid"] for s in subs])) if subs else float("nan")
            mean_sub_ann = float(np.mean([s["annot_nn_homog"] for s in subs if not np.isnan(s.get("annot_nn_homog", float("nan")))])) if subs else float("nan")
            mean_sub_ratio = float(np.mean([s["go_spatial_ratio"] for s in subs if not np.isnan(s.get("go_spatial_ratio", float("nan")))])) if subs else float("nan")
            fh.write("\t".join(str(x) for x in [
                r["cid"], rw.get("n_total", ""), len(r["sub_metrics"]),
                f"{rw.get('mean_dist_centroid', float('nan')):.4f}",
                f"{rw.get('aspect_ratio', float('nan')):.2f}",
                f"{rw.get('annot_nn_homog', float('nan')):.4f}",
                f"{rw.get('go_spatial_ratio', float('nan')):.4f}",
                rw.get("n_species", ""),
                f"{mean_sub_dist:.4f}", f"{mean_sub_ann:.4f}", f"{mean_sub_ratio:.4f}",
            ]) + "\n")
    print(f"  -> {split_tsv}")
    print(f"\n{SEP}")
    print("  Done.")
    print(SEP)

if __name__ == "__main__":
    main()
