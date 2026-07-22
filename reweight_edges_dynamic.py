"""
Reweight alignment edges using a per-edge alpha that accounts for phylogenetic distance between species

w = (1 - α(d,e)) × w_seq + α(d,e) × cosine_sim

where α(d,e) depends on:
1 - Phylogenetic distance d(A, B) between two species
    - estimated from mean cross-species cosine drop
    - distant species -> α -> 0 (trust sequence similarity more)
    - close species -> α -> α_max (trust embeddings more)

2 - Alignment confidence e-value
    - weak alignments (high evalue) -> reduce α further
    - reflects the Jaccard>0.5 finding: embedding signal is most reliable where alignment is already strong

Alpha formula:
    distance_factor(d) = exp(-d / tau)
    confidence_factor(e) = sigmoid((-log(evalue) - e_threshold) / e_scale)
    α(d,e) = α_max * distance_factor * confidence_factor * coh_factor

Parameters
----------
alpha_max   : maximum cosine contribution (reached for same-species edges)
              start with 0.3-0.5 based on fixed-alpha results
tau         : distance decay constant (in cosine-drop units)
              tau=0.05 means drop=0.05 (like folium) → factor=0.37
              tau=0.05 means drop=0.10 → factor=0.14
              smaller tau = faster decay = trust cosine less at distance
e_threshold : -log10(evalue) below which confidence_factor < 0.5
              default 10.0 = evalue < 1e-10
e_scale     : sigmoid sharpness around e_threshold
              larger = sharper transition
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm

COL_Q = 0
COL_S = 4
COL_EVALUE = 12
COL_COSINE = 14
EVALUE_CLAMP = 1e-300
NEG_LOG_MAX = 200.0
SEP = "="*60

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def neg_log10_clamped(evalue: float) -> float:
    return min(-np.log10(max(evalue, EVALUE_CLAMP)), NEG_LOG_MAX)

def normalize_neg_log(nle: float) -> float:
    return nle / NEG_LOG_MAX

def species_from_id(protein_id: str) -> str:
    return protein_id.split("|")[0]

# step 1: precompute species pairwise distance matrix
def precompute_distances(h5_dir: Path, out_dir: Path, n_sample: int = 500):
    """compute in a single pass:
    1. per-species cohesion score
        cohesion(A) = mean_within(A) - mean_cross(A, all_others)
        high cohesion indicates the species forms a tight blob (sequence-biased embedding)
        low/negative means species dispersed (functionally organised)

    2. pairwise cross-species cosine drop:
        drop(A,B) = mean_within(A) - mean_cross(A,B)
        used as phylogenetic distance proxy for distance_factor in alpha

    saved as a single .npz file with:
        - species : (n_sp,)
        - drop_matrix : (n_sp, n_sp)
        - within_mean : (n_sp,)
        - cohesion : (n_sp,)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_files = sorted(h5_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 file found in {h5_dir}")

    species_name = [fp.stem for fp in h5_files]
    n_sp = len(species_name)
    rng = np.random.default_rng(42)
    print(SEP)
    print(f"  Precomputing species embedding geometry")
    print(f"  Species : {n_sp}")
    print(f"  Sample/sp : {n_sample}")
    print(SEP)

    # pass 1: load samples, L2-normalize, compute within-species cosine
    print(f"\n  Pass 1: loading samples + within-species cosine")
    sp_embs = {}
    within_means = {}
    for fp in tqdm(h5_files, desc="  Loading"):
        with h5py.File(fp, "r") as hf:
            embs_all = hf["embeddings"][:].astype(np.float32)
        n = len(embs_all)
        idx = rng.choice(n, size=min(n_sample, n), replace=False)
        embs = embs_all[idx]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.where(norms == 0, 1.0, norms)
        sp_embs[fp.stem] = embs
        k = len(embs)
        triu = (embs @ embs.T)[np.triu_indices(k, k=1)]
        within_means[fp.stem] = float(triu.mean())

    within_arr = np.array([within_means[s] for s in species_name], dtype=np.float32)
    print(f"  Within-species cosine: mean={within_arr.mean():.4f}  min={within_arr.min():.4f}  max={within_arr.max():.4f}")

    # pass 2: pairwise cross-species cosine -> drop matrix
    print(f"\n  Pass 2: computing {n_sp}x{n_sp} cross-species cosine")
    drop_matrix = np.zeros((n_sp, n_sp), dtype=np.float32)
    cross_row = np.zeros(n_sp, dtype=np.float64) # mean cross per species A

    for i, sp_a in enumerate(tqdm(species_name, desc="  Cross-species")):
        embs_a = sp_embs[sp_a]
        cross_vals = []
        for j, sp_b in enumerate(species_name):
            if i == j:
                continue
            cross_mean = float((embs_a @ sp_embs[sp_b].T).mean())
            drop_matrix[i, j] = within_means[sp_a] - cross_mean
            cross_vals.append(cross_mean)
        cross_row[i] = float(np.mean(cross_vals)) if cross_vals else 0.0

    # cohesion = within - mean_cross_all_others
    cohesion_arr = within_arr - cross_row.astype(np.float32)

    off_diag = drop_matrix[drop_matrix != 0]
    print(f"\n  Drop matrix: mean={off_diag.mean():.4f}  std={off_diag.std():.4f}  min={off_diag.min():.4f}  max={off_diag.max():.4f}  n_negative={int((off_diag < 0).sum())}")
    # cohesion table sorted descending
    order = np.argsort(cohesion_arr)[::-1]
    print(f"\n  {'Species':<45} {'within':>7}  {'cross':>7}  {'cohesion':>9}")
    print(f"  {'-' * 73}")
    for idx in order:
        sp = species_name[idx]
        coh = cohesion_arr[idx]
        flag = " <- tight blob" if coh > 0.05 else ""
        print(f"  {sp[:43]:<45} {within_arr[idx]:>7.4f}  {cross_row[idx]:>7.4f}  {coh:>+9.4f}{flag}")

    # top/bottom 5 species pairs by drop
    flat = [(drop_matrix[i, j], species_name[i], species_name[j]) for i in range(n_sp) for j in range(n_sp) if i != j]
    flat.sort(reverse=True)
    print(f"\n  Top 5 most distant pairs:")
    for drop, a, b in flat[:5]:
        print(f"   {drop:+.4f}  {a} ↔ {b}")
    print(f"  Top 5 closes pairs:")
    for drop, a, b in flat[-5:]:
        print(f"   {drop:+.4f}  {a} ↔ {b}")

    # save .npz
    out_npz = out_dir / "species_geometry.npz"
    np.savez(out_npz, species=np.array(species_name), drop_matrix=drop_matrix, within_mean=within_arr, cohesion=cohesion_arr)
    print(f"\n  Saved -> {out_npz}")

    # save human-readable tsv
    tsv_path = out_dir / "species_geometry.tsv"
    with open(tsv_path, "w") as fh:
        fh.write("species\twithin_cosine\tcross_cosine_mean\tcohesion\n")
        for idx in order:
            fh.write(f"{species_name[idx]}\t{within_arr[idx]:.6f}\t"
                     f"{cross_row[idx]:.6f}\t{cohesion_arr[idx]:.6f}\n")
        print(f"  Saved -> {tsv_path}")
        print(SEP)

# step 2: dynamic reweighting
def reweight(cosine_file: Path, dist_npz: Path, out_dir: Path, alpha_maxes: list[float], taus: list[float], e_threshold: float, e_scale: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    # load species geometry
    data = np.load(dist_npz, allow_pickle=True)
    species_list = list(data["species"])
    drop_matrix = data["drop_matrix"] # (n_sp, n_sp) float32
    cohesion_arr = data["cohesion"] # (n_sp,) float32
    sp2idx = {s: i for i, s in enumerate(species_list)}
    n_sp = len(species_list)
    # normalize cohesion to [0, 1] for use as a discount factor
    # cohesion_scale : value at which the cohesion discount halves alpha_max
    coh_p90 = float(np.percentile(cohesion_arr, 90))
    coh_p90 = max(coh_p90, 1e-6)
    print(SEP)
    print("  Dynamic Edge Reweighting")
    print(SEP)
    print(f"  Input           : {cosine_file}")
    print(f"  Species         : {n_sp}")
    print(f"  alpha_max       : {alpha_maxes}")
    print(f"  tau             : {taus}")
    print(f"  e_threshold     : {e_threshold}  (-log10 evalue)")
    print(f"  e_scale         : {e_scale}")
    print(f"  cohesion p90    : {coh_p90:.4f}  (used as cohesion discount scale)")
    print(f"\n  Alpha formula per edge:")
    print(f"    dist_factor   = exp(-drop(A,B) / tau)")
    print(f"    conf_factor   = sigmoid((-log10(evalue) - e_threshold) / e_scale)")
    print(f"    coh_factor    = 1 - max(cohesion(A), cohesion(B)) / cohesion_p90")
    print(f"                    clamped to [0, 1]")
    print(f"    alpha         = alpha_max * dist_factor * conf_factor * coh_factor")

    # open output files for all (alpha_max, tau) combinations
    combos = [(am, t) for am in alpha_maxes for t in taus]
    out_fhs = {}
    for am, t in combos:
        fname = out_dir / f"reweighted_dynamic_amax{am:.2f}_tau{t:.3f}.abc"
        out_fhs[(am, t)] = open(fname, "w")
    print(f"  Combinations: {len(combos)}")

    t0 = time.time()
    n_total = 0
    n_missing = 0
    n_same_sp = 0
    alpha_sums = defaultdict(float)
    w_sums = defaultdict(float)
    # cache : protein_id -> species index
    id2sp = {}

    def get_sp_idx(pid: str) -> int:
        if pid not in id2sp:
            sp = species_from_id(pid)
            id2sp[pid] = sp2idx.get(sp, -1)
        return id2sp[pid]

    print(f"\n  Streaming edges")
    with open(cosine_file) as fh:
        next(fh) # skip header

        for raw_line in fh:
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < 15:
                continue

            cosine_str = parts[COL_COSINE]
            if cosine_str == "NA":
                n_missing += 1
                continue

            q_id = parts[COL_Q]
            s_id = parts[COL_S]

            if q_id == s_id:
                n_same_sp += 1
                continue

            try:
                evalue = float(parts[COL_EVALUE])
                cosine = float(cosine_str)
            except ValueError:
                continue

            # sequence-based weight (normalized)
            nle = neg_log10_clamped(evalue)
            nle_norm = normalize_neg_log(nle)
            cosine = max(0.0, min(1.0, cosine))

            # confidence factor (evalue based, same for all combos)
            conf = float(sigmoid(np.array((nle - e_threshold) / e_scale)))
            # species-pair distance factor
            qi = get_sp_idx(q_id)
            si = get_sp_idx(s_id)
            if qi == -1 or si == -1 or qi == si:
                drop = 0.0
                coh_factor = 1.0
            else:
                # phylogenetic distance: clamp neg drops to 0
                drop = max(0.0, float(drop_matrix[qi, si]))
                # cohesion discount: high-cohesion species get lower alpha (biased embeddings)
                max_coh = max(float(cohesion_arr[qi]), float(cohesion_arr[si]))
                coh_factor = float(np.clip(1.0 - max_coh / coh_p90, 0.0, 1.0))

            for am, t in combos:
                dist_factor = float(np.exp(-drop / t))
                alpha = am * dist_factor * conf * coh_factor
                w = ((1.0 - alpha) * nle_norm + alpha * cosine) * NEG_LOG_MAX
                out_fhs[(am, t)].write(f"{q_id}\t{s_id}\t{w:.6f}\n")
                alpha_sums[(am, t)] += alpha
                w_sums[(am, t)] += w

            n_total += 1

            if n_total % 5_000_000 == 0:
                elapsed = time.time() - t0
                rate = n_total / elapsed / 1000
                print(f"  {n_total:>12,} edges  |  {rate:.0f}k/s  |  {elapsed:.0f}s elapsed", end="\r")

    for fh in out_fhs.values():
        fh.close()

    elapsed = time.time() - t0
    print(f"\n\n  Done.")
    print(f"  Edges written  : {n_total:,}")
    print(f"  Missing cosine : {n_missing:,}")
    print(f"  Self-hits skip : {n_same_sp:,}")
    print(f"  Elapsed        : {elapsed:.1f}s")

    print(f"\n  Per-configuration stats:")
    print(f"  {'alpha_max':>9}  {'tau':>6}  {'mean_alpha':>10}  {'mean_w':>8}  output")
    for am, t in combos:
        mean_a = alpha_sums[(am, t)] / max(n_total, 1)
        mean_w = w_sums[(am, t)] / max(n_total, 1)
        fname = f"reweighted_dynamic_amax{am:.2f}_tau{t:.3f}.abc"
        print(f"  {am:>9.2f}  {t:>6.3f}  {mean_a:>10.4f}  {mean_w:>8.4f}  {fname}")

    print(f"\n{SEP}")
    print("  MCL commands:")
    print(SEP)
    for am, t in combos:
        stem = f"reweighted_dynamic_amax{am:.2f}_tau{t:.3f}"
        abc = out_dir / f"{stem}.abc"
        mci = out_dir / f"{stem}.mci"
        tab = out_dir / f"{stem}.tab"
        print(f"\n  # alpha_max={am:.2f}  tau={t:.3f}")
        print(f"  mcxload -abc {abc} \\")
        print(f"      --stream-mirror -stream-tf 'ceil(200)' \\")
        print(f"      -o {mci} -write-tab {tab}")
        print(f"  mcl {mci} -I 2.0 \\")
        print(f"      -o {out_dir}/clusters_{stem}_I20.txt \\")
        print(f"      -use-tab {tab} -te 8")

    print(f"\n{SEP}")

def main():
    p = argparse.ArgumentParser(
        description="Dynamic edge reweighting with phylogenetic distance-aware alpha",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # precompute
    pre = sub.add_parser("precompute", help="Build species pairwise cosine-drop distance matrix")
    pre.add_argument("--h5_dir", required=True, help="Directory containing .h5 embedding files")
    pre.add_argument("--out_dir", required=True)
    pre.add_argument("--n_sample", type=int, default=500, help="Proteins to sample per species (default=500)")

    # reweight
    rew = sub.add_parser("reweight", help="Stream cosine file and write reweighted ABC files")
    rew.add_argument("--cosine_file", required=True, help="diamond_with_cosine.tsv")
    rew.add_argument("--dist_matrix", required=True, help="species_geometry.npz from precompute step")
    rew.add_argument("--out_dir", required=True)
    rew.add_argument("--alpha_max", nargs="+", type=float, default=[0.4], help="Max cosine weight (default: 0.4)")
    rew.add_argument("--tau", nargs="+", type=float, default=[0.05], help="Distance decay constant (default: 0.05)")
    rew.add_argument("--e_threshold", type=float, default=10.0, help="-log(evalue) midpoint for confidence sigmoid (default: 10)")
    rew.add_argument("--e_scale", type=float, default=5.0, help="sigmoid sharpness (defaukt: 5.0)")

    args = p.parse_args()

    if args.mode == "precompute":
        precompute_distances(h5_dir=Path(args.h5_dir), out_dir=Path(args.out_dir), n_sample=args.n_sample)
    elif args.mode == "reweight":
        reweight(cosine_file=Path(args.cosine_file), dist_npz=Path(args.dist_matrix), out_dir=Path(args.out_dir), alpha_maxes=sorted(args.alpha_max), taus=sorted(args.tau), e_threshold=args.e_threshold, e_scale=args.e_scale)

if __name__ == "__main__":
    main()


