"""
two modes:
    validate - Quality checks on existing .h5 embedding files before running the full dataset
    cosine - Read diamond alignment edges, compute embedding cosine similarity for each pair, write augmented TSV for
             downstream edge-reweighting experiments
"""

import argparse
import os
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import h5py
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

# I/O helpers
def load_h5(path: Path) -> tuple[list[str], np.ndarray]:
    """(ids, float32 embeddings)"""
    with h5py.File(path, "r") as hf:
        ids = [s.decode() if isinstance(s, bytes) else s for s in hf["ids"][:]]
        embs = hf["embeddings"][:].astype(np.float32)
    return ids, embs

def load_h5_dir(h5_dir: Path) -> tuple[dict[str, int], np.ndarray]:
    """load all .h5 files into a single index
    returns :
        id2row : dict protein_id -> row idx in embedding matrix
        emb_matrix : (N, D) float32
    """
    h5_files = sorted(h5_dir.glob("*.h5"))
    if not h5_files:
        sys.exit(f"[ERROR] No .h5 files found in {h5_dir}")

    all_ids, all_embs = [], []
    for fp in h5_files:
        ids, embs = load_h5(fp)
        all_ids.extend(ids)
        all_embs.append(embs)
        print(f"  loaded {fp.name}  ({len(ids):,} proteins)")

    emb_matrix = np.concatenate(all_embs, axis=0)
    id2row = {pid: i for i, pid in enumerate(all_ids)}
    print(f"\n  Total loaded : {len(all_ids):,} proteins | dim = {emb_matrix.shape[1]}")
    return id2row, emb_matrix

def load_go(go_path: Path) -> dict[str, set]:
    """load protein -> go term annot"""
    go_path = Path(go_path)
    if go_path.suffix == ".pkl":
        with open(go_path, "rb") as fh:
            data = pickle.load(fh)
        #normalize, values should be sets
        return {k: set(v) if not isinstance(v, set) else v for k, v in data.items()}
    else:
        protein2go = defaultdict(set)
        with open(go_path) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    protein2go[parts[0]].add(parts[1])
        return dict(protein2go)

def load_domain_annot(annot_dir: Path) -> dict[str, set]:
    """protein_id -> set of domain ids"""
    annot = defaultdict(set)
    for fp in sorted(Path(annot_dir).glob("*.tsv")):
        with open(fp, encoding="utf-8-sig", errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    annot[parts[0]].add(parts[2])
    return dict(annot)

# embedding math
def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1.0, norms)

def cosine_sim_pairs_batch(ids_q: list[str], ids_t: list[str], id2row: dict[str, int], emb_norm: np.ndarray) -> np.ndarray:
    """for parallel arrays ids_q and ids_t, return cosine similarity for each pair. NaN if pairs are missing"""
    sims = np.full(len(ids_q), np.nan, dtype=np.float32)
    q_rows = np.array([id2row.get(pid, -1) for pid in ids_q])
    t_rows = np.array([id2row.get(pid, -1) for pid in ids_t])
    valid = (q_rows >= 0) & (t_rows >= 0)
    if valid.any():
        Q = emb_norm[q_rows[valid]] # (k, D)
        T = emb_norm[t_rows[valid]] # (k, D)
        sims[valid] = (Q * T).sum(axis=1)
    return sims

# validate mode
def run_validate(args):
    h5_path = Path(args.h5)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    SEP = "="*60
    print(SEP)
    print("  Embedding Validation")
    print(SEP)

    # basic stats
    ids, embs = load_h5(h5_path)
    N, D = embs.shape
    print(f"\n[1] Basic stats")
    print(f"  Proteins : {N:,}")
    print(f"  Dim : {D}")
    print(f"  dtype : {embs.dtype}")
    print(f"  mean : {embs.mean():.4f}")
    print(f"  std : {embs.std():.4f}")
    print(f"  min/max : {embs.min():.4f} / {embs.max():.4f}")

    norms = np.linalg.norm(embs, axis=1)
    print(f"  L2 norm : mean = {norms.mean():.3f}  std = {norms.std():.3f}  min = {norms.min():.3f}  max = {norms.max():.3f}")
    # check for NaN / inf
    n_nan = np.isnan(embs).any(axis=1).sum()
    n_inf = np.isinf(embs).any(axis=1).sum()
    print(f"  NaN rows : {n_nan} | Inf rows : {n_inf}")
    if n_nan > 0 or n_inf > 0:
        print(f"  [WARNING] NaN/Inf detected : these proteins will be skipped in downstream")

    # cosine similarity distribution (random sample)
    print(f"\n[2] Cosine similarity distribution (random pairs)")
    emb_norm = l2_normalize(embs)
    rng = np.random.default_rng(42)
    sample_size = min(1000, N)
    idx = rng.choice(N, size=sample_size, replace=False)
    sample = emb_norm[idx]
    # pairwise cosine on the sample
    sim_matrix = sample @ sample.T
    # upper triangle, excluding diagonal
    triu = sim_matrix[np.triu_indices(sample_size, k=1)]
    print(f"  Pairs sampled : {len(triu):,}")
    print(f"  cos sim - mean = {triu.mean():.4f}  std = {triu.std():.4f}  median = {np.median(triu):.4f}")
    print(f"  p5 = {np.percentile(triu, 5):.4f}  p95 = {np.percentile(triu, 95):.4f}")

    # healthy embeddings : mean cosine should be low (<0.3) and spread wide
    if triu.std() < 0.02:
        print("  [WARNING] near-zero std, embeddings may have collapsed")
    elif triu.mean() > 0.95:
        print("  [WARNING] mean cosine sim extremely high, embeddings may have collapsed")
    elif triu.mean() > 0.7:
        print("  [OK] high cosine sim, expected for single-species data (biologically similar)")
    else:
        print("  [OK] good spread, embeddings are well distributed")

    # plot cosine sim histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(triu, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"Pairwise cosine similarity distribution\n"
                 f"{h5_path.stem}  (sample n={sample_size})")
    ax.axvline(triu.mean(), color="red", linestyle="--", label=f"mean={triu.mean():.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cosine_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  -> cosine_distribution.png saved")

    # PCA variance explained
    print(f"\n[3] PCA variance explained")
    pca = PCA(n_components=min(50, N, D))
    pca.fit(embs)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for k in [5, 10, 20, 50]:
        if k <= len(cumvar):
            print(f"  Top-{k:2d} PCs explain {cumvar[k-1]*100:.1f} of variance")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(cumvar) + 1), cumvar * 100, color="#4C72B0")
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title(f"PCA scree — {h5_path.stem}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_scree.png", dpi=150)
    plt.close(fig)
    print(f"  -> pca_scree.png saved")

    # functional validation
    if args.go_tsv:
        print(f"\n[4] Functional validation (GO annotations)")
        protein2go = load_go(Path(args.go_tsv))
        # find annotated proteins in h5
        id_set = set(ids)
        annotated = [(i, pid) for i, pid in enumerate(ids) if pid in protein2go]
        print(f"  Proteins in h5 : {N:,}")
        print(f"  With GO annotations : {len(annotated):,} ({100*len(annotated)/N:.1f}%)")
        if len(annotated) < 20:
            print(f"  [SKIP] too few annotated proteins in h5")
        else:
            # nearest-neighbor GO transfer accuracy
            ann_idx = np.array([i for i, _ in annotated])
            ann_ids = [pid for _, pid in annotated]
            ann_embs = emb_norm[ann_idx] # (M, D)
            ann_goes = [protein2go[pid] for pid in ann_ids]
            # limit to 500 for speed
            eval_n = min(500, len(annotated))
            rng2 = np.random.default_rng(0)
            eval_idx = rng.choice(len(annotated), size=eval_n, replace=False)

            k_values = [1, 5, 10]
            hits = {k: 0 for k in k_values}
            for ei in eval_idx:
                query_emb = ann_embs[ei] # (D, )
                query_go = ann_goes[ei]
                # cosine sim to all annotated proteins except itself
                sims = ann_embs @ query_emb # (M, )
                sims[ei] = -1 # exclude self
                top_k_idx = np.argsort(sims)[::-1][:max(k_values)]

                for k in k_values:
                    neighbors = top_k_idx[:k]
                    neighbor_go = set()
                    for ni in neighbors:
                        neighbor_go |= ann_goes[ni]
                    if query_go & neighbor_go: # at least one GO term shared
                        hits[k] += 1

            print(f"\n KNN GO transfer accuracy (sample n={eval_n}):")
            for k in k_values:
                acc = hits[k] / eval_n
                bar = "█" * int(acc * 30)
                print(f"  k = {k:2d} : {acc*100:5.1f}% {bar}")

            # random baseline : chance of sharing GO term at random
            all_go_counts = Counter()
            for gos in ann_goes:
                all_go_counts.update(gos)
            total_ann = len(annotated)
            # proba that a random protein shares at least one of the query's GO terms (approx)
            random_baseline = np.mean([1-np.prod([1-all_go_counts[g]/total_ann for g in go]) for go in ann_goes[:200]])
            print(f"\n  Random baseline (approx) : {random_baseline*100:.1f}%")
            print(f"  Lift over random at k=1 : {hits[1]/eval_n / max(random_baseline, 1e-9):.1f}x")

    # domain label consistency
    if args.annot_dir:
        print(f"\n[5] Domain annotation consistency")
        domain_annot = load_domain_annot(Path(args.annot_dir))
        # find proteins in this h5 that have a Pfam annotation
        id2idx = {pid: i for i, pid in enumerate(ids)}
        pfam_ids = [pid for pid in ids if pid in domain_annot and any(d.startswith("PF") for d in domain_annot[pid])]
        print(f"  Proteins with Pfam domain : {len(pfam_ids):,}")

        if len(pfam_ids) < 20:
            print(f"  [SKIP] too few PFAM annotated proteins")
        else:
            # pick the 5 most common Pfam families present
            pfam_counter = Counter()
            for pid in pfam_ids:
                for d in domain_annot[pid]:
                    if d.startswith("PF"):
                        pfam_counter[d] += 1
            top_families = [fam for fam, _ in pfam_counter.most_common(5)]
            print(f"\n  Top Pfam families : {top_families}")

            # t-SNE on proteins from these families
            family2pids = defaultdict(list)
            for pid in pfam_ids:
                for fam in top_families:
                    if fam in domain_annot[pid]:
                        family2pids[fam].append(pid)

            # sample up to 100 per family
            plot_ids, plot_labels = [], []
            for fam in top_families:
                members = family2pids[fam][:100]
                plot_ids.extend(members)
                plot_labels.extend([fam] * len(members))
                print(f"  {fam} : {len(members)} proteins")

            print(f"  Total proteins for t-SNE: {len(plot_ids)}")

            if len(plot_ids) < 6:
                print(f"  [SKIP] too few proteins across families for t-SNE")
            else:
                plot_embs = np.stack([embs[id2idx[pid]] for pid in plot_ids])
                # PCA to 50d first (speeding up t-SNE)
                n_pca = min(50, plot_embs.shape[0] - 1, D)
                if n_pca > 2:
                    pre_pca = PCA(n_components=n_pca)
                    plot_pca = pre_pca.fit_transform(plot_embs)
                else:
                    plot_pca = plot_embs

                perplexity = max(2, min(20, len(plot_ids) // 4))
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
                xy = tsne.fit_transform(plot_pca)
                print(f"  t-SNE perplexity={perplexity}")

                # plot
                colors_map = plt.cm.tab10(np.linspace(0, 1, len(top_families)))
                fam2color = dict(zip(top_families, colors_map))

                fig, ax = plt.subplots(figsize=(8, 7))
                for fam in top_families:
                    mask = np.array([l == fam for l in plot_labels])
                    ax.scatter(xy[mask, 0], xy[mask, 1],
                               color=fam2color[fam], label=fam,
                               s=20, alpha=0.7, edgecolors="none")
                ax.legend(fontsize=8, markerscale=2)
                ax.set_title(f"t-SNE of top-5 Pfam families\n{h5_path.stem}")
                ax.set_xlabel("t-SNE 1");
                ax.set_ylabel("t-SNE 2")
                fig.tight_layout()
                fig.savefig(out_dir / "tsne_pfam_families.png", dpi=150)
                plt.close(fig)
                print(f"  -> tsne_pfam_families.png saved")

                # intra vs inter cosine similarity
                norm_plot = l2_normalize(plot_embs)
                sim_mat = norm_plot @ norm_plot.T
                intra, inter = [], []
                for i in range(len(plot_labels)):
                    for j in range(i+1, len(plot_labels)):
                        if plot_labels[i] == plot_labels[j]:
                            intra.append(sim_mat[i, j])
                        else:
                            inter.append(sim_mat[i, j])
                intra = np.array(intra)
                inter = np.array(inter)

                print(f"\n  Intra-family cosine sim : mean = {intra.mean():.3f} std = {intra.std():.3f}")
                print(f"  Inter-family cosine sim : mean = {inter.mean():.3f} std = {inter.std():.3f}")
                separation = intra.mean() - inter.mean()
                print(f"  Separation (intra-inter) : {separation:.3f} - good if separation > 0.05")

                # distribution plot
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.hist(inter, bins=60, alpha=0.6, color="#E74C3C", label="inter-family")
                ax.hist(intra, bins=60, alpha=0.6, color="#2ECC71", label="intra-family")
                ax.set_xlabel("Cosine similarity")
                ax.set_ylabel("Count")
                ax.set_title(f"Intra vs inter-family similarity — {h5_path.stem}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_dir / "intra_inter_family_sim.png", dpi=150)
                plt.close(fig)
                print(f"  -> intra_inter_family_sim.png saved")

    print(f"\n{SEP}")
    print(f"  Validation complete, figs in {out_dir}")
    print(SEP)

# COSINE mode
def run_cosine(args):
    align_path = Path(args.align)
    h5_dir = Path(args.h5_dir)
    out_path = Path(args.out)
    chunk_size = args.chunk_size
    limit = args.limit # None = all edges
    SEP = "="*60

    print(SEP)
    print("  Cosine Similarity on Alignment Edges")
    print(SEP)
    # load embeddings
    print("\nLoading embeddings")
    id2row, emb_matrix = load_h5_dir(h5_dir)
    emb_norm = l2_normalize(emb_matrix)
    del emb_matrix # we keep only normalized version

    # stream alignment file, compute cosine, write output
    print(f"\nProcessing alignment edges")
    print(f"  Input : {align_path}")
    print(f"  Output : {out_path}")
    if limit:
        print(f"  Limit : {limit:,} edges")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_lines = 0
    total_written = 0
    total_missing = 0

    # alignment form : 0:query_id 1:query_len 2:qstart 3:qend 4:subject_id 5:subject_len 6:sstart 7:send 8:aln_len 9:pident 10:query_cov 11:bitscore 12:evalue 13:raw_score
    COL_Q = 0
    COL_S = 4
    with open(align_path) as fin, open(out_path, "w") as fout:
        # write header
        fout.write("query_id\tquery_len\tqstart\tqend\t"
                   "subject_id\tsubject_len\tsstart\tsend\t"
                   "aln_len\tpident\tquery_cov\tbitscore\t"
                   "evalue\traw_score\tcosine_sim\n")
        # process in chunks for efficiency
        chunk_q, chunk_s, chunk_lines = [], [], []

        def flush_chunk():
            nonlocal total_written, total_missing
            sims = cosine_sim_pairs_batch(chunk_q, chunk_s, id2row, emb_norm)
            for raw_line, sim in zip(chunk_lines, sims):
                if np.isnan(sim):
                    total_missing += 1
                    sim_str = "NA"
                else:
                    sim_str = f"{sim:.6f}"
                fout.write(raw_line.rstrip("\n") + "\t" + sim_str + "\n")
                total_written += 1

        for raw_line in fin:
            if not raw_line.strip():
                continue
            parts = raw_line.split("\t")
            if len(parts) < 2:
                continue

            chunk_q.append(parts[COL_Q])
            chunk_s.append(parts[COL_S])
            chunk_lines.append(raw_line)
            total_lines += 1

            if len(chunk_q) >= chunk_size:
                flush_chunk()
                chunk_q, chunk_s, chunk_lines = [], [], []
                # progress
                elapsed = time.time() - t0
                rate = total_lines / elapsed
                print(f"  {total_lines:>12,} edges processed  |  {rate/1000:.1f}k edges/s  |  missing={total_missing:,}  |  {elapsed:.0f}s elapsed", end="\r")

            if limit and total_lines >= limit:
                break

        #flush remainder
        if chunk_q:
            flush_chunk()

    elapsed = time.time() - t0
    coverage = 100 * (total_written - total_missing) / max(total_lines, 1)

    print("\n\n Done")
    print(f"  Edges processed : {total_written:,}")
    print(f"  Missing (NA) : {total_missing:,} (proteins not yet embedded)")
    print(f"  Coverage : {coverage:.1f}%")
    print(f"  Elapsed : {elapsed:.1f}s")
    print(f"  Output : {out_path}")

    # quick stats on cosine values
    print(f"\nComputing cosine similarity stats from output")
    sims_sample = []
    with open(out_path, "w") as fh:
        next(fh) # skip header
        for i, line in enumerate(fh):
            if i > 200_000:
                break
            parts = line.rstrip("\n").split("\t")
            val = parts[-1]
            if val != "NA":
                sims_sample.append(float(val))

    sims_arr = np.array(sims_sample)
    print(f"  Sample size : {len(sims_arr):,}")
    print(f"  mean : {sims_arr.mean():.4f}")
    print(f"  std : {sims_arr.std():.4f}")
    print(f"  median : {np.median(sims_arr):.4f}")
    print(f"  p5 / p95 : {np.percentile(sims_arr, 5):.4f} / {np.percentile(sims_arr, 95):.4f}")

    # correlation between bitscore and cosine sim
    print(f"\nCorrelation: bitscore vs cosine similarity")
    bitscores, cosines = [], []
    with open(out_path) as fh:
        next(fh)
        for i, line in enumerate(fh):
            if i > 200_000:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15 or parts[-1] == "NA":
                continue
            try:
                bitscores.append(float(parts[11])) # bitscore is col 11
                cosines.append(float(parts[-1]))
            except ValueError:
                continue

    if len(bitscores) > 100:
        corr = np.corrcoef(bitscores, cosines)[0, 1]
        print(f"  Pearson r (bitscore, cosine): {corr:.4f}")
        if abs(corr) < 0.3:
            print("  [NOTE] low correlation, embeddings capture information beyond sequence similarity (good for reweighting)")
        elif abs(corr) > 0.7:
            print("  [NOTE] high correlation, embeddings are closely aligned with sequence similarity")

    print(f"\n{SEP}")
    print("  Cosine computation complete")
    print(SEP)

# CLI
def build_parser():
    p = argparse.ArgumentParser(
        description="Embedding validation and cosine similarity on alignment edges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # validate
    val = sub.add_parser("validate", help="Quality checks on .h5 embedding files")
    val.add_argument("--h5", required=True, help="Path to a single .h5 file")
    val.add_argument("--annot_dir", default=None, help="Path to annotation directory (for Pfam t-SNE check)")
    val.add_argument("--go_tsv", default=None, help="GO annotation file : .pkl or .tsv")
    val.add_argument("--out_dir", default="PMIND2026_cluster/figures/embedding_validation", help="Path to output directory")

    # cosine
    cos = sub.add_parser("cosine", help="Cosine Similarity on Alignment Edges")
    cos.add_argument("--align", required=True, help="alignment file path")
    cos.add_argument("--h5_dir", required=True, help="Directory containing .h5 files")
    cos.add_argument("--out", required=True, help="Output TSV path (alignment + cosine_sim col)")
    cos.add_argument("--chunk_size", type=int, default=50_000, help="Edges per processing chunk (default = 50000)")
    cos.add_argument("--limit", type=int, default=None, help="Limit the number of edges to process (for testing)")

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "validate":
        run_validate(args)
    elif args.mode == "cosine":
        run_cosine(args)

if __name__ == "__main__":
    main()






