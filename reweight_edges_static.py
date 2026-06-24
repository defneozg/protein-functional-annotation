import argparse
import time
from pathlib import Path
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  query_id  query_len  qstart  qend  subject_id  subject_len  sstart  send
#    0          1         2       3       4            5          6      7
#  aln_len  pident  query_cov  bitscore  evalue  raw_score  cosine_sim
#    8         9       10         11       12       13          14
# ══════════════════════════════════════════════════════════════════════════════

COL_Q = 0
COL_S = 4
COL_EVALUE = 12
COL_COSINE = 14
EVALUE_CLAMP = 1e-300
NEG_LOG_MAX = 200.0 # mcxload ceil(200)

def neg_log10(evalue: float) -> float:
    return -np.log10(max(evalue, EVALUE_CLAMP))

def normalize_neg_log(x: float) -> float:
    return min(x, NEG_LOG_MAX) / NEG_LOG_MAX

def combine_weights(w_seq_norm: float, cosine: float, alpha: float) -> float:
    """w_combined = (1-alpha) * w_seq_norm + alpha * cosine_sim"""
    return ((1.0 - alpha) * w_seq_norm + alpha * cosine) * NEG_LOG_MAX

def process(cosine_file: Path, out_dir: Path, alphas: list[float], chunk_size:int = 500_000):
    out_dir.mkdir(parents=True, exist_ok=True)
    SEP = "="*60
    # open one output file per alpha
    out_files = {}
    for alpha in alphas:
        fname = out_dir / f"reweighted_alpha{alpha:.2f}.abc"
        out_files[alpha] = open(fname, "w")

    print(SEP)
    print("  Edge Reweighting")
    print(SEP)
    print(f"  Input : {cosine_file}")
    print(f"  Alphas : {alphas}")
    print(f"  Outputs : {out_dir}")

    t0 = time.time()
    n_total = 0
    n_missing = 0
    n_zero_evalue = 0

    # track weight distribution per alpha
    w_sums = {a: 0.0 for a in alphas}
    w_counts = {a: 0 for a in alphas}

    with open(cosine_file) as fh:
        next(fh) # skipping header
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
                continue # skip self-hits
            try:
                evalue = float(parts[COL_EVALUE])
                cosine = float(cosine_str)
            except ValueError:
                 continue

            if evalue == 0.0:
                n_zero_evalue += 1

            nle = neg_log10(evalue)
            nle_norm = normalize_neg_log(nle)
            cosine = max(0.0, min(1.0, cosine))

            for alpha in alphas:
                w = combine_weights(nle_norm, cosine, alpha)
                out_files[alpha].write(f"{q_id}\t{s_id}\t{w:.6f}\n")
                w_sums[alpha] += w
                w_counts[alpha] += 1
            n_total += 1

            if n_total % 5_000_000 == 0:
                elapsed = time.time() - t0
                rate = n_total / elapsed / 1000
                print(f"  {n_total:>12,} edges  |  {rate:.0f}k/s  |  {elapsed:.0f}s elapsed", end="\r")

    for fh in out_files.values():
        fh.close()

    elapsed = time.time() - t0
    print(f"\n\n  Done.")
    print(f"  Edges written    : {n_total:,}")
    print(f"  Missing cosine   : {n_missing:,}  (no embedding, skipped)")
    print(f"  Zero e-values    : {n_zero_evalue:,}  (clamped to {EVALUE_CLAMP})")
    print(f"  Elapsed          : {elapsed:.1f}s")

    print(f"\n  Weight statistics per alpha:")
    print(f"  {'alpha':>6}  {'mean_w':>10}  {'output_file'}")
    for alpha in alphas:
        mean_w = w_sums[alpha] / max(w_counts[alpha], 1)
        fname = out_dir / f"reweighted_alpha{alpha:.2f}.abc"
        print(f"  {alpha:>6.2f}  {mean_w:>10.4f}  {fname.name}")

    # print MCL commands for each output file
    print(f"\n{SEP}")
    print("  MCL commands (run these after reweighting):")
    print(SEP)
    for alpha in alphas:
        stem = f"reweighted_alpha{alpha:.2f}"
        abc = out_dir / f"{stem}.abc"
        mci = out_dir / f"{stem}.mci"
        tab = out_dir / f"{stem}.tab"
        print(f"\n  # alpha={alpha:.2f}")
        print(f"  mcxload -abc {abc} \\")
        print(f"      --stream-mirror -stream-tf 'ceil(200)' \\")
        print(f"      -o {mci} -write-tab {tab}")
        print(f"  mcl {mci} -I 2.0 \\")
        print(f"      -o {out_dir}/clusters_{stem}_I20.txt \\")
        print(f"      -use-tab {tab} -te 8")

    print(f"\n{SEP}")

def main():
    parser = argparse.ArgumentParser(
        description="reweight alignment edges by using embedding cosine similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cosine_file", required=True, help="diamond_with_cosine.tsv from embedding_analysis.py")
    parser.add_argument("--out_dir", required=True, help="output directory")
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.3], help="Mixing coeff for cosine similarity (0=seq only, 1=cosine only, default: 0.3)")
    parser.add_argument("--chunk_size", default=500_000, type=int, help="lines per read chunk (default=500000)")
    args = parser.parse_args()
    # validate alphas
    for a in args.alpha:
        if not 0.0 <= a <= 1.0:
            parser.error(f"alpha {a} is not in [0, 1]")

    process(cosine_file=Path(args.cosine_file), out_dir=Path(args.out_dir), alphas=sorted(args.alpha), chunk_size=args.chunk_size,)

if __name__ == "__main__":
    main()