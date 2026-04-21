"""
Generate per-protein ESM-2 (for now) embeddings
Adding support for three backends : CUDA, MPS, CPU
Output : one HDF5 file per input FASTA (contains dataset ids and embeddings)

Usage :
- small model with a single file :
python generate_embeddings.py --model small --fasta EP00391.fasta

- medium model with one species, on CUDA :
python generate_embeddings.py --model medium --fasta Symbiodinium_sp_*.fasta

- full dataset (when multicore) on CPU :
python generate_embeddings.py --model small --workers 8

- standard model with full dataset on GPU :
python generate_embeddings.py --model large

Models :
- small  -> esm2_t6_8M_UR50D    (8M params, 320-d,  ~10x faster than large)
- medium -> esm2_t12_35M_UR50D  (35M params, 480-d, ~3x faster than large)
- large  -> esm2_t33_650M_UR50D (650M params, 1280-d, best quality)

requirements : fair-esm h5py tqdm torch
"""

import argparse
import os
import sys
import time
import math
from pathlib import Path
from typing import Iterator
import h5py
import numpy as np
import torch
from tqdm import tqdm

BASE = Path("./PMIND2026_cluster")
FASTA_DIR = BASE / "darkdino_fasta"
EMBED_DIR = BASE / "embeddings"

MODEL_ALIASES = {
    "small": "esm2_t6_8M_UR50D",
    "medium": "esm2_t12_35M_UR50D",
    "large": "esm2_t33_650M_UR50D",
}

# max batch tokens per forward pass per device class
BATCH_TOKENS = {
    "cuda": 4_096,
    "mps": 2_048,
    "cpu": 1_024,
}

SEP = "="*60

# FASTA parser
def parse_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """(protein_id, sequence)"""
    header, parts = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header, parts = line[1:].split()[0], []
            else:
                parts.append(line.upper())
    if header is not None:
        yield header, "".join(parts)

# device detection
def detect_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        return device, f"CUDA {name}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "MPS"
    return torch.device("cpu"), "CPU"

# dynamic batching by token count
def make_batches(proteins: list[tuple[str, str]], max_tokens: int, max_seq_len: int = 1022) -> list[list[tuple[str, str]]]:
    """group proteins into batches so that total residues <= max tokens
    proteins longer than max are truncated"""
    # truncate + sort descending by length
    truncated = [(pid, seq[:max_seq_len]) for pid, seq in proteins]
    truncated.sort(key=lambda x: len(x[1]), reverse=True)
    batches, current, current_tokens = [], [], 0
    for pid, seq in truncated:
        n = len(seq)
        if current and current_tokens + n > max_tokens:
            batches.append(current)
            current, current_tokens = [], 0
        current.append((pid, seq))
        current_tokens += n
    if current:
        batches.append(current)
    return batches

# load manifest
def load_manifest(embed_dir: Path) -> set[str]:
    """set of FASTA stems already processed"""
    manifest = embed_dir / "manifest.tsv"
    if not manifest.exists():
        return set()
    done = set()
    with open(manifest) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0] == "OK":
                done.add(parts[1])
    return done

def record_manifest(embed_dir: Path, stem: str, n_proteins: int, elapsed: float):
    manifest = embed_dir / "manifest.tsv"
    with open(manifest, "a") as f: # open file for writing, create if does not exist
        f.write(f"OK\t{stem}\t{n_proteins}\t{elapsed:.1f}s\n")

# main embedding loop for one FASTA file
def embed_fasta(fasta_path: Path, model, alphabet, batch_converter, device: torch.device, device_type:str, repr_layer:int, fp16:bool, embed_dir: Path, repr_kwarg: str = 'repr_layers', repr_arg=None, verbose: bool= True) -> int:
    if repr_arg is None:
        repr_arg = [repr_layer]
    stem = fasta_path.stem
    out_path = embed_dir / f"{stem}.h5"
    max_tokens = BATCH_TOKENS[device_type]
    # load all seqs
    proteins = list(parse_fasta(fasta_path))
    if not proteins:
        print(f"  [SKIP] {stem} -- empty FASTA")
        return 0
    n = len(proteins)
    if verbose:
        print(f"\n  {stem}  ({n:,} proteins)")

    batches = make_batches(proteins, max_tokens)
    all_ids = []
    all_embs = []

    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(batches, desc=f"    batches", leave=False, disable=not verbose):
            labels, seqs = zip(*batch)
            batch_data = [(lbl, seq) for lbl, seq in zip(labels, seqs)]
            _, _, tokens = batch_converter(batch_data)
            tokens = tokens.to(device)

            if fp16 and device_type == "cuda":
                with torch.autocast("cuda"):
                    results = model(tokens, **{repr_kwarg: repr_arg}, return_contacts = False)
            else:
                results = model(tokens, **{repr_kwarg: repr_arg}, return_contacts = False)

            # mean-pool over sequence length (excluding BOS/EOS tokens)
            representations = results["representations"][repr_layer] # (B, L+2, D)
            for i, (pid, seq) in enumerate(batch):
                seq_len = len(seq)
                # slice out only the residue pos [1 : seq_len +1]
                emb = representations[i, 1 : seq_len + 1].mean(0) # (D,)
                all_ids.append(pid)
                all_embs.append(emb.float().cpu().numpy().astype(np.float16))

    elapsed = time.time() - t0
    throughput = n / elapsed

    # save to HDF5
    emb_matrix = np.stack(all_embs, axis=0) # (N, D)
    dt_str = h5py.special_dtype(vlen=str)
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("ids", data=np.array(all_ids, dtype=object), dtype=dt_str, compression="gzip", compression_opts=4)
        hf.create_dataset("embeddings", data=emb_matrix, dtype=np.float16, compression="gzip", compression_opts=4)
        hf.attrs["model"] = model.__class__.__name__
        hf.attrs["repr_layer"] = repr_layer
        hf.attrs["dim"] = emb_matrix.shape[1]
        hf.attrs["n_proteins"] = n
        hf.attrs["fasta_stem"] = stem

    if verbose:
        print(f"  done: {n:,} proteins in {elapsed:.1f} seconds"
              f"({throughput*60:.0f} prot/min) -> {out_path.name}")

    return n, elapsed

# cosine similarity utility (downstream)
def load_embeddings(h_path: Path) -> tuple[list[str], np.ndarray]:
    """load (ids, embeddings float32) from an HDF5 file"""
    with h5py.File(h_path, "r") as hf:
        ids = [s.decode() if isinstance(s, bytes) else s for s in hf["ids"][:]]
        embs = hf["embeddings"][:].astype(np.float32)
    return ids, embs

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """cosine similarity between rows of A and rows of B
    A : (N,D), B : (M, D) -> (N, M) float32"""
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A @ B.T

def cosine_sim_pairs(ids_a: list[str], embs_a: np.ndarray, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
    """given a list of (id1, id2) pairs, return {(id1, id2): cosine_sim}"""
    idx = {pid: i for i, pid in enumerate(ids_a)}
    result = {}
    for id1, id2 in pairs:
        if id1 not in idx or id2 not in idx:
            continue
        v1 = embs_a[idx[id1]]
        v2 = embs_a[idx[id2]]
        sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        result[(id1, id2)] = sim
    return result

# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="generate ESM-2 embeddings for FASTAs of dino proteins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", default="medium",
        choices=list(MODEL_ALIASES.keys()) + list(MODEL_ALIASES.values()),
        help="model size alias or full ESM-2 model name (default: medium)",
    )
    parser.add_argument(
        "--fasta", nargs="*", default=None,
        help="Specific FASTA file(s) to process. If omitted, all FASTAs in FASTA_DIR"
    )
    parser.add_argument(
        "--fasta_dir", type=Path, default=FASTA_DIR,
        help=f"Directory containing FASTA files (default: {FASTA_DIR})",
    )
    parser.add_argument(
        "--embed_dir", type=Path, default=EMBED_DIR,
        help=f"Output directory for embeddings (default: {EMBED_DIR})",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=None,
        help="Override max tokens per batch (default: auto by device)"
    )
    parser.add_argument(
        "--no_fp16", action="store_true",
        help="Disable float16 inference (slower, for debugging)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="CPU threads for tokenisation (default: 1)",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip files already in manifest (default: True)",
    )
    parser.add_argument(
        "--no_resume", dest="resume", action="store_false",
        help="Reprocess all files even if already in manifest",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be done without running inference"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = MODEL_ALIASES.get(args.model, args.model)
    device, device_desc = detect_device()
    device_type = device.type
    fp16 = (device_type == "cuda") and (not args.no_fp16)

    print(SEP)
    print("  Protein embedding generator")
    print(SEP)
    print(f"  Device: {device_desc}")
    print(f"  Model: {model_name}")
    print(f"  fp16: {fp16}")

    # import ESM (deferred sp script is importable without it)
    try:
        import esm
    except ImportError:
        print("\n[ERROR] fair-esm is not installed")
        sys.exit(1)

    # load model
    print(f"\n Loading {model_name}")
    t_load = time.time()
    # support both old and new fair-esm loading APIs
    loader = getattr(esm.pretrained, model_name, None)
    if loader is not None:
        model, alphabet = loader()
    else:
        model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name) # fallback

    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)
    if fp16:
        model = model.half()

    repr_layer = getattr(model, "num_layers", None) or getattr(model, "n_layers", 6) # detect repr_layer
    dim = getattr(model, "embed_dim", 320)

    # probe the forward signature to handle repr_layers bs repr_layer incompatability
    import inspect
    _fwd_sig = inspect.signature(model.forward)
    if "repr_layers" in _fwd_sig.parameters:
        _repr_kwarg = "repr_layers" # standard
        _repr_arg = [repr_layer]
    else:
        _repr_kwarg = "repr_layer" # older / alt builds : takes an int
        _repr_arg = repr_layer

    print(f"  Model loaded in {time.time() - t_load:.1f} seconds | "
          f"repr_layer={repr_layer} | dim={dim} | "
          f"forward kwarg='{_repr_kwarg}'")

    # max tokens per batch
    max_tokens = args.max_tokens or BATCH_TOKENS.get(device_type, 1024)
    print(f"  max_tokens/batch = {max_tokens}")

    # collect FASTA files
    fasta_dir = args.fasta_dir
    if args.fasta:
        fasta_paths = []
        for pattern in args.fasta:
            fasta_paths += list(fasta_dir.glob(pattern)) or [Path(pattern)]
        fasta_paths = sorted(set(fasta_paths))
    else:
        fasta_paths = sorted(
            list(fasta_dir.glob("*.fasta")) +
            list(fasta_dir.glob("*.fa")) +
            list(fasta_dir.glob("*.fna"))
        )

    if not fasta_paths:
        print(f"\n[ERROR] no FASTA files found in {fasta_dir}")
        sys.exit(1)

    # output dir + manifest
    embed_dir = args.embed_dir
    embed_dir.mkdir(parents=True, exist_ok=True)
    done = load_manifest(embed_dir) if args.resume else set()

    to_process = [fp for fp in fasta_paths if fp.stem not in done]
    skipped = len(fasta_paths) - len(to_process)
    print(f"\n  FASTA files found : {len(fasta_paths)}")
    print(f"  Already done : {skipped}")
    print(f"  To process : {len(to_process)}")

    if args.dry_run:
        print("\n  [DRY RUN] Files that would be processed:")
        for fp in to_process:
            print(f"  {fp.name}")
        return

    if not to_process:
        print(f"\n Nothing to do. Use --no_resume to reprocess.")
        return

    # main loop
    print(f"\n{SEP}")
    total_proteins = 0
    total_time = 0.0

    for i, fasta_path in enumerate(to_process, 1):
        print(f"[{i}/{len(to_process)}] {fasta_path.name}")
        try:
            n, elapsed = embed_fasta(
                fasta_path,
                model, alphabet, batch_converter,
                device, device_type,
                repr_layer, fp16,
                embed_dir,
                repr_kwarg=_repr_kwarg, repr_arg=_repr_arg,
                verbose=True,
            )
            total_proteins += n
            total_time += elapsed
            record_manifest(embed_dir, fasta_path.stem, n, elapsed)

            # ETA estimate
            if total_time > 0:
                rate = total_proteins / total_time # proteins/sec
                remaining_files = len(to_process) - i
                # rough estimate, assume average protein count
                avg_n = total_time / i
                eta_s = remaining_files * avg_n / max(rate, 1e-9)
                eta_h = eta_s / 3600
                if eta_s < 60:
                    eta_str = f"{eta_s:.0f}s"
                elif eta_s < 3600:
                    eta_str = f"{eta_s/60:.1f}min"
                else:
                    eta_str = f"{eta_s/3600:.1f}h"
                print(f"   rate : {rate*60:.0f} prot/min"
                      f"ETA : {eta_str} for remaining {remaining_files} files")

        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] {fasta_path.name} -- reducing batch and retrying")
            torch.cuda.empty_cache()
            # halve max_tokens and retry once
            old = BATCH_TOKENS["cuda"]
            BATCH_TOKENS["cuda"] = old // 2
            try:
                n, elapsed = embed_fasta(
                    fasta_path, model, alphabet, batch_converter,
                    device, device_type, repr_layer, fp16, embed_dir,
                    repr_kwarg=_repr_kwarg, repr_arg=_repr_arg)
                total_proteins += n
                total_time += elapsed
                record_manifest(embed_dir, fasta_path.stem, n, elapsed)
            except Exception as e2:
                print(f"  [FAIL] {fasta_path.name} after retry: {e2}")
            finally:
                BATCH_TOKENS["cuda"] = old

        except Exception as e:
            print(f"  [FAIL] {fasta_path.name} : {e}")
            continue

    print(f"\n{SEP}")
    if total_time < 60:
        time_str = f"{total_time:.1f}s"
    elif total_time < 3600:
        time_str = f"{total_time / 60:.1f}min"
    else:
        time_str = f"{total_time / 3600:.2f}h"
    print(f"  Finished embedding {total_proteins:,} proteins in {time_str}")
    print(f"  Output: {embed_dir}")
    print(SEP)

if __name__ == "__main__":
    main()