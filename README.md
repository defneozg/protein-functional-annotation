# Improving Protein Clustering by Combining Sequence Similarity Networks and Protein Embeddings

This repository has the code used to cluster 7.4 million dinoflagellate proteins. It combines two sources of information:

1. **A sequence similarity network (SSN)** built from an all-vs-all DIAMOND search. Nodes are proteins and edges are significant alignments.
2. **Protein language model (pLM) embeddings**, one per protein, from ProtT5-XL-UniRef50 (1024-d) and ESM-2 `esm2_t6_8M_UR50D` (320-d).

The goal is to get clusters that are **larger and functionally more homogeneous**, so that annotations can be transferred to "dark" (unannotated) proteins. Clusters are evaluated with GO Molecular Function (GO-MF) annotations: the main metric is the mean pairwise Jaccard similarity of GO-MF leaf terms within each cluster.

There are two complementary strategies:

| Strategy | Idea | Main scripts |
|---|---|---|
| **Post-clustering refinement** | Cluster the SSN (MCL) and the embeddings (UMAP + HDBSCAN, or k-NN graph + Leiden) separately. Then **merge** MCL clusters that share an embedding cluster and have compatible GO annotations, and **split** MCL clusters that the embedding space divides into more homogeneous sub-groups. | `find_merge_split_candidates.py`, `apply_merge_split.py`, `evaluate_revised_clustering.py` |
| **Pre-clustering edge reweighting** | Before clustering, change the SSN edge weights by mixing the sequence weight (`-log10(E-value)`, capped at 200) with the cosine similarity of the two embeddings. The **static** variant uses a fixed mixing coefficient α. The **dynamic** variant uses a per-edge α that shrinks with embedding-estimated phylogenetic distance and with species-specific embedding bias ("species cohesion"). | `reweight_edges_static.py`, `reweight_edges_dynamic.py`, `reweight_dynamic_sans_conf.py` |

The reweighted graphs are then clustered with **MCL** (`mcl`, inflation 2.0) or **Leiden** (CPM quality function).

---

## 1. Data

### 1.1 Origin

The protein sequences come from two public databases:

- **EukProt (v3)**: genome-scale predicted proteomes across eukaryotes. Files with an `EP*` suffix come from here.
- **METdb**: Marine Microbial Eukaryote Transcriptome Database. Files with a `METDB_*` suffix come from here.

The dataset covers **101 dinoflagellate proteomes/transcriptomes** with **7,426,058 protein sequences** (about 2.17 billion residues, median length 197 aa).

### 1.2 Raw inputs

| Data | Description | Size |
|---|---|---|
| `darkdino_fasta/` | 101 FASTA files, one per proteome. Headers look like `>Species_name_EPxxxxx\|sequenceNNNNN ...` | ~2.7 GB |
| `alignments/diamond_alignments.filter.tsv` | All-vs-all DIAMOND blastp. E-value ≤ 1e-3, max 25 hits per query. 89,280,577 pairs, 14 tab-separated columns: `qseqid qlen qstart qend sseqid slen sstart send length pident ppos score evalue bitscore` | ~14 GB |
| `annotations/darkdino_annotation/` | InterProScan output, one TSV per proteome (e.g. `EP00391.tsv`). 4 columns: `protein_id  database  entry_id  description`. 19 member databases, 48.1M rows | ~5.2 GB |

About 1.58M proteins have no DIAMOND hit at all, so they never appear in the SSN. The embedding-based methods do cover them.

### 1.3 Derived GO files

These files are produced by `go_mapping.py` and `go_prune.py` (see §3.1). Some are also shipped with the data.

| File | Description |
|---|---|
| `go_mappings/*2go` | External GO cross-reference tables (`pfam2go`, `prints2go`, `prosite2go`, `smart2go`, `pirsf2go`, `hamap2go`, `interpro2go`), downloaded from geneontology.org |
| `go_mappings/PANTHER19.0_HMM_classifications` | PANTHER family → GO mapping |
| `go_mappings/go-basic.obo` | GO ontology (release 2026-03-25 was used) |
| `go_mappings/annot_go.tsv` | protein → GO terms (all namespaces) |
| `go_mappings/annot_go_mf.pkl` | protein → GO-MF terms (`dict[str, set[str]]`) |
| `go_mappings/annot_go_mf_pruned.pkl` | **protein → GO-MF leaf terms only.** This is the reference annotation used by every evaluation script. |

### 1.4 Where to put the data

Run all commands from the **repository root**. Put the downloaded data in a folder called `PMIND2026_cluster/` at the root. Intermediate and output folders (embeddings, reweighted graphs, Leiden runs…) are created by the scripts at the paths shown below.

```
.
├── README.md
├── *.py                                   # all scripts
│
├── PMIND2026_cluster/                     # main data folder (download here)
│   ├── darkdino_fasta/                    # 101 FASTA files                [raw input]
│   ├── alignments/
│   │   ├── diamond_alignments.filter.tsv  # DIAMOND all-vs-all            [raw input]
│   │   └── diamond_with_cosine_prott5.tsv # + ProtT5 cosine per edge      [generated, §3.3]
│   ├── annotations/
│   │   └── darkdino_annotation/           # InterProScan TSVs             [raw input]
│   ├── go_mappings/                       # GO mapping files              [generated, §3.1]
│   │   ├── go-basic.obo
│   │   ├── annot_go.tsv
│   │   └── annot_go_mf_pruned.pkl
│   ├── diamond.abc / .mci / .tab          # SSN for MCL (E-value < 1e-20) [generated, §3.2]
│   ├── diamond_thresh20_clusters_I20.txt  # baseline MCL clustering       [generated, §3.2]
│   ├── embeddings_prott5/                 # one .h5 per proteome (ProtT5) [generated, §3.3]
│   ├── umap_output_prott5/                # UMAP 10-d + HDBSCAN labels     [generated, §3.4]
│   ├── leiden_clusters/                   # k-NN graph + Leiden on embeddings [generated, §3.4]
│   ├── figures/                           # all plots / reports
│   ├── revised_clustering_prott5/         # post-clustering outputs (HDBSCAN-based)
│   └── reweighted_static_prott5/          # static reweighting .abc/.mci/.tab + MCL clusters
│
├── embeddings_esm2/                       # one .h5 per proteome (ESM-2)   [generated, §3.3]
├── reweighted_dynamic_prott5/             # dynamic reweighting (with cohesion factor)
├── reweighted_dynamic_prott5_noconf/      # dynamic reweighting (without cohesion factor)
├── leiden_abc_baseline/                   # Leiden on the plain SSN
├── leiden_abc_reweighted/                 # Leiden on a reweighted SSN
├── leiden_dynamic_05/ , leiden_dynamic_01/  # Leiden on dynamic graphs, γ = 0.05 / 0.10
└── leiden_merge_split/                    # post-clustering with Leiden embedding clusters
```

> **Disk space:** ProtT5 embeddings, the cosine-annotated alignment file and the reweighted `.abc` graphs each take tens of GB. You can put any of these folders on an external drive. Just pass the new paths to the scripts.

---

## 2. Requirements

- Python ≥ 3.10
- Python packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `h5py`, `tqdm`, `matplotlib`, `seaborn`, `umap-learn`, `hdbscan`, `faiss-cpu` (or `faiss-gpu`), `python-igraph`, `leidenalg`, `torch`, `fair-esm` (ESM-2), `transformers` + `sentencepiece` (ProtT5)
- [MCL suite](https://micans.org/mcl/) (`mcl`, `mcxload`) on the `PATH`
- `curl` (used by `go_mapping.py` to download the GO mapping tables)
- A CUDA GPU is strongly recommended for ProtT5 embeddings (≈20 h on an A100 for the full dataset). ESM-2 small also runs on Apple MPS or CPU.

---

## 3. Pipeline

Steps are listed in the order they depend on each other.

### 3.1 GO annotation mapping

```bash
# Map InterProScan entries to GO (writes into PMIND2026_cluster/go_mappings/)
python go_mapping.py

# Keep only GO-MF leaf terms (the reference annotation used everywhere below)
python go_prune.py \
    --obo    PMIND2026_cluster/go_mappings/go-basic.obo \
    --input  PMIND2026_cluster/go_mappings/annot_go_mf.pkl \
    --output PMIND2026_cluster/go_mappings/annot_go_mf_pruned.pkl \
    --report
```

### 3.2 Baseline MCL on the SSN

The SSN keeps DIAMOND edges with E-value < 1e-20, weighted by `-log10(E-value)` and capped at 200. Load any `.abc` edge list into MCL format, then cluster with inflation 2.0:

```bash
# Generic: for each .abc file
mcxload -abc <file>.abc --stream-mirror -stream-tf 'ceil(200)' \
    -o <file>.mci -write-tab <file>.tab
mcl <file>.mci -I 2.0 -o clusters_<file>_I20.txt -use-tab <file>.tab

# Baseline run (timed)
time mcl PMIND2026_cluster/diamond.mci \
    -I 2.0 \
    -te 8 \
    -use-tab PMIND2026_cluster/diamond.tab \
    -o PMIND2026_cluster/diamond_thresh20_clusters_I20.txt \
    2>&1 | tee PMIND2026_cluster/mcl_baseline_timing.log
```

### 3.3 Embeddings and cosine similarities

```bash
# ProtT5-XL-UniRef50 (half precision, GPU)
python generate_embeddings_prott5.py \
    --model        prott5 \
    --fasta_dir    PMIND2026_cluster/darkdino_fasta \
    --embed_dir    PMIND2026_cluster/embeddings_prott5 \
    --batch_tokens 16384 \
    --half

# ESM-2 (esm2_t6_8M_UR50D)
python generate_embeddings.py --model small \
    --embed_dir embeddings_esm2

# Add the ProtT5 cosine similarity to every DIAMOND edge
python embedding_analysis.py cosine \
    --align   PMIND2026_cluster/alignments/diamond_alignments.filter.tsv \
    --h5_dir  PMIND2026_cluster/embeddings_prott5 \
    --out     PMIND2026_cluster/alignments/diamond_with_cosine_prott5.tsv

# Sanity check of one embedding file against InterPro / GO annotations
python embedding_analysis.py validate \
    --h5        embeddings_esm2/Akashiwo_sanguinea_METDB_00002.h5 \
    --annot_dir PMIND2026_cluster/annotations/darkdino_annotation \
    --go_tsv    PMIND2026_cluster/go_mappings/annot_go.tsv \
    --out_dir   PMIND2026_cluster/figures/embedding_validation

# Cosine distributions: random pairs vs DIAMOND pairs, by % identity
python cosine_distributions.py \
    --h5_dir      PMIND2026_cluster/embeddings_prott5 \
    --cosine_file PMIND2026_cluster/alignments/diamond_with_cosine_prott5.tsv \
    --out_dir     PMIND2026_cluster/figures/cosine_distributions_prott5 \
    --n_sample    200000 \
    --n_pairs     20000000 \
    --pid_threshold 30.0 40.0 50.0
```

### 3.4 Clustering on embeddings alone

**UMAP + HDBSCAN.** This writes `all_ids.txt`/`umap_ids.txt`, `umap_10d.npy` and `hdbscan_labels.npy`.

```bash
# Full dataset
python cluster_umap_hdbscan.py \
    --h5_dir  PMIND2026_cluster/embeddings_prott5 \
    --out_dir PMIND2026_cluster/umap_output_prott5 \
    --n_jobs  -1

# On a sample (exploration)
python embedding_cluster_viz.py cluster \
    --h5_dir    PMIND2026_cluster/embeddings_prott5 \
    --go_pkl    PMIND2026_cluster/go_mappings/annot_go_mf_pruned.pkl \
    --out_dir   PMIND2026_cluster/figures/umap_prott5

# 2-D UMAP visualisation, coloured by GO-MF and by species
python embedding_cluster_viz.py viz \
    --h5_dir               embeddings_esm2 \
    --go_pkl               PMIND2026_cluster/go_mappings/annot_go_mf_pruned.pkl \
    --out_dir              PMIND2026_cluster/figures/umap \
    --n_sample             50000 \
    --n_annotated_oversample 15000
```

**k-NN graph (FAISS, cosine) + Leiden:**

```bash
# Default settings
python cluster_leiden_embeddings.py \
  --h5_dir  PMIND2026_cluster/embeddings_prott5 \
  --out_dir PMIND2026_cluster/leiden_clusters

# Resolution sweep (like an MCL inflation sweep)
python cluster_leiden_embeddings.py \
  --h5_dir  PMIND2026_cluster/embeddings_prott5 \
  --out_dir PMIND2026_cluster/leiden_clusters \
  --resolutions 0.5 1.0 2.0 5.0

# Tune k-NN graph construction
python cluster_leiden_embeddings.py \
  --h5_dir  PMIND2026_cluster/embeddings_prott5 \
  --out_dir PMIND2026_cluster/leiden_clusters \
  --k 30 --min_cosine 0.3

# Modularity (RBConfigurationVertexPartition) instead of CPM
python cluster_leiden_embeddings.py \
  --h5_dir  PMIND2026_cluster/embeddings_prott5 \
  --out_dir PMIND2026_cluster/leiden_clusters \
  --partition_type modularity --resolutions 1.0 1.5 2.0

# 5 resolutions in parallel (one subprocess per resolution)
python cluster_leiden_embeddings.py \
  --h5_dir  PMIND2026_cluster/embeddings_prott5 \
  --out_dir PMIND2026_cluster/leiden_clusters \
  --resolutions 0.1 0.5 1.0 2.0 5.0 \
  --parallel 5

# Re-run Leiden only, reusing an existing k-NN graph
python cluster_leiden_embeddings.py \
  --h5_dir  PMIND2026_cluster/embeddings_prott5 \
  --out_dir PMIND2026_cluster/leiden_clusters \
  --skip_knn \
  --resolutions 0.1 0.5 1.0 2.0 5.0 \
  --parallel 5
```

---

## 4. Post-clustering approach (merge / split)

`--umap_dir` must contain `all_ids.txt` (or `umap_ids.txt`), `hdbscan_labels.npy` and, optionally, `umap_10d.npy`. To use **Leiden** embedding clusters in place of HDBSCAN, put the Leiden labels in `leiden_merge_split/` under the same file names (`hdbscan_labels.npy` aligned with `all_ids.txt`).

### 4.1 Find candidates

```bash
# HDBSCAN embedding clusters vs baseline MCL (keep all candidates)
python find_merge_split_candidates.py \
    --umap_dir   PMIND2026_cluster/umap_output_prott5 \
    --mcl_file   PMIND2026_cluster/diamond_thresh20_clusters_I20.txt \
    --go_pkl     PMIND2026_cluster/go_mappings/annot_go_mf_pruned.pkl \
    --out_dir    PMIND2026_cluster/figures/merge_split_prott5 \
    --top_n      0 \
    --max_degree 10

# Leiden embedding clusters vs Leiden SSN clustering (γ = 0.1)
python find_merge_split_candidates.py \
    --umap_dir leiden_merge_split \
    --mcl_file leiden_abc_baseline/leiden_cpm_diamond_thresh20_res0.1.txt \
    --go_pkl   PMIND2026_cluster/go_mappings/annot_go_mf_pruned.pkl \
    --out_dir  leiden_merge_split/candidates \
    --top_n    5000

# Leiden embedding clusters vs baseline MCL
python find_merge_split_candidates.py \
    --umap_dir leiden_merge_split \
    --mcl_file PMIND2026_cluster/diamond_thresh20_clusters_I20.txt \
    --go_pkl   PMIND2026_cluster/go_mappings/annot_go_mf_pruned.pkl \
    --out_dir  leiden_merge_split/candidates_mcl_base \
    --min_mcl_size_split 20 \
    --top_n    5000

# Inspect the top candidates in detail
python characterize_candidates.py \
    --merge_candidates PMIND2026_cluster/figures/merge_split_prott5/merge_candidates.tsv \
    --split_candidates PMIND2026_cluster/figures/merge_split_prott5/split_candidates.tsv \
    --mcl_file         PMIND2026_cluster/diamond_thresh20_clusters_I20.txt \
    --umap_dir         PMIND2026_cluster/umap_output_prott5 \
    --go_pkl           PMIND2026_cluster/go_mappings/annot_go_mf_pruned.pkl \
    --out_dir          PMIND2026_cluster/figures/cluster_characterization \
    --top_n            50 \
    --k                10
```

### 4.2 Apply candidates

`--merge_mode transitive` merges whole union-find components. `--merge_mode clique` only merges a group if **every** pair in it meets `--min_jaccard`. Clique mode needs `--go_pkl`.

```bash
# HDBSCAN-based, merges + splits, clique-validated
python apply_merge_split.py \
    --mcl_file         PMIND2026_cluster/diamond_thresh20_clusters_I20.txt \
    --merge_candidates PMIND2026_cluster/figures/merge_split_prott5/merge_candidates.tsv \
    --split_candidates PMIND2026_cluster/figures/merge_split_prott5/split_candidates.tsv \
    --umap_dir         PMIND2026_cluster/umap_output_prott5 \
    --go_pkl           PMIND2026_cluster/go_mappings/annot_go_mf_pruned.pkl \
    --out_dir          PMIND2026_cluster/revised_clustering_prott5 \
    --min_jaccard      0.5 \
    --min_improvement  0.0 \
    --apply            both \
    --merge_mode       clique \
    --max_merged_size  10000

# Leiden-based (MCL baseline)
python apply_merge_split.py \
    --mcl_file         PMIND2026_cluster/diamond_thresh20_clusters_I20.txt \
    --merge_candidates leiden_merge_split/candidates_mcl_base/merge_candidates.tsv \
    --split_candidates leiden_merge_split/candidates_mcl_base/split_candidates.tsv \
    --umap_dir         leiden_merge_split \
    --out_dir          leiden_merge_split/revised \
    --min_jaccard      0.5 \
    --min_improvement  0.1 \
    --merge_mode       clique \
    --go_pkl           PMIND2026_cluster/go_mappings/annot_go_mf_pruned.pkl
```

Ablations to separate the effect of each operation (same inputs, change `--apply` / `--merge_mode`):

```bash
COMMON="--mcl_file PMIND2026_cluster/diamond_thresh20_clusters_I20.txt \
  --merge_candidates PMIND2026_cluster/figures/merge_split_prott5/merge_candidates.tsv \
  --split_candidates PMIND2026_cluster/figures/merge_split_prott5/split_candidates.tsv \
  --umap_dir PMIND2026_cluster/umap_output_prott5 \
  --out_dir  PMIND2026_cluster/revised_clustering_prott5 \
  --min_jaccard 0.5 --min_improvement 0.1"
GO="--go_pkl PMIND2026_cluster/go_mappings/annot_go_mf_pruned.pkl"

# 1. Splits only
python apply_merge_split.py $COMMON --apply splits_only
# 2. Merges only, transitive (original behaviour)
python apply_merge_split.py $COMMON --apply merges_only --merge_mode transitive
# 3. Merges only, clique-validated (conservative)
python apply_merge_split.py $COMMON $GO --apply merges_only --merge_mode clique
# 4. Both, clique-validated
python apply_merge_split.py $COMMON $GO --apply both --merge_mode clique
```

### 4.3 Evaluate

```bash
python evaluate_revised_clustering.py \
    --baseline  PMIND2026_cluster/diamond_thresh20_clusters_I20.txt \
    --revised   leiden_merge_split/revised/revised_clusters_both_clique.txt \
    --go_pkl    PMIND2026_cluster/go_mappings/annot_go_mf_pruned.pkl \
    --changelog leiden_merge_split/revised/changelog.tsv \
    --out_dir   leiden_merge_split/revised/
```

`--changelog` turns on a targeted before/after evaluation of only the clusters that were merged or split.

---

## 5. Pre-clustering approach (edge reweighting)

### 5.1 Static reweighting

`w = [(1 − α)·w_seq/200 + α·cos(u, v)] × 200`

```bash
python reweight_edges_static.py \
    --cosine_file PMIND2026_cluster/alignments/diamond_with_cosine_prott5.tsv \
    --out_dir     PMIND2026_cluster/reweighted_static_prott5 \
    --alpha       0.1 0.2 0.3 0.4 0.5

# Convert to MCL format
for a in 0.10 0.20 0.30 0.40 0.50; do
  mcxload -abc "PMIND2026_cluster/reweighted_static_prott5/reweighted_alpha${a}.abc" \
    --stream-mirror -stream-tf 'ceil(200)' \
    -o "PMIND2026_cluster/reweighted_static_prott5/reweighted_alpha${a}.mci" \
    -write-tab "PMIND2026_cluster/reweighted_static_prott5/reweighted_alpha${a}.tab" || break
done

# Run MCL (5 runs in parallel, 8 threads each)
cd PMIND2026_cluster/reweighted_static_prott5
for alpha in 0.10 0.20 0.30 0.40 0.50; do
    mcl reweighted_alpha${alpha}.mci -I 2.0 \
        -o clusters_reweighted_alpha${alpha}_I20.txt \
        -use-tab reweighted_alpha${alpha}.tab \
        -te 8 &
done
wait
echo "All MCL runs complete"
cd -
```

The MCL loop takes about 1 h per run. On a remote machine, start it inside `screen` or `tmux`.

### 5.2 Dynamic reweighting

Per-edge `α(d, e) = α_max · exp(−drop(A,B)/τ) · f_coh(A,B)`, where `drop(A,B)` is the drop in embedding cosine between species A and B, and `f_coh` penalises species whose embeddings cluster strongly by taxonomy.

```bash
# Step 1: precompute species geometry (species_geometry.npz + species_geometry.tsv)
#         500 proteins per species is enough for stable estimates
python reweight_edges_dynamic.py precompute \
    --h5_dir   PMIND2026_cluster/embeddings_prott5 \
    --out_dir  reweighted_dynamic_prott5 \
    --n_sample 500

# Step 2: reweight (grid over α_max × τ)
python reweight_edges_dynamic.py reweight \
    --cosine_file PMIND2026_cluster/alignments/diamond_with_cosine_prott5.tsv \
    --dist_matrix reweighted_dynamic_prott5/species_geometry.npz \
    --out_dir     reweighted_dynamic_prott5 \
    --alpha_max   0.3 0.4 0.5 \
    --tau         0.03 0.05 0.10

# Variant without the cohesion factor (copy species_geometry.npz into this folder first)
python reweight_dynamic_sans_conf.py reweight \
    --cosine_file PMIND2026_cluster/alignments/diamond_with_cosine_prott5.tsv \
    --dist_matrix reweighted_dynamic_prott5_noconf/species_geometry.npz \
    --out_dir     reweighted_dynamic_prott5_noconf \
    --alpha_max   0.3 0.5 \
    --tau         0.05 0.10
```

Then run `mcxload` + `mcl` on each `reweighted_dynamic_amax*_tau*.abc`, as in §3.2 / §5.1.

### 5.3 Leiden on (reweighted) SSN graphs

`cluster_leiden_abc.py` reads the same `.abc` edge lists as `mcxload`.

```bash
# Resolution sweep on one dynamic graph
python cluster_leiden_abc.py \
    --abc reweighted_dynamic_prott5/reweighted_dynamic_amax0.40_tau0.050.abc \
    --out_dir leiden_abc_reweighted \
    --resolutions 0.001 0.005 0.01 0.05 0.1 0.2 0.5 \
    --parallel 7 --n_iterations 2

# All 9 dynamic graphs at γ = 0.05, in parallel
ABC_DIR=reweighted_dynamic_prott5
OUT_BASE=leiden_dynamic_05
for am in 0.30 0.40 0.50; do
  for tau in 0.030 0.050 0.100; do
    python cluster_leiden_abc.py \
        --abc "${ABC_DIR}/reweighted_dynamic_amax${am}_tau${tau}.abc" \
        --out_dir "${OUT_BASE}/amax${am}_tau${tau}_res0.05" \
        --resolutions 0.05 \
        --n_iterations 2 &
  done
done
echo "Launched 9 jobs (res=0.05). Waiting..."
wait
echo "All done."

# Same at γ = 0.10
OUT_BASE=leiden_dynamic_01
for am in 0.30 0.40 0.50; do
  for tau in 0.030 0.050 0.100; do
    python cluster_leiden_abc.py \
        --abc "${ABC_DIR}/reweighted_dynamic_amax${am}_tau${tau}.abc" \
        --out_dir "${OUT_BASE}/amax${am}_tau${tau}_res0.10" \
        --resolutions 0.10 \
        --n_iterations 2 &
  done
done
echo "Launched 9 jobs (res=0.10). Waiting..."
wait
echo "All done."
```

Leiden on the plain SSN (folder `leiden_abc_baseline/`) uses the same script with the un-reweighted `diamond.abc`.

### 5.4 Evaluate reweighted clusterings

```bash
# Edge-weight distributions before / after reweighting
python plot_weight_distributions.py \
    --cosine_file  PMIND2026_cluster/alignments/diamond_with_cosine_prott5.tsv \
    --static_dir   PMIND2026_cluster/reweighted_static_prott5 \
    --dynamic_dir  reweighted_dynamic_prott5 \
    --out_dir      PMIND2026_cluster/figures/weight_distributions

# ARI / NMI / V-measure + GO-MF homogeneity vs baseline
python compare_reweighted_clusterings.py \
    --baseline    PMIND2026_cluster/diamond_thresh20_clusters_I20.txt \
    --static_dir  PMIND2026_cluster/reweighted_static_prott5 \
    --dynamic_dir reweighted_dynamic_prott5_noconf \
    --go_pkl      PMIND2026_cluster/go_mappings/annot_go_mf_pruned.pkl \
    --out_dir     PMIND2026_cluster/figures/report_figs \
    --max_ari_proteins 3000000

# Homogeneity-tier transitions of dark proteins (strong ≥ 0.8, weak < 0.5, NE = not evaluable)
python characterize_homogeneity_tiers.py \
    --baseline  PMIND2026_cluster/diamond_thresh20_clusters_I20.txt \
    --reweighted \
        reweighted_dynamic_prott5_noconf/clusters_reweighted_dynamic_amax0.30_tau0.050_I20.txt \
        reweighted_dynamic_prott5_noconf/clusters_reweighted_dynamic_amax0.30_tau0.100_I20.txt \
        reweighted_dynamic_prott5_noconf/clusters_reweighted_dynamic_amax0.50_tau0.050_I20.txt \
        reweighted_dynamic_prott5_noconf/clusters_reweighted_dynamic_amax0.50_tau0.100_I20.txt \
    --labels baseline a3t5 a3t10 a5t5 a5t10 \
    --go_pkl  PMIND2026_cluster/go_mappings/annot_go_mf_pruned.pkl \
    --go_obo  PMIND2026_cluster/go_mappings/go-basic.obo \
    --out_dir PMIND2026_cluster/figures/tier_characterization_noconf

# The same tier analysis works on post-clustering outputs
python characterize_homogeneity_tiers.py \
    --baseline  PMIND2026_cluster/diamond_thresh20_clusters_I20.txt \
    --reweighted \
        PMIND2026_cluster/revised_clustering_prott5/revised_clusters_both_clique.txt \
        PMIND2026_cluster/revised_clustering_prott5/revised_clusters_merges_only_clique.txt \
        PMIND2026_cluster/revised_clustering_prott5/revised_clusters_splits_only.txt \
    --labels baseline both_clique merges_only_clique splits_only \
    --go_pkl  PMIND2026_cluster/go_mappings/annot_go_mf_pruned.pkl \
    --go_obo  PMIND2026_cluster/go_mappings/go-basic.obo \
    --out_dir PMIND2026_cluster/figures/tier_characterization

```

---

## 6. Evaluation metrics (short reference)

- **GO-MF homogeneity H(C):** mean pairwise Jaccard between the GO-MF leaf-term sets of the annotated members of a cluster (needs ≥ 2 annotated members). The *filtered* variant only counts clusters where ≥ 30 % of members are annotated.
- **Cluster composition:** fully dark / mixed / fully annotated clusters.
- **Dark proteins receivable:** unannotated proteins in mixed clusters, i.e. candidates for annotation transfer.
- **Partition agreement:** ARI, NMI, homogeneity / completeness / V-measure between two clusterings, computed on their shared proteins.

## 7. Main findings

- Baseline: MCL with inflation 2.0 on the E-value < 1e-20 SSN gives the best homogeneity/coverage trade-off among MCL, PorthoMCL and OrthoFinder. Mean GO-MF homogeneity is 0.931, with 792k receivable dark proteins.
- Post-clustering: **splits** strongly improve the clusters they touch (+0.10 with HDBSCAN, +0.37 with Leiden), but there are too few of them to move the global mean. **Merges** slightly lower homogeneity.
- Pre-clustering: reweighting raises the number of receivable dark proteins to more than 1M with MCL, at a small cost in homogeneity. Dynamic reweighting stays closer to the baseline partition (ARI 0.84–0.87) and is more robust to parameter changes than static reweighting. **Leiden + static reweighting (α = 0.1)** gives the highest homogeneity seen (0.947, filtered 0.956).
