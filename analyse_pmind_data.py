import os, glob, collections
from pathlib import Path

BASE = Path("./PMIND2026_cluster")
FASTA_DIR = BASE / "darkdino_fasta"
ALIGN_DIR = BASE / "alignments"
ANNOT_DIR = BASE / "annotations" / "darkdino_annotation"
SEP = "=" * 25

# helper functions
def section(title):
    print(f"\n{SEP}\n {title}\n{SEP}")

def parse_fasta(filepath):
    """yield (header, sequence) tuples"""
    header, seq_parts = None, []
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header, seq_parts = line[1:], []
            else:
                seq_parts.append(line)
    if header is not None:
        yield header, "".join(seq_parts)

# FASTA analysis
section("1. FASTA files")
fasta_files = sorted(FASTA_DIR.glob("*.fasta")) + sorted(FASTA_DIR.glob("*.fa")) + sorted(FASTA_DIR.glob("*.fna"))
print(f"Files found: {len(fasta_files)}")

all_lengths = []
per_file_counts = {}
fasta_ids = set()

for fp in fasta_files:
    file_lengths = []
    with open(fp) as f:
        current_id = None
        current_len = 0
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id is not None:
                    file_lengths.append(current_len)
                    fasta_ids.add(current_id)
                current_id = line[1:].split()[0] # first token after >
                current_len = 0
            else:
                current_len += len(line) # accumulate residues
        if current_id is not None:
            file_lengths.append(current_len)
            fasta_ids.add(current_id)
    per_file_counts[fp.name] = len(file_lengths)
    all_lengths.extend(file_lengths)

total_seqs = len(all_lengths)
mean_len = sum(all_lengths) / total_seqs if total_seqs else 0
min_len = min(all_lengths) if all_lengths else 0
max_len = max(all_lengths) if all_lengths else 0
sorted_lens = sorted(all_lengths)
median_len = sorted_lens[total_seqs // 2] if total_seqs else 0

print(f"Total sequences found: {total_seqs:,}")
print(f"Mean sequence length (aa): {mean_len:,}")
print(f"Median sequence length (aa): {median_len:,}")
print(f"Max sequence length (aa): {max_len:,}")
print(f"Min sequence length (aa): {min_len:,}")
print(f"Total residues: {sum(all_lengths):,}")

# per-file breakdown
counts = list(per_file_counts.values())
print(f"\n Sequences per file : Min : {min(counts)}  |  Max : {max(counts)}  |  Mean : {sum(counts)/len(counts):.1f}")

# files with fewest and most seqs
top3 = sorted(per_file_counts.items(), key=lambda x: -x[1])[:3]
bottom3 = sorted(per_file_counts.items(), key=lambda x: x[1])[:3]
print(f"Top 3 largest files : {top3}")
print(f"Top 3 smallest files : {bottom3}")

# len distribution buckets
buckets = collections.Counter()
for l in all_lengths:
    if l < 100 : buckets["<100 aa"] += 1
    elif l < 300 : buckets["100-299 aa"] += 1
    elif l < 600 : buckets["300-599 aa"] += 1
    elif l < 1000 : buckets["600-999 aa"] += 1
    else: buckets[">=1000 aa"] += 1
print(f"Length distribution :")
for k in ["<100 aa", "100-299 aa", "300-599 aa", "600-999 aa", ">=1000 aa"]:
    pct = 100 * buckets[k] / total_seqs if total_seqs else 0
    print(f"  {k:<10}: {buckets[k]>6,} ({pct:.1f}%)")

# Alignment files
section("2. Alignment files")
# query_id query_len qstart qend subject_id subject_len sstart send aln_len pident query_cov bitscore evalue raw_score
COL = dict(query_id = 0, query_len = 1, q_start = 2, q_end = 3, subject_id = 4, subject_len = 5, s_start = 6, s_end = 7,
           aln_len = 8, pident = 9, query_cov = 10, bitscore = 11, evalue = 12)
align_files = sorted(ALIGN_DIR.glob("*.tsv"))
print(f"Files found: {len(align_files)}")

for fp in align_files:
    # encoding fallback, handles UTF-8, BOM from windows created files
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(fp, encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue

    else:
        print(f"\n -- {fp.name} -> Could not decode, skipping")
        continue
    if not lines:
        print(f"\n -- {fp.name} -> EMPTY")
        continue

    n_lines = len(lines)
    n_cols = len(lines[0].strip().split("\t"))

    queries, subjects = set(), set()
    pidents, evals = [], []
    query_covs, bitscores = [], []
    hits_per_query = collections.Counter()

    for line in lines:
        p = line.strip().split("\t")
        if len(p) < 13:
            continue
        qid = p[COL["query_id"]]
        sid = p[COL["subject_id"]]
        queries.add(qid)
        subjects.add(sid)
        hits_per_query[qid] += 1

        try: pidents.append(float(p[COL["pident"]]))
        except ValueError: pass
        try: query_covs.append(float(p[COL["query_cov"]]))
        except ValueError: pass
        try: bitscores.append(float(p[COL["bitscore"]]))
        except ValueError: pass
        try:
            e = float(p[COL["evalue"]])
            evals.append(e)
        except ValueError: pass

    hpc = list(hits_per_query.values())

    print(f"\n -- {fp.name}")
    print(f"  Total lines: {n_lines:,}, (columns): {n_cols:,})")
    print(f"  Unique queries: {len(queries):,}")
    print(f"  Unique subjects: {len(subjects):,}")
    print(f"  Hits per query  : min={min(hpc)}  mean={sum(hpc) / len(hpc):.1f}  max={max(hpc)}")

    if pidents:
        print(f"  % Identity : min={min(pidents):.1f}  mean={sum(pidents) / len(pidents):.1f}  max={max(pidents):.1f}")
        id_buckets = {"<30%": 0, "30–50%": 0, "50–70%": 0, "70–90%": 0, "≥90%": 0}
        for v in pidents:
            if v < 30:
                id_buckets["<30%"] += 1
            elif v < 50:
                id_buckets["30–50%"] += 1
            elif v < 70:
                id_buckets["50–70%"] += 1
            elif v < 90:
                id_buckets["70–90%"] += 1
            else:
                id_buckets["≥90%"] += 1
        print(f"  Identity distribution:")
        for k, v in id_buckets.items():
            pct = 100 * v / len(pidents)
            bar = "█" * int(pct / 3)
            print(f"  {k:<8}: {v:>8,}  ({pct:5.1f}%)  {bar}")

    if query_covs:
        print(f"  Query coverage : min={min(query_covs):.1f}  mean={sum(query_covs) / len(query_covs):.1f}  max={max(query_covs):.1f}")

    if evals:
        nonzero = [e for e in evals if e > 0]
        if nonzero:
            print(f"  E-value : min={min(nonzero):.2e}  max={max(nonzero):.2e}")
        zeros = len(evals) - len(nonzero)
        if zeros:
            print(f"    ({zeros:,} zero e-values = perfect/near-perfect hits)")

    if bitscores:
        print(f"  Bitscore : min={min(bitscores):.0f}  mean={sum(bitscores) / len(bitscores):.0f}  max={max(bitscores):.0f}")

# Annotation files
section("3. Annotation files")
annot_files = sorted(ANNOT_DIR.glob("*.tsv"))
print(f"Files found: {len(annot_files)}")

all_proteins = set() # all annotated protein IDs
db_counts = collections.Counter() # enteries par db
domain_counts = collections.Counter() # enteries par domain id
protein_domains = collections.defaultdict(set) # protein : set of domain ids
protein_dbs = collections.defaultdict(set) # protein : set of db
total_rows = 0

for fp in annot_files:
    rows = 0
    with open(fp) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            prot_id = parts[0]
            database = parts[1]
            domain_id = parts[2]

            all_proteins.add(prot_id)
            db_counts[database] += 1
            domain_counts[domain_id] += 1
            protein_domains[prot_id].add(domain_id)
            protein_dbs[prot_id].add(database)
            rows += 1

    total_rows += rows
    print(f"  -- {fp.name} -> {rows:,} annotation rows")

# per protein domain counts
dom_per_prot = [len(v) for v in protein_domains.values()]
db_per_prot = [len(v) for v in protein_dbs.values()]

print(f"  Total annotation rows: {total_rows:,}")
print(f"  Unique proteins annotated: {len(all_proteins):,}")
print(f"  Unique domains/entry ids: {len(domain_counts):,}")
print(f"  Unique dbs : {len(db_counts):,}")
print(f"\n  Domains per protein - min:{min(dom_per_prot)}  mean:{sum(dom_per_prot)/len(dom_per_prot):.1f}  max:{max(dom_per_prot)}")
print(f"  DB per protein — min:{min(db_per_prot)}  mean:{sum(db_per_prot)/len(db_per_prot):.1f}  max:{max(db_per_prot)}")

print("Annotation counts per db :")
for db, cnt in db_counts.most_common():
    pct = 100 * cnt / total_rows if total_rows else 0
    bar = "█" * int(pct / 2)
    print(f"  {db:<20} {cnt:>8,}  ({pct:5.1f}%) {bar}")

print("\n Top 20 most frequent domain IDs :")
for dom, cnt in domain_counts.most_common(20):
    print(f"  {dom:<35}  x{cnt:,}")

# proteins with only one db vs multi-db coverage
single_db = sum(1 for v in protein_dbs.values() if len(v) == 1)
multi_db = len(all_proteins) - single_db
print(f"\n Proteins annotated by one db only : {single_db:,}")
print(f"\n Proteins annotated by multiple db : {multi_db:,}")

# cross analysis
section("4. Cross analysis between FASTA and annotations")
overlap = fasta_ids & all_proteins
unannotated = fasta_ids - all_proteins

print(f"Proteins in FASTA files : {len(fasta_ids):,}")
print(f"Proteins with annotation : {len(overlap):,} ({100*len(overlap)/len(fasta_ids):.2f}%)")
print(f"Proteins with no annotation (dark) : {len(unannotated):,} ({100*len(unannotated)/len(fasta_ids):.2f}%)")

# db covering FASTA proteins
fasta_db_coverage = collections.Counter()
for prot in overlap:
    for db in protein_dbs[prot]:
        fasta_db_coverage[db] += 1

print("Database coverage across FASTA proteins :")
for db, cnt in fasta_db_coverage.most_common():
    pct = 100 * cnt / len(fasta_ids)
    print(f"  {db:<20} {cnt:>6,} proteins ({pct:.1f}%)")

# pfam subset
pfam_proteins = {p for p, dbs in protein_dbs.items() if "Pfam" in dbs} & fasta_ids
print(f"\n FASTA proteins with Pfam : {len(pfam_proteins):,} ({100*len(pfam_proteins)/len(fasta_ids):.1f}%)")
