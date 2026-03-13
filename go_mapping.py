"""
Pfam, PRINTS, ProSite, SMART, TIGRFAM, PIRSF
→ direct *2go files from geneontology.org

PANTHER
→ ftp.pantherdb.org/hmm_classifications/current_release/
→ PTHR ID → GO terms directly, no InterPro needed

Gene3D, SUPERFAMILY, CDD
→ interpro2go via InterPro REST API (signature → IPR → GO)
"""

import glob, re, os, json, time, gzip, urllib.request, urllib.parse
from collections import defaultdict
import subprocess
import pickle
from collections import Counter

ANNOT_DIR = "PMIND2026_cluster/annotations/darkdino_annotation"
MAPPING_DIR = "PMIND2026_cluster/go_mappings"
os.makedirs(MAPPING_DIR, exist_ok=True)

EXCLUDE_DBS = {"MobiDBLite", "Phobius", "TMHMM", "Coils"}
EXCLUDE_DOMAIN_IDS = {
    "mobidb-lite", "TMhelix", "TRANSMEMBRANE",
    "NON_CYTOPLASMIC_DOMAIN", "CYTOPLASMIC_DOMAIN",
    "SIGNAL_PEPTIDE", "SIGNAL_PEPTIDE_C_REGION",
    "SIGNAL_PEPTIDE_H_REGION", "SIGNAL_PEPTIDE_N_REGION", "Coil"
}

GO_BASE = "https://current.geneontology.org/ontology/external2go/"
PANTHER_HMM = "https://data.pantherdb.org/ftp/hmm_classifications/current_release/PANTHER_GO_SLIM_MF_molecular_function"
# full GO classification file
PANTHER_CLASSIFICATIONS_URL = "https://data.pantherdb.org/ftp/hmm_classifications/current_release/"

DIRECT_DB_MAP = {
    "Pfam"            : ("pfam2go",    "Pfam"),
    "PRINTS"          : ("prints2go",  "PRINTS"),
    "ProSiteProfiles" : ("prosite2go", "PROSITE"),
    "ProSitePatterns" : ("prosite2go", "PROSITE"),
    "SMART"           : ("smart2go",   "SMART"),
    "PIRSF"           : ("pirsf2go",   "PIRSF"),
    "Hamap"           : ("hamap2go",   "HAMAP"),
}

def download_once(url, dest, valid_starts=(b"!",)):
    if not os.path.exists(dest):
        print(f"  Downloading {url}")
        result = subprocess.run(
            [
                "curl", "-L", "-f",
                "--silent", "--show-error",
                "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
                "-H", "Accept: text/plain,*/*",
                "-H", "Accept-Language: en-US,en;q=0.9",
                "-H", "Referer: https://geneontology.org/",
                url, "-o", dest
            ],
            capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"curl failed for {url}:\n{result.stderr.decode()}"
            )
        with open(dest, "rb") as fh:
            head = fh.read(10)
        if not any(head.startswith(s) for s in valid_starts):
            os.remove(dest)
            raise ValueError(
                f"Downloaded file failed sanity check.\n"
                f"First bytes: {head!r}\n"
                f"URL: {url}"
            )
    else:
        print(f"  Cached: {os.path.basename(dest)}")

"""print("Testing download...")
test_dest = os.path.join(MAPPING_DIR, "_test.txt")
if os.path.exists(test_dest):
    os.remove(test_dest)
download_once("https://current.geneontology.org/ontology/external2go/pfam2go", test_dest)
with open(test_dest) as fh:
    lines = [l for l in fh if not l.startswith("!") and l.strip()]
print(f"  Test: {len(lines)} non-comment lines in pfam2go")
os.remove(test_dest)

with open(os.path.join(MAPPING_DIR, "pfam2go"), "rb") as fh:
    print(repr(fh.read(200)))"""

def parse_x2go(path, prefix):
    mapping = defaultdict(set)
    go_re = re.compile(r'GO:(\d{7})')
    prefix_colon = prefix + ":"
    print(f"    DEBUG: looking for prefix '{prefix_colon}' in {os.path.basename(path)}")
    matched = 0
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            if not line.startswith(prefix_colon):
                if matched == 0:  # print first non-comment non-matching line
                    print(f"    DEBUG: first non-matching line: {repr(line[:80])}")
                continue
            matched += 1
            dom_id = line[len(prefix_colon):].split()[0]
            for t in go_re.findall(line):
                mapping[dom_id].add(f"GO:{t}")
    print(f"    DEBUG: matched {matched} lines")
    return mapping

go_re = re.compile(r'GO:(\d{7})')

# 1) direct *2go files
print("="*60)
print("Step 1 : Direct *2go files")

domain2go = defaultdict(set)
seen_files = set()
for db, (fname, prefix) in DIRECT_DB_MAP.items():
    dest = os.path.join(MAPPING_DIR, fname)
    if fname not in seen_files:
        download_once(GO_BASE + fname, dest)
        seen_files.add(fname)
    m = parse_x2go(dest, prefix)
    for dom_id, go_set in m.items():
        domain2go[dom_id].update(go_set)
    print(f"  {db:<22}: {len(m):,} entries")

pfam_keys = [k for k in domain2go if k.startswith("PF")]
smart_keys = [k for k in domain2go if k.startswith("SM")]

# 2) PANTHER native GO mapping
print("="*60)
print("Step 2 : PANTHER native GO mapping")
# Col 0: PTHR ID (e.g. PTHR12345 or PTHR12345:SF6)
# Col 2: MF GO terms (pipe-separated)
# Col 3: BP GO terms (pipe-separated)
# Col 4: CC GO terms (pipe-separated)

PANTHER_BASE = "http://data.pantherdb.org/ftp/hmm_classifications/current_release/"

#  actual filename (it changes with each version e.g. PANTHER19.0_HMM_classifications)
panther_index_dest = os.path.join(MAPPING_DIR, "panther_index.html")
if not os.path.exists(panther_index_dest):
    result = subprocess.run(
        ["curl", "-L", "-f", "--silent", "--show-error",
         "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
         PANTHER_BASE, "-o", panther_index_dest],
        capture_output=True
    )

with open(panther_index_dest) as fh:
    index_content = fh.read()

# extract filename from directory listing
import re as _re
panther_fname_match = _re.search(r'(PANTHER[\d.]+_HMM_classifications)', index_content)
if not panther_fname_match:
    raise RuntimeError(
        f"Could not find PANTHER HMM classifications filename in index.\n"
        f"Index content: {index_content[:500]}"
    )
panther_fname = panther_fname_match.group(1)
print(f"  Found file: {panther_fname}")

panther_dest = os.path.join(MAPPING_DIR, panther_fname)
download_once(
    PANTHER_BASE + panther_fname,
    panther_dest,
    valid_starts=(b"PTHR", b"#")
)

panther2go = defaultdict(set)
n_with_go = 0

with open(panther_dest, encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        if line.startswith("#") or not line.strip():
            continue
        cols = line.strip().split("\t")
        if len(cols) < 3:
            continue
        pthr_id   = cols[0].strip()
        family_id = pthr_id.split(":")[0]

        go_terms = set(
            f"GO:{t}"
            for col_idx in [2, 3, 4]
            if col_idx < len(cols)
            for t in go_re.findall(cols[col_idx])
        )
        if go_terms:
            panther2go[pthr_id].update(go_terms)
            panther2go[family_id].update(go_terms)
            n_with_go += 1

print(f"  Rows with GO terms          : {n_with_go:,}")
print(f"  Unique PANTHER keys with GO : {len(panther2go):,}")

# merge into domain2go
for pthr_id, go_set in panther2go.items():
    domain2go[pthr_id].update(go_set)

print(f"domain2go after PANTHER: {len(domain2go)} keys")
print(f"PTHR46769: {len(domain2go.get('PTHR46769', set()))} GO terms")

# 3) interpro2go for Gene3D, SUPERFAMILY, CDD
print("\n" + "="*60)
print("Step 3 : InterPro bridge for Gene3D, SUPERFAMILY, CDD and SFLD")

interpro2go_dest = os.path.join(MAPPING_DIR, "interpro2go")
download_once(GO_BASE + "interpro2go", interpro2go_dest)

ipr2go = defaultdict(set)
with open(interpro2go_dest, encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("!") or not line.strip():
            continue
        if not line.startswith("InterPro"):
            continue
        ipr_id = line.split()[0].replace("InterPro:", "")
        for t in go_re.findall(line):
            ipr2go[ipr_id].add(f"GO:{t}")
print(f"  Loaded {len(ipr2go):,} IPR -> GO mappings")

BRIDGE_DBS = {"Gene3D", "SUPERFAMILY", "CDD", "HAMAP", "SFLD", "NCBIfam", "TIGRFAM"}
bridge_ids = set()
for f in glob.glob(f"{ANNOT_DIR}/*.tsv"):
    for line in open(f, encoding="utf-8-sig", errors="ignore"):
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        _, db, dom_id = parts[0], parts[1], parts[2]
        if db in BRIDGE_DBS and db not in EXCLUDE_DBS and dom_id not in EXCLUDE_DOMAIN_IDS:
            bridge_ids.add((db, dom_id))

print(f"  Unique bridge-DB domain ids : {len(bridge_ids):,}")

# load or bridge cache
cache_path = os.path.join(MAPPING_DIR, "bridge_cache.json")
bridge_cache = {}
if os.path.exists(cache_path):
    with open(cache_path) as f:
        bridge_cache = json.load(f)

missing = [(db, dom) for db, dom in bridge_ids if dom not in bridge_cache]
print(f"  Not yet cached : {len(missing):,}")

def query_interpro(accession):
    url = (f"https://www.ebi.ac.uk/interpro/api/entry/interpro/?"
           f"member_db_acc={urllib.parse.quote(accession)}&format=json")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (research script; contact: defneozguven2001@gmail.com)"
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        go_terms = set()
        for entry in data.get("results", []):
            ipr_acc = entry["metadata"]["accession"]
            go_terms.update(ipr2go.get(ipr_acc, set()))
        return sorted(go_terms)
    except Exception:
        return []

if missing:
    print(f"  Querying InterPro API for {len(missing):,} IDs")
    for idx, (db, acc) in enumerate(missing):
        bridge_cache[acc] = query_interpro(acc)
        time.sleep(0.05)
        if (idx + 1) % 200 == 0:
            with open(cache_path, "w") as f:
                json.dump(bridge_cache, f)
            mapped = sum(1 for v in bridge_cache.values() if v)
            print(f"  {idx+1}/{len(missing)} — {mapped} mapped", end="\r")
    with open(cache_path, "w") as f:
        json.dump(bridge_cache, f)
    print()

# merge into domain2go
for acc, go_list in bridge_cache.items():
    if go_list:
        domain2go[acc].update(go_list)

bridge_mapped = sum(1 for _, dom in bridge_ids if domain2go.get(dom))
print(f"  Bridge IDs with >=1 GO term     : {bridge_mapped:,} / {len(bridge_ids):,} "
      f"({100*bridge_mapped/max(len(bridge_ids),1):.1f}%)")

# Quick sanity check after building domain2go
test_ids = ["G3DSA:3.40.800.20", "SSF103511", "PTHR46769", "PF00850", "SM00249"]
print("\nSanity check:")
for tid in test_ids:
    go = domain2go.get(tid, set())
    print(f"  {tid}: {len(go)} GO terms — {list(go)[:3]}")

# 4) build annot_go
print("\n" + "="*60)
print("Step 4 : Building protein -> GO dict")

annot_go = defaultdict(set)
db_stats = defaultdict(lambda: {"rows": 0, "mapped": 0})

for f in glob.glob(f"{ANNOT_DIR}/*.tsv"):
    for line in open(f, encoding="utf-8-sig", errors="ignore"):
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        prot, db, dom_id = parts[0], parts[1], parts[2]
        if db in EXCLUDE_DBS or dom_id in EXCLUDE_DOMAIN_IDS:
            continue
        db_stats[db]["rows"] += 1
        go_terms = domain2go.get(dom_id, set())
        if go_terms:
            annot_go[prot].update(go_terms)
            db_stats[db]["mapped"] += 1

print(f"\n{'Database':<25} {'Rows':>8} {'GO-mapped':>10} {'Coverage':>9}")
print("-" * 55)
for db, s in sorted(db_stats.items(), key=lambda x: -x[1]["rows"]):
    pct = 100 * s["mapped"] / s["rows"] if s["rows"] else 0
    print(f"{db:<25} {s['rows']:>8,} {s['mapped']:>10,} {pct:>8.1f}%")

print(f"\nProteins with ≥1 GO term : {len(annot_go):,}")
mean_go = sum(len(v) for v in annot_go.values()) / max(len(annot_go), 1)
print(f"Mean GO terms per protein: {mean_go:.1f}")

with open(os.path.join(MAPPING_DIR, "domain2go.pkl"), "wb") as f:
    pickle.dump(dict(domain2go), f)
with open(os.path.join(MAPPING_DIR, "annot_go.pkl"), "wb") as f:
    pickle.dump(dict(annot_go), f)
print(f"\nSaved domain2go and annot_go to {MAPPING_DIR}/")

annot_go_path = os.path.join(MAPPING_DIR, "annot_go.tsv")
with open(annot_go_path, "w") as f:
    for prot, go_set in annot_go.items():
        f.write(f"{prot}\t{';'.join(sorted(go_set))}\n")
print(f"Saved annot_go TSV: {annot_go_path}")

term_counts = Counter(len(v) for v in annot_go.values())
for n in [1, 2, 5, 10, 20, 50, 100]:
    above = sum(c for k, c in term_counts.items() if k >= n)
    print(f"  Proteins with >= {n:3d} GO terms: {above:,}")

# extra analysis
print("\n" + "="*60)
print("Step 5 : Coverage diagnostics")

# fraction of unique domain ids that are mapped
print("\n--- Unique domain ID coverage (not row coverage) ---")
db_unique = defaultdict(lambda: {"total": set(), "mapped": set()})
for f in glob.glob(f"{ANNOT_DIR}/*.tsv"):
    for line in open(f, encoding="utf-8-sig", errors="ignore"):
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        _, db, dom_id = parts[0], parts[1], parts[2]
        if db in EXCLUDE_DBS or dom_id in EXCLUDE_DOMAIN_IDS:
            continue
        db_unique[db]["total"].add(dom_id)
        if domain2go.get(dom_id):
            db_unique[db]["mapped"].add(dom_id)

print(f"{'Database':<25} {'Unique IDs':>10} {'Mapped':>8} {'Coverage':>9}")
print("-" * 55)
for db, s in sorted(db_unique.items(), key=lambda x: -len(x[1]["total"])):
    total, mapped = len(s["total"]), len(s["mapped"])
    pct = 100 * mapped / total if total else 0
    print(f"{db:<25} {total:>10,} {mapped:>8,} {pct:>8.1f}%")

# protein level coverage
total_proteins = set()
for f in glob.glob(f"{ANNOT_DIR}/*.tsv"):
    for line in open(f, encoding="utf-8-sig", errors="ignore"):
        parts = line.strip().split("\t")
        if len(parts) < 1:
            continue
        total_proteins.add(parts[0])

print("""--- Protein-level coverage ---""")
print(f"Total proteins in dataset : {len(total_proteins):,}")
print(f"Proteins with >= 1 GO term: {len(annot_go):,}  ({100*len(annot_go)/len(total_proteins):.1f}%)")
print(f"Proteins with 0 GO terms  : {len(total_proteins) - len(annot_go):,}")

# GO term distribution
print("""\n--- GO term count distribution (per protein) ---""")
thresholds = [1, 2, 5, 10, 20, 50]
for n in thresholds:
    above = sum(1 for v in annot_go.values() if len(v) >= n)
    print(f"  >= {n:3d} GO terms: {above:>8,}  ({100*above/max(len(annot_go),1):.1f}% of mapped proteins)")

# most frequent GO terms
print(f"\n--- Top 15 most frequent GO terms ---")
go_freq = Counter(t for go_set in annot_go.values() for t in go_set)
for term, count in go_freq.most_common(15):
    print(f"  {term}  {count:>8,} proteins")

# how many terms are MF vs BP vs CC
print("\n--- GO namespace breakdown (requires go-basic.obo) ---")
obo_dest = os.path.join(MAPPING_DIR, "go-basic.obo")
download_once("http://purl.obolibrary.org/obo/go/go-basic.obo", obo_dest, valid_starts=(b"format",b"!"))

# parse obo to get namespace per term
go_namespace = {}
current_id = None
with open(obo_dest) as fh:
    for line in fh:
        line = line.strip()
        if line.startswith("id: GO:"):
            current_id = line.split("id: ")[1].strip()
        elif line.startswith("namespace:") and current_id:
            go_namespace[current_id] = line.split("namespace: ")[1].strip()

ns_counts = Counter()
for go_set in annot_go.values():
    for t in go_set:
        ns = go_namespace.get(t, "unknown")
        ns_counts[ns] += 1

total_terms = sum(ns_counts.values())
print(f"  molecular_function : {ns_counts['molecular_function']:>10,}  ({100*ns_counts['molecular_function']/total_terms:.1f}%)")
print(f"  biological_process : {ns_counts['biological_process']:>10,}  ({100*ns_counts['biological_process']/total_terms:.1f}%)")
print(f"  cellular_component : {ns_counts['cellular_component']:>10,}  ({100*ns_counts['cellular_component']/total_terms:.1f}%)")

# per-database GO namespace profile
print("\n--- Per-database namespace profile ---")
db_ns = defaultdict(lambda: Counter())
for f in glob.glob(f"{ANNOT_DIR}/*.tsv"):
    for line in open(f, encoding="utf-8-sig", errors="ignore"):
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        _, db, dom_id = parts[0], parts[1], parts[2]
        if db in EXCLUDE_DBS or dom_id in EXCLUDE_DOMAIN_IDS:
            continue
        for t in domain2go.get(dom_id, set()):
            db_ns[db][go_namespace.get(t, "unknown")] += 1

print(f"{'Database':<25} {'MF%':>6} {'BP%':>6} {'CC%':>6} {'total terms':>12}")
print("-" * 60)
for db, ns in sorted(db_ns.items(), key=lambda x: -sum(x[1].values())):
    total = sum(ns.values())
    if total == 0:
        continue
    mf = 100*ns['molecular_function']/total
    bp = 100*ns['biological_process']/total
    cc = 100*ns['cellular_component']/total
    print(f"{db:<25} {mf:>5.1f}% {bp:>5.1f}% {cc:>5.1f}% {total:>12,}")

# Save MF-only annot_go for homogeneity analysis
print("\n--- Building MF-only annot_go_mf ---")
mf_terms = {t for t, ns in go_namespace.items() if ns == "molecular_function"}

annot_go_mf = defaultdict(set)
for prot, go_set in annot_go.items():
    mf = go_set & mf_terms
    if mf:
        annot_go_mf[prot] = mf

print(f"  Proteins with >= 1 MF GO term: {len(annot_go_mf):,}  ({100*len(annot_go_mf)/len(total_proteins):.1f}% of all proteins)")
mean_mf = sum(len(v) for v in annot_go_mf.values()) / max(len(annot_go_mf), 1)
print(f"  Mean MF GO terms per protein : {mean_mf:.1f}")

with open(os.path.join(MAPPING_DIR, "annot_go_mf.pkl"), "wb") as f:
    pickle.dump(dict(annot_go_mf), f)
print(f"  Saved annot_go_mf.pkl")

# is the inflation coming from a few generic MF terms?
mf_freq = Counter(t for go_set in annot_go_mf.values() for t in go_set)
print("Top 10 most frequent MF terms:")
for term, count in mf_freq.most_common(10):
    # look up term name
    term_name = "?"
    with open(obo_dest) as fh:
        found = False
        for line in fh:
            if line.strip() == f"id: {term}":
                found = True
            elif found and line.startswith("name:"):
                term_name = line.split("name: ")[1].strip()
                break
    print(f"  {term}  {term_name:<40} {count:>8,} proteins")

# Rebuild annot_go using only reliable functional databases
print("\n--- Rebuilding annot_go with functional DBs only ---")

FUNCTIONAL_DBS = {"Pfam", "PRINTS", "SMART", "ProSiteProfiles",
                  "ProSitePatterns", "PANTHER", "PIRSF", "Hamap"}

annot_go_clean = defaultdict(set)
db_stats_clean = defaultdict(lambda: {"rows": 0, "mapped": 0})

for f in glob.glob(f"{ANNOT_DIR}/*.tsv"):
    for line in open(f, encoding="utf-8-sig", errors="ignore"):
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        prot, db, dom_id = parts[0], parts[1], parts[2]
        if db not in FUNCTIONAL_DBS:
            continue
        if dom_id in EXCLUDE_DOMAIN_IDS:
            continue
        db_stats_clean[db]["rows"] += 1
        go_terms = domain2go.get(dom_id, set()) & mf_terms  # MF only
        if go_terms:
            annot_go_clean[prot].update(go_terms)
            db_stats_clean[db]["mapped"] += 1

print(f"\n{'Database':<25} {'Rows':>8} {'GO-mapped':>10} {'Coverage':>9}")
print("-" * 55)
for db, s in sorted(db_stats_clean.items(), key=lambda x: -x[1]["rows"]):
    pct = 100 * s["mapped"] / s["rows"] if s["rows"] else 0
    print(f"{db:<25} {s['rows']:>8,} {s['mapped']:>10,} {pct:>8.1f}%")

print(f"\nProteins with >= 1 MF GO term : {len(annot_go_clean):,}  ({100*len(annot_go_clean)/len(total_proteins):.1f}%)")
mean_mf = sum(len(v) for v in annot_go_clean.values()) / max(len(annot_go_clean), 1)
print(f"Mean MF GO terms per protein  : {mean_mf:.1f}")

# check top terms now
mf_freq_clean = Counter(t for go_set in annot_go_clean.values() for t in go_set)
print("\nTop 10 MF terms after cleaning:")
for term, count in mf_freq_clean.most_common(10):
    print(f"  {term}  {count:>8,} proteins")

# save
with open(os.path.join(MAPPING_DIR, "annot_go_clean.pkl"), "wb") as f:
    pickle.dump(dict(annot_go_clean), f)
print(f"\nSaved annot_go_clean.pkl")

print(annot_go_clean.get("Durusdinium_trenchii_EP00913|sequence06893", set()))

print(f"Proteins with >= 1 MF GO term : {len(annot_go_clean):,}")
mean_mf = sum(len(v) for v in annot_go_clean.values()) / max(len(annot_go_clean), 1)
print(f"Mean MF GO terms per protein  : {mean_mf:.1f}")
mf_freq_clean = Counter(t for go_set in annot_go_clean.values() for t in go_set)
print("\nTop 10 MF terms:")
for term, count in mf_freq_clean.most_common(10):
    print(f"  {term}  {count:>8,} proteins")