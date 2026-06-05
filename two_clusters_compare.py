import pickle
from collections import defaultdict

CLUSTER_FILE = "orthofinder_backup/Orthogroups/Orthogroups.txt"
MAPPING_DIR  = "PMIND2026_cluster/go_mappings"

# cluster indices to compare (0-based, from the near-perfect cluster analysis)
# cluster 85 = index 85, cluster 114 = index 114
TARGET_CLUSTERS = {85: "Cluster 85 (size 593)", 114: "Cluster 114 (size 524)"}

# load GO annotations for context
print("Loading GO annotations")
with open(f"{MAPPING_DIR}/annot_go_mf_pruned.pkl", "rb") as f:
    annot_go = pickle.load(f)

print("Loading clusters")
clusters = []
with open(CLUSTER_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        og_id, rest = line.split(":", 1)
        members = [m for m in rest.strip().split() if m]
        if len(members) >= 2:
            clusters.append((og_id, members))

print(f"  {len(clusters):,} clusters loaded")

# extract the two target clusters
target_data = {}
for idx, target_idx in enumerate(sorted(TARGET_CLUSTERS.keys())):
    og_id, members = clusters[target_idx]
    species = defaultdict(list)
    for m in members:
        if "|" in m:
            sp = m.split("|")[0]
            species[sp].append(m)
    target_data[target_idx] = {
        "og_id":   og_id,
        "label":   TARGET_CLUSTERS[target_idx],
        "members": members,
        "species": dict(species),
        "annotated": [m for m in members if m in annot_go],
    }

c1 = target_data[85]
c2 = target_data[114]

sp1 = set(c1["species"].keys())
sp2 = set(c2["species"].keys())

shared_sp   = sp1 & sp2
only_in_c1  = sp1 - sp2
only_in_c2  = sp2 - sp1
jaccard_sp  = len(shared_sp) / len(sp1 | sp2) if sp1 | sp2 else 0.0

# protein overlap
members1 = set(c1["members"])
members2 = set(c2["members"])
shared_proteins = members1 & members2
jaccard_prot = len(shared_proteins) / len(members1 | members2) if members1 | members2 else 0.0

print(f"\n{'='*70}")
print(f"COMPARISON: {c1['label']} vs {c2['label']}")
print(f"{'='*70}")

print(f"\n--- Cluster 85 ({c1['og_id']}) ---")
print(f"  Total proteins : {len(c1['members'])}")
print(f"  Annotated      : {len(c1['annotated'])}")
print(f"  Unique species : {len(sp1)}")

print(f"\n--- Cluster 114 ({c2['og_id']}) ---")
print(f"  Total proteins : {len(c2['members'])}")
print(f"  Annotated      : {len(c2['annotated'])}")
print(f"  Unique species : {len(sp2)}")

print(f"\n--- Species overlap ---")
print(f"  Shared species       : {len(shared_sp)}")
print(f"  Only in Cluster 85   : {len(only_in_c1)}")
print(f"  Only in Cluster 114  : {len(only_in_c2)}")
print(f"  Jaccard (species)    : {jaccard_sp:.3f}")

print(f"\n--- Protein overlap ---")
print(f"  Shared proteins      : {len(shared_proteins)}")
print(f"  Jaccard (proteins)   : {jaccard_prot:.3f}")

print(f"\n--- Shared species detail ---")
print(f"  (species present in BOTH clusters, with protein counts)\n")
print(f"  {'Species':<55} {'C85 proteins':>12} {'C114 proteins':>13}")
print(f"  {'-'*82}")
for sp in sorted(shared_sp):
    n1 = len(c1["species"].get(sp, []))
    n2 = len(c2["species"].get(sp, []))
    print(f"  {sp:<55} {n1:>12} {n2:>13}")

print(f"\n--- Species only in Cluster 85 ---")
for sp in sorted(only_in_c1):
    n = len(c1["species"].get(sp, []))
    print(f"  {sp:<55} {n:>5} proteins")

print(f"\n--- Species only in Cluster 114 ---")
for sp in sorted(only_in_c2):
    n = len(c2["species"].get(sp, []))
    print(f"  {sp:<55} {n:>5} proteins")

# proteins per species breakdown
print(f"\n--- Proteins per species (both clusters) ---")
all_sp_union = sorted(sp1 | sp2)
print(f"\n  {'Species':<55} {'C85':>6} {'C114':>6}")
print(f"  {'-'*70}")
for sp in all_sp_union:
    n1 = len(c1["species"].get(sp, []))
    n2 = len(c2["species"].get(sp, []))
    marker = " <-- SHARED" if sp in shared_sp else ""
    print(f"  {sp:<55} {n1:>6} {n2:>6}{marker}")

# interpretation
print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")
if jaccard_prot > 0:
    print(f"  WARNING: {len(shared_proteins)} proteins appear in BOTH clusters!")
    print(f"  These are duplicate entries in OrthoFinder output.")
else:
    print(f"  Protein overlap: NONE, no protein belongs to both clusters.")

if jaccard_sp >= 0.8:
    print(f"  Species overlap is HIGH (Jaccard={jaccard_sp:.3f}).")
    print(f"  Both clusters likely represent paralogous gene families")
    print(f"  within the same set of species, possibly an ancient duplication")
    print(f"  that OrthoFinder correctly split into two ortholog groups.")
elif jaccard_sp >= 0.4:
    print(f"  Species overlap is MODERATE (Jaccard={jaccard_sp:.3f}).")
    print(f"  Partial species complement, clusters may cover different")
    print(f"  subsets of the dinoflagellate tree.")
else:
    print(f"  Species overlap is LOW (Jaccard={jaccard_sp:.3f}).")
    print(f"  Clusters largely cover DIFFERENT species, likely not paralogs")
    print(f"  of each other but rather the same function in different lineages.")