from collections import defaultdict, Counter
import glob

annot = defaultdict(set)
for f in glob.glob("PMIND2026_cluster/annotations/darkdino_annotation/*.tsv"):
    for line in open(f, encoding="utf-8-sig"):
        parts = line.strip().split("\t")
        if len(parts) >= 3 and parts[1] == "Pfam":  # we focus specifically on Pfam for functional annotation
            annot[parts[0]].add(parts[2])

# loading clusters
clusters = []
with open("PMIND2026_cluster/diamond_bitscore_clusters_I20.txt") as fh: # can change depending on the file we are focusing on
    for line in fh:
        members = line.strip().split("\t")
        if len(members) >= 2:
            clusters.append(members)

# homogeneity score per cluster, fraction of members sharing the most common annotation
scores = []
annotated_counts = []
for cluster in clusters:
    annotated = [m for m in cluster if m in annot]
    annotated_counts.append(len(annotated))
    if len(annotated) < 2:
        scores.append(None)
        continue

    all_domains = [] # collect all domain ids across annotated members
    for m in annotated:
        all_domains.extend(annot[m])
    if not all_domains:
        scores.append(None)
        continue
    top_domain, top_count = Counter(all_domains).most_common(1)[0]
    score = top_count / len(annotated) # fraction of annotated members that have the top domain
    scores.append(score)

valid = [s for s in scores if s is not None]
print(f"Clusters with more than 2 annotated members : {len(valid)}")
print(f"Mean homogeneity score : {sum(valid)/len(valid):.3f}")
print(f"Clusters with score >= 0.8 : {sum(1 for s in valid if s >= 0.8)}")
print(f"Clusters with score < 0.5 : {sum(1 for s in valid if s < 0.5)}")
