"""
Given an MCL cluster ID (or pair), print:
 - all protein ids
 - species composition
 - GO term composition
 - for pairs : which GO terms are shared vs unique to each side
"""

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

def load_go(pkl_path: Path) -> dict[str, set]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return {k: set(v) if not isinstance(v, set) else v for k, v in data.items()}

def load_mcl_clusters_full(mcl_file: Path) -> dict[int, list[str]]:
    clusters = {}
    with open(mcl_file) as fh:
        for cid, line in enumerate(fh):
            members = [p for p in line.strip().split("\t") if p]
            if members:
                clusters[cid] = members
    return clusters

def species_from_id(pid: str) -> str:
    return pid.split("|")[0]

def _find(d: Path, *candidates):
    for name in candidates:
        p = d / name
        if p.exists():
            return p
    return d / candidates[0]

def describe_cluster(cluster_id: int, members: list[str], protein2go: dict):
    print(f"\n{'='*60}")
    print(f"  MCL cluster: {cluster_id}")
    print(f"{'=' * 60}")
    print(f"  Total proteins : {len(members)}")
    species_counts = Counter(species_from_id(p) for p in members)
    print(f"  Species represented : {len(species_counts)}")
    for sp, n in species_counts.most_common(10):
        print(f"  {sp:<45} {n}")
    if len(species_counts) > 10:
        print(f" ... and {len(species_counts) - 10} more")

    annotated = [(p, protein2go[p]) for p in members if protein2go.get(p)]
    print(f"\n  GO-annotated proteins: {len(annotated)} / {len(members)}")

    term_counter = Counter()
    for _, go_set in annotated:
        term_counter.update(go_set)

    print(f"  Unique GO terms: {len(term_counter)}")
    print(f"\n  Top GO terms:")
    for term, n in term_counter.most_common(15):
        print(f"  {term:<15} n={n}")

    if annotated:
        print(f"\n  Sample annotated proteins")
        for p, go_set in annotated[:10]:
            terms_str = ", ".join(sorted(go_set)[:5])
            more = f" (+{len(go_set)-5} more)" if len(go_set) > 5 else ""
            print(f"    {p:<55} {terms_str}{more}")

    return term_counter, set(p for p, _ in annotated)

def compare_pair(cid1, members1, term_counter1, cid2, members2, term_counter2, protein2go):
    print(f"\n{'='*60}")
    print(f"  Comparison: Cluster {cid1} vs Cluster {cid2}")
    print(f"{'=' * 60}")

    terms1 = set(term_counter1.keys())
    terms2 = set(term_counter2.keys())
    shared = terms1 & terms2
    only1 = terms1 - terms2
    only2 = terms2 - terms1
    print(f"\n  GO terms in cluster {cid1} only : {len(only1)}")
    print(f"  GO terms in cluster {cid2} only : {len(only2)}")
    print(f"  Shared GO terms: {len(shared)}")

    if shared:
        print(f"\n  Shared GO terms (with counts in each cluster):")
        for term in sorted(shared, key=lambda t: term_counter1[t] + term_counter2[t], reverse=True):
            print(f"    {term:<15} cluster_{cid1}={term_counter1[term]:>3}  cluster_{cid2}={term_counter2[term]:>3}")

    if only1:
        print(f"\n  Terms only in cluster {cid1}:")
        for term, n in sorted(((t, term_counter1[t]) for t in only1), key=lambda x: -x[1])[:10]:
            print(f"    {term:<15} n={n}")

    if only2:
        print(f"\n  Terms only in cluster {cid2}:")
        for term, n in sorted(((t, term_counter2[t]) for t in only2), key=lambda x: -x[1])[:10]:
            print(f"    {term:<15} n={n}")

    # species overlap check: same species (paralog split) or different (cross-species merge)
    sp1 = set(species_from_id(p) for p in members1)
    sp2 = set(species_from_id(p) for p in members2)
    print(f"\n  Species in cluster {cid1} : {len(sp1)}")
    print(f"  Species in cluster {cid2} : {len(sp2)}")
    print(f" Species overlap : {len(sp1 & sp2)}")
    if sp1 & sp2:
        print(f"  Shared species: {sorted(sp1 & sp2)[:5]} {'...' if len(sp1 & sp2) > 5 else ''}")
    if sp1 - sp2:
        print(f"  Species only in {cid1}: {sorted(sp1 - sp2)[:5]} {'...' if len(sp1 - sp2) > 5 else ''}")
    if sp2 - sp1:
        print(f"  Species only in {cid2}: {sorted(sp2 - sp1)[:5]} {'...' if len(sp2 - sp1) > 5 else ''}")

def show_subgroups(cluster_id, members, umap_dir, protein2go):
    print(f"\n{'='*60}")
    print(f"  HDBSCAN sub-group breakdown for MCL cluster: {cluster_id}")
    print(f"{'='*60}")

    ids_path = _find(umap_dir, "umap_ids.txt", "all_ids.txt")
    hdb_path = _find(umap_dir, "hdbscan_labels.npy")

    all_ids = ids_path.read_text().splitlines()
    hdb_lines = np.load(hdb_path)
    id2hdb = dict(zip(all_ids, hdb_lines))

    sub_groups = defaultdict(list)
    for p in members:
        h = id2hdb.get(p, -2)
        sub_groups[int(h)].append(p)

    for hdb, group_members in sorted(sub_groups.items(), key=lambda x: -len(x[1])):
        ann = [(p, protein2go[p]) for p in group_members if protein2go.get(p)]
        term_counter = Counter()
        for _, go_set in ann:
            term_counter.update(go_set)

        label = f"HDBSCAN cluster {hdb}" if hdb >= 0 else \
                ("noise (-1)" if hdb == -1 else "not in embedding set")
        print(f"\n  {label}: {len(group_members)} proteins,  {len(ann)} annotated")
        if term_counter:
            top3 = term_counter.most_common(3)
            print(f"    Top GO terms: " + ", ".join(f"{t}(n={n})" for t, n in top3))
        species_here = Counter(species_from_id(p) for p in group_members)
        top_sp = species_here.most_common(3)
        print(f"    Top species : " + ", ".join(f"{s}(n={n})" for s, n in top_sp))

def main():
    p = argparse.ArgumentParser(
        description="Inspect MCL cluster(s) for merge/split case studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mcl_file", required=True)
    p.add_argument("--go_pkl", required=True)
    p.add_argument("--cluster", type=int, nargs="+", required=True, help="One cluster ID to inspect or two for pairwise comparison")
    p.add_argument("--umap_dir", default=None, help="Required if --show_subgroups is set")
    p.add_argument("--show_subgroups", action="store_true", help="Show HDBSCAN sub-group breakdown (for split candidates)")
    args = p.parse_args()

    print("Loading MCL clusters")
    clusters = load_mcl_clusters_full(Path(args.mcl_file))
    print("Loading GO annotations")
    protein2go = load_go(Path(args.go_pkl))

    if len(args.cluster) == 1:
        cid = args.cluster[0]
        members = clusters.get(cid)
        if members is None:
            print(f"[ERROR] No cluster {cid} in MCL file")
            return
        term_counter, _ = describe_cluster(cid, members, protein2go)

        if args.show_subgroups:
            if not args.umap_dir:
                print("[ERROR] -umap_dir required for --show_subgroups")
                return
            show_subgroups(cid, members, Path(args.umap_dir), protein2go)

    elif len(args.cluster) == 2:
        cid1, cid2 = args.cluster
        members1 = clusters.get(cid1)
        members2 = clusters.get(cid2)
        if members1 is None or members2 is None:
            print(f"[ERROR] one or both clusters not found in MCL file")
            return

        term_counter1, _ = describe_cluster(cid1, members1, protein2go)
        term_counter2, _ = describe_cluster(cid2, members2, protein2go)
        compare_pair(cid1, members1, term_counter2, cid2, members2, term_counter2, protein2go)

    else:
        print("[ERROR] --cluster takes 1 or 2 IDs")

if __name__ == "__main__":
    main()