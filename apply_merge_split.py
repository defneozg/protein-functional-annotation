"""
Builds new clustering file by applying validated merge and split candidates to the baseline MCL clustering

Order of operations:
1- conflict check : check for any MCL cluster IDs appearing in both merge and split candidates; excluded
2- apply split
3- apply merge
4- find unions for merge chains: if A merges with B, and B merges with C, all three are merged into same cluster

Conflict resolution rule (--resolve_conflicts):
Split applied first, merge reevaluated at sub-group level. If a sub-group cannot be identified, cluster if left untouched
"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import pickle

SEP = "="*60

def _find(d: Path, *candidates: str) -> Path:
    for name in candidates:
        p = d / name
        if p.exists():
            return p
    return d / candidates[0]

def load_mcl_clusters_full(mcl_file: Path) -> dict[int, list[str]]:
    clusters = {}
    with open(mcl_file) as fh:
        for cid, line in enumerate(fh):
            members = [p for p in line.strip().split("\t") if p]
            if members:
                clusters[cid] = members
    return clusters

def load_merge_candidates(path: Path, min_jaccard: float) -> list[dict]:
    rows = []
    with open(path) as fh:
        header = fh.readline().strip().split("\t")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            row = dict(zip(header, parts))
            if float(row["go_jaccard_cross"]) >= min_jaccard:
                rows.append(row)
    return rows

def load_split_candidates(path: Path, min_improvement: float) -> list[dict]:
    rows = []
    with open(path) as fh:
        header = fh.readline().strip().split("\t")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            row = dict(zip(header, parts))
            if float(row["improvement"]) >= min_improvement:
                rows.append(row)
    return rows

def parse_sub_details(sub_details_str: str) -> list[tuple[int, int, int, float]]:
    """parse 'hdb:size:n_annot:homog'"""
    out = []
    for chunk in sub_details_str.split(";"):
        if not chunk:
            continue
        h, sz, na, ho = chunk.split(":")
        out.append((int(h), int(sz), int(na), int(ho)))
    return out

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            self.parent[rx] = ry

    def groups(self) -> dict:
        out = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return out

def load_go(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    return {k: set(v) if not isinstance(v, set) else v for k, v in data.items()}

def pair_go_jaccard(members_a: list[str], members_b: list[str], protein2go: dict, max_sample: int = 100, rng=None) -> float | None:
    """mean cross-cluster GO jaccard between two member lists"""
    if rng is None:
        rng = np.random.default_rng(0)
    ann_a = [protein2go[m] for m in members_a if m in protein2go]
    ann_b = [protein2go[m] for m in members_b if m in protein2go]
    if not ann_a or not ann_b:
        return None
    if len(ann_a) > max_sample:
        ann_a = list(rng.choice(ann_a, max_sample, replace=False))
    if len(ann_b) > max_sample:
        ann_b = list(rng.choice(ann_b, max_sample, replace=False))
    vals = []
    for gi in ann_a:
        for gj in ann_b:
            union = len(gi | gj)
            vals.append(len(gi & gj) / union if union > 0 else 0.0)
    return float(np.mean(vals)) if vals else None


def main():
    p = argparse.ArgumentParser(
        description="apply merge/split candidates to baseline MCL clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mcl_file", required=True)
    p.add_argument("--merge_candidates", required=True)
    p.add_argument("--split_candidates", required=True)
    p.add_argument("--umap_dir", required=True, help="Needed to resolve HDBSCAN sub-groups for split")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_jaccard", default=0.5, type=float, help="min go_jaccard_cross to accept merge candidates (default: 0.5)")
    p.add_argument("--min_improvement", type=float, default=0.1, help="min homogeneity improvement to accept split candidates (default: 0.1)")
    p.add_argument("--resolve_conflicts",action="store_true", help="resolve clusters appearing in both lists (default: leave them untouched and flag)")
    p.add_argument("--merge_mode", choices=["transitive", "clique"], default="transitive",
                   help="transitive: union-find, chains A-B + B-C into one group "
                        "even if A-C was never directly evaluated (default, matches "
                        "original behaviour). "
                        "clique: only merge a group if EVERY pair within it meets "
                        "--min_jaccard, re-checking pairs not in the candidate file "
                        "directly against the GO annotations. More conservative.")
    p.add_argument("--go_pkl", default=None, help="Required for --merge_mode clique (to re-check pairwise Jaccard for pairs not in merge_candidates.tsv)")
    p.add_argument("--apply", choices=["both", "splits_only", "merges_only"], default="both", help="Which operations to apply (default: both). Use splits_only or merges_only to isolate each effect for evaluation.")
    args = p.parse_args()

    if args.merge_mode == "clique" and args.apply != "splits_only" and not args.go_pkl:
        p.error("--merge_mode clique requires --go_pkl")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(SEP)
    print("Apply Merge / Split candidates")
    print(SEP)

    print(f"\n  Loading MCL clusters")
    clusters = load_mcl_clusters_full(Path(args.mcl_file))
    print(f"  {len(clusters):,} clusters loaded")

    print(f"  Loading merge candidates (jaccard >= {args.min_jaccard})")
    merges = load_merge_candidates(Path(args.merge_candidates), args.min_jaccard)
    print(f"  {len(merges):,} merge candidates accepted")

    print(f"  Loading split candidates (improvement >= {args.min_improvement})")
    splits = load_split_candidates(Path(args.split_candidates), args.min_improvement)
    print(f"  {len(splits):,} split candidates accepted")

    merge_cluster_ids = set()
    for row in merges:
        merge_cluster_ids.add(int(row["mcl_cluster_1"]))
        merge_cluster_ids.add(int(row["mcl_cluster_2"]))

    split_cluster_ids = {int(row["mcl_cluster"]) for row in splits}

    # conflict detection
    conflicts = merge_cluster_ids & split_cluster_ids
    print(f"\n  Conflict check:")
    print(f"  Clusters in merge list : {len(merge_cluster_ids):,}")
    print(f"  Clusters in split list : {len(split_cluster_ids):,}")
    print(f"  CONFLICTS (in both) : {len(conflicts):,}")
    if conflicts:
        print(f"  Conflicting cluster IDs : {sorted(conflicts)[:20]} {'...' if len(conflicts) > 20 else ''}")

    conflicts_path = out_dir / "conflicts.tsv"
    with open(conflicts_path, "w") as fh:
        fh.write("mcl_cluster\tin_merge\tin_split\n")
        for cid in sorted(conflicts):
            fh.write(f"{cid}\tyes\tyes\n")
    print(f"  -> {conflicts_path}")
    changelog = [] # list of (cluster_id(s), action, reason)

    # load HDBSCAN labels for split resolution
    umap_dir = Path(args.umap_dir)
    ids_path = _find(umap_dir, "umap_ids.txt", "all_ids.txt")
    hdb_path = _find(umap_dir, "hdbscan_labels.npy")
    all_ids = np.load(hdb_path)
    id2hdb = dict(zip(all_ids, all_ids))

    # apply splits
    new_clusters: dict = {}
    split_piece_map: dict[int, dict[int, object]] = {} # cluster_id -> {hdb_label: new_key}
    skip_set = set() if args.resolve_conflicts else conflicts

    print(f"\n  Applying splits")
    n_splits_applied = 0
    apply_splits = args.apply in ("both", "splits_only")
    apply_merges = args.apply in ("both", "merges_only")
    if not apply_splits:
        print(f"  [SKIPPED] --apply={args.apply} excludes splits")
    for row in (splits if apply_splits else []):
        cid = int(row["mcl_cluster"])
        if cid in skip_set:
            changelog.append((cid, "split_skipped", "conflict with merge list"))
            continue

        members = clusters.get(cid)
        if members is None:
            continue

        sub_groups = defaultdict(list)
        for pid in members:
            h = id2hdb.get(pid, -2)
            sub_groups[int(h)].append(pid)

        piece_keys = {}
        for h, group_members in sub_groups.items():
            new_key = (cid, h)
            new_clusters[new_key] = group_members
            piece_keys[h] = new_key

        split_piece_map[cid] = piece_keys
        n_splits_applied += 1
        changelog.append((cid, "split_applied", f"{len(sub_groups)} sub-groups: " + ",".join(f"hdb{h}(n={len(g)})" for h, g in sub_groups.items())))

    print(f"  {n_splits_applied:,} splits applied")

    # clusters not touched by splits
    for cid, members in clusters.items():
        if cid not in split_piece_map and cid not in skip_set:
            new_clusters[cid] = members
        elif cid in skip_set:
            new_clusters[cid] = members # untouched conflict cluster

    # apply merges via union find
    print(f"\n  Applying merges")
    uf = UnionFind()
    n_merges_applied = 0
    n_merges_skipped = 0

    if not apply_merges:
        print(f"  [SKIPPED] --apply={args.apply} excludes merges")

    for row in (merges if apply_merges else []):
        c1 = int(row["mcl_cluster_1"])
        c2 = int(row["mcl_cluster_2"])

        if c1 in skip_set or c2 in skip_set:
            changelog.append(((c1, c2), "merge_skipped", "conflict with split list"))
            n_merges_skipped += 1
            continue

        # resolve to current key(s) if either side was split
        hdb_id = int(row["hdbscan_cluster"])
        key1 = split_piece_map.get(c1, {}).get(hdb_id, c1) if c1 in split_piece_map else c1
        key2 = split_piece_map.get(c2, {}).get(hdb_id, c2) if c2 in split_piece_map else c2

        if key1 not in new_clusters or key2 not in new_clusters:
            changelog.append(((c1, c2), "merge_skipped", "resolved key not found in new_clusters"))
            n_merges_skipped += 1
            continue

        uf.union(key1, key2)
        n_merges_applied += 1
        changelog.append(((c1, c2), "merge_applied", f"jaccard={row['go_jaccard_cross']} via hdbscan_cluster={hdb_id}"))

    print(f"  {n_merges_applied:,} merges applied")
    print(f"  {n_merges_skipped:,} merges skipped")

    # clique validation for union find groups
    if apply_merges and args.merge_mode == "clique":
        print(f"\n  Clique validation: re-checking all intra-group pairs against min_jaccard = {args.min_jaccard}")
        protein2go = load_go(Path(args.go_pkl))
        rng_clique = np.random.default_rng(0)
        proposed_groups = uf.groups()
        n_groups_checked = 0
        n_groups_accepted = 0
        n_groups_rejected = 0
        n_groups_partial = 0

        uf_validated = UnionFind() # rebuild only with clique-validated merges
        for root, keys in proposed_groups.items():
            if len(keys) < 2:
                continue
            n_groups_checked += 1

            # check every pair in the group
            all_pairs_ok = True
            failing_pairs = []
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    ka, kb = keys[i], keys[j]
                    members_a = new_clusters.get(ka, [])
                    members_b = new_clusters.get(kb, [])
                    jacc = pair_go_jaccard(members_a, members_b, protein2go, rng=rng_clique)
                    if jacc is None:
                        continue
                    if jacc < args.min_jaccard:
                        all_pairs_ok = False
                        failing_pairs.append((ka, kb, jacc))

            if all_pairs_ok:
                for k in keys[1:]:
                    uf_validated.union(keys[0], k)
                n_groups_accepted += 1
                changelog.append((tuple(keys), "clique_accepted", f"all {len(keys)*(len(keys)-1)//2} intra-group pairs >= {args.min_jaccard}"))

            else:
                n_groups_rejected += 1
                changelog.append((tuple(keys), "clique_rejected", f"{len(failing_pairs)} pair(s) below {args.min_jaccard}: " + ";".join(f"{a}-{b}={j:.3f}" for a, b, j in failing_pairs[:5])))
                for r in merges:
                    c1, c2 = int(r["mcl_cluster_1"]), int(r["mcl_cluster_2"])
                    hdb_id = int(r["hdbscan_cluster"])
                    k1 = split_piece_map.get(c1, {}).get(hdb_id, c1) if c1 in split_piece_map else c1
                    k2 = split_piece_map.get(c2, {}).get(hdb_id, c2) if c2 in split_piece_map else c2
                    if k1 in keys and k2 in keys:
                        jacc_direct = pair_go_jaccard(new_clusters.get(k1, []), new_clusters.get(k2, []), protein2go, rng=rng_clique)
                        if jacc_direct is not None and jacc_direct >= args.min_jaccard:
                            uf_validated.union(k1, k2)
                            n_groups_partial += 1

        uf = uf_validated # validated structure for final collapse
        print(f"  Groups checked   : {n_groups_checked:,}")
        print(f"  Groups accepted  : {n_groups_accepted:,}  (all intra-pairs valid)")
        print(f"  Groups rejected  : {n_groups_rejected:,}  (kept only valid sub-pairs)")
        print(f"  Valid sub-pairs kept from rejected groups: {n_groups_partial:,}")


    # collapse union-find groups into final clusters
    print(f"\n  Collapsing merged groups")
    final_groups = defaultdict(list)
    for key, members in new_clusters.items():
        root = uf.find(key) if key in uf.parent else key
        final_groups[root].extend(members)

    print(f"  Final cluster count: {len(final_groups):,} (was {len(clusters):,} originally)")

    # write output
    suffix = f"{args.apply}_{args.merge_mode}" if args.apply != "splits_only" else args.apply
    out_path = out_dir / f"revised_clusters_{suffix}.txt"
    with open(out_path, "w") as fh:
        for members in final_groups.values():
            fh.write("\t".join(members) + "\n")
    print(f"  -> {out_path}")

    changelog_path = out_dir / "changelog.tsv"
    with open(changelog_path, "w") as fh:
        fh.write("cluster_id(s)\taction\treason\n")
        for cid, action, reason in changelog:
            fh.write(f"{cid}\t{action}\t{reason}\n")
    print(f"  -> {changelog_path}")

    # summary stats
    print(f"\n{SEP}")
    print("  Summary")
    print(SEP)
    sizes_before = [len(m) for m in clusters.values()]
    sizes_after = [len(m) for m in final_groups.values()]
    print(f"  Apply mode      : {args.apply}")
    print(f"  Merge mode      : {args.merge_mode}")
    print(f"  Clusters before : {len(clusters):,}  "
          f"(median size {np.median(sizes_before):.0f}, max {max(sizes_before)})")
    print(f"  Clusters after  : {len(final_groups):,}  "
          f"(median size {np.median(sizes_after):.0f}, max {max(sizes_after)})")
    print(f"  Splits applied  : {n_splits_applied:,}")
    print(f"  Merges applied  : {n_merges_applied:,}")
    print(f"  Merges skipped  : {n_merges_skipped:,}")
    print(f"  Conflicts found : {len(conflicts):,}  "
          f"({'resolved' if args.resolve_conflicts else 'left untouched'})")
    print(SEP)

if __name__ == "__main__":
    main()




