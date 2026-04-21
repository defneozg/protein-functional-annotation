from __future__ import annotations
import re
import pickle
from collections import defaultdict, deque, Counter

# parse go-basic.obo
def build_go_graph(obo_path: str) -> dict:
    """ 'go_leaves'  : set[str] — terms with NO children in the full GO graph
        'ancestors'  : dict[str, set[str]] — all transitive ancestors per term
        'children'   : dict[str, set[str]] — direct children per term
        'namespace'  : dict[str, str]
        'obsolete'   : set[str]
    Only is_a and part_of relations are followed (safe for annotation grouping).
    has_part is excluded"""
    go_id_re = re.compile(r'^id:\s+(GO:\d{7})')
    is_a_re  = re.compile(r'^is_a:\s+(GO:\d{7})')
    rel_re   = re.compile(r'^relationship:\s+part_of\s+(GO:\d{7})')
    ns_re    = re.compile(r'^namespace:\s+(\S+)')
    obs_re   = re.compile(r'^is_obsolete:\s+true')

    direct_parents: dict[str, set[str]] = defaultdict(set)
    go_ns:  dict[str, str] = {}
    go_obs: set[str]       = set()
    current_id: str | None = None
    in_term = False

    with open(obo_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            if line == "[Term]":
                in_term = True; current_id = None; continue
            if line.startswith("[") and line != "[Term]":
                in_term = False; continue
            if not in_term:
                continue
            m = go_id_re.match(line)
            if m: current_id = m.group(1); continue
            if current_id is None: continue
            m = ns_re.match(line)
            if m: go_ns[current_id] = m.group(1); continue
            if obs_re.match(line): go_obs.add(current_id); continue
            m = is_a_re.match(line)
            if m: direct_parents[current_id].add(m.group(1)); continue
            m = rel_re.match(line)
            if m: direct_parents[current_id].add(m.group(1)); continue

    # build children map (reverse of direct_parents)
    children: dict[str, set[str]] = defaultdict(set)
    for child, parents in direct_parents.items():
        for p in parents:
            children[p].add(child)

    all_terms = set(go_ns.keys())

    # GO-global leaves: terms that have no children at all; a term whose only children are obsolete is still considered a leaf
    go_leaves: set[str] = set()
    for term in all_terms:
        if term in go_obs:
            continue
        real_children = children.get(term, set()) - go_obs
        if not real_children:
            go_leaves.add(term)

    # transitive ancestor closure per term (BFS)
    ancestors: dict[str, set[str]] = {}
    for term in all_terms:
        visited: set[str] = set()
        q = list(direct_parents.get(term, set()))
        while q:
            p = q.pop()
            if p in visited:
                continue
            visited.add(p)
            q.extend(direct_parents.get(p, set()) - visited)
        ancestors[term] = visited

    n_mf_leaves = sum(
        1 for t in go_leaves
        if go_ns.get(t) == "molecular_function"
    )
    print(f"[go_prune] Parsed {len(all_terms):,} GO terms")
    print(f"           GO-global leaves (non-obsolete) : {len(go_leaves):,}")
    print(f"           MF leaves                       : {n_mf_leaves:,}")

    return {
        "go_leaves": go_leaves,
        "ancestors": ancestors,
        "children":  dict(children),
        "namespace": go_ns,
        "obsolete":  go_obs,
    }

# pruning functions
def keep_go_leaves(go_set: set[str], go_leaves: set[str],) -> set[str]:
    """From a protein's GO set, keep only terms that are leaves in the full GO graph (no children anywhere in GO).
    Returns the filtered set, which may be EMPTY if all terms are
    internal nodes — this is intentional (protein becomes dark).
    """
    return go_set & go_leaves

def prune_annotation_dict(annot: dict[str, set[str]], graph: dict, verbose: bool = True,) -> dict[str, set[str]]:
    """apply to all dict"""
    go_leaves = graph["go_leaves"]
    pruned: dict[str, set[str]] = {}
    total_before = 0
    total_after = 0
    dropped_proteins = 0

    for prot, go_set in annot.items():
        total_before += len(go_set)
        leaf_set = keep_go_leaves(go_set, go_leaves)
        if not leaf_set:
            dropped_proteins += 1
            continue
        total_after += len(leaf_set)
        pruned[prot] = leaf_set

    if verbose:
        n_in  = len(annot)
        n_out = len(pruned)
        removed_terms = total_before - total_after
        print(f"[go_prune] GO-leaf pruning results:")
        print(f"           Proteins in          : {n_in:,}")
        print(f"           Proteins out         : {n_out:,}")
        print(f"           Proteins turned dark : {dropped_proteins:,} "
              f"({100*dropped_proteins/max(n_in,1):.1f}%)")
        print(f"           Avg terms before     : {total_before/max(n_in,1):.2f}")
        print(f"           Avg terms after      : {total_after/max(n_out,1):.2f}")
        print(f"           Terms removed        : {removed_terms:,} "
              f"({100*removed_terms/max(total_before,1):.1f}%)")

    return pruned


# verifying with specific terms
def term_report(terms: list[tuple[str, str]], graph: dict,) -> None:
    """sanity-checking before running on the full dataset."""
    go_leaves = graph["go_leaves"]
    children  = graph["children"]
    go_obs    = graph["obsolete"]

    print(f"\n{'Term':<12} {'Leaf?':>6}  {'#Children':>10}  Name")
    print("-" * 65)
    for term, name in terms:
        is_leaf    = term in go_leaves
        n_children = len({c for c in children.get(term, set()) if c not in go_obs})
        flag = "YES" if is_leaf else "no"
        print(f"{term}  {flag:>6}  {n_children:>10}  {name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Prune GO annotation pkl to GO-global leaf terms only. "
            "Proteins with no leaf annotations are dropped (become dark)."
        )
    )
    parser.add_argument("--obo",    required=True,
                        help="Path to go-basic.obo")
    parser.add_argument("--input",  required=True,
                        help="Input .pkl  (dict[str, set[str]])")
    parser.add_argument("--output", required=True,
                        help="Output .pkl path")
    parser.add_argument("--report", action="store_true",
                        help="Print leaf/internal status of common noisy terms")
    args = parser.parse_args()

    graph = build_go_graph(args.obo)

    if args.report:
        term_report([
            ("GO:0005488", "binding"),
            ("GO:0003824", "catalytic activity"),
            ("GO:0005515", "protein binding"),
            ("GO:0016740", "transferase activity"),
            ("GO:0097159", "obsolete organic cyclic compound binding"),
            ("GO:0003676", "nucleic acid binding"),
            ("GO:0003723", "RNA binding"),
            ("GO:0046872", "metal ion binding"),
            ("GO:0043167", "ion binding"),
            ("GO:0008270", "zinc ion binding"),
            ("GO:0005524", "ATP binding"),
            ("GO:0004672", "protein kinase activity"),
        ], graph)

    with open(args.input, "rb") as f:
        annot = pickle.load(f)
    print(f"\n[go_prune] Loaded {len(annot):,} proteins from {args.input}")

    pruned = prune_annotation_dict(annot, graph)

    with open(args.output, "wb") as f:
        pickle.dump(pruned, f)
    print(f"[go_prune] Saved to {args.output}")