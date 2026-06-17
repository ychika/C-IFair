"""
Possible-parent-set enumeration for cluster CPDAGs.

Implements Algorithm 1 helper functions:
  - allow_A_center: admissibility check for orienting U -> A <- V
  - noDTS_cut:      directed-triangle-separation check
  - possible_parent_sets: enumerate all valid possible-parent sets of A

The CPDAG graph object G is expected to be a cloc.ClusterCPDAG instance
(or any object with compatible attributes: undirected, dir_succ,
adjacent, get_triple, directed_path_exists).
"""

from itertools import combinations


def allow_A_center(G, A, U, V):
    # Shielded triple: always admissible to orient U -> A <- V.
    if G.adjacent(U, V):
        return True
    # Unshielded triple: admissible iff the recorded type is 'never'.
    ann = G.get_triple(A, U, V)
    if ann is None:
        return False
    return ann.get("type") == "never"


def noDTS_cut(G, A, U, V):
    clause1_violation = G.adjacent(U, V) and (U in G.dir_succ[V])
    if clause1_violation:
        return False
    if G.directed_path_exists(V, U):
        return False
    return True


def possible_parent_sets(G, A):
    """Return all valid possible-parent sets of cluster A in CPDAG G.

    Returns a sorted list of tuples, each tuple being one candidate set.
    """
    sibs_set = set()
    und = getattr(G, 'undirected', set())
    for e in und:
        try:
            if A in e:
                u, v = tuple(e)
                sibs_set.add(v if u == A else u)
        except Exception:
            continue
    sibs = sorted(sibs_set)
    all_sets = []
    for r in range(len(sibs) + 1):
        for S in combinations(sibs, r):
            S = set(S)
            okC = True
            for U, V in combinations(S, 2):
                if not allow_A_center(G, A, U, V):
                    okC = False
                    break
            if not okC:
                continue
            okD = True
            for U in S:
                for V in set(sibs) - S:
                    if not noDTS_cut(G, A, U, V):
                        okD = False
                        break
                if not okD:
                    break
            if okD:
                all_sets.append(tuple(sorted(S)))
    return sorted(all_sets)


# ---------------------------------------------------------------------------
# Demo examples (use cloc.ClusterCPDAG directly)
# ---------------------------------------------------------------------------

def build_example1():
    from cloc import ClusterCPDAG
    clusters = ["X", "Y", "Z", "W", "R", "Q"]
    G = ClusterCPDAG(clusters)
    G.add_undirected("X", "Z")
    G.add_undirected("Z", "Y")
    G.add_directed("Z", "W")
    G.add_directed("X", "R")
    G.add_directed("Y", "R")
    G.add_directed("R", "Q")
    for center, a, b in [
        ("Z", "X", "W"), ("Z", "Y", "W"),
        ("X", "Z", "R"), ("Y", "Z", "R"),
        ("X", "R", "Q"), ("Y", "R", "Q"),
    ]:
        G.add_triple(center, a, b, "marginal")
    G.add_triple("R", "X", "Y", "conditional")
    G.add_triple("Z", "X", "Y", "never", conn_mark={"W"})
    return G


def build_example2():
    G = build_example1()
    G.add_triple("Z", "X", "Y", "never", conn_mark=None)
    return G
