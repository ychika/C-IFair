# cloc.py
# CLOC (Algorithm 1) — cluster-level causal discovery.
# Reference: "Causal Discovery over Clusters of Variables in Markovian Systems" (Anand et al.)
#
# Two CI back-ends are provided with the same interface (test(Xc, Yc, Sc) -> bool):
#   ClusterCI           — data-based Fisher-Z / chi-square test (default)
#   TrueGraphClusterCI  — oracle d-separation test from a known variable-level DAG
#
# Usage:
#   # Data-based (default):
#   P = cloc_learn(data, clusters, var_index_by_cluster, ...)
#
#   # Oracle mode — pass a TrueGraphClusterCI as ci_tester:
#   ci = TrueGraphClusterCI(adj, var_index_by_cluster)
#   P = cloc_learn(None, clusters, var_index_by_cluster, ..., ci_tester=ci)
#   # Convenience wrapper that does the two lines above:
#   P = cloc_learn_oracle(adj, clusters, var_index_by_cluster, ...)

import numpy as np
from itertools import combinations, chain
import math
import sys

try:
    from causallearn.utils.cit import CIT  # type: ignore
    _HAS_CAUSALLEARN = True
except Exception:
    _HAS_CAUSALLEARN = False


# ---------------------------------------------------------------------------
# Low-level helpers (used by ClusterCI)
# ---------------------------------------------------------------------------

def pinv(mat, rcond=1e-8):
    try:
        return np.linalg.pinv(mat, rcond=rcond)
    except Exception:
        eps = 1e-6
        return np.linalg.pinv(mat + eps * np.eye(mat.shape[0]))


def gauss_ci_test(data, X_idx, Y_idx, Z_idx=()):
    """Gaussian Fisher-Z CI test for a variable pair (x, y) with multi-dim Z."""
    X_idx = list(X_idx) if isinstance(X_idx, (list, tuple, set)) else [X_idx]
    Y_idx = list(Y_idx) if isinstance(Y_idx, (list, tuple, set)) else [Y_idx]
    Z_idx = list(Z_idx) if isinstance(Z_idx, (list, tuple, set)) else [Z_idx]
    x = X_idx[0]
    y = Y_idx[0]
    idx = [x, y] + Z_idx
    sub = data[:, idx]
    C = np.cov(sub, rowvar=False)
    P = pinv(C)
    rho = -P[0, 1] / math.sqrt(max(P[0, 0], 1e-12) * max(P[1, 1], 1e-12))
    n = data.shape[0]
    k = len(Z_idx)
    rho = float(np.clip(rho, -0.999999, 0.999999))
    z = 0.5 * math.log((1 + rho) / (1 - rho))
    se = 1 / math.sqrt(max(n - k - 3, 1))
    from math import erf, sqrt
    z_stat = abs(z) / se
    p = 2 * (1 - 0.5 * (1 + erf(z_stat / sqrt(2))))
    alpha = 0.01
    return (p > alpha, p, rho)


# ---------------------------------------------------------------------------
# Cluster CPDAG data structure
# ---------------------------------------------------------------------------

class ClusterCPDAG:
    def __init__(self, clusters):
        self.clusters = list(clusters)
        self.undirected = set()   # {frozenset({u,v})}
        self.dir_edges = set()    # {(u,v)}
        # (center, min(x,y), max(x,y)) -> {"type", "conn_mark", "sep_mark"}
        self.trip = {}
        # SepSet: (min(x,y), max(x,y)) -> set
        self.sep_sets = {}

    def add_undirected(self, a, b): self.undirected.add(frozenset((a, b)))
    def add_directed(self, a, b):   self.dir_edges.add((a, b))

    def neighbors(self, a):
        nbrs = set()
        for e in self.undirected:
            if a in e:
                u, v = tuple(e); nbrs.add(v if u == a else u)
        for u, v in self.dir_edges:
            if u == a: nbrs.add(v)
            if v == a: nbrs.add(u)
        return nbrs

    def adj_undirected(self, a, b): return frozenset((a, b)) in self.undirected
    def adj_any(self, a, b):
        return self.adj_undirected(a, b) or (a, b) in self.dir_edges or (b, a) in self.dir_edges

    def adjacent(self, u, v):
        return frozenset((u, v)) in self.undirected or (u, v) in self.dir_edges or (v, u) in self.dir_edges

    def set_trip(self, center, x, y, typ, conn_mark=None, sep_mark=None):
        key = (center, min(x, y), max(x, y))
        self.trip.setdefault(key, {"type": typ, "conn_mark": set(), "sep_mark": set()})
        self.trip[key]["type"] = typ
        if conn_mark: self.trip[key]["conn_mark"].update(conn_mark)
        if sep_mark:  self.trip[key]["sep_mark"].update(sep_mark)

    def add_sep_mark(self, center, x, y, targetC):
        key = (center, min(x, y), max(x, y))
        if key not in self.trip:
            self.trip[key] = {"type": "marginal", "conn_mark": set(), "sep_mark": set()}
        self.trip[key]["sep_mark"].add(targetC)

    def get_trip(self, center, x, y):
        return self.trip.get((center, min(x, y), max(x, y)))

    # add_triple / get_triple: alternate interface used by pparent_new.py
    def add_triple(self, center, x, y, typ, conn_mark=None):
        a, b = sorted((x, y))
        key = (center, a, b)
        self.trip[key] = {
            "type": typ,
            "conn_mark": None if conn_mark is None else set(conn_mark),
            "sep_mark": None,
        }

    def get_triple(self, center, x, y):
        a, b = sorted((x, y))
        return self.trip.get((center, a, b))

    def set_sps(self, x, y, S):
        self.sep_sets[(min(x, y), max(x, y))] = set(S)

    def get_sps(self, x, y):
        return self.sep_sets.get((min(x, y), max(x, y)), set())

    def show(self):
        und  = sorted([tuple(sorted(list(e))) for e in self.undirected])
        dirr = sorted(list(self.dir_edges))
        lines = ["Undirected: " + str(und), "Directed:   " + str(dirr), "Triplets:"]
        for (c, x, y), v in sorted(self.trip.items()):
            lines.append(f"  <{x}, {c}, {y}> : {v}")
        lines.append("SepSets:")
        for (x, y), S in sorted(self.sep_sets.items()):
            lines.append(f"  SepSet({x},{y}) = {sorted(S)}")
        return "\n".join(lines)

    def _succ(self, u):
        return {v for (x, v) in self.dir_edges if x == u}

    def _pred(self, v):
        return {u for (u, y) in self.dir_edges if y == v}

    def directed_path_exists(self, s, t):
        from collections import deque
        q = deque([s])
        seen = {s}
        while q:
            u = q.popleft()
            if u == t:
                return True
            for v in self._succ(u):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return False

    def descendants(self, a):
        out = set()
        stack = [a]
        while stack:
            u = stack.pop()
            for v in self._succ(u):
                if v not in out:
                    out.add(v)
                    stack.append(v)
        return out

    @property
    def dir_succ(self):
        from collections import defaultdict
        d = defaultdict(set)
        for (u, v) in self.dir_edges:
            d[u].add(v)
        return d

    @property
    def dir_pred(self):
        from collections import defaultdict
        d = defaultdict(set)
        for (u, v) in self.dir_edges:
            d[v].add(u)
        return d


# ---------------------------------------------------------------------------
# CI back-end 1: data-based (Fisher-Z / chi-square via causallearn)
# ---------------------------------------------------------------------------

class ClusterCI:
    """Cluster-level CI tester backed by data.

    Ensures X and Y are never included in the conditioning set passed to the
    underlying variable-level tester.
    """
    def __init__(self, data, var_index_by_cluster, cluster_types=None, alpha=0.01):
        self.data = data
        self.map  = {k: list(v) for k, v in var_index_by_cluster.items()}
        self.alpha = alpha
        self.cluster_types = dict(cluster_types) if cluster_types is not None \
            else {k: "cont" for k in self.map.keys()}

        if _HAS_CAUSALLEARN:
            try:
                self.cit_chi = CIT(data, "chisq")
            except Exception:
                self.cit_chi = None
            try:
                self.cit_fz  = CIT(data, "fisherz")
            except Exception:
                self.cit_fz  = None
            try:
                self.cit_kci = CIT(data, "kci")
            except Exception:
                self.cit_kci = None
        else:
            self.cit_chi = self.cit_kci = self.cit_fz = None

    def _is_discrete(self, c: str) -> bool:
        return self.cluster_types.get(c, "cont") != "cont"

    def _all_discrete(self, Cs) -> bool:
        return all(self._is_discrete(c) for c in Cs)

    def test(self, Xc, Yc, Sc=()):
        Sc = tuple([s for s in Sc if s not in (Xc, Yc)])
        X_idx = self.map[Xc]
        Y_idx = self.map[Yc]
        S_idx = list(chain.from_iterable(self.map[s] for s in Sc))

        use_chi = self._all_discrete([Xc, Yc] + list(Sc))

        for x in X_idx:
            for y in Y_idx:
                if _HAS_CAUSALLEARN:
                    if use_chi and (self.cit_chi is not None):
                        p = self.cit_chi(x, y, S_idx)
                    else:
                        if self.cit_fz is not None:
                            p = self.cit_fz(x, y, S_idx)
                        elif self.cit_kci is not None:
                            p = self.cit_kci(x, y, S_idx)
                        else:
                            print("[Warning] causallearn CIT fisherz/kci not available; falling back to gauss_ci_test.")
                            sys.exit(1)
                            indep, p, _ = gauss_ci_test(self.data, x, y, S_idx)
                            return indep
                    indep = (p > self.alpha)
                else:
                    indep, p, _ = gauss_ci_test(self.data, x, y, S_idx)

                if not indep:
                    return False
        return True


# ---------------------------------------------------------------------------
# CI back-end 2: oracle (d-separation on the true variable-level DAG)
# ---------------------------------------------------------------------------

class TrueGraphClusterCI:
    """Cluster-level CI oracle using d-separation on a known variable-level DAG.

    Drop-in replacement for ClusterCI: call test(Xc, Yc, Sc) with cluster names.
    adj[i, j] != 0 means i -> j.  Returns True iff every variable in Xc is
    d-separated from every variable in Yc given all variables in clusters Sc.
    """
    def __init__(self, adj, var_index_by_cluster, clusters=None, *,
                 max_path_len=None, check_acyclic=True):
        self.adj = np.asarray(adj)
        if self.adj.ndim != 2 or self.adj.shape[0] != self.adj.shape[1]:
            raise ValueError("adj must be a square adjacency matrix.")
        self.p = int(self.adj.shape[0])

        if clusters is None:
            clusters = list(var_index_by_cluster.keys())
        self.clusters = list(clusters)
        self.map = {c: list(var_index_by_cluster[c]) for c in self.clusters}

        for c, idxs in self.map.items():
            bad = [i for i in idxs if i < 0 or i >= self.p]
            if bad:
                raise ValueError(f"Cluster {c!r} contains out-of-range variable indices: {bad}")

        self.max_path_len = self.p if max_path_len is None else int(max_path_len)
        self.children  = {i: set(np.flatnonzero(self.adj[i, :] != 0).tolist()) for i in range(self.p)}
        self.parents   = {j: set(np.flatnonzero(self.adj[:, j] != 0).tolist()) for j in range(self.p)}
        self.skeleton  = {i: set(self.children[i]) | set(self.parents[i]) for i in range(self.p)}

        if check_acyclic and not self._is_acyclic():
            raise ValueError("adj must represent a DAG; d-separation is not defined for directed cycles.")

    def _is_acyclic(self):
        temp = set(); perm = set()
        def visit(u):
            if u in perm: return True
            if u in temp: return False
            temp.add(u)
            for v in self.children[u]:
                if not visit(v): return False
            temp.remove(u); perm.add(u)
            return True
        return all(visit(u) for u in range(self.p))

    def _ancestors_of(self, nodes):
        anc = set(nodes); stack = list(nodes)
        while stack:
            v = stack.pop()
            for u in self.parents[v]:
                if u not in anc:
                    anc.add(u); stack.append(u)
        return anc

    def _is_collider_on_path(self, prev_node, node, next_node):
        return (self.adj[prev_node, node] != 0) and (self.adj[next_node, node] != 0)

    def _path_is_active(self, path, Z):
        Z = set(Z); anc_Z = self._ancestors_of(Z)
        for i in range(1, len(path) - 1):
            a, b, c = path[i - 1], path[i], path[i + 1]
            if self._is_collider_on_path(a, b, c):
                if b not in anc_Z: return False
            else:
                if b in Z: return False
        return True

    def _active_path_exists(self, x, y, Z):
        if x == y: return True
        Z = set(Z); stack = [(x, [x])]
        while stack:
            u, path = stack.pop()
            if len(path) > self.max_path_len: continue
            for v in self.skeleton[u]:
                if v in path: continue
                new_path = path + [v]
                if v == y:
                    if self._path_is_active(new_path, Z): return True
                else:
                    stack.append((v, new_path))
        return False

    def d_separated(self, x, y, Z=()):
        return not self._active_path_exists(int(x), int(y), set(Z))

    def test(self, Xc, Yc, Sc=()):
        Sc    = tuple([s for s in Sc if s not in (Xc, Yc)])
        X_idx = self.map[Xc]
        Y_idx = self.map[Yc]
        Z_idx = list(chain.from_iterable(self.map[s] for s in Sc))
        for x in X_idx:
            for y in Y_idx:
                if not self.d_separated(x, y, Z_idx):
                    return False
        return True

    def explain(self, Xc, Yc, Sc=()):
        Sc    = tuple([s for s in Sc if s not in (Xc, Yc)])
        Z_idx = list(chain.from_iterable(self.map[s] for s in Sc))
        active_pairs = [
            (x, y) for x in self.map[Xc] for y in self.map[Yc]
            if not self.d_separated(x, y, Z_idx)
        ]
        return {
            "X": Xc, "Y": Yc, "S": tuple(Sc),
            "conditioning_indices": tuple(Z_idx),
            "independent": len(active_pairs) == 0,
            "active_pairs": active_pairs,
        }


# ---------------------------------------------------------------------------
# Internal graph utility
# ---------------------------------------------------------------------------

def _simple_paths(P, start, end, max_len=8):
    """Enumerate simple paths over the undirected neighborhood (direction ignored)."""
    adj   = lambda u: list(P.neighbors(u))
    stack = [(start, [start])]
    out   = []
    while stack:
        u, path = stack.pop()
        if len(path) > max_len: continue
        if u == end and len(path) >= 2:
            out.append(path[:]); continue
        for v in adj(u):
            if v in path: continue
            stack.append((v, path + [v]))
    return out


# ---------------------------------------------------------------------------
# CLOC algorithm
# ---------------------------------------------------------------------------

def cloc_learn(data, clusters, var_index_by_cluster, cluster_types=None, *,
               y_cluster=None, ci_tester=None):
    """Learn a ClusterCPDAG via CLOC (Algorithm 1).

    Parameters
    ----------
    data : np.ndarray of shape (n, p), or None when ci_tester is provided.
    clusters : list of cluster name strings.
    var_index_by_cluster : dict mapping cluster name -> list of variable indices.
    cluster_types : optional dict mapping cluster name -> "cont" | "binary".
    y_cluster : optional cluster name to exclude from structure learning (e.g. label Y).
    ci_tester : optional CI back-end object with a test(Xc, Yc, Sc) -> bool interface.
        When None (default), a ClusterCI backed by ``data`` is constructed automatically.
        Pass a TrueGraphClusterCI instance to use the oracle d-separation back-end.
    """
    if y_cluster is not None and y_cluster in clusters:
        C = [c for c in clusters if c != y_cluster]
        var_index_by_cluster = {c: var_index_by_cluster[c] for c in C}
        if cluster_types is not None:
            cluster_types = {c: cluster_types.get(c, "cont") for c in C}
    else:
        C = list(clusters)

    ci = ci_tester if ci_tester is not None \
        else ClusterCI(data, var_index_by_cluster, cluster_types=cluster_types, alpha=0.01)
    P  = ClusterCPDAG(C)

    # (1) Complete graph
    for a, b in combinations(C, 2):
        P.add_undirected(a, b)

    # (2) Skeleton & SepSets
    for X, Y in combinations(C, 2):
        others = [c for c in C if c not in (X, Y)]
        sep = None
        for k in range(len(others) + 1):
            if sep is not None: break
            for S in combinations(others, k):
                if ci.test(X, Y, S):
                    sep = set(S); break
        if sep is not None:
            P.set_sps(X, Y, sep)
            if P.adj_undirected(X, Y):
                P.undirected.remove(frozenset((X, Y)))

    # (3) Unshielded triplets: assign arc types + orient colliders
    for Z in C:
        adj = [v for v in P.neighbors(Z)]
        for X, Y in combinations(sorted(adj), 2):
            if not P.adj_any(X, Y):  # unshielded triple <X, Z, Y>
                S = P.get_sps(X, Y)

                if Z not in S:
                    dep_given_S_plus_Z = not ci.test(X, Y, set(S) | {Z})
                    if dep_given_S_plus_Z:
                        typ = "conditional"
                        P.set_trip(Z, X, Y, typ)
                        if P.adj_undirected(X, Z):
                            P.undirected.discard(frozenset((X, Z))); P.add_directed(X, Z)
                        if P.adj_undirected(Y, Z):
                            P.undirected.discard(frozenset((Y, Z))); P.add_directed(Y, Z)
                        continue
                if Z in S:
                    P.set_trip(Z, X, Y, "marginal")
                else:
                    P.set_trip(Z, X, Y, "never")

    # (3b) Shielded triplets (Alg.2 lines 15–27)
    # For every shielded triple <L, CEN, R>:
    #   (1) evaluate all eligible W (not just the first), keep strongest arc type.
    #   (2) "{B,C} in SepSet(X,W)" is interpreted as {CEN, R} in SepSet(L, W).
    for X, Z, Y in combinations(C, 3):
        for (L, CEN, R) in [(X, Z, Y), (Y, Z, X)]:
            if not (P.adj_any(L, CEN) and P.adj_any(CEN, R) and P.adj_any(L, R)):
                continue
            Ws = [w for w in C if w not in (L, CEN, R) and P.adj_any(R, w) and not P.adj_any(L, w)]
            if not Ws:
                continue

            decided_type = None
            do_orient    = False

            for W in Ws:
                def arc_type(center, a, b):
                    Sxy = P.get_sps(a, b)
                    if center not in Sxy and not ci.test(a, b, set(Sxy) | {center}):
                        return "conditional"
                    elif center in Sxy:
                        return "marginal"
                    else:
                        return "never"

                AZRW    = arc_type(CEN, R, W)
                ALRW    = arc_type(L, R, W)
                Sep_XW  = P.get_sps(L, W)
                cond1   = (R not in Sep_XW) and (not ci.test(L, W, set(P.get_sps(L, R)) | {R}))

                cand_type   = None
                cand_orient = False

                if AZRW == "conditional" and ALRW != "conditional":
                    cand_type = "marginal" if cond1 else "never"
                elif AZRW == "marginal" and ALRW != "marginal":
                    if cond1:
                        cand_type = "conditional"; cand_orient = True
                    else:
                        cand_type = "marginal" if (CEN in Sep_XW and R in Sep_XW) else "never"

                if cand_type is None:
                    continue

                if cand_type == "conditional":
                    decided_type = "conditional"; do_orient = cand_orient; break
                elif cand_type == "marginal" and decided_type != "conditional":
                    decided_type = "marginal"; do_orient = False
                elif cand_type == "never" and decided_type is None:
                    decided_type = "never"; do_orient = False

            if decided_type is not None:
                P.set_trip(CEN, L, R, decided_type)
                if decided_type == "conditional" and do_orient:
                    if P.adj_undirected(L, CEN):
                        P.undirected.discard(frozenset((L, CEN))); P.add_directed(L, CEN)
                    if P.adj_undirected(R, CEN):
                        P.undirected.discard(frozenset((R, CEN))); P.add_directed(R, CEN)

    # (4) Path-based separation marks
    for C0, Cn in combinations(C, 2):
        for p in _simple_paths(P, C0, Cn, max_len=8):
            if len(p) < 4: continue
            ok = True; K = set()
            for i in range(1, len(p) - 1):
                a_prev, a, a_next = p[i - 1], p[i], p[i + 1]
                t = P.get_trip(a, a_prev, a_next)
                if t is None or t["type"] not in ("marginal", "conditional") or t["sep_mark"]:
                    ok = False; break
                if t["type"] == "conditional": K.add(a)
            if not ok: continue

            S0    = P.get_sps(C0, Cn)
            K_sub = set(K) - {C0, Cn}              # K \ {Ci, Cj}
            S_sub = K_sub | (set(S0) - set(p))     # K\{Ci,Cj} ∪ (SepSet(C0,Cn)\p)

            if not ci.test(C0, Cn, S_sub): continue

            best = None
            for i in range(0, len(p) - 3):
                for j in range(i + 3, len(p)):
                    Ci, Cj = p[i], p[j]
                    if ci.test(Ci, Cj, set(S_sub) - {Ci, Cj}):
                        if best is None or (j - i) < (best[1] - best[0]):
                            best = (i, j)
                        break
            if best is None: continue
            i, j = best; Ci, Cj = p[i], p[j]

            if i + 2 < len(p):
                P.add_sep_mark(p[i + 1], p[i], p[i + 2], targetC=Cj)
            if j - 2 >= 0:
                P.add_sep_mark(p[j - 1], p[j - 2], p[j], targetC=Ci)

    # (5) Connection marks
    for (center, x, y), info in list(P.trip.items()):
        if info["type"] == "never":
            for W in [v for v in C if P.adj_undirected(center, v)]:
                if W in (x, y): continue
                S = P.get_sps(x, y)
                if ci.test(x, y, S) and not ci.test(x, y, set(S) | {W}):
                    info["conn_mark"].add(W)

    # (6) Orientation closure
    changed = True
    while changed:
        changed = False
        # Rule 1
        for Z in C:
            for X in C:
                if X == Z or (X, Z) not in P.dir_edges: continue
                for Y in C:
                    if Y in (X, Z) or not P.adj_undirected(Z, Y) or P.adj_any(X, Y): continue
                    t = P.get_trip(Z, X, Y)
                    if t and t["type"] == "marginal":
                        P.undirected.discard(frozenset((Z, Y)))
                        if (Z, Y) not in P.dir_edges:
                            P.add_directed(Z, Y); changed = True
        # Rule 2
        for e in list(P.undirected):
            X, Y = tuple(e)
            for Z in C:
                if (X, Z) in P.dir_edges and (Z, Y) in P.dir_edges:
                    P.undirected.discard(e)
                    if (X, Y) not in P.dir_edges:
                        P.add_directed(X, Y); changed = True
                    break
                if (Y, Z) in P.dir_edges and (Z, X) in P.dir_edges:
                    P.undirected.discard(e)
                    if (Y, X) not in P.dir_edges:
                        P.add_directed(Y, X); changed = True
                    break
        # Rule 3
        for Z in C:
            parents = [X for (X, Z2) in P.dir_edges if Z2 == Z]
            if len(parents) < 2: continue
            for X, Y in combinations(parents, 2):
                for W in C:
                    if W in (X, Y, Z): continue
                    if not P.adj_undirected(W, Z): continue
                    if not (P.adj_undirected(X, W) and P.adj_undirected(W, Y)): continue
                    if P.adj_any(X, Y): continue
                    t = P.get_trip(W, X, Y)
                    if t and t["type"] == "marginal":
                        P.undirected.discard(frozenset((W, Z)))
                        if (W, Z) not in P.dir_edges:
                            P.add_directed(W, Z); changed = True
        # Rule 4
        for Z in C:
            for X in C:
                if (X, Z) not in P.dir_edges: continue
                for Y in C:
                    if (Z, Y) not in P.dir_edges or Y in (X, Z): continue
                    for W in C:
                        if W in (X, Y, Z): continue
                        if not (P.adj_undirected(W, Z) and P.adj_undirected(W, Y)): continue
                        if not (P.adj_undirected(X, W) and P.adj_undirected(W, Y)): continue
                        if P.adj_any(X, Y): continue
                        t = P.get_trip(W, X, Y)
                        if t and t["type"] == "marginal":
                            P.undirected.discard(frozenset((W, Y)))
                            if (W, Y) not in P.dir_edges:
                                P.add_directed(W, Y); changed = True
        # Rule 5: connection mark
        for Z in C:
            nbrs = [v for v in C if P.adj_undirected(Z, v)]
            for X, Y in combinations(nbrs, 2):
                t = P.get_trip(Z, X, Y)
                if not t or t["type"] not in ("never", "conditional"): continue
                for W in nbrs:
                    if W in (X, Y): continue
                    if W in t["conn_mark"]:
                        P.undirected.discard(frozenset((Z, W)))
                        if (Z, W) not in P.dir_edges:
                            P.add_directed(Z, W); changed = True
    return P


def cloc_learn_oracle(adj, clusters, var_index_by_cluster, cluster_types=None, *,
                      y_cluster=None, max_path_len=None, check_acyclic=True):
    """Convenience wrapper: run CLOC with the oracle d-separation CI back-end.

    Equivalent to:
        ci = TrueGraphClusterCI(adj, var_index_by_cluster, max_path_len=max_path_len)
        P  = cloc_learn(None, clusters, var_index_by_cluster, ..., ci_tester=ci)
    """
    ci = TrueGraphClusterCI(
        adj, var_index_by_cluster,
        clusters=clusters,
        max_path_len=max_path_len,
        check_acyclic=check_acyclic,
    )
    return cloc_learn(
        data=None,
        clusters=clusters,
        var_index_by_cluster=var_index_by_cluster,
        cluster_types=cluster_types,
        y_cluster=y_cluster,
        ci_tester=ci,
    )


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def sample_toy_G2(n=7000, seed=7):
    rng = np.random.default_rng(seed)
    Z3 = rng.normal(size=n); Y1 = rng.normal(size=n); X1 = rng.normal(size=n)
    Z2 = 0.8 * Z3 + 0.7 * Y1 + rng.normal(scale=0.8, size=n)
    Z1 = 0.9 * Z2 + 0.6 * X1 + rng.normal(scale=0.7, size=n)
    W1 = 0.8 * Z1 + rng.normal(scale=0.7, size=n)
    R1 = 0.9 * X1 + 0.9 * Y1 + rng.normal(scale=0.7, size=n)
    Q1 = 0.9 * R1 + rng.normal(scale=0.7, size=n)
    data  = np.column_stack([X1, Y1, Z1, Z2, Z3, W1, R1, Q1])
    names = ["X1", "Y1", "Z1", "Z2", "Z3", "W1", "R1", "Q1"]
    return data, names


if __name__ == "__main__":
    clusters = ["X", "Y", "Z", "W", "R", "Q"]
    cluster_to_vars = {
        "X": ["X1"], "Y": ["Y1"], "Z": ["Z1", "Z2", "Z3"],
        "W": ["W1"], "R": ["R1"], "Q": ["Q1"],
    }
    data, varnames = sample_toy_G2()
    var_to_idx   = {v: i for i, v in enumerate(varnames)}
    cluster_to_idx = {c: [var_to_idx[v] for v in vs] for c, vs in cluster_to_vars.items()}

    P = cloc_learn(data, clusters, cluster_to_idx)
    print("=== Learned cluster CP-DAG ===")
    print(P.show())
