"""
Adjustment candidate set computation for cluster CPDAGs.

Provides utilities to enumerate valid adjustment sets Z for causal inference
with sensitive attribute A (and optional mediator Xad) given a learned CPDAG P.

Main public API:
  return_adjustment_candidates(P, A_cluster, ...)
  learn_adjustment_candidates_with_refinement(data_train, clusters, ...)
  rename_reset_clusters(G, refine_state, ...)
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Any, Optional, Iterable, Mapping

import cloc
from . import possible_parent_sets as pparent


def _filter_inputs_excluding_y(
    clusters: List[str],
    cluster_to_idx: Dict[str, List[int]],
    cluster_types: Optional[Mapping[str, str]],
    y_cluster: Optional[str],
) -> Tuple[List[str], Dict[str, List[int]], Optional[Dict[str, str]]]:
    """Return (clusters, cluster_to_idx, cluster_types) with y_cluster removed."""
    cur_clusters = list(clusters)
    cur_map = {c: list(vs) for c, vs in cluster_to_idx.items()}
    cur_types: Optional[Dict[str, str]] = None
    if cluster_types is not None:
        cur_types = {k: str(v) for k, v in cluster_types.items()}

    if y_cluster is None or y_cluster not in cur_map:
        if cur_types is not None:
            cur_types = {c: cur_types.get(c, "cont") for c in cur_clusters if c in cur_map}
        return cur_clusters, cur_map, cur_types

    cur_clusters = [c for c in cur_clusters if c != y_cluster]
    cur_map.pop(y_cluster, None)
    if cur_types is not None:
        cur_types.pop(y_cluster, None)
        cur_types = {c: cur_types.get(c, "cont") for c in cur_clusters if c in cur_map}
    return cur_clusters, cur_map, cur_types


def parents_from_P(P: Any, target_cluster: str) -> Set[str]:
    """Return definite parents of target_cluster in the learned cluster CPDAG."""
    return {u for (u, v) in getattr(P, "dir_edges", set()) if v == target_cluster}


def _singletonize_cluster(
    clusters: List[str],
    cluster_to_idx: Dict[str, List[int]],
    center: str,
) -> Tuple[List[str], Dict[str, List[int]], List[str]]:
    """Replace cluster 'center' with one singleton cluster per variable index."""
    if center not in cluster_to_idx:
        return clusters, cluster_to_idx, []

    var_idxs = list(cluster_to_idx[center])
    new_clusters = [c for c in clusters if c != center]
    new_cluster_to_idx = {c: list(vs) for c, vs in cluster_to_idx.items() if c != center}

    created: List[str] = []
    for idx in var_idxs:
        name = f"{center}__v{idx}"
        if name in new_cluster_to_idx:
            j = 1
            while f"{name}_{j}" in new_cluster_to_idx:
                j += 1
            name = f"{name}_{j}"
        new_clusters.append(name)
        new_cluster_to_idx[name] = [int(idx)]
        created.append(name)

    return new_clusters, new_cluster_to_idx, created


def rename_reset_clusters(
    G: dict,
    refine_state: dict,
    *,
    data_train=None,
    device=None,
):
    """Rebuild cluster-related fields of G after refinement/splitting.

    Re-learns a cluster CPDAG via cloc and updates meta['cluster_edges'].
    var_edges are NOT modified.

    Returns
    -------
    G_new : dict
    P_new : Any
    """
    import numpy as np
    import torch
    from copy import deepcopy

    if refine_state is None:
        return G, None

    for k in ("clusters", "cluster_to_idx", "cluster_types"):
        if k not in refine_state:
            raise ValueError(f"refine_state must contain key: {k}")

    meta = G.get("meta", {})
    Y_name = meta.get("Y", None)

    refined_clusters = list(refine_state["clusters"])
    if Y_name and (Y_name not in refined_clusters) and (Y_name in G.get("clusters", [])):
        refined_clusters = refined_clusters + [Y_name]

    cluster_to_idx = {c: list(idxs) for c, idxs in refine_state["cluster_to_idx"].items()}
    cluster_types = dict(refine_state["cluster_types"])
    if Y_name and Y_name not in cluster_types:
        cluster_types[Y_name] = meta.get("cluster_types", {}).get(Y_name, "cont")

    if device is None:
        ct = G.get("cluster_tensors", {})
        if isinstance(ct, dict) and len(ct) > 0:
            device = next(iter(ct.values())).device
        else:
            device = torch.device("cpu")

    data = G["data"]
    data_np = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
    varnames = list(G.get("varnames", [f"v{i}" for i in range(data_np.shape[1])]))

    varnames_new = list(varnames)
    for c in refined_clusters:
        idxs = cluster_to_idx.get(c, [])
        for j, gi in enumerate(idxs):
            gi = int(gi)
            if 0 <= gi < len(varnames_new):
                varnames_new[gi] = f"{c}_{j+1}"
    varnames = varnames_new

    cluster_to_vars = {}
    for c in refined_clusters:
        idxs = cluster_to_idx.get(c, [])
        cluster_to_vars[c] = [varnames[int(i)] for i in idxs if 0 <= int(i) < len(varnames)]

    T = torch.from_numpy(data_np).float().to(device)
    cluster_tensors = {}
    for c in refined_clusters:
        idxs = cluster_to_idx.get(c, [])
        if len(idxs) == 0:
            cluster_tensors[c] = torch.ones((T.size(0), 1), device=device)
        else:
            cluster_tensors[c] = T[:, [int(i) for i in idxs]]

    if data_train is None:
        data_train = data_np
    data_train_np = data_train.detach().cpu().numpy() if isinstance(data_train, torch.Tensor) else np.asarray(data_train)

    P_new = cloc.cloc_learn(
        data_train_np,
        refined_clusters,
        cluster_to_idx,
        cluster_types=cluster_types,
        y_cluster=Y_name,
    )

    cluster_edges = list(getattr(P_new, "dir_edges", []))

    G_new = deepcopy(G)
    G_new["clusters"] = refined_clusters
    G_new["cluster_to_vars"] = cluster_to_vars
    G_new["cluster_tensors"] = cluster_tensors
    meta_new = dict(meta)
    meta_new["cluster_types"] = cluster_types
    meta_new["cluster_edges"] = cluster_edges
    G_new["meta"] = meta_new
    G_new["varnames"] = varnames

    return G_new, P_new


def _refine_centers_to_singletons(
    clusters: List[str],
    cluster_to_idx: Dict[str, List[int]],
    centers: Set[str],
    protected: Set[str],
    cluster_types: Optional[Dict[str, str]],
) -> Tuple[List[str], Dict[str, List[int]], Optional[Dict[str, str]]]:
    """Refine clusters in 'centers' (except 'protected') into singleton clusters."""
    new_clusters = list(clusters)
    new_map = {c: list(vs) for c, vs in cluster_to_idx.items()}
    new_types = dict(cluster_types) if cluster_types is not None else None

    for C in sorted(centers):
        if C in protected:
            continue
        base_type = (new_types.get(C) if new_types is not None else None)
        new_clusters, new_map, created = _singletonize_cluster(new_clusters, new_map, C)
        if new_types is not None:
            if C in new_types:
                del new_types[C]
            t = base_type if base_type is not None else "cont"
            for nm in created:
                new_types[nm] = t

    return new_clusters, new_map, new_types


def _centers_to_refine_for_candidates(
    P: Any,
    endpoints: List[str],
    Z_candidates: List[Tuple[str, ...]],
) -> Set[str]:
    """Return center clusters to refine based on connection-mark collisions."""
    endpoints_set = set(endpoints)
    centers: Set[str] = set()

    trip = getattr(P, "trip", {})
    Z_sets = [set(z) for z in Z_candidates]

    for (center, x, y), info in trip.items():
        if not info:
            continue
        typ = info.get("type")
        if typ not in ("never", "conditional"):
            continue
        conn = info.get("conn_mark")
        if not conn:
            continue
        if (x not in endpoints_set) and (y not in endpoints_set):
            continue
        conn_set = set(conn)
        for Z in Z_sets:
            if center not in Z:
                continue
            if conn_set & Z:
                centers.add(center)
                break

    return centers


def _build_Z_candidates(
    parents_A: Set[str],
    poss_A: List[Tuple[str, ...]],
    *,
    has_xad: bool,
    parents_X: Optional[Set[str]] = None,
    poss_X: Optional[List[Tuple[str, ...]]] = None,
) -> List[Tuple[str, ...]]:
    """Build Z candidates from possible-parent sets of A (and Xad when present)."""
    if not has_xad:
        Z_cands_base = [tuple(sorted(set(psa))) for psa in poss_A]
        Z_cands = [tuple(sorted(set(z) | set(parents_A))) for z in Z_cands_base]
    else:
        parents_X = parents_X or set()
        poss_X = poss_X or [tuple()]
        Z_cands_base = [
            tuple(sorted(set(psa) | set(psx)))
            for psa in poss_A
            for psx in poss_X
        ]
        Z_cands = [
            tuple(sorted(set(z) | set(parents_A) | set(parents_X)))
            for z in Z_cands_base
        ]

    return sorted(set(Z_cands), key=lambda t: (len(t), t))


def _iter_clusters(obj: Any) -> Iterable[str]:
    """Iterate cluster names from common container shapes."""
    if obj is None:
        return
    if isinstance(obj, (set, frozenset, list, tuple)):
        for x in obj:
            if isinstance(x, (set, frozenset, list, tuple)):
                for y in x:
                    yield str(y)
            else:
                yield str(x)
    else:
        yield str(obj)


def _all_connection_mark_clusters(P: Any) -> Set[str]:
    """Return all cluster names that appear in any connection mark in P."""
    out: Set[str] = set()
    trip = getattr(P, "trip", {})
    if not isinstance(trip, dict):
        return out
    for _, info in trip.items():
        if not isinstance(info, dict):
            continue
        cm = info.get("conn_mark")
        if not cm:
            continue
        for c in cm:
            out.add(c)
    return out


def _max_poss_parent_set(poss_sets: List[Tuple[str, ...]]) -> Tuple[str, ...]:
    """Return the largest possible-parent set (tie-broken lexicographically)."""
    if not poss_sets:
        return tuple()
    return max(poss_sets, key=lambda t: (len(t), t))


def _has_connection_marks_in_pparents(
    P: Any,
    A_cluster: str,
    Xad_cluster: Optional[str],
    HAS_XAD: bool,
) -> bool:
    """Return True iff connection marks are relevant for A (and Xad) adjustment sets."""
    conn_mark_clusters = _all_connection_mark_clusters(P)

    parents_A = parents_from_P(P, A_cluster)
    poss_A = pparent.possible_parent_sets(P, A_cluster)
    Z0: Set[str] = set(parents_A) | set(_max_poss_parent_set(poss_A))

    if HAS_XAD:
        if Xad_cluster is None:
            raise ValueError("HAS_XAD=True requires Xad_cluster")
        parents_X = parents_from_P(P, Xad_cluster)
        poss_X = pparent.possible_parent_sets(P, Xad_cluster)
        Z0 |= set(parents_X) | set(_max_poss_parent_set(poss_X))

    return len(Z0 & conn_mark_clusters) > 0


def _undirected_neighbors(P: Any, c: str) -> Set[str]:
    sibs: Set[str] = set()
    for e in getattr(P, "undirected", set()):
        if c in e:
            u, v = tuple(e)
            sibs.add(v if u == c else u)
    return sibs


def _neighbors_any(P: Any, c: str) -> Set[str]:
    """All neighbors via directed or undirected adjacency."""
    nbrs: Set[str] = set()
    nbrs |= _undirected_neighbors(P, c)
    for (u, v) in getattr(P, "dir_edges", set()):
        if u == c:
            nbrs.add(v)
        if v == c:
            nbrs.add(u)
    return nbrs


def _get_triplet_info(P: Any, center: str, x: str, y: str) -> Optional[Dict[str, Any]]:
    """Return info dict for triplet <x, center, y> if stored, else None."""
    if hasattr(P, "get_trip"):
        out = P.get_trip(center, x, y)
        if out is not None:
            return out
    if hasattr(P, "get_triple"):
        out = P.get_triple(center, x, y)
        if out is not None:
            return out
    trip = getattr(P, "trip", {})
    if isinstance(trip, dict):
        key = (center, min(x, y), max(x, y))
        return trip.get(key)
    return None


def _normalize_trip_type(t: Optional[str]) -> str:
    """Map raw labels to {'marg', 'never', 'cond', 'unknown'}."""
    if not t:
        return "unknown"
    s = str(t).lower()
    if "never" in s: return "never"
    if "cond"  in s: return "cond"
    if "marg"  in s: return "marg"
    return "unknown"


def _possible_descendants(P: Any, start: str) -> Set[str]:
    """Conservative possible descendants of start in a CPDAG.

    Traversal: u -> v (directed out) and u - v (undirected) are followed;
    v -> u is not followed.
    """
    seen: Set[str] = {start}
    out: Set[str] = set()
    q: List[str] = [start]
    dir_edges  = getattr(P, "dir_edges",  set())
    undirected = getattr(P, "undirected", set())

    while q:
        u = q.pop(0)
        for (a, b) in dir_edges:
            if a == u and b not in seen:
                seen.add(b); out.add(b); q.append(b)
        for e in undirected:
            if u in e:
                a, b = tuple(e)
                v = b if a == u else a
                if v not in seen:
                    seen.add(v); out.add(v); q.append(v)
    return out


def _possible_descendants_plus(P: Any, start: str) -> Set[str]:
    return {start} | _possible_descendants(P, start)


def _has_separation_mark(info: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(info, dict):
        return False
    sm = info.get("sep_mark")
    if sm is None:
        return False
    if isinstance(sm, (set, frozenset, list, tuple, dict, str)):
        return len(sm) > 0
    return bool(sm)


@dataclass
class ExtendSearchResult:
    ok: bool
    Z: Set[str]
    M: Set[str]


def final_certify(P: Any, root: str, Z: Set[str]) -> ExtendSearchResult:
    """Check-only pass: re-verify all reached conditional triplets against the final Z."""
    final_Z: Set[str] = set(Z)
    M: Set[str] = set()
    queue: List[str] = list(final_Z)
    processed: Set[str] = set()

    while queue:
        D = queue.pop(0)
        if D in processed:
            continue
        processed.add(D)

        for N in _neighbors_any(P, D):
            if N == root:
                continue
            info = _get_triplet_info(P, D, root, N)
            t = _normalize_trip_type(info.get("type") if info else None)

            if t == "cond":
                if _has_separation_mark(info):
                    continue
                B = _possible_descendants_plus(P, D)
                H = final_Z & B
                if H:
                    M.add(D); M.update(H)
                    return ExtendSearchResult(ok=False, Z=final_Z, M=M)

            elif t == "never":
                cm = set(_iter_clusters(info.get("conn_mark") if info else None))
                for c in sorted(cm & final_Z):
                    M.add(c)
                    if (c not in processed) and (c not in queue):
                        queue.append(c)

            elif t == "marg":
                continue

            else:
                M.add(D); M.add(N)
                return ExtendSearchResult(ok=False, Z=final_Z, M=M)

    return ExtendSearchResult(ok=True, Z=final_Z, M=M)


def extend_search(P: Any, root: str, Z0: Set[str]) -> ExtendSearchResult:
    """Extend-search procedure: build adjustment candidate Z from initial Z0.

    Construction pass adds required clusters, then a final certification pass
    re-checks all conditional triplets against the completed Z.
    """
    Z: Set[str] = set(Z0)
    M: Set[str] = set()
    queue: List[str] = list(Z0)
    processed: Set[str] = set()

    while queue:
        D = queue.pop(0)
        if D in processed:
            continue
        processed.add(D)

        for N in _neighbors_any(P, D):
            if N == root:
                continue
            info = _get_triplet_info(P, D, root, N)
            t = _normalize_trip_type(info.get("type") if info else None)

            if t == "marg":
                Z.add(D)

            elif t == "never":
                Z.add(D)
                if info:
                    cm = info.get("conn_mark")
                    for c in _iter_clusters(cm):
                        M.add(c)
                        if (c not in processed) and (c not in queue):
                            queue.append(c)

            elif t == "cond":
                if _has_separation_mark(info):
                    continue
                B = _possible_descendants_plus(P, D)
                H = Z & B
                if H:
                    M.add(D); M.update(H)
                    return ExtendSearchResult(ok=False, Z=Z, M=M)

            else:
                M.add(D); M.add(N)
                return ExtendSearchResult(ok=False, Z=Z, M=M)

    cert = final_certify(P, root, Z)
    if not cert.ok:
        M.update(cert.M)
        return ExtendSearchResult(ok=False, Z=Z, M=M)

    return ExtendSearchResult(ok=True, Z=Z, M=M)


def return_adjustment_candidates(
    P: Any,
    A_cluster: str,
    *,
    HAS_XAD: bool = False,
    Xad_cluster: Optional[str] = None,
    do_refine: bool = False,
    data_train: Any = None,
    clusters: Optional[List[str]] = None,
    cluster_to_idx: Optional[Dict[str, List[int]]] = None,
    cluster_types: Optional[Mapping[str, str]] = None,
    y_cluster: Optional[str] = None,
    max_iters: int = 20,
    verbose: bool = False,
) -> Tuple[List[Tuple[str, ...]], Optional[Tuple[str, ...]], Optional[Dict[str, Any]]]:
    """Return adjustment candidates for the given learned graph P.

    Returns
    -------
    Z_cands : list[tuple[str, ...]]
    fail_M  : tuple[str, ...] | None
    refine_state : dict | None
    """
    if HAS_XAD and not Xad_cluster:
        raise ValueError("HAS_XAD=True requires Xad_cluster")

    has_conn = _has_connection_marks_in_pparents(P, A_cluster, Xad_cluster, HAS_XAD)

    if not has_conn:
        parents_A = parents_from_P(P, A_cluster)
        poss_A = pparent.possible_parent_sets(P, A_cluster)
        parents_X: Set[str] = set()
        poss_X: List[Tuple[str, ...]] = [tuple()]
        if HAS_XAD:
            parents_X = parents_from_P(P, Xad_cluster)
            poss_X = pparent.possible_parent_sets(P, Xad_cluster)
        Z_cands = _build_Z_candidates(
            parents_A=parents_A, poss_A=poss_A,
            has_xad=HAS_XAD, parents_X=parents_X, poss_X=poss_X,
        )
        return Z_cands, None, None

    parents_A = set(parents_from_P(P, A_cluster))
    poss_A = pparent.possible_parent_sets(P, A_cluster) or [set()]
    res_As = [extend_search(P, A_cluster, parents_A | set(S)) for S in poss_A]

    if not HAS_XAD:
        okA = all(r.ok for r in res_As)
        if okA:
            return sorted({tuple(sorted(r.Z)) for r in res_As}), None, None
        if not do_refine:
            fail_M = sorted({m for r in res_As if not r.ok for m in r.M})
            return [], tuple(fail_M), None

        if data_train is None or clusters is None or cluster_to_idx is None:
            raise ValueError("do_refine=True requires data_train, clusters, cluster_to_idx")
        P_ref, Z_cands, info = learn_adjustment_candidates_with_refinement(
            data_train, clusters, cluster_to_idx, A_cluster,
            HAS_XAD=False, Xad_cluster=None,
            cluster_types=cluster_types, y_cluster=y_cluster,
            max_iters=max_iters, verbose=verbose,
        )
        refine_state = info.get("final_state", None) if isinstance(info, dict) else None
        if refine_state is not None:
            refine_state = dict(refine_state)
            refine_state["P"] = P_ref
        return Z_cands, None, refine_state

    parents_X = set(parents_from_P(P, Xad_cluster))
    poss_X = pparent.possible_parent_sets(P, Xad_cluster) or [set()]
    res_Xs = [extend_search(P, Xad_cluster, parents_X | set(S)) for S in poss_X]

    from itertools import product

    okA = all(r.ok for r in res_As)
    okX = all(r.ok for r in res_Xs)

    if okA and okX:
        Z_cands = sorted({
            tuple(sorted(set(rA.Z) | set(rX.Z)))
            for rA, rX in product(res_As, res_Xs)
        })
        return Z_cands, None, None

    if not do_refine:
        fail_M: Set[str] = set()
        if not okA:
            fail_M |= {m for r in res_As if not r.ok for m in r.M}
        if not okX:
            fail_M |= {m for r in res_Xs if not r.ok for m in r.M}
        return [], tuple(sorted(fail_M)), None

    if data_train is None or clusters is None or cluster_to_idx is None:
        raise ValueError("do_refine=True requires data_train, clusters, cluster_to_idx")
    P_ref, Z_cands, info = learn_adjustment_candidates_with_refinement(
        data_train, clusters, cluster_to_idx, A_cluster,
        HAS_XAD=True, Xad_cluster=Xad_cluster,
        cluster_types=cluster_types, y_cluster=y_cluster,
        max_iters=max_iters, verbose=verbose,
    )
    refine_state = info.get("final_state", None) if isinstance(info, dict) else None
    if refine_state is not None:
        refine_state = dict(refine_state)
        refine_state["P"] = P_ref
    return Z_cands, None, refine_state


def learn_adjustment_candidates_with_refinement(
    data_train,
    clusters: List[str],
    cluster_to_idx: Dict[str, List[int]],
    A_cluster: str,
    *,
    HAS_XAD: bool = False,
    Xad_cluster: Optional[str] = None,
    cluster_types: Optional[Mapping[str, str]] = None,
    y_cluster: Optional[str] = None,
    max_iters: int = 20,
    verbose: bool = False,
) -> Tuple[Any, List[Tuple[str, ...]], Dict[str, Any]]:
    """Iterative refinement: re-learn CPDAG and split problematic clusters until stable.

    Each iteration:
      1. Learns a cluster CPDAG via cloc.cloc_learn.
      2. If no connection marks near A/Xad, builds Z_cands from possible-parent sets.
      3. If connection marks are relevant, uses extend_search.
      4. Splits centers with connection-mark collisions into singletons and repeats.
    """
    if HAS_XAD and not Xad_cluster:
        raise ValueError("HAS_XAD=True requires Xad_cluster to be provided")

    info: Dict[str, Any] = {"iters": []}

    cur_clusters, cur_map, cur_types = _filter_inputs_excluding_y(
        list(clusters),
        {c: list(vs) for c, vs in cluster_to_idx.items()},
        cluster_types,
        y_cluster,
    )

    protected = {A_cluster}
    endpoints = [A_cluster]
    if HAS_XAD:
        protected.add(Xad_cluster)
        endpoints.append(Xad_cluster)

    def _cands_from_extend(P_now: Any) -> Tuple[List[Tuple[str, ...]], bool]:
        parents_A = set(parents_from_P(P_now, A_cluster))
        res_As = [
            extend_search(P_now, A_cluster, parents_A | set(S))
            for S in (pparent.possible_parent_sets(P_now, A_cluster) or [set()])
        ]
        if not HAS_XAD:
            return [tuple(sorted(r.Z)) for r in res_As], all(bool(r.ok) for r in res_As)

        parents_X = set(parents_from_P(P_now, Xad_cluster))
        res_Xs = [
            extend_search(P_now, Xad_cluster, parents_X | set(S))
            for S in (pparent.possible_parent_sets(P_now, Xad_cluster) or [set()])
        ]
        from itertools import product
        Z_cands = sorted({
            tuple(sorted(set(rA.Z) | set(rX.Z)))
            for rA, rX in product(res_As, res_Xs)
        })
        extend_ok = all(bool(r.ok) for r in res_As) and all(bool(r.ok) for r in res_Xs)
        return Z_cands, extend_ok

    for t in range(max_iters):
        P = cloc.cloc_learn(data_train, cur_clusters, cur_map,
                            cluster_types=cur_types, y_cluster=y_cluster)

        has_conn = _has_connection_marks_in_pparents(P, A_cluster, Xad_cluster, HAS_XAD)

        extend_ok = None
        if not has_conn:
            parents_A = parents_from_P(P, A_cluster)
            poss_A    = pparent.possible_parent_sets(P, A_cluster)
            parents_X: Set[str] = set()
            poss_X: List[Tuple[str, ...]] = [tuple()]
            if HAS_XAD:
                parents_X = parents_from_P(P, Xad_cluster)
                poss_X    = pparent.possible_parent_sets(P, Xad_cluster)
            Z_cands = _build_Z_candidates(
                parents_A=parents_A, poss_A=poss_A,
                has_xad=HAS_XAD, parents_X=parents_X, poss_X=poss_X,
            )
        else:
            Z_cands, extend_ok = _cands_from_extend(P)

        centers = _centers_to_refine_for_candidates(P, endpoints=endpoints, Z_candidates=Z_cands)

        info["iters"].append({
            "t": t,
            "n_clusters": len(cur_clusters),
            "has_conn": bool(has_conn),
            "extend_ok": extend_ok,
            "n_Z_cands": len(Z_cands),
            "refine_centers": tuple(sorted(centers)),
        })

        if verbose:
            print(
                f"[iter {t}] #clusters={len(cur_clusters)} has_conn={has_conn} "
                f"extend_ok={extend_ok} #Z={len(Z_cands)} refine={sorted(centers)}"
            )

        if not centers:
            info["final_state"] = {
                "clusters": list(cur_clusters),
                "cluster_to_idx": {k: list(v) for k, v in cur_map.items()},
                "cluster_types": dict(cur_types),
            }
            return P, Z_cands, info

        cur_clusters, cur_map, cur_types = _refine_centers_to_singletons(
            cur_clusters, cur_map, centers, protected, cluster_types=cur_types,
        )

    # max_iters reached: run one final pass
    P = cloc.cloc_learn(data_train, cur_clusters, cur_map,
                        cluster_types=cur_types, y_cluster=y_cluster)
    has_conn = _has_connection_marks_in_pparents(P, A_cluster, Xad_cluster, HAS_XAD)
    if not has_conn:
        parents_A = parents_from_P(P, A_cluster)
        poss_A    = pparent.possible_parent_sets(P, A_cluster)
        parents_X = parents_from_P(P, Xad_cluster) if HAS_XAD else set()
        poss_X    = pparent.possible_parent_sets(P, Xad_cluster) if HAS_XAD else [tuple()]
        Z_cands = _build_Z_candidates(
            parents_A=parents_A, poss_A=poss_A,
            has_xad=HAS_XAD, parents_X=parents_X, poss_X=poss_X,
        )
    else:
        Z_cands, _ = _cands_from_extend(P)

    info["final_state"] = {
        "clusters": list(cur_clusters),
        "cluster_to_idx": {k: list(v) for k, v in cur_map.items()},
        "cluster_types": dict(cur_types),
    }
    return P, Z_cands, info
