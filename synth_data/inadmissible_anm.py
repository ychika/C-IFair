"""
Inadmissible-partition cluster ANM generator (linear only).

Graph structure: Erdos-Renyi DAG with a cluster partition that is crafted
to contain a directed 2-cycle at the cluster level even though the
variable-level graph is acyclic.  This violates the admissibility assumption
and is used to stress-test methods that rely on it.

The data generation loop is identical to random_anm (same linear ANM with
Y_set noise scaling).  Only the graph / partition construction differs.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Set, Optional
import math
import random

import numpy as np
import torch

from lib_data import _topo_order_from_edges
from ._dag_utils import (
    _sample_strong_uniform,
    _er_dag_edges_with_order,
    _ancestors_of,
    _ensure_A_children,
)
from ._gen_utils import _simulate_linear


# ---------------------------------------------------------------------
# Inadmissible-partition specific helpers
# ---------------------------------------------------------------------

def _directed_cycle_dfs(n: int, edges: List[Tuple[int, int]]) -> Optional[List[int]]:
    """Return one directed cycle as a list of cluster indices, or None if acyclic."""
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n:
            adj[u].append(v)

    state = [0] * n
    parent = [-1] * n

    def dfs(u: int) -> Optional[List[int]]:
        state[u] = 1
        for v in adj[u]:
            if state[v] == 0:
                parent[v] = u
                cyc = dfs(v)
                if cyc is not None:
                    return cyc
            elif state[v] == 1:
                cycle = [v]
                cur = u
                while cur != -1 and cur != v:
                    cycle.append(cur)
                    cur = parent[cur]
                cycle.append(v)
                cycle.reverse()
                return cycle
        state[u] = 2
        return None

    for s in range(n):
        if state[s] == 0:
            cyc = dfs(s)
            if cyc is not None:
                return cyc
    return None


def _pick_witness_edges_for_2cycle(
    V: int,
    var_edges: List[Tuple[int, int]],
    order: List[int],
    *,
    g: torch.Generator,
    max_tries: int = 2000,
) -> Tuple[List[Tuple[int, int]], Tuple[int, int, int, int]]:
    """Pick or create two variable edges (u1->v1, u2->v2) with 4 distinct nodes.

    Clusters are then built as C0={u1,v2,...} and C1={v1,u2,...}, guaranteeing
    a 2-cycle C0->C1 and C1->C0 at the cluster level.
    """
    edges = list(dict.fromkeys(var_edges))
    if len(edges) >= 2:
        idxs = list(range(len(edges)))
        random.shuffle(idxs)
        for _ in range(min(max_tries, len(idxs) * 3)):
            e1 = edges[random.choice(idxs)]
            e2 = edges[random.choice(idxs)]
            if e1 == e2:
                continue
            u1, v1 = e1
            u2, v2 = e2
            if len({u1, v1, u2, v2}) == 4:
                return edges, (u1, v1, u2, v2)

    if V < 4:
        raise ValueError("V must be >= 4 to force an inadmissible partition via a 2-cycle witness.")
    pos = {v: i for i, v in enumerate(order)}
    sorted_nodes = sorted(range(V), key=lambda x: pos[x])
    u1 = sorted_nodes[0]
    v1 = sorted_nodes[V // 2]
    u2 = sorted_nodes[(V // 2) - 1]
    v2 = sorted_nodes[-1]

    uniq = {u1, v1, u2, v2}
    if len(uniq) < 4:
        uniq_list = []
        for n in sorted_nodes:
            if n not in uniq_list:
                uniq_list.append(n)
            if len(uniq_list) == 4:
                break
        u1, v1, u2, v2 = uniq_list[0], uniq_list[1], uniq_list[2], uniq_list[3]
        if pos[u1] > pos[v1]:
            u1, v1 = v1, u1
        if pos[u2] > pos[v2]:
            u2, v2 = v2, u2

    edge_set = set(edges)
    if (u1, v1) not in edge_set:
        edges.append((u1, v1))
        edge_set.add((u1, v1))
    if (u2, v2) not in edge_set:
        edges.append((u2, v2))
        edge_set.add((u2, v2))
    return edges, (u1, v1, u2, v2)


# =========================
# Main generator: linear synthetic data on an inadmissible cluster partition
# =========================

def gen_random_cluster_lin_anm_inadmissible(
    seed: int = 23,
    d: int = 6,
    N: int = 4000,
    expected_outdeg: float = 2.0,
    s_A: float = 0.5,
    s_X: float = 0.15,
    ratio_anc: float = 0.6,
    y_noise_std: float = 1.0,
    base_weight_std: float = 1.0,
    weight_low: float = 1.0,
    weight_high: float = 2.0,
    A_boost: float = 5.0,
    y_weight_mode: str = "strong_uniform",
    r_min_from_A: int = 3,
    strict_check: bool = True,
    HAS_XAD: bool = True,
    num_nodes_in_cluster: int = 3,
):
    # Inline seed setting (preserves original sequence)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator().manual_seed(seed)

    d = max(4, int(d))
    num_nodes_in_cluster = int(num_nodes_in_cluster)
    if num_nodes_in_cluster < 2:
        raise ValueError("num_nodes_in_cluster must be >= 2 to construct an inadmissible partition.")

    sizes = [num_nodes_in_cluster for _ in range(d)]
    V = sum(sizes)

    order = list(range(V))
    random.shuffle(order)
    var_edges = _er_dag_edges_with_order(
        V=V, expected_outdeg=expected_outdeg, order=order, g=g, strict=strict_check,
    )

    # Craft partition with guaranteed 2-cycle at cluster level
    var_edges, (u1, v1, u2, v2) = _pick_witness_edges_for_2cycle(V, var_edges, order, g=g)

    used = {u1, v1, u2, v2}
    remaining = [i for i in range(V) if i not in used]
    random.shuffle(remaining)

    c0 = [u1, v2]
    while len(c0) < num_nodes_in_cluster:
        c0.append(remaining.pop())

    c1 = [v1, u2]
    while len(c1) < num_nodes_in_cluster:
        c1.append(remaining.pop())

    clusters: List[List[int]] = [c0, c1]
    for _ in range(d - 2):
        cl = [remaining.pop() for _ in range(num_nodes_in_cluster)]
        clusters.append(cl)

    flat = [v for cl in clusters for v in cl]
    assert len(flat) == V and len(set(flat)) == V, "Partition construction failed (duplicate/missing nodes)."

    C = len(clusters)
    cluster_names = [f"C{i+1}" for i in range(C)]

    idx2cl: Dict[int, Tuple[int, int]] = {}
    for ci, cl in enumerate(clusters):
        for li, var_idx in enumerate(cl):
            idx2cl[var_idx] = (ci, li)

    cluster_to_vars = {
        cluster_names[i]: [f"{cluster_names[i]}_{j+1}" for j in range(len(clusters[i]))]
        for i in range(C)
    }
    varnames = [None] * V
    for v in range(V):
        ci, li = idx2cl[v]
        varnames[v] = cluster_to_vars[cluster_names[ci]][li]

    cluster_edge_set: Set[Tuple[int, int]] = set()
    for u, v in var_edges:
        cu, _ = idx2cl[u]
        cv, _ = idx2cl[v]
        if cu != cv:
            cluster_edge_set.add((cu, cv))
    cluster_edges = sorted(cluster_edge_set)

    cyc = _directed_cycle_dfs(C, cluster_edges)

    deg = [0] * C
    for a, b in cluster_edges:
        deg[a] += 1
        deg[b] += 1

    types = ["cont"] * C
    bin_cands = list(range(C))
    random.shuffle(bin_cands)
    for i in bin_cands[:max(2, C // 3)]:
        types[i] = "binary"

    cand_A = [i for i in range(C) if deg[i] >= expected_outdeg]
    if not cand_A:
        mdeg = max(deg)
        cand_A = [i for i in range(C) if deg[i] == mdeg]
    A_idx = random.choice(cand_A)
    types[A_idx] = "binary"

    cycle_clusters = {0, 1}
    candidates_Y = [i for i in range(C) if i != A_idx and i not in cycle_clusters]
    if not candidates_Y:
        candidates_Y = [i for i in range(C) if i != A_idx]
    Y_idx = max(candidates_Y, key=lambda i: (deg[i], -i))
    types[Y_idx] = "cont"

    Xad_idx = None
    if HAS_XAD:
        A_children = sorted({v for u, v in cluster_edges if u == A_idx and v != A_idx})
        if not A_children:
            target_pool = [j for j in range(C) if j != A_idx and j != Y_idx]
            if not target_pool:
                target_pool = [j for j in range(C) if j != A_idx]
            target_c = random.choice(target_pool)
            pos = {v: i for i, v in enumerate(order)}
            uA = min(clusters[A_idx], key=lambda x: pos[x])
            vT = max(clusters[target_c], key=lambda x: pos[x])
            if pos[uA] < pos[vT]:
                var_edges.append((uA, vT))
            else:
                added = False
                for uu in clusters[A_idx]:
                    for vv in clusters[target_c]:
                        if pos[uu] < pos[vv]:
                            var_edges.append((uu, vv))
                            added = True
                            break
                    if added:
                        break
            cluster_edge_set = set()
            for uu, vv in var_edges:
                cu, _ = idx2cl[uu]
                cv, _ = idx2cl[vv]
                if cu != cv:
                    cluster_edge_set.add((cu, cv))
            cluster_edges = sorted(cluster_edge_set)
            A_children = sorted({v for u, v in cluster_edges if u == A_idx and v != A_idx})

        if A_children:
            Xad_idx = random.choice(A_children)
        else:
            Xad_idx = random.choice([i for i in range(C) if i not in {A_idx, Y_idx}])
        types[Xad_idx] = "binary"

    nonY_vars = [v for v in range(V) if idx2cl[v][0] != Y_idx]
    Y_vars = [v for v in range(V) if idx2cl[v][0] == Y_idx]
    pos = {v: i for i, v in enumerate(order)}
    edge_set = set(var_edges)

    for u in nonY_vars:
        for v in Y_vars:
            if (u, v) in edge_set:
                continue
            if pos[u] < pos[v]:
                var_edges.append((u, v))
                edge_set.add((u, v))
            else:
                for vv in Y_vars:
                    if pos[u] < pos[vv] and (u, vv) not in edge_set:
                        var_edges.append((u, vv))
                        edge_set.add((u, vv))
                        break

    nonY_edges = [(u, v) for (u, v) in var_edges if idx2cl[u][0] != Y_idx and idx2cl[v][0] != Y_idx]
    if nonY_edges:
        map_nonY = {node: i for i, node in enumerate(nonY_vars)}
        edges_nonY_mapped = [(map_nonY[u], map_nonY[v]) for (u, v) in nonY_edges]
        order_nonY_idx = _topo_order_from_edges(len(nonY_vars), edges_nonY_mapped)
        ordered_nonY = [nonY_vars[i] for i in order_nonY_idx]
    else:
        ordered_nonY = list(nonY_vars)
    order_full = ordered_nonY + list(Y_vars)
    pos2 = {v: i for i, v in enumerate(order_full)}
    var_edges = [(u, v) for (u, v) in var_edges if pos2[u] < pos2[v]]
    edge_set = set(var_edges)

    parents: List[List[int]] = [[] for _ in range(V)]
    for u, v in var_edges:
        parents[v].append(u)

    A_vars = list(clusters[A_idx])
    Y_set = set(Y_vars)
    allowed_targets = [i for i in range(V) if i not in set(A_vars) and i not in Y_set]
    # Note: no pos constraint (inadmissible behavior); _ensure_A_children defaults pos=None
    var_edges = _ensure_A_children(var_edges, A_vars, allowed_targets, r_min=r_min_from_A, forbid=Y_set)

    parents = [[] for _ in range(V)]
    for u, v in var_edges:
        parents[v].append(u)

    torch.set_default_dtype(torch.float32)
    W: List[torch.Tensor] = [None for _ in range(V)]  # type: ignore
    b = torch.zeros(V)

    for v in range(V):
        if v in Y_set:
            continue
        ps = parents[v]
        if len(ps) == 0:
            W[v] = torch.empty(0)
        else:
            W[v] = _sample_strong_uniform(
                (len(ps),), low=weight_low, high=weight_high, generator=g, device="cpu"
            ) * base_weight_std

    A_par = [u for u in nonY_vars if idx2cl[u][0] == A_idx]
    X_par = [u for u in nonY_vars if (HAS_XAD and Xad_idx is not None and idx2cl[u][0] == Xad_idx)] if HAS_XAD else []

    if HAS_XAD and Xad_idx is not None:
        rest = [u for u in nonY_vars if idx2cl[u][0] not in {A_idx, Xad_idx}]
    else:
        rest = [u for u in nonY_vars if idx2cl[u][0] != A_idx]

    ancA = _ancestors_of(set(A_par), parents)
    rest_ancA  = [u for u in rest if u in ancA]
    rest_nonAnc = [u for u in rest if u not in ancA]

    if not (0.0 <= s_A <= 1.0 and 0.0 <= s_X <= 1.0 and 0.0 <= ratio_anc <= 1.0):
        raise ValueError("s_A, s_X, ratio_anc should be in [0,1]")

    if HAS_XAD:
        s_X_eff = s_X
        s_rest = 1.0 - s_A - s_X_eff
        if s_rest < -1e-8:
            raise ValueError(f"s_A + s_X = {s_A+s_X_eff} > 1.0.")
        s_rest = max(0.0, s_rest)
    else:
        s_X_eff = 0.0
        s_rest = 1.0 - s_A
        if s_rest < -1e-8:
            raise ValueError(f"s_A = {s_A} > 1.0.")
        s_rest = max(0.0, s_rest)

    s_rest_anc  = s_rest * ratio_anc
    s_rest_nanc = s_rest * (1.0 - ratio_anc)

    def make_group_weights(idx_list, total_mass: float):
        if len(idx_list) == 0 or total_mass <= 0.0:
            return {}
        tilde = torch.rand(len(idx_list)) + 0.2
        w = (tilde / tilde.sum()) * total_mass
        return {idx_list[i]: float(w[i].item()) for i in range(len(idx_list))}

    wA   = make_group_weights(A_par,        s_A)
    wX   = make_group_weights(X_par,        s_X_eff)
    wAnc = make_group_weights(rest_ancA,    s_rest_anc)
    wNac = make_group_weights(rest_nonAnc,  s_rest_nanc)
    group_weight_map: Dict[int, float] = {}
    group_weight_map.update(wA)
    group_weight_map.update(wX)
    group_weight_map.update(wAnc)
    group_weight_map.update(wNac)

    for v in Y_vars:
        ps = list(parents[v])
        if len(ps) == 0:
            W[v] = torch.empty(0)
            continue
        if y_weight_mode == "strong_uniform":
            coeff = torch.empty(len(ps))
            for i, u in enumerate(ps):
                if u in set(A_par):
                    low_i  = weight_low  * A_boost
                    high_i = weight_high * A_boost
                else:
                    low_i  = weight_low
                    high_i = weight_high
                coeff[i] = _sample_strong_uniform(
                    (1,), low=low_i, high=high_i, generator=g, device="cpu"
                )[0]
            W[v] = coeff
        else:
            coeff = torch.zeros(len(ps))
            for i, u in enumerate(ps):
                coeff[i] = group_weight_map.get(u, 0.0)
            if float(coeff.abs().sum().item()) < 1e-12:
                coeff = torch.ones(len(ps)) / len(ps)
            W[v] = coeff
        W[v] = W[v] / math.sqrt(len(ps))

    # --- simulate (identical loop to random_anm linear generator) ---
    var_type = [types[idx2cl[v][0]] for v in range(V)]
    X = _simulate_linear(
        N, V, order_full, parents, W, b, var_type,
        Y_set=Y_set, y_noise_std=y_noise_std,
    )

    adj = np.zeros((V, V), dtype=int)
    for u, v in var_edges:
        adj[u, v] = 1

    var_is_binary = [False] * V
    for v in range(V):
        c = idx2cl[v][0]
        var_is_binary[v] = False if c == Y_idx else types[c] == "binary"

    sigma = [float(y_noise_std) if v in Y_set else 1.0 for v in range(V)]
    ftype = ["linear"] * V
    params = {
        "W": W,
        "b": [float(bi) for bi in b],
        "sigma": sigma,
        "ftype": ftype,
        "parents": parents,
    }

    cluster_edge_set = set()
    for uu, vv in var_edges:
        cu, _ = idx2cl[uu]
        cv, _ = idx2cl[vv]
        if cu != cv:
            cluster_edge_set.add((cu, cv))
    cluster_edges_final = sorted(cluster_edge_set)
    cyc_final = _directed_cycle_dfs(C, cluster_edges_final)

    cluster_tensors: Dict[str, torch.Tensor] = {}
    for ci, cname in enumerate(cluster_names):
        cluster_tensors[cname] = X[:, clusters[ci]]

    y0 = X[:, clusters[Y_idx][0]].float()
    Y_cont = y0.clone()
    thr = torch.median(y0)
    Y_bin = (y0 >= thr).float()
    Y_is_cont = True

    var_to_cluster = [idx2cl[v][0] for v in range(V)]
    partition_var_indices = [list(cl) for cl in clusters]

    meta = {
        "A":   cluster_names[A_idx],
        "Xad": (cluster_names[Xad_idx] if HAS_XAD and Xad_idx is not None else []),
        "Y":   cluster_names[Y_idx],
        "cluster_types": {cluster_names[i]: ("cont" if i == Y_idx else types[i]) for i in range(C)},
        "cluster_edges": [(cluster_names[i], cluster_names[j]) for (i, j) in cluster_edges_final],
        "var_edges": list(var_edges),
        "inadmissible_partition": True,
        "cluster_has_cycle": (cyc_final is not None),
        "cluster_cycle": ([cluster_names[i] for i in cyc_final] if cyc_final is not None else None),
        "partition_var_indices": partition_var_indices,
        "var_to_cluster": var_to_cluster,
        "cycle_witness_var_edges": [(u1, v1), (u2, v2)],
        "cycle_witness_clusters": (
            (cluster_names[idx2cl[u1][0]], cluster_names[idx2cl[v1][0]]),
            (cluster_names[idx2cl[u2][0]], cluster_names[idx2cl[v2][0]]),
        ),
    }

    return {
        "data":            X.numpy(),
        "varnames":        varnames,
        "clusters":        cluster_names,
        "cluster_to_vars": cluster_to_vars,
        "cluster_tensors": cluster_tensors,
        "meta":            meta,
        "Y_bin":           Y_bin,
        "Y_cont":          Y_cont,
        "Y_is_cont":       Y_is_cont,
        "params":          params,
        "adj":             adj,
        "var_is_binary":   var_is_binary,
    }
