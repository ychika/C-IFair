"""
Random-graph cluster ANM generators (linear and nonlinear).

Graph structure: Erdos-Renyi DAG with random cluster partition.
The nonlinear variant applies a single global nonlinearity to all nodes
(including binary nodes before the logistic link) — this is the key
behavioral difference from fixed_anm, which uses per-node functions and
skips nonlinearity for binary nodes.
"""

from typing import List, Tuple, Dict, Set, Optional, Callable
import numpy as np
import torch
import random
import math

from lib_data import _topo_order_from_edges
from ._dag_utils import (
    _sample_strong_uniform,
    _er_dag_edges_with_order,
    _ancestors_of,
    _ensure_A_children,
)
from ._gen_utils import get_nonlinear, _simulate_linear, _simulate_nonlinear


def _set_seed_all(sd: int):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)


def _make_group_weights(idx_list, total_mass: float):
    if len(idx_list) == 0 or total_mass <= 0.0:
        return {}
    tilde = torch.rand(len(idx_list)) + 0.2
    w = (tilde / tilde.sum()) * total_mass
    return {idx_list[i]: float(w[i].item()) for i in range(len(idx_list))}


# =========================
# Main generator: linear synthetic data on a random graph
# =========================
def gen_random_cluster_lin_anm(
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
    import torch, random, numpy as np
    from typing import List, Tuple, Dict, Set

    _set_seed_all(seed)
    g = torch.Generator().manual_seed(seed)

    d = max(4, int(d))

    sizes = [num_nodes_in_cluster for _ in range(d)]
    V = sum(sizes)
    order = list(range(V))
    random.shuffle(order)

    var_edges = _er_dag_edges_with_order(
        V=V, expected_outdeg=expected_outdeg, order=order, g=g, strict=strict_check
    )

    clusters = []
    cur = 0
    for k in range(d):
        clusters.append(list(range(cur, cur + sizes[k])))
        cur += sizes[k]

    C = len(clusters)
    cluster_names = [f"C{i+1}" for i in range(C)]

    idx2cl = {}
    for ci, cl in enumerate(clusters):
        for li, old_idx in enumerate(cl):
            idx2cl[old_idx] = (ci, li)

    cluster_to_vars = {cluster_names[i]: [f"{cluster_names[i]}_{j+1}" for j in range(len(clusters[i]))]
                       for i in range(C)}
    varnames = [None] * V
    for old in range(V):
        ci, li = idx2cl[old]
        varnames[old] = cluster_to_vars[cluster_names[ci]][li]

    pos = {v: i for i, v in enumerate(order)}
    cl_rep = [min(cl, key=lambda v: pos[v]) for cl in clusters]
    cl_pos = [pos[v] for v in cl_rep]
    cl_edges = set()
    for (u, v) in var_edges:
        cu, _ = idx2cl[u]; cv, _ = idx2cl[v]
        if cu == cv: continue
        if cl_pos[cu] < cl_pos[cv]:
            cl_edges.add((cu, cv))
        elif cl_pos[cv] < cl_pos[cu]:
            cl_edges.add((cv, cu))
    cluster_edges = sorted(list(cl_edges))

    deg = [0] * C
    for u, v in cluster_edges:
        deg[u] += 1; deg[v] += 1

    types = ["cont"] * C
    bin_cands = list(range(C)); random.shuffle(bin_cands)
    for i in bin_cands[:max(2, C // 3)]:
        types[i] = "binary"

    cand_A = [i for i in range(C) if deg[i] >= expected_outdeg]
    if not cand_A:
        mdeg = max(deg); cand_A = [i for i in range(C) if deg[i] == mdeg]
    A_idx = random.choice(cand_A)
    types[A_idx] = "binary"

    deg_order = sorted(range(C), key=lambda i: (-deg[i], i))
    Y_idx = None
    for i in deg_order:
        if i != A_idx:
            Y_idx = i; break
    if Y_idx is None:
        Y_idx = deg_order[0] if deg_order[0] != A_idx else deg_order[1]
    types[Y_idx] = "cont"

    Xad_idx = None

    if HAS_XAD:
        A_children = sorted({v for u, v in cluster_edges if u == A_idx})
        if not A_children:
            target_pool = [j for j in range(C) if j != A_idx and j != Y_idx]
            if not target_pool:
                target_pool = [j for j in range(C) if j != A_idx]
            target_c = random.choice(target_pool)
            A_rep = cl_rep[A_idx]
            T_rep = cl_rep[target_c]
            if pos[A_rep] < pos[T_rep]:
                u = random.choice(clusters[A_idx])
                w = random.choice(clusters[target_c])
                if (u, w) not in set(var_edges):
                    var_edges.append((u, w))
            else:
                added = False
                for u in clusters[A_idx]:
                    for w in clusters[target_c]:
                        if pos[u] < pos[w] and (u, w) not in set(var_edges):
                            var_edges.append((u, w)); added = True; break
                    if added: break
            cl_edges = set()
            for (u, v) in var_edges:
                cu, _ = idx2cl[u]; cv, _ = idx2cl[v]
                if cu != cv:
                    if pos[u] < pos[v]: cl_edges.add((cu, cv))
                    else: cl_edges.add((cv, cu))
            cluster_edges = sorted(list(cl_edges))
            A_children = sorted({v for u, v in cluster_edges if u == A_idx})

        tries = 0
        while (not A_children) and (tries < 20):
            minposA = min(pos[u] for u in clusters[A_idx])
            cand = []
            for j in range(C):
                if j == A_idx or j == Y_idx: continue
                mx = max(pos[w] for w in clusters[j])
                if mx > minposA:
                    cand.append(j)
            if not cand:
                cand = [j for j in range(C) if j != A_idx and j != Y_idx] or [j for j in range(C) if j != A_idx]
            target_c = random.choice(cand)
            u_min = min(clusters[A_idx], key=lambda uu: pos[uu])
            w_max = max(clusters[target_c], key=lambda ww: pos[ww])
            if pos[u_min] < pos[w_max] and (u_min, w_max) not in set(var_edges):
                var_edges.append((u_min, w_max))
            else:
                added = False
                for uu in clusters[A_idx]:
                    for ww in clusters[target_c]:
                        if pos[uu] < pos[ww] and (uu, ww) not in set(var_edges):
                            var_edges.append((uu, ww)); added = True; break
                    if added: break
            cl_edges = set()
            for (uu, vv) in var_edges:
                cu, _ = idx2cl[uu]; cv, _ = idx2cl[vv]
                if cu != cv:
                    if pos[uu] < pos[vv]:
                        cl_edges.add((cu, cv))
                    else:
                        cl_edges.add((cv, cu))
            cluster_edges = sorted(list(cl_edges))
            A_children = sorted({v for u, v in cluster_edges if u == A_idx})
            tries += 1

        if not A_children:
            cand2 = [j for j in range(C) if j != A_idx and j != Y_idx] or [j for j in range(C) if j != A_idx]
            Xad_idx = random.choice(cand2)
            u_min = min(clusters[A_idx], key=lambda uu: pos[uu])
            w_max = max(clusters[Xad_idx], key=lambda ww: pos[ww])
            if pos[u_min] < pos[w_max] and (u_min, w_max) not in set(var_edges):
                var_edges.append((u_min, w_max))
        else:
            Xad_idx = random.choice(A_children)
        types[Xad_idx] = "binary"

    nonY_vars = [v for v in range(V) if idx2cl[v][0] != Y_idx]
    Y_vars    = [v for v in range(V) if idx2cl[v][0] == Y_idx]
    for u in nonY_vars:
        for v in Y_vars:
            if (u, v) not in set(var_edges):
                if pos[u] < pos[v]:
                    var_edges.append((u, v))
                else:
                    for vv in Y_vars:
                        if pos[u] < pos[vv] and (u, vv) not in set(var_edges):
                            var_edges.append((u, vv))

    nonY_edges = [(u, v) for (u, v) in var_edges if (idx2cl[u][0] != Y_idx and idx2cl[v][0] != Y_idx)]
    if nonY_edges:
        map_nonY = {node: i for i, node in enumerate(nonY_vars)}
        edges_nonY_mapped = [(map_nonY[u], map_nonY[v]) for (u, v) in nonY_edges]
        order_nonY_idx = _topo_order_from_edges(len(nonY_vars), edges_nonY_mapped)
        ordered_nonY = [nonY_vars[i] for i in order_nonY_idx]
    else:
        ordered_nonY = nonY_vars[:]
    order_full = ordered_nonY + Y_vars
    pos2 = {v: i for i, v in enumerate(order_full)}
    var_edges = [(u, v) for (u, v) in var_edges if pos2[u] < pos2[v]]

    parents = [[] for _ in range(V)]
    for u, v in var_edges: parents[v].append(u)

    A_vars = [i for i in range(V) if idx2cl[i][0] == A_idx]
    Y_set  = set(Y_vars)
    allowed_targets = [i for i in range(V) if i not in A_vars and i not in Y_set]
    var_edges = _ensure_A_children(var_edges, A_vars, allowed_targets, r_min_from_A, Y_set, pos=pos2)
    parents = [[] for _ in range(V)]
    for u, v in var_edges: parents[v].append(u)

    if HAS_XAD:
        cl_edges2 = set()
        for (u, v) in var_edges:
            cu, _ = idx2cl[u]; cv, _ = idx2cl[v]
            if cu != cv:
                cl_edges2.add((cu, cv))
        cluster_edges = sorted(list(cl_edges2))
        A_children = sorted({v for u, v in cluster_edges if u == A_idx})
        if Xad_idx not in A_children:
            if A_children:
                Xad_idx = random.choice(A_children)
                types[Xad_idx] = "binary"

    torch.set_default_dtype(torch.float32)
    W = [None for _ in range(V)]
    b = torch.zeros(V)

    for v in range(V):
        if v in Y_set: continue
        ps = parents[v]
        if len(ps) == 0:
            W[v] = torch.empty(0)
        else:
            W[v] = _sample_strong_uniform((len(ps),), low=weight_low, high=weight_high, generator=g, device='cpu') * base_weight_std

    A_par = [u for u in nonY_vars if idx2cl[u][0] == A_idx]
    X_par = [u for u in nonY_vars if idx2cl[u][0] == Xad_idx] if HAS_XAD else []
    rest  = [u for u in nonY_vars if idx2cl[u][0] not in ({A_idx, Xad_idx} if HAS_XAD else {A_idx})]

    ancA = _ancestors_of(set(A_par), parents)
    rest_ancA  = [u for u in rest if u in ancA]
    rest_nonAnc = [u for u in rest if u not in ancA]

    if not (0.0 <= s_A <= 1.0 and 0.0 <= s_X <= 1.0 and 0.0 <= ratio_anc <= 1.0):
        raise ValueError("s_A, s_X, ratio_anc should be [0,1]")

    if HAS_XAD:
        s_X_eff = s_X
        s_rest = 1.0 - s_A - s_X_eff
        if s_rest < -1e-8:
            raise ValueError(f"s_A + s_X = {s_A+s_X_eff} > 1.0. Be sure to be less than 1.")
        s_rest = max(0.0, s_rest)
    else:
        s_X_eff = 0.0
        s_rest = 1.0 - s_A
        if s_rest < -1e-8:
            raise ValueError(f"s_A = {s_A} > 1.0. Be sure to be in [0..1]")
        s_rest = max(0.0, s_rest)

    s_rest_anc  = s_rest * ratio_anc
    s_rest_nanc = s_rest * (1.0 - ratio_anc)

    wA   = _make_group_weights(A_par,        s_A)
    wX   = _make_group_weights(X_par,        s_X_eff)
    wAnc = _make_group_weights(rest_ancA,    s_rest_anc)
    wNac = _make_group_weights(rest_nonAnc,  s_rest_nanc)

    group_weight_map = {}
    group_weight_map.update(wA); group_weight_map.update(wX)
    group_weight_map.update(wAnc); group_weight_map.update(wNac)

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
                coeff[i] = _sample_strong_uniform((1,), low=low_i, high=high_i, generator=g, device='cpu')[0]
            W[v] = coeff
        else:
            coeff = torch.zeros(len(ps))
            for i, u in enumerate(ps):
                w_u = group_weight_map.get(u, 0.0)
                coeff[i] = w_u
            if float(coeff.abs().sum().item()) < 1e-12:
                coeff = torch.ones(len(ps)) / len(ps)
            W[v] = coeff
        W[v] = W[v] / math.sqrt(len(ps))

    # --- simulate (linear ANM; Y cluster gets y_noise_std, others noise_scale=1) ---
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
        var_is_binary[v] = False if c == Y_idx else (types[c] == "binary")

    sigma = [(float(y_noise_std) if v in Y_set else 1.0) for v in range(V)]
    ftype = ["linear"] * V

    params = {
        "W": W,
        "b": [float(bi) for bi in b],
        "sigma": sigma,
        "ftype": ftype,
        "parents": parents,
    }

    cluster_tensors = {}
    for ci, cname in enumerate(cluster_names):
        idxs = [i for i in range(V) if idx2cl[i][0] == ci]
        cluster_tensors[cname] = X[:, idxs]

    Y_vars_oldidx = [i for i in range(V) if idx2cl[i][0] == Y_idx]
    y0 = X[:, Y_vars_oldidx[0]].float()
    Y_cont = y0.clone()
    thr = torch.median(y0)
    Y_bin = (y0 >= thr).float()
    Y_is_cont = True

    meta = {
        "A":   cluster_names[A_idx],
        "Xad": (cluster_names[Xad_idx] if HAS_XAD else []),
        "Y":   cluster_names[Y_idx],
        "cluster_types": {cluster_names[i]: ("cont" if i == Y_idx else types[i]) for i in range(C)},
        "cluster_edges": [(cluster_names[i], cluster_names[j]) for (i, j) in cluster_edges],
        "var_edges": var_edges,
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


# =========================
# Main generator: nonlinear synthetic data on a random graph
# =========================
def gen_random_cluster_nonlin_anm(
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
    func: str = "tanh",
    assign_f_per_node=None,
):
    import torch, random, numpy as np
    from typing import List, Tuple, Dict, Set

    _set_seed_all(seed)
    g = torch.Generator().manual_seed(seed)

    d = max(4, int(d))

    sizes = [int(num_nodes_in_cluster) for _ in range(d)]
    V = sum(sizes)

    order = list(range(V))
    random.shuffle(order)

    var_edges = _er_dag_edges_with_order(
        V=V, expected_outdeg=expected_outdeg, order=order, g=g, strict=strict_check
    )

    clusters = []
    cur = 0
    for k in range(d):
        clusters.append(list(range(cur, cur + sizes[k])))
        cur += sizes[k]

    C = len(clusters)
    cluster_names = [f"C{i+1}" for i in range(C)]

    idx2cl = {}
    for ci, cl in enumerate(clusters):
        for li, old_idx in enumerate(cl):
            idx2cl[old_idx] = (ci, li)

    cluster_to_vars = {cluster_names[i]: [f"{cluster_names[i]}_{j+1}" for j in range(len(clusters[i]))]
                       for i in range(C)}
    varnames = [None] * V
    for old in range(V):
        ci, li = idx2cl[old]
        varnames[old] = cluster_to_vars[cluster_names[ci]][li]

    pos = {v: i for i, v in enumerate(order)}
    cl_rep = [min(cl, key=lambda v: pos[v]) for cl in clusters]
    cl_pos = [pos[v] for v in cl_rep]
    cl_edges = set()
    for (u, v) in var_edges:
        cu, _ = idx2cl[u]; cv, _ = idx2cl[v]
        if cu == cv: continue
        if cl_pos[cu] < cl_pos[cv]:
            cl_edges.add((cu, cv))
        elif cl_pos[cv] < cl_pos[cu]:
            cl_edges.add((cv, cu))
    cluster_edges = sorted(list(cl_edges))

    deg = [0] * C
    for u, v in cluster_edges:
        deg[u] += 1; deg[v] += 1

    types = ["cont"] * C
    bin_cands = list(range(C)); random.shuffle(bin_cands)
    for i in bin_cands[:max(2, C // 3)]:
        types[i] = "binary"

    cand_A = [i for i in range(C) if deg[i] >= expected_outdeg]
    if not cand_A:
        mdeg = max(deg); cand_A = [i for i in range(C) if deg[i] == mdeg]
    A_idx = random.choice(cand_A)
    types[A_idx] = "binary"

    deg_order = sorted(range(C), key=lambda i: (-deg[i], i))
    Y_idx = None
    for i in deg_order:
        if i != A_idx:
            Y_idx = i; break
    if Y_idx is None:
        Y_idx = deg_order[0] if deg_order[0] != A_idx else deg_order[1]
    types[Y_idx] = "cont"

    Xad_idx = None

    if HAS_XAD:
        A_children = sorted({v for u, v in cluster_edges if u == A_idx})
        if not A_children:
            target_pool = [j for j in range(C) if j != A_idx and j != Y_idx]
            if not target_pool:
                target_pool = [j for j in range(C) if j != A_idx]
            target_c = random.choice(target_pool)
            A_rep = cl_rep[A_idx]
            T_rep = cl_rep[target_c]
            if pos[A_rep] < pos[T_rep]:
                u = random.choice(clusters[A_idx])
                w = random.choice(clusters[target_c])
                if (u, w) not in set(var_edges):
                    var_edges.append((u, w))
            else:
                added = False
                for u in clusters[A_idx]:
                    for w in clusters[target_c]:
                        if pos[u] < pos[w] and (u, w) not in set(var_edges):
                            var_edges.append((u, w)); added = True; break
                    if added: break
            cl_edges = set()
            for (u, v) in var_edges:
                cu, _ = idx2cl[u]; cv, _ = idx2cl[v]
                if cu != cv:
                    if pos[u] < pos[v]: cl_edges.add((cu, cv))
                    else: cl_edges.add((cv, cu))
            cluster_edges = sorted(list(cl_edges))
            A_children = sorted({v for u, v in cluster_edges if u == A_idx})

        tries = 0
        while (not A_children) and (tries < 20):
            minposA = min(pos[u] for u in clusters[A_idx])
            cand = []
            for j in range(C):
                if j == A_idx or j == Y_idx: continue
                mx = max(pos[w] for w in clusters[j])
                if mx > minposA:
                    cand.append(j)
            if not cand:
                cand = [j for j in range(C) if j != A_idx and j != Y_idx] or [j for j in range(C) if j != A_idx]
            target_c = random.choice(cand)
            u_min = min(clusters[A_idx], key=lambda uu: pos[uu])
            w_max = max(clusters[target_c], key=lambda ww: pos[ww])
            if pos[u_min] < pos[w_max] and (u_min, w_max) not in set(var_edges):
                var_edges.append((u_min, w_max))
            else:
                added = False
                for uu in clusters[A_idx]:
                    for ww in clusters[target_c]:
                        if pos[uu] < pos[ww] and (uu, ww) not in set(var_edges):
                            var_edges.append((uu, ww)); added = True; break
                    if added: break
            cl_edges = set()
            for (uu, vv) in var_edges:
                cu, _ = idx2cl[uu]; cv, _ = idx2cl[vv]
                if cu != cv:
                    if pos[uu] < pos[vv]:
                        cl_edges.add((cu, cv))
                    else:
                        cl_edges.add((cv, cu))
            cluster_edges = sorted(list(cl_edges))
            A_children = sorted({v for u, v in cluster_edges if u == A_idx})
            tries += 1

        if not A_children:
            cand2 = [j for j in range(C) if j != A_idx and j != Y_idx] or [j for j in range(C) if j != A_idx]
            Xad_idx = random.choice(cand2)
            u_min = min(clusters[A_idx], key=lambda uu: pos[uu])
            w_max = max(clusters[Xad_idx], key=lambda ww: pos[ww])
            if pos[u_min] < pos[w_max] and (u_min, w_max) not in set(var_edges):
                var_edges.append((u_min, w_max))
        else:
            Xad_idx = random.choice(A_children)
        types[Xad_idx] = "binary"

    nonY_vars = [v for v in range(V) if idx2cl[v][0] != Y_idx]
    Y_vars    = [v for v in range(V) if idx2cl[v][0] == Y_idx]
    for u in nonY_vars:
        for v in Y_vars:
            if (u, v) not in set(var_edges):
                if pos[u] < pos[v]:
                    var_edges.append((u, v))
                else:
                    for vv in Y_vars:
                        if pos[u] < pos[vv] and (u, vv) not in set(var_edges):
                            var_edges.append((u, vv))

    nonY_edges = [(u, v) for (u, v) in var_edges if (idx2cl[u][0] != Y_idx and idx2cl[v][0] != Y_idx)]
    if nonY_edges:
        map_nonY = {node: i for i, node in enumerate(nonY_vars)}
        edges_nonY_mapped = [(map_nonY[u], map_nonY[v]) for (u, v) in nonY_edges]
        order_nonY_idx = _topo_order_from_edges(len(nonY_vars), edges_nonY_mapped)
        ordered_nonY = [nonY_vars[i] for i in order_nonY_idx]
    else:
        ordered_nonY = nonY_vars[:]
    order_full = ordered_nonY + Y_vars
    pos2 = {v: i for i, v in enumerate(order_full)}
    var_edges = [(u, v) for (u, v) in var_edges if pos2[u] < pos2[v]]

    parents = [[] for _ in range(V)]
    for u, v in var_edges: parents[v].append(u)

    A_vars = [i for i in range(V) if idx2cl[i][0] == A_idx]
    Y_set  = set(Y_vars)
    allowed_targets = [i for i in range(V) if i not in A_vars and i not in Y_set]
    var_edges = _ensure_A_children(var_edges, A_vars, allowed_targets, r_min_from_A, Y_set, pos=pos2)
    parents = [[] for _ in range(V)]
    for u, v in var_edges: parents[v].append(u)

    if HAS_XAD:
        cl_edges2 = set()
        for (u, v) in var_edges:
            cu, _ = idx2cl[u]; cv, _ = idx2cl[v]
            if cu != cv:
                cl_edges2.add((cu, cv))
        cluster_edges = sorted(list(cl_edges2))
        A_children = sorted({v for u, v in cluster_edges if u == A_idx})
        if Xad_idx not in A_children:
            if A_children:
                Xad_idx = random.choice(A_children)
                types[Xad_idx] = "binary"

    torch.set_default_dtype(torch.float32)
    W = [None for _ in range(V)]
    b = torch.zeros(V)

    for v in range(V):
        if v in Y_set: continue
        ps = parents[v]
        if len(ps) == 0:
            W[v] = torch.empty(0)
        else:
            W[v] = _sample_strong_uniform((len(ps),), low=weight_low, high=weight_high, generator=g, device='cpu') * base_weight_std

    A_par = [u for u in nonY_vars if idx2cl[u][0] == A_idx]
    X_par = [u for u in nonY_vars if idx2cl[u][0] == Xad_idx] if HAS_XAD else []
    rest  = [u for u in nonY_vars if idx2cl[u][0] not in ({A_idx, Xad_idx} if HAS_XAD else {A_idx})]

    ancA = _ancestors_of(set(A_par), parents)
    rest_ancA  = [u for u in rest if u in ancA]
    rest_nonAnc = [u for u in rest if u not in ancA]

    if not (0.0 <= s_A <= 1.0 and 0.0 <= s_X <= 1.0 and 0.0 <= ratio_anc <= 1.0):
        raise ValueError("s_A, s_X, ratio_anc should be [0,1]")

    if HAS_XAD:
        s_X_eff = s_X
        s_rest = 1.0 - s_A - s_X_eff
        if s_rest < -1e-8:
            raise ValueError(f"s_A + s_X = {s_A+s_X_eff} > 1.0. Be sure to be less than 1")
        s_rest = max(0.0, s_rest)
    else:
        s_X_eff = 0.0
        s_rest = 1.0 - s_A
        if s_rest < -1e-8:
            raise ValueError(f"s_A = {s_A} > 1.0. Be sure to be in [0..1]")
        s_rest = max(0.0, s_rest)

    s_rest_anc  = s_rest * ratio_anc
    s_rest_nanc = s_rest * (1.0 - ratio_anc)

    wA   = _make_group_weights(A_par,        s_A)
    wX   = _make_group_weights(X_par,        s_X_eff)
    wAnc = _make_group_weights(rest_ancA,    s_rest_anc)
    wNac = _make_group_weights(rest_nonAnc,  s_rest_nanc)

    group_weight_map = {}
    group_weight_map.update(wA); group_weight_map.update(wX)
    group_weight_map.update(wAnc); group_weight_map.update(wNac)

    A_par_set = set(A_par)

    for v in Y_vars:
        ps = list(parents[v])
        if len(ps) == 0:
            W[v] = torch.empty(0)
            continue
        if y_weight_mode == "strong_uniform":
            coeff = torch.empty(len(ps))
            for i, u in enumerate(ps):
                if u in A_par_set:
                    low_i  = weight_low  * A_boost
                    high_i = weight_high * A_boost
                else:
                    low_i  = weight_low
                    high_i = weight_high
                coeff[i] = _sample_strong_uniform(
                    (1,), low=low_i, high=high_i, generator=g, device='cpu'
                )[0]
            coeff = coeff / math.sqrt(len(ps))
            W[v] = coeff
        else:
            coeff = torch.zeros(len(ps))
            for i, u in enumerate(ps):
                coeff[i] = group_weight_map.get(u, 0.0)
            if float(coeff.abs().sum().item()) < 1e-12:
                coeff = torch.ones(len(ps)) / len(ps)
            coeff = coeff / math.sqrt(len(ps))
            W[v] = coeff

    # Resolve func before simulation (preserves original random-state order)
    if func is None:
        rng = random.Random(seed)
        func = rng.choice(["sin", "cos", "tanh"])

    # --- simulate (nonlinear ANM; apply_nonlin_to_binary=True matches original behavior) ---
    var_type = [types[idx2cl[v][0]] for v in range(V)]
    ftype_per_var = ["linear" if not parents[v] else func for v in range(V)]
    X = _simulate_nonlinear(
        N, V, order_full, parents, W, b, var_type, ftype_per_var,
        apply_nonlin_to_binary=True,
        Y_set=Y_set, y_noise_std=y_noise_std,
    )

    adj = np.zeros((V, V), dtype=int)
    for u, v in var_edges:
        adj[u, v] = 1

    var_is_binary = [False] * V
    for v in range(V):
        c = idx2cl[v][0]
        var_is_binary[v] = False if c == Y_idx else (types[c] == "binary")

    sigma = [(float(y_noise_std) if v in Y_set else 1.0) for v in range(V)]
    ftype_list = ["nonlinear"] * V

    params = {
        "W": W,
        "b": [float(bi) for bi in b],
        "sigma": sigma,
        "ftype": ftype_list,
        "parents": parents,
        "nonlinear_func_default": func,
        "nonlinear_func_per_node": assign_f_per_node,
    }

    cluster_tensors = {}
    for ci, cname in enumerate(cluster_names):
        idxs = [i for i in range(V) if idx2cl[i][0] == ci]
        cluster_tensors[cname] = X[:, idxs]

    Y_vars_oldidx = [i for i in range(V) if idx2cl[i][0] == Y_idx]
    y0 = X[:, Y_vars_oldidx[0]].float()
    Y_cont = y0.clone()
    thr = torch.median(y0)
    Y_bin = (y0 >= thr).float()
    Y_is_cont = True

    meta = {
        "A":   cluster_names[A_idx],
        "Xad": (cluster_names[Xad_idx] if HAS_XAD else []),
        "Y":   cluster_names[Y_idx],
        "cluster_types": {cluster_names[i]: ("cont" if i == Y_idx else types[i]) for i in range(C)},
        "cluster_edges": [(cluster_names[i], cluster_names[j]) for (i, j) in cluster_edges],
        "var_edges": var_edges,
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


def sample_do_from_scm(
    N: int,
    adj: np.ndarray,
    params: Dict[str, list],
    do_assign: Dict[int, float],
    var_is_binary: Optional[List[bool]] = None,
) -> np.ndarray:
    V = adj.shape[0]
    parents = [list(np.where(adj[:, j] != 0)[0]) for j in range(V)]
    edges = [(u, v) for u in range(V) for v in range(V) if adj[u, v]]
    order = _topo_order_from_edges(V, edges)

    X = torch.zeros((N, V))
    if var_is_binary is None:
        var_is_binary = [False] * V

    for v in order:
        if v in do_assign:
            X[:, v] = float(do_assign[v])
            continue

        ps = parents[v]
        w  = params["W"][v]
        b  = float(params["b"][v])
        s  = float(params["sigma"][v])
        ftype = params["ftype"][v] if params["ftype"][v] is not None else "linear"
        f = get_nonlinear(ftype) if ftype != "linear" else (lambda x: x)
        eps = torch.randn(N) * s

        if len(ps) == 0 or w is None or (hasattr(w, "numel") and w.numel() == 0):
            z = torch.full((N,), float(b))
        else:
            Wv = torch.as_tensor(w, dtype=torch.float32)
            z = X[:, ps].matmul(Wv) + b

        if var_is_binary[v]:
            X[:, v] = torch.bernoulli(torch.sigmoid(z + eps))
        else:
            X[:, v] = f(z) + eps

    return X.numpy()


def make_cluster_do_assign(G, cluster_values):
    name2idx = {v: i for i, v in enumerate(G["varnames"])}
    do_assign = {}
    for cname, val in cluster_values.items():
        vars_in_c = G["cluster_to_vars"][cname]
        if np.isscalar(val):
            vals = [float(val)] * len(vars_in_c)
        else:
            vals = list(val)
            if len(vals) != len(vars_in_c):
                raise ValueError(f"Length mismatch for {cname}: expected {len(vars_in_c)}, got {len(vals)}")
            vals = [float(x) for x in vals]
        for vname, x in zip(vars_in_c, vals):
            do_assign[name2idx[vname]] = x
    return do_assign
