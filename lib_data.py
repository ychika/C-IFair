import torch, random
from typing import List, Tuple, Dict, Optional
import numpy as np


def sample_interventional_dataset(G, dataname: str, do_assign: dict, N_do: int):
    if dataname in ("linear", "nonlinear", "nonlin", "lin_conn", "nonlin_conn", "lin_inadmissible"):
        from synth_data.random_anm import sample_do_from_scm
        V = len(G["varnames"])
        adj = np.zeros((V, V), dtype=int)
        for u, v in G["meta"]["var_edges"]:
            adj[u, v] = 1
        params = G.get("params", None)
        if params is None:
            raise ValueError("params is None: true SCM parameters are required for interventional sampling.")
        var_is_binary = [False] * V
        for cname, typ in G["meta"]["cluster_types"].items():
            if typ == "binary":
                for vname in G["cluster_to_vars"][cname]:
                    var_is_binary[G["varnames"].index(vname)] = True
        return sample_do_from_scm(N=N_do, adj=adj, params=params, do_assign=do_assign, var_is_binary=var_is_binary)
    elif dataname in ("adult", "german"):
        from real_data.adult import sample_adult_interventional_cgmm
        return sample_adult_interventional_cgmm(G, do_assign=do_assign, N=N_do)
    elif dataname in ("oulad",):
        from real_data.oulad import sample_oulad_interventional_cgmm
        return sample_oulad_interventional_cgmm(G, do_assign=do_assign, N=N_do)

    else:
        raise ValueError(f"Unknown dataname={dataname}")


def _set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _topo_order_from_edges(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    indeg = [0] * n
    succ  = [[] for _ in range(n)]
    for u, v in edges:
        indeg[v] += 1; succ[u].append(v)
    Q = [i for i in range(n) if indeg[i] == 0]
    order = []
    while Q:
        u = Q.pop()
        order.append(u)
        for w in succ[u]:
            indeg[w] -= 1
            if indeg[w] == 0: Q.append(w)
    return order if len(order) == n else list(range(n))


def _er_upper_dag(V: int, p: float, g: Optional[torch.Generator] = None) -> List[Tuple[int, int]]:
    if g is None: g = torch.Generator()
    edges = []
    for i in range(V):
        for j in range(i + 1, V):
            if torch.rand(1, generator=g).item() < p:
                edges.append((i, j))
    return edges


def _cluster_edges_from_var_edges(
    V: int,
    var_edges: List[Tuple[int, int]],
    idx2cl: Dict[int, int],
    cl_order: Optional[List[int]] = None,
) -> List[Tuple[int, int]]:
    reps = {}
    for v in range(V):
        c = idx2cl[v]
        reps.setdefault(c, v)
        reps[c] = min(reps[c], v)
    pos = (
        {v: i for i, v in enumerate(sorted(reps.values()))}
        if cl_order is None
        else {v: i for i, v in enumerate(cl_order)}
    )
    out = set()
    for u, v in var_edges:
        cu, cv = idx2cl[u], idx2cl[v]
        if cu == cv: continue
        ru, rv = reps[cu], reps[cv]
        if pos[ru] < pos[rv]:
            out.add((cu, cv))
        else:
            out.add((cv, cu))
    return sorted(out)
