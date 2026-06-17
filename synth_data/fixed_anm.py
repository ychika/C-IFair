"""
Fixed-graph cluster ANM generators (linear and nonlinear).

Graph structure: hardcoded variable-level DAG (Figure A or C style),
with a new sink Y cluster receiving edges from all other variables.

Key behavioral difference from random_anm:
  - nonlinearity is NOT applied to binary nodes (apply_nonlin_to_binary=False)
  - per-node nonlinearity functions (ftype_list) instead of a single global func
  - uniform noise scaling via noise_scale / bin_noise_scale instead of Y_set handling
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional
import math
import random

import numpy as np
import torch

from lib_data import _set_seed, _topo_order_from_edges
from ._gen_utils import _simulate_linear, _simulate_nonlinear


# ---------------------------------------------------------------------
# Fixed variable-level graphs
# ---------------------------------------------------------------------

def _fixed_varnames() -> List[str]:
    return ["X1", "S1", "Z1", "Z2", "Z3", "W1", "R1", "Q1", "A1", "Y1"]


def _fixed_clusters() -> Tuple[List[str], Dict[str, List[str]], Dict[str, str]]:
    """Return (clusters, cluster_to_vars, cluster_types) for the fixed SCM."""
    cluster_to_vars: Dict[str, List[str]] = {
        "X": ["X1"],
        "S": ["S1"],
        "Z": ["Z1", "Z2", "Z3"],
        "W": ["W1"],
        "R": ["R1"],
        "Q": ["Q1"],
        "A": ["A1"],
        "Y": ["Y1"],
    }
    clusters = list(cluster_to_vars.keys())
    cluster_types: Dict[str, str] = {c: "cont" for c in clusters}
    cluster_types["A"] = "bin"
    return clusters, cluster_to_vars, cluster_types


def _fixed_var_edges(graph: str) -> List[Tuple[int, int]]:
    """Return variable-level directed edges as index pairs for graph A or C."""
    if graph == "A":
        base_names = [
            ("X1", "Z1"), ("Z1", "Z2"), ("Z2", "W1"), ("Z3", "Z2"),
            ("Z3", "S1"), ("X1", "R1"), ("S1", "R1"), ("R1", "Q1"), ("W1", "A1"),
        ]
    elif graph == "C":
        base_names = [
            ("X1", "Z1"), ("Z1", "W1"), ("Z2", "Z1"), ("Z3", "Z2"),
            ("S1", "Z2"), ("X1", "R1"), ("S1", "R1"), ("R1", "Q1"), ("W1", "A1"),
        ]
    else:
        raise ValueError(f"Unknown graph: {graph}")

    varnames = _fixed_varnames()
    for v in varnames:
        if v != "Y1":
            base_names.append((v, "Y1"))

    idx = {v: i for i, v in enumerate(varnames)}
    return [(idx[u], idx[v]) for (u, v) in base_names]


# ---------------------------------------------------------------------
# Weight construction helpers (fixed-graph specific)
# ---------------------------------------------------------------------

def _make_linear_W(
    d: int,
    edges: List[Tuple[int, int]],
    *,
    w_low: float = 0.5,
    w_high: float = 1.5,
) -> torch.Tensor:
    W = torch.zeros((d, d), dtype=torch.float32)
    for u, v in edges:
        w = random.uniform(w_low, w_high)
        if random.random() < 0.5:
            w = -w
        W[v, u] = w
    return W


def _make_nonlinear_params(
    d: int,
    edges: List[Tuple[int, int]],
    *,
    w_low: float = 0.5,
    w_high: float = 1.5,
    ftypes: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, Dict[Tuple[int, int], str]]:
    if ftypes is None:
        ftypes = ["tanh", "relu", "sin", "cos", "square", "cube"]
    W = torch.zeros((d, d), dtype=torch.float32)
    fn: Dict[Tuple[int, int], str] = {}
    for u, v in edges:
        w = random.uniform(w_low, w_high)
        if random.random() < 0.5:
            w = -w
        W[v, u] = w
        fn[(u, v)] = random.choice(ftypes)
    return W, fn


# ---------------------------------------------------------------------
# Simulation wrappers (convert fixed-graph params then delegate to _gen_utils)
# ---------------------------------------------------------------------

def _simulate_linear_anm(
    n: int,
    varnames: List[str],
    edges: List[Tuple[int, int]],
    *,
    bin_nodes: Optional[List[int]] = None,
    noise_scale: float = 1.0,
    bin_noise_scale: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Linear ANM on a fixed graph.

    Behavioral difference from random_anm:
      No Y_set handling — all variables share the same noise_scale scheme.
    """
    d = len(varnames)
    if bin_nodes is None:
        bin_nodes = []
    bin_set = set(bin_nodes)

    topo = _topo_order_from_edges(d, edges)
    parents: List[List[int]] = [[] for _ in range(d)]
    for u, v in edges:
        parents[v].append(u)
    parents = [sorted(ps) for ps in parents]

    W_full = _make_linear_W(d, edges)
    b = torch.zeros(d, dtype=torch.float32)

    W_list: List[Optional[torch.Tensor]] = [None] * d
    sigma_list: List[float] = [0.0] * d
    ftype_list: List[str] = ["linear"] * d
    for v in range(d):
        ps = parents[v]
        if ps:
            W_list[v] = torch.tensor([float(W_full[v, u]) for u in ps], dtype=torch.float32)
        sigma_list[v] = float(bin_noise_scale if v in bin_set else noise_scale)
        ftype_list[v] = "sigmoid" if v in bin_set else "linear"

    var_type = ["binary" if v in bin_set else "cont" for v in range(d)]
    X_tensor = _simulate_linear(
        n, d, topo, parents, W_list, b, var_type,
        Y_set=None, noise_scale=noise_scale, bin_noise_scale=bin_noise_scale,
    )

    params = {
        "W": W_list,
        "b": [float(x) for x in b.tolist()],
        "sigma": sigma_list,
        "ftype": ftype_list,
        "edges": edges,
        "topo": topo,
        "bin_nodes": sorted(bin_set),
    }
    return X_tensor.numpy(), params


def _simulate_nonlinear_anm(
    n: int,
    varnames: List[str],
    edges: List[Tuple[int, int]],
    *,
    bin_nodes: Optional[List[int]] = None,
    noise_scale: float = 1.0,
    bin_noise_scale: float = 1.0,
    ftypes: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Nonlinear ANM on a fixed graph with per-node nonlinearity.

    Behavioral difference from random_anm:
      - Binary nodes use linear pre-activation (apply_nonlin_to_binary=False).
      - Each non-binary node independently draws its nonlinearity from ftypes.
    """
    d = len(varnames)
    if bin_nodes is None:
        bin_nodes = []
    bin_set = set(bin_nodes)
    if ftypes is None:
        ftypes = ["tanh", "relu", "sin", "cos", "square", "cube"]

    topo = _topo_order_from_edges(d, edges)
    parents: List[List[int]] = [[] for _ in range(d)]
    for u, v in edges:
        parents[v].append(u)
    parents = [sorted(ps) for ps in parents]

    W_full = _make_linear_W(d, edges)
    b = torch.zeros(d, dtype=torch.float32)

    ftype_list: List[str] = ["linear"] * d
    for v in range(d):
        if v in bin_set:
            ftype_list[v] = "sigmoid"
        else:
            ftype_list[v] = random.choice(ftypes) if parents[v] else "linear"

    W_list: List[Optional[torch.Tensor]] = [None] * d
    sigma_list: List[float] = [0.0] * d
    for v in range(d):
        ps = parents[v]
        if ps:
            W_list[v] = torch.tensor([float(W_full[v, u]) for u in ps], dtype=torch.float32)
        sigma_list[v] = float(bin_noise_scale if v in bin_set else noise_scale)

    var_type = ["binary" if v in bin_set else "cont" for v in range(d)]
    X_tensor = _simulate_nonlinear(
        n, d, topo, parents, W_list, b, var_type, ftype_list,
        apply_nonlin_to_binary=False,
        Y_set=None, noise_scale=noise_scale, bin_noise_scale=bin_noise_scale,
    )

    params = {
        "W": W_list,
        "b": [float(x) for x in b.tolist()],
        "sigma": sigma_list,
        "ftype": ftype_list,
        "edges": edges,
        "topo": topo,
        "bin_nodes": sorted(bin_set),
    }
    return X_tensor.numpy(), params


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def get_fixed_cluster_anm(
    *,
    n: int = 5000,
    seed: int = 0,
    graph: str = "C",
    noise_scale: float = 0.1,
    bin_noise_scale: float = 0.1,
) -> Dict[str, Any]:
    """Fixed-graph linear cluster ANM generator."""
    _set_seed(seed)

    varnames = _fixed_varnames()
    clusters, cluster_to_vars, cluster_types = _fixed_clusters()
    edges = _fixed_var_edges(graph)

    bin_nodes = [varnames.index("A1")]

    data, params = _simulate_linear_anm(
        n=n, varnames=varnames, edges=edges,
        bin_nodes=bin_nodes, noise_scale=noise_scale, bin_noise_scale=bin_noise_scale,
    )

    d = len(varnames)
    adj = np.zeros((d, d), dtype=int)
    for u, v in edges:
        adj[u, v] = 1

    v2i = {v: i for i, v in enumerate(varnames)}
    X = torch.from_numpy(data).float()

    cluster_tensors = {}
    for c, vs in cluster_to_vars.items():
        idxs = [v2i[v] for v in vs]
        cluster_tensors[c] = X[:, idxs]

    v2cl: Dict[int, int] = {}
    for ci, cname in enumerate(clusters):
        for v in cluster_to_vars[cname]:
            v2cl[v2i[v]] = ci
    cluster_edge_idx = set()
    for u, v in edges:
        cu, cv = v2cl[u], v2cl[v]
        if cu != cv:
            cluster_edge_idx.add((cu, cv))
    cluster_edges = [(clusters[i], clusters[j]) for (i, j) in sorted(cluster_edge_idx)]

    y_idx = v2i["Y1"]
    Y_cont = torch.from_numpy(data[:, y_idx].copy()).to(dtype=torch.float32)
    med = torch.median(Y_cont)
    Y_bin = (Y_cont > med).to(dtype=torch.long)
    Y_is_cont = True

    var_is_binary = [False] * d
    for j in bin_nodes:
        var_is_binary[j] = True

    meta = {
        "A":             "A",
        "Xad":           [],
        "Y":             "Y",
        "var_edges":     edges,
        "cluster_types": cluster_types,
        "cluster_edges": cluster_edges,
        "graph":         str(graph).upper().strip(),
        "scm":           "linear-anm-fixed",
    }

    return {
        "data":            data,
        "varnames":        varnames,
        "clusters":        clusters,
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


def get_fixed_cluster_nonlin_anm(
    *,
    n: int = 5000,
    seed: int = 0,
    graph: str = "C",
    noise_scale: float = 0.1,
    bin_noise_scale: float = 0.1,
    ftypes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fixed-graph nonlinear cluster ANM generator."""
    _set_seed(seed)

    varnames = _fixed_varnames()
    clusters, cluster_to_vars, cluster_types = _fixed_clusters()
    edges = _fixed_var_edges(graph)

    bin_nodes = [varnames.index("A1")]

    data, params = _simulate_nonlinear_anm(
        n=n, varnames=varnames, edges=edges,
        bin_nodes=bin_nodes, noise_scale=noise_scale, bin_noise_scale=bin_noise_scale,
        ftypes=ftypes,
    )

    d = len(varnames)
    adj = np.zeros((d, d), dtype=int)
    for u, v in edges:
        adj[u, v] = 1

    v2i = {v: i for i, v in enumerate(varnames)}
    X = torch.from_numpy(data).float()

    cluster_tensors = {}
    for c, vs in cluster_to_vars.items():
        idxs = [v2i[v] for v in vs]
        cluster_tensors[c] = X[:, idxs]

    v2cl: Dict[int, int] = {}
    for ci, cname in enumerate(clusters):
        for v in cluster_to_vars[cname]:
            v2cl[v2i[v]] = ci
    cluster_edge_idx = set()
    for u, v in edges:
        cu, cv = v2cl[u], v2cl[v]
        if cu != cv:
            cluster_edge_idx.add((cu, cv))
    cluster_edges = [(clusters[i], clusters[j]) for (i, j) in sorted(cluster_edge_idx)]

    y_idx = v2i["Y1"]
    Y_cont = torch.from_numpy(data[:, y_idx].copy()).to(dtype=torch.float32)
    med = torch.median(Y_cont)
    Y_bin = (Y_cont > med).to(dtype=torch.long)
    Y_is_cont = True

    var_is_binary = [False] * d
    for j in bin_nodes:
        var_is_binary[j] = True

    meta = {
        "A":             "A",
        "Xad":           [],
        "Y":             "Y",
        "var_edges":     edges,
        "cluster_types": cluster_types,
        "cluster_edges": cluster_edges,
        "graph":         str(graph).upper().strip(),
        "scm":           "nonlinear-anm-fixed",
    }

    return {
        "data":            data,
        "varnames":        varnames,
        "clusters":        clusters,
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
