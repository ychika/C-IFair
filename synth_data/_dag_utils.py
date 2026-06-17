"""
Shared DAG construction utilities for synthetic data generators.

These helpers appear in random_anm and inadmissible_anm; they are
collected here to avoid duplication without changing their behavior.
"""

from typing import List, Tuple, Set, Optional, Dict
import random
import torch


def _sample_strong_uniform(
    shape,
    low: float = 0.5,
    high: float = 2.0,
    *,
    generator=None,
    device=None,
):
    """Sample coefficients from U([-high,-low] ∪ [low,high]) to ensure non-trivial magnitude."""
    if device is None:
        device = "cpu"
    r = torch.rand(shape, generator=generator, device=device)
    mag = low + (high - low) * r
    sgn = torch.where(
        torch.rand(shape, generator=generator, device=device) < 0.5,
        torch.tensor(-1.0, device=device),
        torch.tensor(1.0, device=device),
    )
    return mag * sgn


def _er_dag_edges_with_order(
    V: int,
    expected_outdeg: float,
    order: List[int],
    *,
    g: torch.Generator,
    strict: bool = True,
) -> List[Tuple[int, int]]:
    """Generate an Erdos-Renyi DAG consistent with a given topological order."""
    if V <= 1:
        return []
    if expected_outdeg > (V - 1) and strict:
        raise ValueError(
            f"[ER-DAG] expected_outdeg={expected_outdeg} > V-1={V-1}, so impossible"
        )
    p = max(0.0, min(1.0, expected_outdeg / (V - 1)))
    edges: List[Tuple[int, int]] = []
    for i in range(V):
        for j in range(i + 1, V):
            u, v = order[i], order[j]
            if torch.rand(1, generator=g).item() < p:
                edges.append((u, v))
    return edges


def _ancestors_of(nodes: Set[int], parents: List[List[int]]) -> Set[int]:
    """Return the set of all ancestors of the given node set."""
    anc: Set[int] = set()
    st = list(nodes)
    while st:
        v = st.pop()
        for u in parents[v]:
            if u not in anc:
                anc.add(u)
                st.append(u)
    return anc


def _ensure_A_children(
    var_edges: List[Tuple[int, int]],
    A_vars: List[int],
    allowed_targets: List[int],
    r_min: int,
    forbid: set,
    *,
    pos: Optional[Dict[int, int]] = None,
) -> List[Tuple[int, int]]:
    """Ensure each variable in the A cluster has out-degree >= r_min to non-forbidden nodes.

    When ``pos`` is provided, only edges respecting the topological ordering
    (pos[u] < pos[w]) are added.  When ``pos`` is None, no ordering constraint
    is applied (used by the inadmissible generator).
    """
    out = {u: 0 for u in A_vars}
    edge_set = set(var_edges)
    for u, v in var_edges:
        if u in A_vars and v not in forbid:
            out[u] += 1
    edges_add = []
    for u in A_vars:
        need = max(0, r_min - out[u])
        if need == 0:
            continue
        cand = [
            w for w in allowed_targets
            if w not in forbid
            and w != u
            and (u, w) not in edge_set
            and (pos is None or pos[u] < pos[w])
        ]
        random.shuffle(cand)
        for w in cand[:need]:
            edges_add.append((u, w))
            edge_set.add((u, w))
    return var_edges + edges_add
