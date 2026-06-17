from typing import Dict
import numpy as np

from lib_data import _topo_order_from_edges


def _cgmm_sample_interventional(
    G: dict,
    do_assign: Dict[int, float],
    N: int,
    *,
    cgmm_cls,
    cgmm_kwargs: dict,
    postprocess_binary: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    """
    Approximate interventional sampling via per-node conditional GMM regressors.

    Fits one CGMM per non-root variable on observed data, then samples in
    topological order while respecting do-assignments.
    """
    X_obs = np.asarray(G["data"], dtype=np.float32)
    V = X_obs.shape[1]

    var_edges = list(G["meta"]["var_edges"])
    parents = [[] for _ in range(V)]
    for u, v in var_edges:
        parents[v].append(u)
    order = _topo_order_from_edges(V, var_edges)

    var_is_binary = [False] * V
    cluster_types = G["meta"]["cluster_types"]
    for cname, vs in G["cluster_to_vars"].items():
        if cluster_types.get(cname, "cont") == "binary":
            for vname in vs:
                var_is_binary[G["varnames"].index(vname)] = True

    models = [None] * V
    roots_empirical = [None] * V
    for v in range(V):
        ps = parents[v]
        y = X_obs[:, v]
        if len(ps) == 0:
            roots_empirical[v] = y.copy()
            continue
        model = cgmm_cls(**cgmm_kwargs)
        model.fit(X_obs[:, ps], y)
        models[v] = model

    rng = np.random.default_rng(random_state)
    X_do = np.zeros((N, V), dtype=np.float32)

    scalar_like = (int, float, np.floating)
    for k, val in do_assign.items():
        if isinstance(val, (list, np.ndarray)):
            arr = np.asarray(val, dtype=np.float32)
            X_do[:, k] = arr if arr.shape == (N,) else float(arr.reshape(-1)[0])
        elif isinstance(val, scalar_like):
            X_do[:, k] = float(val)
        else:
            X_do[:, k] = float(val)

    for v in order:
        if v in do_assign:
            continue
        ps = parents[v]
        if len(ps) == 0:
            y_emp = roots_empirical[v]
            if y_emp is None or len(y_emp) == 0:
                X_do[:, v] = 0.0
            else:
                X_do[:, v] = rng.choice(y_emp, size=N, replace=True).astype(np.float32)
        else:
            y_smp = models[v].sample(X_do[:, ps], n_samples=1)
            y_smp = np.asarray(y_smp)
            if y_smp.ndim == 3:
                y_smp = y_smp[:, 0, 0]
            elif y_smp.ndim == 2:
                if y_smp.shape[0] == N and y_smp.shape[1] == 1:
                    y_smp = y_smp[:, 0]
                elif y_smp.shape[1] == N and y_smp.shape[0] == 1:
                    y_smp = y_smp[0, :]
                else:
                    y_smp = y_smp.reshape(N)
            else:
                y_smp = y_smp.reshape(N)
            X_do[:, v] = y_smp.astype(np.float32)

        if postprocess_binary and var_is_binary[v]:
            X_do[:, v] = np.clip(np.rint(X_do[:, v]), 0, 1).astype(np.float32)

    return X_do
