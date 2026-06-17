from typing import Dict
import numpy as np
import pandas as pd


def _cat_to_codes(s: pd.Series, unknown_label: str = "Unknown") -> np.ndarray:
    s = s.astype("object").fillna(unknown_label)
    cats = pd.Categorical(s)
    return cats.codes.astype(np.int64)


def _binarize_series(s: pd.Series, positive_values) -> np.ndarray:
    return s.astype("object").isin(positive_values).astype(np.int64)


def make_cluster_do_assign(G: dict, cluster_values: dict) -> Dict[int, float]:
    """Convert cluster-level value assignments to variable-index assignments."""
    name2idx = {v: i for i, v in enumerate(G["varnames"])}
    do_assign: Dict[int, float] = {}
    for cname, val in cluster_values.items():
        vars_in_c = G["cluster_to_vars"][cname]
        if np.isscalar(val):
            vals = [float(val)] * len(vars_in_c)
        else:
            vals = list(val)
            if len(vals) != len(vars_in_c):
                raise ValueError(
                    f"Length mismatch for {cname}: expected {len(vars_in_c)}, got {len(vals)}"
                )
            vals = [float(x) for x in vals]
        for vname, x in zip(vars_in_c, vals):
            do_assign[name2idx[vname]] = x
    return do_assign
