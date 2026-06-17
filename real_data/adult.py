import numpy as np
import pandas as pd
import torch
from fairlearn.datasets import fetch_adult as adult

from lib_data import _set_seed, _topo_order_from_edges
from ._utils import _cat_to_codes, _binarize_series
from ._edges import _build_true_edges_adult
from ._sampling import _cgmm_sample_interventional
from cgmm import ConditionalGMMRegressor


def load_adult_as_clusters(N: int = None, seed: int = 23, HAS_XAD: bool = True):
    """
    Load the Adult (fairlearn) dataset as a cluster graph dict.

      - A   = [sex_is_male(0/1), race_is_white(0/1)]
      - Xad = [occupation]
      - C1  = [marital-status, relationship]
      - C2  = [age]
      - C4  = [hours-per-week]
      - C5  = [workclass]
      - C3  = [education, native-country]
      - Y   = [Y_1]  (target: >50K->1, <=50K->0)

    Returns a dict with keys:
      data, varnames, clusters, cluster_to_vars, cluster_tensors, meta,
      Y_bin, Y_cont, Y_is_cont
    """
    _set_seed(seed)

    ds = adult(as_frame=True)
    X_df: pd.DataFrame = ds["data"].copy()
    y_raw: pd.Series = ds["target"].copy()

    y_bin = (y_raw.astype(str) == ">50K").astype(np.int64).to_numpy()

    need_cols = [
        "sex", "race",
        "occupation",
        "marital-status", "relationship",
        "age",
        "hours-per-week",
        "workclass",
        "education", "native-country",
    ]
    X = X_df[need_cols].copy()

    for col in ["age", "hours-per-week"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

    A_sex  = _binarize_series(X["sex"],  positive_values={"Male"})
    A_race = _binarize_series(X["race"], positive_values={"White"})

    Xad_occ = _cat_to_codes(X["occupation"])
    C1_mar  = _cat_to_codes(X["marital-status"])
    C1_rel  = _cat_to_codes(X["relationship"])
    C2_age  = X["age"].to_numpy().astype(np.float32)
    C4_hour = X["hours-per-week"].to_numpy().astype(np.float32)
    C5_wc   = _cat_to_codes(X["workclass"])
    C3_edu  = _cat_to_codes(X["education"])
    C3_nat  = _cat_to_codes(X["native-country"])

    M = len(X)
    idx = np.arange(M)
    if N is not None and N < M:
        rng = np.random.default_rng(seed)
        idx = rng.choice(M, size=N, replace=False)

    A_sex, A_race   = A_sex[idx], A_race[idx]
    Xad_occ         = Xad_occ[idx]
    C1_mar, C1_rel  = C1_mar[idx], C1_rel[idx]
    C2_age          = C2_age[idx]
    C4_hour         = C4_hour[idx]
    C5_wc           = C5_wc[idx]
    C3_edu, C3_nat  = C3_edu[idx], C3_nat[idx]
    y_bin           = y_bin[idx]

    XAD_CLUSTER_NAME = "Xad" if HAS_XAD else "C6"
    Xad_varnames = ["Xad_1"] if HAS_XAD else ["C6_1"]
    clusters = ["A", XAD_CLUSTER_NAME, "C1", "C2", "C4", "C5", "C3", "Y"]
    cluster_to_vars = {
        "A":              ["A_1", "A_2"],
        XAD_CLUSTER_NAME: Xad_varnames,
        "C1":             ["C1_1", "C1_2"],
        "C2":             ["C2_1"],
        "C4":             ["C4_1"],
        "C5":             ["C5_1"],
        "C3":             ["C3_1", "C3_2"],
        "Y":              ["Y_1"],
    }
    varnames = (
        cluster_to_vars["A"] +
        cluster_to_vars[XAD_CLUSTER_NAME] +
        cluster_to_vars["C1"] +
        cluster_to_vars["C2"] +
        cluster_to_vars["C4"] +
        cluster_to_vars["C5"] +
        cluster_to_vars["C3"] +
        cluster_to_vars["Y"]
    )

    cols_as_arrays = [
        A_sex.astype(np.float32),
        A_race.astype(np.float32),
        Xad_occ.astype(np.float32),
        C1_mar.astype(np.float32),
        C1_rel.astype(np.float32),
        C2_age.astype(np.float32),
        C4_hour.astype(np.float32),
        C5_wc.astype(np.float32),
        C3_edu.astype(np.float32),
        C3_nat.astype(np.float32),
        y_bin.astype(np.float32),
    ]
    data_mat = np.stack(cols_as_arrays, axis=1)

    T = torch.from_numpy(data_mat).float()
    var_to_idx = {v: i for i, v in enumerate(varnames)}
    def cols(names): return [var_to_idx[n] for n in names]
    cluster_tensors = {
        "A":              T[:, cols(cluster_to_vars["A"])],
        XAD_CLUSTER_NAME: T[:, cols(cluster_to_vars[XAD_CLUSTER_NAME])],
        "C1":             T[:, cols(cluster_to_vars["C1"])],
        "C2":             T[:, cols(cluster_to_vars["C2"])],
        "C4":             T[:, cols(cluster_to_vars["C4"])],
        "C5":             T[:, cols(cluster_to_vars["C5"])],
        "C3":             T[:, cols(cluster_to_vars["C3"])],
        "Y":              T[:, cols(cluster_to_vars["Y"])],
    }

    var_edges, cluster_edges = _build_true_edges_adult(cluster_to_vars, HAS_XAD)

    cluster_types = {
        "A":              "binary",
        XAD_CLUSTER_NAME: "binary",
        "C1":             "binary",
        "C2":             "cont",
        "C4":             "cont",
        "C5":             "binary",
        "C3":             "binary",
        "Y":              "binary",
    }
    meta = {
        "A":             "A",
        "Xad":           (XAD_CLUSTER_NAME if HAS_XAD else []),
        "Y":             "Y",
        "cluster_types": cluster_types,
        "cluster_edges": cluster_edges,
        "var_edges":     var_edges,
    }

    Y_bin = cluster_tensors["Y"][:, 0].clone()
    Y_cont = Y_bin.clone()
    Y_is_cont = False

    return {
        "data":            data_mat,
        "varnames":        varnames,
        "clusters":        clusters,
        "cluster_to_vars": cluster_to_vars,
        "cluster_tensors": cluster_tensors,
        "meta":            meta,
        "Y_bin":           Y_bin,
        "Y_cont":          Y_cont,
        "Y_is_cont":       Y_is_cont,
    }


def sample_adult_interventional_cgmm(
    G: dict,
    do_assign: dict,
    N: int,
    *,
    cgmm_cls=ConditionalGMMRegressor,
    cgmm_kwargs: dict = None,
    postprocess_binary: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    if cgmm_kwargs is None:
        cgmm_kwargs = {"n_components": 5, "random_state": random_state}
    return _cgmm_sample_interventional(
        G, do_assign, N,
        cgmm_cls=cgmm_cls,
        cgmm_kwargs=cgmm_kwargs,
        postprocess_binary=postprocess_binary,
        random_state=random_state,
    )
