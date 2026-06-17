import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml

from lib_data import _set_seed
from ._utils import _cat_to_codes
from ._edges import (
    _build_true_edges_german,
    _extract_gender_from_personal_status,
    _robust_zscore,
    _age_over_25,
)


def load_german_as_clusters(N: int = None, seed: int = 23, HAS_XAD: bool = True):
    """
    Load the German Credit (OpenML credit-g) dataset as a cluster graph dict.

      - A   = [A_1(gender_is_male 0/1), A_2(age_over25 0/1)]
      - Xad = [Xad_1(credit_amount), Xad_2(duration)]
      - C1  = [C1_1(job)]
      - C2  = [C2_1(savings_status)]
      - C3  = [C3_1(checking_status)]
      - C4  = [C4_1(housing)]
      - C5  = [C5_1(purpose)]
      - Y   = [Y_1(class: good=1, bad=0)]

    Returns a dict with keys:
      data, varnames, clusters, cluster_to_vars, cluster_tensors, meta,
      Y_bin, Y_cont, Y_is_cont
    """
    _set_seed(seed)

    frame = fetch_openml("credit-g", version=1, as_frame=True)
    X_df: pd.DataFrame = frame.data.copy()
    y_raw: pd.Series   = frame.target.copy()

    y_bin = (y_raw.astype(str).str.lower() == "good").astype(np.int64).to_numpy()

    need = {
        "personal_status": "gender",
        "age": "age",
        "credit_amount": "credit",
        "duration": "duration",
        "job": "job",
        "savings_status": "savings",
        "checking_status": "checking",
        "housing": "housing",
        "purpose": "purpose",
    }
    X = X_df[list(need.keys())].copy()

    A_gender = _extract_gender_from_personal_status(X["personal_status"])
    A_over25 = _age_over_25(X["age"])

    Xad_credit_raw = pd.to_numeric(X["credit_amount"], errors="coerce").fillna(0).to_numpy().astype(np.float32)
    Xad_dur_raw    = pd.to_numeric(X["duration"], errors="coerce").fillna(0).to_numpy().astype(np.float32)

    normalize_cont = True; clip_percentiles = (1, 99); log1p_credit: bool = True
    if normalize_cont:
        def _proc(x, do_log):
            return _robust_zscore(x, clip=clip_percentiles, log1p=do_log)

        Xad_credit, stats_credit = _proc(Xad_credit_raw, log1p_credit)
        Xad_dur,    stats_dur    = _proc(Xad_dur_raw,    False)

        cont_preproc = {
            "Xad_1": {"name": "credit_amount",  **stats_credit},
            "Xad_2": {"name": "duration",       **stats_dur},
        }
    else:
        Xad_credit = Xad_credit_raw
        Xad_dur    = Xad_dur_raw
        cont_preproc = {
            "Xad_1": {"name": "credit_amount",  "lo": None, "hi": None,
                      "mean": float(np.mean(Xad_credit)), "std": float(np.std(Xad_credit)) or 1.0, "log1p": False},
            "Xad_2": {"name": "duration",       "lo": None, "hi": None,
                      "mean": float(np.mean(Xad_dur)),    "std": float(np.std(Xad_dur)) or 1.0,    "log1p": False},
        }

    C1_job      = _cat_to_codes(X["job"])
    C2_savings  = _cat_to_codes(X["savings_status"])
    C3_checking = _cat_to_codes(X["checking_status"])
    C4_housing  = _cat_to_codes(X["housing"])
    C5_purpose  = _cat_to_codes(X["purpose"])

    M = len(X)
    idx = np.arange(M)
    if N is not None and N < M:
        rng = np.random.default_rng(seed)
        idx = rng.choice(M, size=N, replace=False)

    A_gender    = A_gender[idx]
    A_over25    = A_over25[idx]
    Xad_credit  = Xad_credit[idx]
    Xad_dur     = Xad_dur[idx]
    C1_job      = C1_job[idx]
    C2_savings  = C2_savings[idx]
    C3_checking = C3_checking[idx]
    C4_housing  = C4_housing[idx]
    C5_purpose  = C5_purpose[idx]
    y_bin       = y_bin[idx]

    Xad_cluster_name = "Xad" if HAS_XAD else "C6"
    Xad_varnames = ["Xad_1", "Xad_2"] if HAS_XAD else ["C6_1", "C6_2"]

    clusters = ["A", Xad_cluster_name, "C1", "C2", "C3", "C4", "C5", "Y"]
    cluster_to_vars = {
        "A":              ["A_1", "A_2"],
        Xad_cluster_name: Xad_varnames,
        "C1":             ["C1_1"],
        "C2":             ["C2_1"],
        "C3":             ["C3_1"],
        "C4":             ["C4_1"],
        "C5":             ["C5_1"],
        "Y":              ["Y_1"],
    }
    varnames = (
        cluster_to_vars["A"]   +
        cluster_to_vars[Xad_cluster_name] +
        cluster_to_vars["C1"]  +
        cluster_to_vars["C2"]  +
        cluster_to_vars["C3"]  +
        cluster_to_vars["C4"]  +
        cluster_to_vars["C5"]  +
        cluster_to_vars["Y"]
    )

    cols_as_arrays = [
        A_gender.astype(np.float32),
        A_over25.astype(np.float32),
        Xad_credit.astype(np.float32),
        Xad_dur.astype(np.float32),
        C1_job.astype(np.float32),
        C2_savings.astype(np.float32),
        C3_checking.astype(np.float32),
        C4_housing.astype(np.float32),
        C5_purpose.astype(np.float32),
        y_bin.astype(np.float32),
    ]
    data_mat = np.stack(cols_as_arrays, axis=1)

    T = torch.from_numpy(data_mat).float()
    var_to_idx = {v: i for i, v in enumerate(varnames)}
    def cols(names): return [var_to_idx[n] for n in names]
    cluster_tensors = {
        "A":              T[:, cols(cluster_to_vars["A"])],
        Xad_cluster_name: T[:, cols(cluster_to_vars[Xad_cluster_name])],
        "C1":             T[:, cols(cluster_to_vars["C1"])],
        "C2":             T[:, cols(cluster_to_vars["C2"])],
        "C3":             T[:, cols(cluster_to_vars["C3"])],
        "C4":             T[:, cols(cluster_to_vars["C4"])],
        "C5":             T[:, cols(cluster_to_vars["C5"])],
        "Y":              T[:, cols(cluster_to_vars["Y"])],
    }

    var_edges, cluster_edges = _build_true_edges_german(cluster_to_vars, HAS_XAD)

    cluster_types = {
        "A":              "binary",
        Xad_cluster_name: "cont",
        "C1":             "binary",
        "C2":             "binary",
        "C3":             "binary",
        "C4":             "binary",
        "C5":             "binary",
        "Y":              "binary",
    }
    meta = {
        "A":             "A",
        "Xad":           (Xad_cluster_name if HAS_XAD else []),
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
