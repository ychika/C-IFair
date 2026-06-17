"""
Preprocessing for Open University Learning Analytics Dataset (OULAD).

Output format matches load_adult_as_clusters:
  {
    "data": np.ndarray [N, V],
    "varnames": List[str],
    "clusters": List[str],
    "cluster_to_vars": Dict[str, List[str]],
    "cluster_tensors": Dict[str, torch.Tensor],
    "meta": {
        "A": "A",
        "Xad": [],
        "Y": "Y",
        "cluster_types": Dict[str, str],
        "cluster_edges": List[Tuple[str, str]],
        "var_edges": List[Tuple[int, int]],
    },
    "Y_bin": torch.Tensor [N],
    "Y_cont": torch.Tensor [N],
    "Y_is_cont": bool,
  }

Causal graph (variable-level):
  gender -> age_band
  imd_band -> highest_education
  imd_band -> final_grade
  disability -> highest_education
  studied_credits -> highest_education
  studied_credits -> num_of_prev_attempts
  studied_credits -> final_grade
  highest_education -> num_of_prev_attempts
  highest_education -> final_grade
  num_of_prev_attempts -> final_grade
"""

from __future__ import annotations

import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from lib_data import _set_seed
from ._utils import _cat_to_codes, _binarize_series
from ._edges import _build_true_edges_oulad
from ._sampling import _cgmm_sample_interventional


_UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00349/OULAD.zip"


def _download_and_extract_oulad(root: Path) -> Path:
    """
    Ensure OULAD csv files exist under `root / 'OULAD' / 'anonymiseddata'`.
    Returns the directory path containing the csvs.
    """
    root = Path(root)
    out_dir = root / "OULAD"
    csv_dir = out_dir / "anonymiseddata"
    expected = csv_dir / "studentInfo.csv"
    if expected.exists():
        return csv_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "OULAD.zip"
    if not zip_path.exists():
        print(f"[OULAD] downloading: {_UCI_ZIP_URL} -> {zip_path}")
        urllib.request.urlretrieve(_UCI_ZIP_URL, zip_path)

    print(f"[OULAD] extracting: {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    if expected.exists():
        return csv_dir

    for p in out_dir.rglob("studentInfo.csv"):
        return p.parent

    raise FileNotFoundError("studentInfo.csv not found after extracting OULAD.zip")


def _compute_final_grade(csv_dir: Path) -> pd.DataFrame:
    """
    Compute final_grade per (code_module, code_presentation, id_student) as:
      final_grade = sum(score * weight) / 100

    Returns a DataFrame with columns:
      code_module, code_presentation, id_student, final_grade
    """
    csv_dir = Path(csv_dir)
    assessments    = pd.read_csv(csv_dir / "assessments.csv")
    student_assess = pd.read_csv(csv_dir / "studentAssessment.csv")

    merged = student_assess.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "weight"]],
        on="id_assessment",
        how="left",
        validate="many_to_one",
    )

    merged["score"]  = pd.to_numeric(merged["score"],  errors="coerce").fillna(0.0)
    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0)
    merged["weighted"] = merged["score"] * merged["weight"]

    agg = (
        merged.groupby(["code_module", "code_presentation", "id_student"], as_index=False)["weighted"]
        .sum()
        .rename(columns={"weighted": "weighted_sum"})
    )
    agg["final_grade"] = (agg["weighted_sum"] / 100.0).astype(np.float32)
    return agg[["code_module", "code_presentation", "id_student", "final_grade"]]


def load_oulad_as_clusters(
    N: Optional[int] = None,
    seed: int = 23,
    data_root: str | os.PathLike = "./data",
    use_all_modules: bool = True,
) -> Dict:
    """
    Load and preprocess OULAD into the standard cluster graph dict format.

    Clusters (fixed):
      - A   = [A_1]   disability (0/1)
      - C1  = [C1_1]  gender (0/1; Male=1)
      - C2  = [C2_1]  age_band (categorical codes)
      - C3  = [C3_1]  imd_band (categorical codes)
      - C4  = [C4_1]  studied_credits (continuous)
      - C5  = [C5_1]  highest_education (categorical codes)
      - C6  = [C6_1]  num_of_prev_attempts (continuous)
      - Y   = [Y_1]   final_result (binary; Pass/Distinction=1, else 0)
    """
    _set_seed(seed)
    csv_dir = _download_and_extract_oulad(Path(data_root))

    info = pd.read_csv(Path(csv_dir) / "studentInfo.csv")
    final_grade = _compute_final_grade(csv_dir)
    info = info.merge(
        final_grade,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
        validate="one_to_one",
    )
    info["final_grade"] = pd.to_numeric(info["final_grade"], errors="coerce").fillna(0.0)

    if "final_result" not in info.columns:
        info["final_result"] = np.where(info["final_grade"] >= 40.0, "Pass", "Fail")

    if not use_all_modules:
        info = info.sort_values(["id_student", "code_module", "code_presentation"]).drop_duplicates(["id_student"])

    need_cols = [
        "disability", "gender", "age_band", "imd_band", "studied_credits",
        "highest_education", "num_of_prev_attempts", "final_grade", "final_result",
    ]
    X = info[need_cols].copy()

    A_dis = _binarize_series(
        X["disability"],
        positive_values={"Y", "y", 1, "1", True, "True", "true"},
    ).astype(np.int64)

    C1_gender = _binarize_series(
        X["gender"],
        positive_values={"M", "m", "Male", "male", 1, "1", True},
    ).astype(np.int64)

    C2_age = _cat_to_codes(X["age_band"])

    X["imd_band"] = X["imd_band"].replace({"?": "Unknown"})
    C3_imd = _cat_to_codes(X["imd_band"])

    C4_credits = pd.to_numeric(X["studied_credits"], errors="coerce")
    C4_credits = C4_credits.fillna(C4_credits.median()).astype(np.float32).to_numpy()

    C5_edu = _cat_to_codes(X["highest_education"])

    C6_prev = pd.to_numeric(X["num_of_prev_attempts"], errors="coerce")
    C6_prev = C6_prev.fillna(C6_prev.median()).astype(np.float32).to_numpy()

    res = X["final_result"].astype(str).str.strip()
    Y1 = res.isin(["Pass", "Distinction"]).astype(np.float32).to_numpy()

    M = len(X)
    idx = np.arange(M)
    if N is not None and N < M:
        rng = np.random.default_rng(seed)
        idx = rng.choice(M, size=N, replace=False)

    A_dis     = A_dis[idx]
    C1_gender = C1_gender[idx]
    C2_age    = C2_age[idx]
    C3_imd    = C3_imd[idx]
    C4_credits = C4_credits[idx]
    C5_edu    = C5_edu[idx]
    C6_prev   = C6_prev[idx]
    Y1        = Y1[idx]

    clusters = ["A", "C1", "C2", "C3", "C4", "C5", "C6", "Y"]
    cluster_to_vars = {
        "A":  ["A_1"],
        "C1": ["C1_1"],
        "C2": ["C2_1"],
        "C3": ["C3_1"],
        "C4": ["C4_1"],
        "C5": ["C5_1"],
        "C6": ["C6_1"],
        "Y":  ["Y_1"],
    }
    varnames = (
        cluster_to_vars["A"]  +
        cluster_to_vars["C1"] +
        cluster_to_vars["C2"] +
        cluster_to_vars["C3"] +
        cluster_to_vars["C4"] +
        cluster_to_vars["C5"] +
        cluster_to_vars["C6"] +
        cluster_to_vars["Y"]
    )

    data_mat = np.stack(
        [
            A_dis.astype(np.float32),
            C1_gender.astype(np.float32),
            C2_age.astype(np.float32),
            C3_imd.astype(np.float32),
            C4_credits.astype(np.float32),
            C5_edu.astype(np.float32),
            C6_prev.astype(np.float32),
            Y1.astype(np.float32),
        ],
        axis=1,
    )

    T = torch.from_numpy(data_mat).float()
    var_to_idx = {v: i for i, v in enumerate(varnames)}
    def cols(names: List[str]) -> List[int]:
        return [var_to_idx[n] for n in names]
    cluster_tensors = {c: T[:, cols(vs)] for c, vs in cluster_to_vars.items()}

    var_edges, cluster_edges = _build_true_edges_oulad(cluster_to_vars)

    cluster_types = {
        "A":  "binary",
        "C1": "binary",
        "C2": "cont",
        "C3": "cont",
        "C4": "cont",
        "C5": "cont",
        "C6": "cont",
        "Y":  "binary",
    }
    meta = {
        "A":             "A",
        "Xad":           [],
        "Y":             "Y",
        "cluster_types": cluster_types,
        "cluster_edges": cluster_edges,
        "var_edges":     var_edges,
    }

    Y_cont = cluster_tensors["Y"][:, 0].clone()
    Y_bin  = Y_cont.clone()
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


def sample_oulad_interventional_cgmm(
    G: Dict,
    do_assign: Dict[int, float],
    N: int,
    *,
    cgmm_cls=None,
    cgmm_kwargs: Optional[dict] = None,
    postprocess_binary: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    """
    Approximate interventional sampling via conditional GMM regressors.
    WARNING: This is an approximation for evaluation; not a true SCM sampler.
    """
    if cgmm_cls is None:
        from cgmm import ConditionalGMMRegressor
        cgmm_cls = ConditionalGMMRegressor
    if cgmm_kwargs is None:
        cgmm_kwargs = {"n_components": 5, "random_state": random_state}
    return _cgmm_sample_interventional(
        G, do_assign, N,
        cgmm_cls=cgmm_cls,
        cgmm_kwargs=cgmm_kwargs,
        postprocess_binary=postprocess_binary,
        random_state=random_state,
    )


if __name__ == "__main__":
    G = load_oulad_as_clusters(N=5000, seed=23, data_root="./data")
    print("Loaded OULAD:", G["data"].shape, "V=", len(G["varnames"]))
    print("cluster_edges:", G["meta"]["cluster_edges"])
