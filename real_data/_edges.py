from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# German-specific feature helpers
# ------------------------------------------------------------

def _extract_gender_from_personal_status(series: pd.Series) -> np.ndarray:
    s = series.astype("object").fillna("")
    return s.str.contains("male", case=False, regex=False).astype(np.int64).to_numpy()


def _robust_zscore(x: np.ndarray, clip=(1, 99), log1p=False):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x[~np.isnan(x)], clip)
    x_c = np.clip(x, lo, hi)
    if log1p:
        x_c = np.log1p(np.maximum(x_c, 0.0))
    mu = float(np.mean(x_c))
    sd = float(np.std(x_c)) if float(np.std(x_c)) > 0 else 1.0
    x_z = (x_c - mu) / sd
    return x_z.astype(np.float32), {
        "lo": float(lo), "hi": float(hi), "mean": mu, "std": sd, "log1p": bool(log1p)
    }


def _age_over_25(series: pd.Series) -> np.ndarray:
    a = pd.to_numeric(series, errors="coerce")
    a = a.fillna(a.median())
    return (a.astype(float) > 25.0).astype(np.int64).to_numpy()


# ------------------------------------------------------------
# Dataset-specific DAG edge builders
# ------------------------------------------------------------

def _build_true_edges_adult(cluster_to_vars, HAS_XAD):
    """
    Build variable-level and cluster-level edges for the Adult dataset.

      - A   = [A_1(sex_is_male), A_2(race_is_white)]
      - Xad = [Xad_1(occupation)]
      - C1  = [C1_1(marital_status), C1_2(relationship)]
      - C2  = [C2_1(age)]
      - C4  = [C4_1(hours_per_week)]
      - C5  = [C5_1(work_class)]
      - C3  = [C3_1(edu), C3_2(native_country)]
      - Y   = [Y_1(target)]
    """
    canonical_to_current = {
        "sex":             "A_1",
        "race":            "A_2",
        "occupation":      "Xad_1",
        "marital_status":  "C1_1",
        "relationship":    "C1_2",
        "age":             "C2_1",
        "hour":            "C4_1",
        "work_class":      "C5_1",
        "edu":             "C3_1",
        "native_country":  "C3_2",
        "target":          "Y_1",
    }

    if HAS_XAD:
        canonical_to_current["occupation"] = "Xad_1"
    else:
        canonical_to_current["occupation"] = "C6_1"

    varnames = []
    for cname, vs in cluster_to_vars.items():
        varnames.extend(vs)
    var_to_idx = {v: i for i, v in enumerate(varnames)}

    def vidx_by_canonical(name):
        cur = canonical_to_current.get(name)
        return var_to_idx[cur] if (cur in var_to_idx) else None

    raw_edges_canonical = [
        ("race", "edu"), ("race", "hour"), ("race", "work_class"),
        ("race", "marital_status"), ("race", "occupation"),
        ("race", "relationship"), ("race", "target"),

        ("sex",  "edu"), ("sex",  "hour"), ("sex",  "work_class"),
        ("sex",  "marital_status"), ("sex",  "occupation"),
        ("sex",  "relationship"), ("sex",  "target"),

        ("native_country", "edu"), ("native_country", "hour"), ("native_country", "work_class"),
        ("native_country", "marital_status"), ("native_country", "occupation"),
        ("native_country", "relationship"), ("native_country", "target"),

        ("age", "edu"), ("age", "hour"), ("age", "work_class"),
        ("age", "marital_status"), ("age", "occupation"),
        ("age", "relationship"), ("age", "target"),

        ("edu", "hour"), ("edu", "work_class"),
        ("edu", "marital_status"), ("edu", "occupation"),
        ("edu", "relationship"),   ("edu", "target"),

        ("hour", "marital_status"), ("hour", "occupation"),
        ("hour", "relationship"), ("hour", "target"),

        ("work_class", "marital_status"), ("work_class", "occupation"),
        ("work_class", "relationship"), ("work_class", "target"),

        ("marital_status", "relationship"),
        ("marital_status", "occupation"),
        ("marital_status", "target"),

        ("occupation", "relationship"),
        ("occupation", "target"),

        ("relationship", "target"),
    ]

    var_edges = []
    for u_c, v_c in raw_edges_canonical:
        u = vidx_by_canonical(u_c)
        v = vidx_by_canonical(v_c)
        if u is not None and v is not None and u != v:
            var_edges.append((u, v))

    y_idx = vidx_by_canonical("target")
    if y_idx is not None:
        for j in range(len(varnames)):
            if j != y_idx:
                var_edges.append((j, y_idx))

    var_edges = sorted(set(var_edges))

    inv_var_to_cluster = {}
    for cname, vs in cluster_to_vars.items():
        for v in vs:
            if v in var_to_idx:
                inv_var_to_cluster[var_to_idx[v]] = cname
    cl_edges = set()
    for (u, v) in var_edges:
        cu, cv = inv_var_to_cluster[u], inv_var_to_cluster[v]
        if cu != cv:
            cl_edges.add((cu, cv))
    cluster_edges = sorted(cl_edges)

    return var_edges, cluster_edges


def _build_true_edges_german(cluster_to_vars, HAS_XAD):
    canonical_to_current = {
        "sex":       "A_1",
        "age":       "A_2",
        "job":       "C1_1",
        "savings":   "C2_1",
        "checking":  "C3_1",
        "housing":   "C4_1",
        "duration":  "Xad_2",
        "credit":    "Xad_1",
        "purpose":   "C5_1",
        "target":    "Y_1",
    }

    if HAS_XAD:
        canonical_to_current["credit"]   = "Xad_1"
        canonical_to_current["duration"] = "Xad_2"
    else:
        canonical_to_current["credit"]   = "C6_1"
        canonical_to_current["duration"] = "C6_2"

    varnames = []
    for _, vs in cluster_to_vars.items():
        varnames.extend(vs)
    var_to_idx = {v: i for i, v in enumerate(varnames)}

    def vidx(name):
        cur = canonical_to_current.get(name)
        return var_to_idx[cur] if cur in var_to_idx else None

    raw_edges_canonical = [
        ("sex", "job"), ("sex", "savings"),

        ("age", "job"), ("age", "savings"), ("age", "housing"),
        ("age", "credit"), ("age", "duration"), ("age", "purpose"),

        ("job", "savings"), ("job", "checking"),

        ("savings", "checking"), ("savings", "housing"), ("savings", "duration"),

        ("checking", "duration"),

        ("housing", "duration"), ("housing", "purpose"),

        ("credit", "duration"), ("credit", "purpose"),

        ("duration", "purpose"),
    ]

    var_edges = []
    for u_c, v_c in raw_edges_canonical:
        u = vidx(u_c); v = vidx(v_c)
        if u is not None and v is not None and u != v:
            var_edges.append((u, v))

    y_idx = vidx("target")
    if y_idx is not None:
        for j in range(len(varnames)):
            if j != y_idx:
                var_edges.append((j, y_idx))

    var_edges = sorted(set(var_edges))

    inv_var_to_cluster = {}
    for cname, vs in cluster_to_vars.items():
        for v in vs:
            inv_var_to_cluster[var_to_idx[v]] = cname
    cl_edges = set()
    for (u, v) in var_edges:
        cu, cv = inv_var_to_cluster[u], inv_var_to_cluster[v]
        if cu != cv:
            cl_edges.add((cu, cv))
    cluster_edges = sorted(cl_edges)
    return var_edges, cluster_edges


def _build_true_edges_oulad(cluster_to_vars: Dict[str, List[str]]):
    """
    Build variable-level and cluster-level edges for OULAD.

    Variables (canonical -> current variable names):
      disability            -> A_1
      gender                -> C1_1
      age_band              -> C2_1
      imd_band              -> C3_1
      studied_credits       -> C4_1
      highest_education     -> C5_1
      num_of_prev_attempts  -> C6_1
      final_grade           -> Y_1

    Edges (from the provided causal graph):
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

    All non-Y variables also have an edge into Y for evaluation compatibility.
    """
    canonical_to_current = {
        "disability":            "A_1",
        "gender":                "C1_1",
        "age_band":              "C2_1",
        "imd_band":              "C3_1",
        "studied_credits":       "C4_1",
        "highest_education":     "C5_1",
        "num_of_prev_attempts":  "C6_1",
        "final_grade":           "Y_1",
    }

    varnames: List[str] = []
    for _, vs in cluster_to_vars.items():
        varnames.extend(vs)

    var_to_idx = {v: i for i, v in enumerate(varnames)}

    def vidx(canonical: str):
        cur = canonical_to_current.get(canonical)
        return var_to_idx.get(cur, None)

    raw_edges = [
        ("gender", "age_band"),
        ("imd_band", "highest_education"),
        ("imd_band", "final_grade"),
        ("disability", "highest_education"),
        ("studied_credits", "highest_education"),
        ("studied_credits", "num_of_prev_attempts"),
        ("studied_credits", "final_grade"),
        ("highest_education", "num_of_prev_attempts"),
        ("highest_education", "final_grade"),
        ("num_of_prev_attempts", "final_grade"),
    ]

    var_edges: List[Tuple[int, int]] = []
    for u_c, v_c in raw_edges:
        u = vidx(u_c)
        v = vidx(v_c)
        if u is not None and v is not None and u != v:
            var_edges.append((u, v))

    y_idx = vidx("final_grade")
    if y_idx is not None:
        for j in range(len(varnames)):
            if j != y_idx:
                var_edges.append((j, y_idx))

    var_edges = sorted(set(var_edges))

    inv_var_to_cluster: Dict[int, str] = {}
    for cname, vs in cluster_to_vars.items():
        for v in vs:
            if v in var_to_idx:
                inv_var_to_cluster[var_to_idx[v]] = cname

    cl_edges = set()
    for u, v in var_edges:
        cu, cv = inv_var_to_cluster[u], inv_var_to_cluster[v]
        if cu != cv:
            cl_edges.add((cu, cv))
    cluster_edges = sorted(cl_edges)

    return var_edges, cluster_edges
