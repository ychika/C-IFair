import os, json, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
import numpy as np
import sys, math, itertools
from itertools import combinations
import matplotlib.pyplot as plt
import time

# libs from your project
import cloc
from graph_alg import adjustment_candidate_sets as ac

from lib_unfairness import (
    set_seed, detect_xad_types,
    PropensityAX_General, PropensityA_General,
    kernel_on_Xad, rff_features,
    measure_unfairness, barycenter_mmd_for_x_given_z,
    median_heuristic_sigma_1d, kde1d, gaussian_mmd1d,
    _batches, slice_Z_models, train_mlp, auc_from_logits_sklearn
)
from graph_alg.adjustment_candidate_sets import parents_from_P
from lib_data import (
    _set_seed as set_seed_data,
    sample_interventional_dataset,
)

from real_data import load_adult_as_clusters, load_german_as_clusters, load_oulad_as_clusters
from synth_data import (
    gen_random_cluster_lin_anm, gen_random_cluster_nonlin_anm,
    get_fixed_cluster_anm, get_fixed_cluster_nonlin_anm,
    gen_random_cluster_lin_anm_inadmissible,
)


# =========================
# Device auto-select (CUDA -> MPS -> CPU)
# =========================
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

DEVICE = get_device()
print(f"[INFO] Using device: {DEVICE}")

def now():
    return time.perf_counter()

def sync_if_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

def make_x_targets_from_Xad(
    Xad_tr: torch.Tensor,
    *,
    Rrep: int,
    DEVICE: torch.device,
    binary_only_ok: bool = True,
    max_enum_bits: int = 12,   
):
    """
    Returns:
      x_targets: (R, d_x) for continuous-ish
                 (2^d_x, d_x) for binary
    """
    if Xad_tr is None:
        return None

    Xad_tr = Xad_tr.to(DEVICE)

    # shape: (N, d_x)
    if Xad_tr.dim() == 1:
        Xad_tr = Xad_tr.unsqueeze(1)

    # ---- whether it is binary ----
    vals = torch.unique(Xad_tr.detach())
    is_binary = (vals.numel() <= 2) and torch.all((vals == 0) | (vals == 1))

    if is_binary:
        d_x = Xad_tr.size(1)
        if d_x > max_enum_bits:
            raise ValueError(f"Xad is binary but too many bits to enumerate: d_x={d_x} > {max_enum_bits}")

        codes = torch.arange(2**d_x, device=DEVICE, dtype=torch.long)  # (2^d_x,)
        shifts = torch.arange(d_x, device=DEVICE, dtype=torch.long)    # (d_x,)
        x_targets = ((codes.unsqueeze(1) >> shifts) & 1).to(Xad_tr.dtype)  # (2^d_x, d_x)
        return x_targets

    # is not binary (categorical)
    if binary_only_ok:
        pass

    R = min(int(Rrep), Xad_tr.size(0))
    idx = torch.randperm(Xad_tr.size(0), device=DEVICE, dtype=torch.long)[:R]  # long
    x_targets = Xad_tr[idx]  # (R, d_x)

    return x_targets


# =========================
# Main
# =========================
def main(args):
    dataname=args.dataname; seed=args.seed; N=args.N 
    epochs=args.epochs; prop_steps=args.prop_steps; batch_size=args.batch_size 
    lr_mlp=args.lr_mlp; lr_prop=args.lr_prop
    Rrep=args.R; lmbda=args.lmbda; d_clusters=args.d_clusters; hard_pred=args.hard_pred; tau=args.tau

    set_seed(seed); set_seed_data(seed)
    bool_HAS_XAD = False
    # (1) Data
    if dataname == "adult":
        G = load_adult_as_clusters(HAS_XAD=bool_HAS_XAD)
    elif dataname == "german":
        bool_HAS_XAD = False
        G = load_german_as_clusters(HAS_XAD=bool_HAS_XAD)
    elif dataname == "oulad":
        bool_HAS_XAD = False
        G = load_oulad_as_clusters()        
    elif dataname == "linear":
        G = gen_random_cluster_lin_anm(seed=seed, d=d_clusters, N=N, HAS_XAD=bool_HAS_XAD)
    elif dataname == "nonlinear" or dataname == "nonlin":
        G = gen_random_cluster_nonlin_anm(seed=seed, d=d_clusters, N=N, func=None, HAS_XAD=bool_HAS_XAD)
    elif dataname == "lin_conn":
        G = get_fixed_cluster_anm(seed=seed, n=N)
    elif dataname == "nonlin_conn":
        G = get_fixed_cluster_nonlin_anm(seed=seed, n=N, ftypes=["sin", "cos"])
    elif dataname == "lin_inadmissible":
        G = gen_random_cluster_lin_anm_inadmissible(seed=seed, d=d_clusters, N=N, HAS_XAD=bool_HAS_XAD)
    else:
        raise ValueError(f"Unknown dataname: {dataname}")
    data, varnames = G["data"], G["varnames"]

    # === train/val/test split ===
    N_all = len(data)
    rng = np.random.default_rng(seed + 1234)
    perm = rng.permutation(N_all)

    n_train = max(1, int(0.8 * N_all))
    n_val   = max(1, int(0.1 * N_all))

    train_idx = torch.tensor(perm[:n_train], dtype=torch.long)
    val_idx   = torch.tensor(perm[n_train:n_train+n_val], dtype=torch.long)
    test_idx  = torch.tensor(perm[n_train+n_val:], dtype=torch.long)
    graph_idx = train_idx
    # split cluster tensors
    def split_cluster_tensors(cluster_tensors, idx):
        return {c: t[idx].to(DEVICE) for c, t in cluster_tensors.items()}

    cluster_tensors_tr = split_cluster_tensors(G["cluster_tensors"], train_idx)
    cluster_tensors_va = split_cluster_tensors(G["cluster_tensors"], val_idx)    
    cluster_tensors_te = split_cluster_tensors(G["cluster_tensors"], test_idx)

    meta_info = G["meta"]
    A_cluster = meta_info["A"]
    Xad_cluster = meta_info["Xad"]
    HAS_XAD = (isinstance(Xad_cluster, str) and (len(Xad_cluster) > 0) and (Xad_cluster in cluster_tensors_tr))    
    Y_is_cont = G.get("Y_is_cont", False)
    Y_all = (G["Y_cont"] if Y_is_cont else G["Y_bin"]).to(DEVICE).float()
    Y_tr = Y_all[train_idx.to(DEVICE)]
    Y_va = Y_all[val_idx.to(DEVICE)]    
    Y_te = Y_all[test_idx.to(DEVICE)]
    normalize_Y = True
    if Y_is_cont and normalize_Y:
        mu = Y_tr.mean()
        sd = Y_tr.std(unbiased=False).clamp_min(1e-12)
        Y_tr = (Y_tr - mu)/sd
        Y_va = (Y_va - mu)/sd
        Y_te = (Y_te - mu)/sd


    def build_A_labels(t_cluster):
        A_bits = t_cluster[A_cluster].long()
        if A_bits.dim() == 1:
            A_bits = A_bits.unsqueeze(1)
        mA = A_bits.size(1)
        K_A_local = int(2 ** mA)

        # big-endian: leftmost is MSB
        lab = torch.zeros(A_bits.size(0), dtype=torch.long, device=A_bits.device)
        for j in range(mA):
            lab = (lab << 1) + A_bits[:, j]   # shift then add current bit
        return lab, K_A_local


    A_lab_tr, K_A = build_A_labels(cluster_tensors_tr)
    A_lab_va, _   = build_A_labels(cluster_tensors_va)    
    A_lab_te, _   = build_A_labels(cluster_tensors_te)

    # Xad split
    if HAS_XAD:
        Xad_tr = cluster_tensors_tr[Xad_cluster].float().to(DEVICE)
        Xad_va = cluster_tensors_va[Xad_cluster].float().to(DEVICE)    
        Xad_te = cluster_tensors_te[Xad_cluster].float().to(DEVICE)

    clusters = G["clusters"]
    varnames = G["varnames"]
    cluster_to_vars = G["cluster_to_vars"]
    cluster_tensors = G["cluster_tensors"]
    meta_info = G["meta"]




    # === build_X_pred AFTER K_A is known ===
    def build_X_pred(cluster_tensors_split):
        parts = [t.float().to(DEVICE) for c, t in cluster_tensors_split.items() if c != meta_info["Y"]]
        # A one-hot at the end
        A_bits = cluster_tensors_split[A_cluster].long()
        if A_bits.dim() == 1:
            A_bits = A_bits.unsqueeze(1)
        lab = torch.zeros(A_bits.size(0), dtype=torch.long, device=DEVICE)
        for j in range(A_bits.size(1)):
            lab = lab + (A_bits[:, j] << j)
        return torch.cat(parts, dim=1)


    Y_target = G["Y_cont"] if Y_is_cont else G["Y_bin"]

    clusters, cluster_to_vars = G["clusters"], G["cluster_to_vars"]
    cluster_tensors = G["cluster_tensors"]

    # (2) Learn Cluster CP-DAG via cloc
    var_to_idx = {v:i for i,v in enumerate(varnames)}
    cluster_to_idx = {c:[var_to_idx[v] for v in vs] for c,vs in cluster_to_vars.items()}
    # Data for graph estimation
    data_graph = data[graph_idx.numpy(), :]
    # Data for model training
    data_train = data[train_idx.numpy(), :]
    # 1) learn ClusterCPDAG
    y_cluster = G["meta"]["Y"]
    sync_if_cuda(); t0 = now()
    P = cloc.cloc_learn(data_graph, clusters, cluster_to_idx, cluster_types=G["meta"]["cluster_types"], y_cluster=y_cluster)
    sync_if_cuda(); t1 = now(); cloc_time = t1 - t0
    print(f"[TIME] cloc_learn: {cloc_time:.3f} sec")

    flag = ac._has_connection_marks_in_pparents(P, A_cluster, Xad_cluster, HAS_XAD)
    # 2) get adjustment candidates
    _Xad_cluster = Xad_cluster if HAS_XAD else None
    Z_cands, fail_M, refine_state = ac.return_adjustment_candidates(
        P, A_cluster=A_cluster,
        HAS_XAD=HAS_XAD,
        Xad_cluster=_Xad_cluster,
        do_refine=True,
        data_train=data_train,
        clusters=clusters,
        cluster_to_idx=cluster_to_idx,
        cluster_types=G["meta"]["cluster_types"],
        y_cluster=y_cluster,
        max_iters=20,
        verbose=True
    )
    if refine_state is not None:
        G, _P_ref = ac.rename_reset_clusters(G, refine_state, data_train=data_train, device=DEVICE)
        clusters = G["clusters"]
        varnames = G["varnames"]
        cluster_to_vars = G["cluster_to_vars"]
        cluster_tensors = G["cluster_tensors"]
        meta_info = G["meta"] 
        cluster_tensors_tr = split_cluster_tensors(G["cluster_tensors"], train_idx)
        cluster_tensors_va = split_cluster_tensors(G["cluster_tensors"], val_idx)
        cluster_tensors_te = split_cluster_tensors(G["cluster_tensors"], test_idx)
        meta_info = G["meta"]
        A_cluster = meta_info["A"]
        Xad_cluster = meta_info["Xad"]
        HAS_XAD = (isinstance(Xad_cluster, str) and (len(Xad_cluster) > 0) and (Xad_cluster in cluster_tensors_tr))
        # refresh A labels / K_A
        A_lab_tr, K_A = build_A_labels(cluster_tensors_tr)
        A_lab_va, _ = build_A_labels(cluster_tensors_va)
        A_lab_te, _ = build_A_labels(cluster_tensors_te)
        # refresh Xad tensors used in fairness computation
        if HAS_XAD:
            Xad_tr = cluster_tensors_tr[Xad_cluster]
            Xad_va = cluster_tensors_va[Xad_cluster]
            Xad_te = cluster_tensors_te[Xad_cluster]
    if len(Z_cands) == 0:
        print("ERROR: No need to do fairness training, as there is no confounding")
        raise NotImplementedError

    # (5) Build tensors for fairness (train/test)
    X_pred_tr = torch.cat([cluster_tensors_tr[c].float().to(DEVICE) for c in clusters if c != meta_info["Y"]], dim=1)
    X_pred_va = torch.cat([cluster_tensors_va[c].float().to(DEVICE) for c in clusters if c != meta_info["Y"]], dim=1)
    X_pred_te = torch.cat([cluster_tensors_te[c].float().to(DEVICE) for c in clusters if c != meta_info["Y"]], dim=1)


    # (6) Joint propensity models for each Z
    Z_models = []; Z_models_val = []
    if HAS_XAD:
        meta_xad = detect_xad_types(Xad_tr)    
    for Sset in Z_cands:
        def build_Zmat(c_tensors):
            if len(Sset) == 0:
                return torch.ones(c_tensors[A_cluster].size(0), 1, device=DEVICE)
            mats = []
            for c in Sset:
                if c in {A_cluster, meta_info["Y"]}:
                    continue
                mats.append(c_tensors[c].float().to(DEVICE))
            return torch.cat(mats, dim=1) if mats else torch.ones(c_tensors[A_cluster].size(0), 1, device=DEVICE)

        Z_tr = build_Zmat(cluster_tensors_tr)
        Z_va = build_Zmat(cluster_tensors_va)        
        Z_te = build_Zmat(cluster_tensors_te)
        if HAS_XAD: # joint propensity P(A, Xad | Z)
            jprop_model = PropensityAX_General(z_dim=Z_tr.size(1), xad_meta=meta_xad, K_A=K_A).to(DEVICE)
        else: # propensity P(A | Z)
            jprop_model = PropensityA_General(z_dim=Z_tr.size(1), K_A=K_A).to(DEVICE)            
        jprop_opt = torch.optim.AdamW(jprop_model.parameters(), lr=lr_prop)
        for _ in range(prop_steps):
            if HAS_XAD:
                jprop_loss = - jprop_model.log_g(A_lab_tr, Xad_tr, Z_tr).mean()
            else:
                jprop_loss = - jprop_model.log_g(A_lab_tr, Z_tr).mean()                
            jprop_opt.zero_grad(); jprop_loss.backward(); jprop_opt.step()
        Z_models.append((Sset, Z_tr, Z_te, jprop_model))
        Z_models_val.append((Sset, Z_va, Z_va, jprop_model))        

    # (7) Predictor + fairness
    pred = nn.Sequential(nn.Linear(X_pred_tr.size(1), 32), nn.ReLU(), nn.Linear(32,1)).to(DEVICE)
    pred_opt = torch.optim.AdamW(pred.parameters(), lr=lr_mlp)
    if HAS_XAD:
        x_targets = make_x_targets_from_Xad(Xad_tr, Rrep=Rrep, DEVICE=DEVICE)
        x_targets_va = make_x_targets_from_Xad(Xad_va, Rrep=Rrep, DEVICE=DEVICE)
        x_targets_te = make_x_targets_from_Xad(Xad_te, Rrep=Rrep, DEVICE=DEVICE)
    else:
        x_targets = None; x_targets_va = None; x_targets_te = None

    D_FEAT = 128
    logs = []

    # init sigma0 by performing median heuristic with the initial predictor 
    with torch.no_grad():
        sc0 = pred(X_pred_tr).squeeze(1)
        sigma0 = median_heuristic_sigma_1d(torch.sigmoid(sc0) if not Y_is_cont else sc0)
    # for early stopping
    best_val_obj = float("inf"); best_state = None; wait = 0
    patience = 10**9 

    for ep in range(epochs):
        pred_opt.zero_grad()
        # === whether to separate into minibatches ===
        if batch_size and batch_size > 0:
            n = X_pred_tr.size(0)
            n_batches = (n + batch_size - 1) // batch_size
            task_epoch_val = 0.0

            for Xb, Yb, idx_sel in _batches(X_pred_tr, Y_tr, batch_size, shuffle=True):
                # --- minibatch loss---
                score_b = pred(Xb).squeeze(1)
                loss_task_b = (torch.mean((score_b - Yb)**2) if Y_is_cont
                            else torch.nn.functional.binary_cross_entropy_with_logits(score_b, Yb))
                # accumulate loss
                (loss_task_b / n_batches).backward(retain_graph=True)
                task_epoch_val += float(loss_task_b.detach().item())

                # --- minibatch unfairness ---
                Z_models_b = slice_Z_models(Z_models, idx_sel)
                fair_b = measure_unfairness(
                    pred=pred,
                    X_pred_tr=Xb,
                    Y_is_cont=Y_is_cont,
                    hard_pred=hard_pred,
                    tau=tau,
                    D_FEAT=D_FEAT,
                    sigma0=sigma0,
                    A_lab_tr=A_lab_tr[idx_sel],
                    Z_models=Z_models_b,
                    K_A=K_A,
                    device=DEVICE,
                    HAS_XAD=HAS_XAD,                              
                    Xad_tr=(Xad_tr[idx_sel] if HAS_XAD else None),
                    meta_xad=(meta_xad if HAS_XAD else None),     
                    x_targets=(x_targets if HAS_XAD else None)    
                )
                (lmbda * fair_b / n_batches).backward()  
        else:
            # === Full data ===
            score = pred(X_pred_tr).squeeze(1)
            loss_task = (torch.mean((score - Y_tr)**2) if Y_is_cont
                        else torch.nn.functional.binary_cross_entropy_with_logits(score, Y_tr))
            loss_task.backward(retain_graph=True)

            fair_full = measure_unfairness(
                pred=pred,
                X_pred_tr=X_pred_tr,              
                Y_is_cont=Y_is_cont,
                hard_pred=hard_pred,
                tau=tau,
                D_FEAT=D_FEAT,
                sigma0=sigma0,
                A_lab_tr=A_lab_tr,
                Z_models=Z_models,
                K_A=K_A,
                meta_xad=(meta_xad if HAS_XAD else None),     
                x_targets=(x_targets if HAS_XAD else None),   
                Xad_tr=(Xad_tr if HAS_XAD else None),         
                device=DEVICE,
                HAS_XAD=HAS_XAD                               
            )
            (lmbda * fair_full).backward()

        # === parameter update ===
        pred_opt.step()

        # (optional) logging
        if (ep + 1) % 10 == 0:
            if batch_size and batch_size > 0:
                task_disp = task_epoch_val / n_batches
                fair_disp = float(fair_b.detach())  
            else:
                task_disp = float(loss_task.detach())
                fair_disp = float(fair_full.detach())
            print(f"ep {ep+1:03d} pred_loss={task_disp:.4f}  fair_reg={fair_disp:.4f}  |  #M={len(Z_models)}")




        with torch.no_grad():
            # task on val
            sc_val = pred(X_pred_va).squeeze(1)
            val_task = (torch.mean((sc_val - Y_va)**2) if Y_is_cont
                        else torch.nn.functional.binary_cross_entropy_with_logits(sc_val, Y_va))
            # fairness on val
            fair_val = measure_unfairness(
                pred=pred,
                X_pred_tr=X_pred_va,              
                Y_is_cont=Y_is_cont,
                hard_pred=hard_pred,
                tau=tau,
                D_FEAT=D_FEAT,
                sigma0=sigma0,
                A_lab_tr=A_lab_va,
                Z_models=Z_models_val,
                K_A=K_A,
                meta_xad=(meta_xad if HAS_XAD else None),     
                x_targets=(x_targets_va if HAS_XAD else None),   
                Xad_tr=(Xad_va if HAS_XAD else None),         
                device=DEVICE,
                HAS_XAD=HAS_XAD                               
            )            
            val_obj = float((val_task + lmbda * fair_val).detach().item())

        if val_obj < best_val_obj - 1e-4:
            best_val_obj = val_obj
            best_state = {k: v.detach().cpu().clone() for k, v in pred.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[EarlyStopping] epoch={ep+1}, best_val_obj={best_val_obj:.6f}")
                break
        # ================================================================

    if best_state is not None:
        pred.load_state_dict(best_state)
    # ==============================

    def build_pred_inputs_from_X(
        cluster_tensors: dict,
        *,
        clusters_order: list,           
        A_cluster: str,
        Y_cluster: str,
        K_A: int,
        idx_sel=None,                   # torch.LongTensor | np.ndarray | None
        drop_clusters=None,             # list[str] | None
        include_A_onehot: bool = True,
        return_A_labels: bool = False,
        device=None
    ):
        if device is None:
            device = next(iter(cluster_tensors.values())).device

        if idx_sel is None:
            N = next(iter(cluster_tensors.values())).size(0)
            idx_sel = torch.arange(N, device=device)
        elif not torch.is_tensor(idx_sel):
            idx_sel = torch.as_tensor(idx_sel, dtype=torch.long, device=device)

        drop_set = set(drop_clusters or [])

        parts = []
        for c in clusters_order:
            if c == Y_cluster:
                continue
            if c in drop_set:
                continue
            t = cluster_tensors[c].to(device)
            parts.append(t[idx_sel].float())

        X_cat = torch.cat(parts, dim=1) if parts else torch.ones((idx_sel.numel(), 1), device=device)

        Abits = cluster_tensors[A_cluster][idx_sel].long().to(device)
        if Abits.dim() == 1:
            Abits = Abits.unsqueeze(1)
        Alab = torch.zeros(Abits.size(0), dtype=torch.long, device=device)
        for j in range(Abits.size(1)):
            Alab = (Alab << 1) + Abits[:, j]   # big-endian


        if include_A_onehot:
            Aoh = torch.nn.functional.one_hot(Alab, num_classes=K_A).float().to(device)
            X_cat = torch.cat([X_cat, Aoh], dim=1)

        return (X_cat, Alab) if return_A_labels else X_cat


    def build_pred_inputs_from_X_drop_vars(
        cluster_tensors: dict,
        *,
        varnames: list,
        clusters_order: list,
        A_cluster: str,
        Y_cluster: str,
        K_A: int,
        idx_sel=None,
        drop_var_idxs=None,             # set[int] | list[int] | None  (global variable indices in G['data'])
        include_A_onehot: bool = False,
        return_A_labels: bool = False,
        device=None
    ):
        """Variant of build_pred_inputs_from_X that can drop *variable-level* columns.

        drop_var_idxs: global indices (column indices of G['data'] / G['varnames']) to remove.
        This allows baselines that remove A/Xad and their descendants even if they share clusters with other variables.
        """
        if device is None:
            device = next(iter(cluster_tensors.values())).device

        # idx_sel normalize
        if idx_sel is None:
            N = next(iter(cluster_tensors.values())).size(0)
            idx_sel = torch.arange(N, device=device)
        elif not torch.is_tensor(idx_sel):
            idx_sel = torch.as_tensor(idx_sel, dtype=torch.long, device=device)

        drop_var_set = set(drop_var_idxs or [])
        cl2gidx = {
            c: [i for i, vn in enumerate(varnames)
                if vn.startswith(c + "_") or (vn.startswith(c) and vn[len(c):].isdigit())]
            for c in clusters_order
        }


        parts = []
        for c in clusters_order:
            if c == Y_cluster:
                continue
            t = cluster_tensors[c].to(device)
            gidxs = cl2gidx.get(c, [])
            if len(gidxs) != t.size(1):
                raise ValueError(f"[build_pred_inputs_from_X_drop_vars] size mismatch for cluster {c}: "
                                 f"len(gidxs)={len(gidxs)} vs tensor_dim={t.size(1)}. "
                                 f"Check varnames naming and cluster_tensors construction.")
            keep_cols = [j for j, gi in enumerate(gidxs) if gi not in drop_var_set]
            if len(keep_cols) == 0:
                continue
            parts.append(t[idx_sel][:, keep_cols].float())

        X_cat = torch.cat(parts, dim=1) if parts else torch.ones((idx_sel.numel(), 1), device=device)

        # A label (always computable even if A vars were dropped from X)
        Abits = cluster_tensors[A_cluster][idx_sel].long().to(device)
        if Abits.dim() == 1:
            Abits = Abits.unsqueeze(1)
        Alab = torch.zeros(Abits.size(0), dtype=torch.long, device=device)
        for j in range(Abits.size(1)):
            Alab = Alab + (Abits[:, j] << j)

        if include_A_onehot:
            Aoh = torch.nn.functional.one_hot(Alab, num_classes=K_A).float().to(device)
            X_cat = torch.cat([X_cat, Aoh], dim=1)

        return (X_cat, Alab) if return_A_labels else X_cat


    def _descendants_in_var_dag(var_edges, start_vars):
        """Return descendants (excluding starts) in variable-level DAG given as list[(u,v)] with u->v."""
        from collections import defaultdict, deque
        adj = defaultdict(list)
        for u, v in var_edges:
            adj[int(u)].append(int(v))
        starts = set(int(s) for s in start_vars)
        q = deque(list(starts))
        seen = set(starts)
        desc = set()
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    desc.add(v)
                    q.append(v)
        return desc


    with torch.no_grad():
        sc_te = pred(X_pred_te).squeeze(1)
        if Y_is_cont:
            rmse_full = float(torch.sqrt(torch.mean((sc_te - Y_te)**2)).item())
        else:
            rmse_full = float(auc_from_logits_sklearn(sc_te, Y_te))

    def build_true_parent_Z_matrices(
            G, varnames,
            A_cluster: str,
            train_idx: torch.Tensor,
            test_idx: torch.Tensor,
            HAS_XAD: bool = True,
            Xad_cluster: str = None,
            bool_cluster_level: bool = True
        ):
            """
            Build (oracle) adjustment Z matrices.

            When bool_cluster_level=False (legacy):
            - Use variable-level DAG: G['meta']['var_edges'] over (V variables).
            - Z is the union of *variable* parents of variables in A-cluster (and Xad-cluster if HAS_XAD).

            When bool_cluster_level=True:
            - Use cluster-level DAG: G['meta']['cluster_edges'] over clusters.
            - Z is the union of *cluster* parents of A_cluster (and Xad_cluster if HAS_XAD).
            - Returned Z matrices are built by concatenating tensors of those parent clusters
                from G['cluster_tensors'] (so Z is still variable-valued, but selected via clusters).

            If Z is empty/unknown, return a constant vector.
            """

            # -----------------------------
            # Cluster-level oracle Z
            # -----------------------------
            if bool_cluster_level:
                if HAS_XAD and (Xad_cluster is None or Xad_cluster == ""):
                    raise ValueError("HAS_XAD=True, but Xad_cluster is None/empty.")

                cluster_edges = G["meta"].get("cluster_edges", [])
                cluster_tensors = G.get("cluster_tensors", None)
                cluster_to_vars = G.get("cluster_to_vars", None)
                if cluster_tensors is None or cluster_to_vars is None:
                    raise ValueError("bool_cluster_level=True, but G['cluster_tensors'] or G['cluster_to_vars'] not found")

                target_clusters = {A_cluster}
                # parents at cluster-level
                parents_A = sorted({u for (u, v) in cluster_edges if v in target_clusters})
                if HAS_XAD:
                    target_clusters = {Xad_cluster}      
                    parents_X = sorted({u for (u, v) in cluster_edges if v in target_clusters})

                if HAS_XAD:
                    parent_clusters = sorted(set(parents_A) | set(parents_X))
                else:
                    parent_clusters = sorted(set(parents_A))

                if len(parent_clusters) == 0:
                    Z_tr = torch.ones((len(train_idx), 1), dtype=torch.float32, device=DEVICE)
                    Z_te = torch.ones((len(test_idx), 1), dtype=torch.float32, device=DEVICE)
                    return Z_tr, Z_te, 1, []

                # Concatenate parent-cluster tensors (each is (N, |cluster|))
                Z_all_list = [cluster_tensors[c].float().to(DEVICE) for c in parent_clusters]
                Z_all = torch.cat(Z_all_list, dim=1)

                Z_tr = Z_all[train_idx.to(DEVICE)].contiguous()
                Z_te = Z_all[test_idx.to(DEVICE)].contiguous()

                # Column names (variable names) consistent with concatenation order
                z_names = []
                for c in parent_clusters:
                    z_names.extend(cluster_to_vars[c])

                return Z_tr, Z_te, Z_tr.size(1), z_names

            # -----------------------------
            # Variable-level oracle Z 
            # -----------------------------
            data_np = G["data"]
            edges = G["meta"]["var_edges"]  # list of (u, v), u->v

            A_vars = [i for i, vn in enumerate(varnames) if vn.startswith(A_cluster + "_")]

            if HAS_XAD:
                if Xad_cluster is None or Xad_cluster == "":
                    raise ValueError("HAS_XAD=True, but Xad_cluster is None/empty.")
                Xad_vars = [i for i, vn in enumerate(varnames) if vn.startswith(Xad_cluster + "_")]
                target_vars = set(A_vars) | set(Xad_vars)
            else:
                target_vars = set(A_vars)

            parents_union = set()
            for (u, v) in edges:
                if v in target_vars:
                    parents_union.add(u)

            Z_cols = sorted(list(parents_union))
            X_all = torch.from_numpy(data_np).float().to(DEVICE)

            if len(Z_cols) == 0:
                Z_tr = torch.ones((len(train_idx), 1), dtype=torch.float32, device=DEVICE)
                Z_te = torch.ones((len(test_idx), 1), dtype=torch.float32, device=DEVICE)
                return Z_tr, Z_te, 1, []

            Z_tr = X_all[train_idx.to(DEVICE)][:, Z_cols].contiguous()
            Z_te = X_all[test_idx.to(DEVICE)][:, Z_cols].contiguous()
            z_names = [varnames[j] for j in Z_cols]
            return Z_tr, Z_te, Z_tr.size(1), z_names



    Z_tr_true, Z_te_true, z_dim, z_names = build_true_parent_Z_matrices( 
        G, varnames, A_cluster, train_idx, test_idx,
        HAS_XAD=HAS_XAD,
        Xad_cluster=Xad_cluster if HAS_XAD else None
    )


    # Propensity on true Z (train)
    if HAS_XAD:  
        jprop_model_trueZ = PropensityAX_General(z_dim=Z_tr_true.size(1), xad_meta=meta_xad, K_A=K_A).to(DEVICE)
        jprop_opt_trueZ = torch.optim.AdamW(jprop_model_trueZ.parameters(), lr=lr_prop)
        for _ in range(prop_steps):
            jprop_loss = - jprop_model_trueZ.log_g(A_lab_tr, Xad_tr.float(), Z_tr_true.float()).mean()
            jprop_opt_trueZ.zero_grad(); jprop_loss.backward(); jprop_opt_trueZ.step()
    else:  
        jprop_model_trueZ = PropensityA_General(z_dim=Z_tr_true.size(1), K_A=K_A).to(DEVICE)
        jprop_opt_trueZ = torch.optim.AdamW(jprop_model_trueZ.parameters(), lr=lr_prop)
        for _ in range(prop_steps):
            jprop_loss = - jprop_model_trueZ.log_g(A_lab_tr, Z_tr_true.float()).mean()
            jprop_opt_trueZ.zero_grad(); jprop_loss.backward(); jprop_opt_trueZ.step()


    # Test MMD with true Z (reuse z once)
    def interventional_mmd_test_trueZ(  
        pred_model,
        X_pred_te_local: torch.Tensor,
        A_lab_te: torch.Tensor,
        Z_te: torch.Tensor,
        model_trueZ,
        Y_is_cont: bool,
        sigma_for_rff: float,
        HAS_XAD: bool = True,            
        Xad_te: torch.Tensor = None,     
        x_targets: torch.Tensor = None,  
        meta_xad: dict = None            
    ):
        with torch.no_grad():
            score_te = pred_model(X_pred_te_local).squeeze(1)
            mmd_input_te = torch.sigmoid(score_te) if not Y_is_cont else score_te

        z_te = rff_features(mmd_input_te, D=128, sigma=sigma_for_rff, device=X_pred_te_local.device)

        K_A_local = int(A_lab_te.max().item()) + 1 if A_lab_te.numel() > 0 else 1

        if HAS_XAD:
            if Xad_te is None:
                raise ValueError("HAS_XAD=True, but Xad_te is None")
            if x_targets is None:
                raise ValueError("HAS_XAD=True, but x_targets is None")
            if meta_xad is None:
                raise ValueError("HAS_XAD=True, but meta_xad is None.")

            with torch.no_grad():
                weights_per_x = []
                for r in range(x_targets.size(0)):
                    x0 = x_targets[r].view(1, -1)
                    Kx = kernel_on_Xad(Xad_te.float(), x0.float(), meta_xad, h_cont=None)
                    ws = []
                    for a in range(K_A_local):
                        logg = model_trueZ.log_g_at(a, x0.float(), Z_te.float())
                        g = torch.exp(logg).clamp(1e-12)
                        numer = (A_lab_te == a).float() * Kx
                        w = numer / g
                        w = w / (w.sum() + 1e-12)
                        ws.append(w)
                    weights_per_x.append(ws)

            pens_list = []
            for ws in weights_per_x:
                pens_list.append(barycenter_mmd_for_x_given_z(z_te, ws, chunk=4096))
            return float(torch.stack(pens_list).mean().item())

        else:
            # w_i^a ¥propto 1[A_i=a]/g(a|Z_i)
            with torch.no_grad():
                ws = []
                for a in range(K_A_local):
                    logg = model_trueZ.log_g_at(a, Z_te.float())  # PropensityA_General
                    g = torch.exp(logg).clamp(1e-12)
                    numer = (A_lab_te == a).float()
                    w = numer / g
                    w = w / (w.sum() + 1e-12)
                    ws.append(w)

            pens = barycenter_mmd_for_x_given_z(z_te, ws, chunk=4096)
            return float(pens.item())

    mmd_full = interventional_mmd_test_trueZ(
        pred_model=pred,
        X_pred_te_local=X_pred_te,
        A_lab_te=A_lab_te,
        Z_te=Z_te_true,
        model_trueZ=jprop_model_trueZ,
        Y_is_cont=Y_is_cont,
        sigma_for_rff=sigma0,
        HAS_XAD=HAS_XAD,
        Xad_te=(Xad_te if HAS_XAD else None), 
        x_targets=(x_targets_te if HAS_XAD else None),
        meta_xad=(meta_xad if HAS_XAD else None)
    )

    # baseline0: No fairness constraint
    rmse_b0, pred_b0 = train_mlp(X_pred_tr, Y_tr, X_pred_te, Y_te,epochs=int(epochs), lr=lr_mlp, is_binary=(not Y_is_cont), batch_size=batch_size,Xval=X_pred_va, Yval=Y_va)
    mmd_b0 = interventional_mmd_test_trueZ(
        pred_model=pred_b0,
        X_pred_te_local=X_pred_te,
        A_lab_te=A_lab_te,
        Z_te=Z_te_true,
        model_trueZ=jprop_model_trueZ,
        Y_is_cont=Y_is_cont,
        sigma_for_rff=sigma0,
        HAS_XAD=HAS_XAD,
        Xad_te=(Xad_te if HAS_XAD else None),
        x_targets=(x_targets_te if HAS_XAD else None),
        meta_xad=(meta_xad if HAS_XAD else None)
    )

 
    # ---------- Baseline1 (Oracle): variable-level true DAG descendants ----------
    var_edges_true = G["meta"]["var_edges"]  # list of (u,v) with u->v at *variable* level
    A_vars = [i for i, vn in enumerate(varnames) if vn.startswith(A_cluster + "_")]
    start_vars = set(A_vars)
    if HAS_XAD:
        Xad_vars = [i for i, vn in enumerate(varnames) if vn.startswith(Xad_cluster + "_")]
        start_vars |= set(Xad_vars)

    desc_vars = _descendants_in_var_dag(var_edges_true, start_vars)
    drop_var_idxs_oracle = start_vars | desc_vars

    X_pred_tr_b1 = build_pred_inputs_from_X_drop_vars(
        cluster_tensors,
        varnames=varnames,
        clusters_order=G["clusters"],
        A_cluster=A_cluster,
        Y_cluster=meta_info["Y"],
        K_A=K_A,
        idx_sel=train_idx,
        drop_var_idxs=drop_var_idxs_oracle,
        include_A_onehot=False,
        device=DEVICE
    )
    X_pred_va_b1 = build_pred_inputs_from_X_drop_vars(
        cluster_tensors,
        varnames=varnames,
        clusters_order=G["clusters"],
        A_cluster=A_cluster,
        Y_cluster=meta_info["Y"],
        K_A=K_A,
        idx_sel=val_idx,
        drop_var_idxs=drop_var_idxs_oracle,
        include_A_onehot=False,
        device=DEVICE
    )
    X_pred_te_b1 = build_pred_inputs_from_X_drop_vars(
        cluster_tensors,
        varnames=varnames,
        clusters_order=G["clusters"],
        A_cluster=A_cluster,
        Y_cluster=meta_info["Y"],
        K_A=K_A,
        idx_sel=test_idx,
        drop_var_idxs=drop_var_idxs_oracle,
        include_A_onehot=False,
        device=DEVICE
    )
    rmse_b1, pred_b1 = train_mlp(
        X_pred_tr_b1, Y_tr, X_pred_te_b1, Y_te,
        epochs=int(epochs // 10), lr=lr_mlp, is_binary=(not Y_is_cont), batch_size=batch_size,
        Xval=X_pred_va_b1, Yval=Y_va
    )
    mmd_b1 = interventional_mmd_test_trueZ(
        pred_model=pred_b1,
        X_pred_te_local=X_pred_te_b1,
        A_lab_te=A_lab_te,
        Z_te=Z_te_true,
        model_trueZ=jprop_model_trueZ,
        Y_is_cont=Y_is_cont,
        sigma_for_rff=sigma0,
        HAS_XAD=HAS_XAD,
        Xad_te=(Xad_te if HAS_XAD else None),
        x_targets=(x_targets_te if HAS_XAD else None),
        meta_xad=(meta_xad if HAS_XAD else None)
    )

    is_real_data = dataname in ("adult", "german", "oulad")
    perf_label = "AUC" if is_real_data else "RMSE"
    print("\n=== Test metrics ===")
    print(f"Full (proposed; with fairness) : {perf_label}={rmse_full:.4f}")
    print(f"Baseline0 full features (λ=0)   : {perf_label}={rmse_b0:.4f}")
    print(f"Baseline1 Oracle (true var-desc dropped): {perf_label}={rmse_b1:.4f}")

    print(f"[INFO] A={A_cluster}, X_ad={Xad_cluster}, Y={meta_info['Y']}")

    ### ==== compare interventional distribution densities P(Y|do(A), do(Xad)) for different do(A=a) and do(A=a') ==== ###
    ## == utility functions == ##
    # ---- helpers: build cluster tensors from a raw X-matrix (N,V) ----
    def build_cluster_tensors_from_matrix(X_mat: np.ndarray, varnames: list, cluster_to_vars: dict):
        T = torch.from_numpy(X_mat).float().to(DEVICE)
        var_to_idx = {v: i for i, v in enumerate(varnames)}
        def cols(names): return [var_to_idx[n] for n in names]
        return {c: T[:, cols(vs)] for c, vs in cluster_to_vars.items()}

    # ---- helpers: fixed value for Xad (mode if binary-like, else mean) ----
    ## mode (most frequent value) for binary Xad; mean for continuous Xad
    def _mode_1d(x_np: np.ndarray) -> float:
        vals, counts = np.unique(x_np, return_counts=True)
        return float(vals[np.argmax(counts)])

    def pick_fixed_value_for_cluster(G, cluster_name: str) -> float:
        ctype = G["meta"]["cluster_types"].get(cluster_name, "cont")
        cols = [G["varnames"].index(v) for v in G["cluster_to_vars"][cluster_name]]
        x = G["data"][:, cols].astype(np.float32)
        # take only the first variable value of Xad?
        col0 = x[:, 0]
        if ctype == "binary":
            return _mode_1d(col0)
        return float(np.mean(col0))
    # ---- helpers: make do_assign dict for A-bits & Xad ----
    def make_do_assign_pattern(G, A_bits: list, Xad_value: float) -> dict:
        """
        A_bits: e.g., [0,0], [0,1], ...
        Xad_value: fixed
        """
        name2idx = {v: i for i, v in enumerate(G["varnames"])}
        do = {}
        A_vars = G["cluster_to_vars"][G["meta"]["A"]]
        assert len(A_bits) == len(A_vars), f"A_bits length {len(A_bits)} != len(A_vars) {len(A_vars)}"
        for b, vname in zip(A_bits, A_vars):
            do[name2idx[vname]] = float(b)

        if HAS_XAD:
            Xad_first = G["cluster_to_vars"][G["meta"]["Xad"]][0]
            do[name2idx[Xad_first]] = float(Xad_value)
        return do

    Xad_fixed_value = pick_fixed_value_for_cluster(G, Xad_cluster) if ('HAS_XAD' in globals() and HAS_XAD) else None 
    A_dim = len(G["cluster_to_vars"][A_cluster])
    assert A_dim >= 1, "A cluster must have >=1 dimension"
    A_patterns = [list(bits) for bits in itertools.product([0, 1], repeat=A_dim)]

    N_do = len(test_idx)
    X_do_map = {}
    for pat in A_patterns:
        do_assign = make_do_assign_pattern(G, A_bits=pat, Xad_value=(Xad_fixed_value if Xad_fixed_value is not None else 0.0))  
        X_do = sample_interventional_dataset(G, dataname, do_assign, N_do=N_do)
        X_do_map[tuple(pat)] = X_do

    def model_outputs_on_DoX(pred_model, Xp, cluster_tensors_do: dict, is_binary_target: bool) -> np.ndarray:
        with torch.no_grad():
            sc = pred_model(Xp).squeeze(1)
            if is_binary_target:
                out = torch.sigmoid(sc)
            else:
                out = sc
        return out.detach().cpu().numpy()
    ## == utility functions == ##


    # baseline: A=00
    A_base = tuple(A_patterns[0])
    outs_map = {}
    for pat in A_patterns[1:]:
        key = tuple(pat)
        # cluster_tensors for baseline & this pattern
        ct_base = build_cluster_tensors_from_matrix(X_do_map[A_base], varnames, cluster_to_vars)
        ct_this = build_cluster_tensors_from_matrix(X_do_map[key],     varnames, cluster_to_vars)
        # build inputs for each predictor
        X_pred_do_full_base = build_X_pred(ct_base)
        X_pred_do_full_this = build_X_pred(ct_this)            
        X_pred_do_b1_base = build_pred_inputs_from_X_drop_vars(
            ct_base,
            varnames=varnames,
            clusters_order=G["clusters"],
            A_cluster=A_cluster,
            Y_cluster=meta_info["Y"],
            K_A=K_A,
            idx_sel=None,
            drop_var_idxs=drop_var_idxs_oracle,
            include_A_onehot=False,
            device=DEVICE
        )
        X_pred_do_b1_this = build_pred_inputs_from_X_drop_vars(
            ct_this,
            varnames=varnames,
            clusters_order=G["clusters"],
            A_cluster=A_cluster,
            Y_cluster=meta_info["Y"],
            K_A=K_A,
            idx_sel=None,
            drop_var_idxs=drop_var_idxs_oracle,
            include_A_onehot=False,
            device=DEVICE
        )
        # collect outputs for each model
        outs = {}
        outs["Proposed"]  = (model_outputs_on_DoX(pred, X_pred_do_full_base, ct_base, not Y_is_cont),
                         model_outputs_on_DoX(pred, X_pred_do_full_this,  ct_this, not Y_is_cont))
        outs["Baseline0"] = (model_outputs_on_DoX(pred_b0, X_pred_do_full_base, ct_base, not Y_is_cont),
                         model_outputs_on_DoX(pred_b0, X_pred_do_full_this, ct_this, not Y_is_cont))
        outs["Baseline1"] = (model_outputs_on_DoX(pred_b1, X_pred_do_b1_base,  ct_base, not Y_is_cont),
                         model_outputs_on_DoX(pred_b1, X_pred_do_b1_this, ct_this, not Y_is_cont))
        outs_map[key] = outs

    mmd_results = {
        "baseline_A": None,
        "xad_fixed_value": (float(Xad_fixed_value) if Xad_fixed_value is not None else None),
        "patterns": {}
    }
    def _mmd_1d_pair(v_base: np.ndarray, v_this: np.ndarray) -> float:
        sigma = sigma0
        xb = torch.from_numpy(v_base).to(DEVICE).float()
        xt = torch.from_numpy(v_this).to(DEVICE).float()
        if not Y_is_cont:
            return float(torch.abs(xb.mean() - xt.mean()).detach().item())
        try:
            m = gaussian_mmd1d(xb, xt, sigma=sigma)
            return float(m.detach().item())
        except Exception:
            def _rbf(x, y, s):
                x = x.view(-1, 1); y = y.view(1, -1)
                return torch.exp(-(x - y)**2 / (2.0 * max(s, 1e-6)**2))
            mmd2 = _rbf(xb, xb, sigma).mean() + _rbf(xt, xt, sigma).mean() - 2.0 * _rbf(xb, xt, sigma).mean()
            return float(mmd2.detach().item())
    for pat in A_patterns[1:]:
        key = tuple(pat)
        outs = outs_map[key]
        pat_key = f"A={''.join(map(str, key))}"
        mmd_results["patterns"][pat_key] = {}
        for name, (v_base, v_this) in outs.items():
            mmd_results["patterns"][pat_key][name] = _mmd_1d_pair(v_base, v_this)

    # ===== RETURN METRICS (for multiple experiments aggregation) =====
    names = ["Proposed", "Baseline0", "Baseline1"]
    metrics = np.array([
        [rmse_full, mmd_full],
        [rmse_b0,   mmd_b0],
        [rmse_b1,   0.0],
    ], dtype=float)

    return metrics, names, mmd_results

     


if __name__ == "__main__":
    import argparse
    from argparse import Namespace

    ap = argparse.ArgumentParser()
    ap.add_argument('--dataname', type=str, default="synth")
    ap.add_argument('--seed', type=int, default=23)
    ap.add_argument('--N', type=int, default=5000)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--prop_steps', type=int, default=120)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--lr_mlp', type=float, default=2e-03)
    ap.add_argument('--lr_prop', type=float, default=1e-02)
    ap.add_argument('--R', type=int, default=8)
    ap.add_argument('--lambda', dest='lmbda', type=float, default=1.0)
    ap.add_argument('--d', dest='d_clusters', type=int, default=6, help='number of clusters (>=4 recommended)')
    ap.add_argument('--hard_pred', action='store_true', help='for binary Y: use straight-through Gumbel-Softmax for MMD input')
    ap.add_argument('--tau', type=float, default=0.5, help='Gumbel-Softmax temperature')

    ap.add_argument('--num_experiments', type=int, default=1)

    args = ap.parse_args()

    all_metrics = []; all_mmd_results = []
    names = None

    # repeat experiments (vary seed by +i)
    for i in range(int(args.num_experiments)):
        args_i = Namespace(**vars(args))
        args_i.seed = int(args.seed) + i
        metrics_i, names_i, mmd_results_i = main(args_i)

        all_metrics.append(metrics_i)
        all_mmd_results.append(mmd_results_i)

        if names is None:
            names = names_i

    all_metrics = np.stack(all_metrics, axis=0)  # (E, 5, 2)
    mean = all_metrics.mean(axis=0)              # (5, 2)
    std  = all_metrics.std(axis=0, ddof=1) if all_metrics.shape[0] > 1 else np.zeros_like(mean)


    # --- build per-experiment, per-method averaged interventional MMD over A-patterns ---
    # shape: (E, K)  where K=len(names)
    mmd_pat_mat = None
    if len(all_mmd_results) > 0 and all_mmd_results[0] is not None:
        K = len(names)
        E = len(all_mmd_results)
        mmd_pat_mat = np.full((E, K), np.nan, dtype=float)

        for e, res in enumerate(all_mmd_results):
            if res is None or "patterns" not in res:
                continue
            pats = res["patterns"]  # dict: pat_key -> {method_name: value, ...}
            if len(pats) == 0:
                continue

            # average over patterns for each method
            for k, method in enumerate(names):
                vals = []
                for _pat_key, row in pats.items():
                    if method in row:
                        vals.append(float(row[method]))
                if len(vals) > 0:
                    mmd_pat_mat[e, k] = float(np.mean(vals))

        mmd_pat_mean = np.nanmean(mmd_pat_mat, axis=0)
        mmd_pat_std  = (np.nanstd(mmd_pat_mat, axis=0, ddof=1)
                        if mmd_pat_mat.shape[0] > 1 else np.zeros_like(mmd_pat_mean))

    is_real_data = args.dataname in ("adult", "german", "oulad")
    perf_label = "AUC" if is_real_data else "RMSE"
    print(f"\n=== Aggregate over num_experiments={int(args.num_experiments)} (seed={int(args.seed)}..{int(args.seed)+int(args.num_experiments)-1}) ===")
    for k, name in enumerate(names):
        rmse_mu, mmd_mu = mean[k, 0], mean[k, 1]
        rmse_sd, mmd_sd = std[k, 0], std[k, 1]

        if not is_real_data and mmd_pat_mat is not None:
            mmdI_mu = mmd_pat_mean[k]
            mmdI_sd = mmd_pat_std[k]
            print(f"{name:9s}: {perf_label}={rmse_mu:.4f} +- {rmse_sd:.4f}, "
                f"MMD={mmdI_mu:.4f} +- {mmdI_sd:.4f}")
        else:
            print(f"{name:9s}: {perf_label}={rmse_mu:.4f} +- {rmse_sd:.4f}, "
                f"MMD={mmd_mu:.4f} +- {mmd_sd:.4f}")

    def print_stats(x, name="M", ps=(5, 25, 50, 75, 95), ddof=1):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        n = x.size
        if n == 0:
            print(f"{name}: (no data)")
            return

        mean = x.mean()
        std  = x.std(ddof=ddof) if n > 1 else 0.0
        vmin, vmax = x.min(), x.max()
        q = np.percentile(x, ps)
        qdict = {p: v for p, v in zip(ps, q)}
        if 25 in qdict and 75 in qdict:
            iqr = qdict[75] - qdict[25]
            iqr_str = f", IQR={iqr:.4f}"
        else:
            iqr_str = ""

        p_str = ", ".join([f"p{p}={qdict[p]:.4f}" for p in ps])
        print(f"{name}: n={n}, mean={mean:.4f}, std={std:.4f}, min={vmin:.4f}, max={vmax:.4f}{iqr_str}, {p_str}")

