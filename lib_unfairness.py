
import os, math, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import roc_auc_score

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

# ---------------- Reproducibility ----------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- Heuristics ----------------
def median_heuristic_sigma_1d(y: torch.Tensor) -> float:
    y = y.detach().flatten()
    if y.numel()<2: return 0.2
    m = min(y.numel(), 4096)
    idx = torch.randperm(y.numel(), device=y.device)[:m]
    ys = y[idx]
    diffs = (ys.unsqueeze(0)-ys.unsqueeze(1)).abs().flatten()
    diffs = diffs[diffs>0]
    if diffs.numel()==0: return 0.2
    med = torch.median(diffs).item()
    return max(float(med), 1e-3)

def is_discrete_column(col: torch.Tensor, max_unique: int = 20) -> bool:
    if col.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return True
    v = col.flatten()
    if v.numel() > 5000:
        idx = torch.randperm(v.numel(), device=v.device)[:5000]
        v = v[idx]
    return torch.unique(v).numel() <= min(max_unique, max(2, v.numel()//10))

def detect_xad_types(Xad: torch.Tensor, max_unique: int = 20) -> Dict:
    if Xad.dim()==1: Xad = Xad.unsqueeze(1)
    disc_idx, cont_idx, cardinalities = [], [], {}
    for j in range(Xad.size(1)):
        col = Xad[:, j]
        if is_discrete_column(col, max_unique=max_unique):
            disc_idx.append(j)
            cardinalities[j] = int(torch.unique(col).numel())
        else:
            cont_idx.append(j)
    return {'disc_idx': disc_idx, 'cont_idx': cont_idx, 'cardinalities': cardinalities}

def median_bandwidth_per_dim(X: torch.Tensor) -> torch.Tensor:
    hs=[]
    for j in range(X.size(1)):
        hs.append(median_heuristic_sigma_1d(X[:, j]))
    return torch.tensor(hs, device=X.device, dtype=X.dtype)

# ---------------- Feature builders ----------------
def onehot_from_discrete_cols(X_disc: torch.Tensor, cardinalities: List[int]) -> torch.Tensor:
    if X_disc.numel()==0: return torch.zeros((X_disc.shape[0],0), device=X_disc.device)
    outs=[]
    for j in range(X_disc.shape[1]):
        C = cardinalities[j]
        outs.append(F.one_hot(X_disc[:, j].long().clamp(min=0, max=C-1), num_classes=C).float())
    return torch.cat(outs, dim=1) if outs else torch.zeros((X_disc.shape[0],0), device=X_disc.device)

def build_xad_features_for_A_head(Xad: torch.Tensor, meta: Dict) -> torch.Tensor:
    if Xad.dim()==1: Xad = Xad.unsqueeze(1)
    cont = Xad[:, meta['cont_idx']] if meta['cont_idx'] else torch.zeros((Xad.size(0),0), device=Xad.device)
    if meta['disc_idx']:
        Xd = Xad[:, meta['disc_idx']]
        cards = [meta['cardinalities'][j] for j in meta['disc_idx']]
        oh = onehot_from_discrete_cols(Xd, cards)
    else:
        oh = torch.zeros((Xad.size(0),0), device=Xad.device)
    return torch.cat([cont, oh], dim=1)

# =========================
# Density plots (Visualization)
# =========================

# ---- KDE (Gaussian, 1D, simple implementation) ----
def kde1d(values: np.ndarray, grids: np.ndarray = None, bandwidth: float = None) -> tuple[np.ndarray, np.ndarray]:
    """
    returns (grid_x, density_y)
    """
    v = values[~np.isnan(values)]
    if v.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    if grids is None:
        lo, hi = np.percentile(v, [0.5, 99.5])
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        grids = np.linspace(lo, hi, 512)
    if bandwidth is None:
        # Silverman's rule
        s = np.std(v)
        iqr = np.subtract(*np.percentile(v, [75, 25]))
        sigma = min(s, iqr / 1.349) if iqr > 0 else s
        n = max(1, v.size)
        bandwidth = 0.9 * sigma * (n ** (-1/5)) if sigma > 1e-12 else 0.1
    h = max(1e-6, bandwidth)
    # Gaussian kernels
    diff = (grids[:, None] - v[None, :]) / h
    dens = np.exp(-0.5 * diff**2).mean(axis=1) / (h * np.sqrt(2 * np.pi))
    return grids, dens


# ---------------- RFF for 1D p_hat ----------------
def rff_features(y: torch.Tensor, D: int = 128, sigma: Optional[float] = None, device=None) -> torch.Tensor:
    if device is None: device=y.device
    if sigma is None: sigma = median_heuristic_sigma_1d(y)
    g = torch.Generator(device=device)
    rff_seed = 0; g.manual_seed(int(rff_seed))    
    w = torch.randn(D, device=device, generator=g) / sigma
    b = torch.rand(D, device=device, generator=g) * (2*math.pi)
    return torch.cos(y.unsqueeze(-1)*w + b) * math.sqrt(2.0/D)

def multi_rff_features_1d(y: torch.Tensor, D: int, sigmas: list[float], device=None):
    """
    y: (N,)  / sigmas: list of bandwidths
    Return: (N, D_total) where D_total = len(sigmas) * (D_per)
    """
    M = len(sigmas)
    assert M > 0
    D_per = max(1, D // M)
    feats = []
    for s in sigmas:
        feats.append(rff_features(y, D=D_per, sigma=float(s), device=device))
    z = torch.cat(feats, dim=1)
    return z / math.sqrt(M)

# ---------------- Models ----------------
class SoftmaxA(nn.Module):
    def __init__(self, in_dim: int, K: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, K))
    def forward(self, x):
        return self.net(x)  # logits

class ContDensityDiagGaussian(nn.Module):
    def __init__(self, z_dim: int, d_cont: int):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, d_cont))
        self.logsig = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, d_cont))
    def forward(self, Z):
        mu = self.mu(Z)
        logs = self.logsig(Z).clamp(-5, 3)
        return mu, logs

class DiscMassProduct(nn.Module):
    def __init__(self, z_dim: int, cardinals: List[int]):
        super().__init__()
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, C)) for C in cardinals])
        self.cardinals = cardinals
    def log_prob(self, Z: torch.Tensor, X_disc: torch.Tensor) -> torch.Tensor:
        if X_disc.numel()==0: return torch.zeros(Z.size(0), device=Z.device)
        lp = 0.0
        for j, head in enumerate(self.heads):
            logits = head(Z)
            lp_j = F.log_softmax(logits, dim=1)
            idx = X_disc[:, j].long().clamp(min=0, max=self.cardinals[j]-1).unsqueeze(1)
            lp = lp + lp_j.gather(1, idx).squeeze(1)
        return lp
    def log_prob_at_x0(self, Z: torch.Tensor, x0_disc: torch.Tensor) -> torch.Tensor:
        if x0_disc.numel()==0:
            return torch.zeros(Z.size(0), device=Z.device)
        lp = 0.0
        N = Z.size(0)
        for j, head in enumerate(self.heads):
            logits = head(Z)                     # (N, C_j)
            lp_j = F.log_softmax(logits, dim=1)  # (N, C_j)
            v = x0_disc[..., j].long().clamp(0, self.cardinals[j]-1)
            # normalize shape of index to (N,1)
            if v.dim() == 0:
                idx = v.view(1,1).expand(N,1)
            elif v.dim() == 1:
                if v.numel() == N:
                    idx = v.view(N,1)
                else:
                    idx = v.view(1,1).expand(N,1)
            else:
                idx = v.unsqueeze(-1)
                if idx.size(0) != N:
                    idx = idx.view(1,1).expand(N,1)
            lp = lp + lp_j.gather(1, idx).squeeze(1)
        return lp

class PropensityAX_General(nn.Module):
    """
    g(a, x | Z) = P(A=a | Z, x_feat) * f_cont(x_cont | Z) * P_disc(x_disc | Z)
    """
    def __init__(self, z_dim: int, xad_meta: Dict, K_A: int):
        super().__init__()
        self.meta = xad_meta
        self.K_A = K_A
        d_cont = len(self.meta['cont_idx'])
        feat_dim = d_cont + sum([self.meta['cardinalities'][j] for j in self.meta['disc_idx']])
        self.A_head = SoftmaxA(in_dim = z_dim + feat_dim, K = K_A)
        self.has_cont = d_cont > 0
        self.has_disc = len(self.meta['disc_idx']) > 0
        if self.has_cont:
            self.cont = ContDensityDiagGaussian(z_dim=z_dim, d_cont=d_cont)
        if self.has_disc:
            cards = [self.meta['cardinalities'][j] for j in self.meta['disc_idx']]
            self.disc = DiscMassProduct(z_dim=z_dim, cardinals=cards)

    def log_g(self, A: torch.Tensor, Xad: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        if Xad.dim()==1: Xad = Xad.unsqueeze(1)
        xfeat = build_xad_features_for_A_head(Xad, self.meta)
        logits = self.A_head(torch.cat([Z, xfeat], dim=1))
        lpA = F.log_softmax(logits, dim=1).gather(1, A.long().unsqueeze(1)).squeeze(1)
        lp = lpA
        if self.has_cont:
            Xc = Xad[:, self.meta['cont_idx']]
            mu, logs = self.cont(Z)
            lp_cont = -0.5 * (((Xc - mu) * torch.exp(-logs))**2 + 2*logs + math.log(2*math.pi)).sum(dim=1)
            lp = lp + lp_cont
        if self.has_disc:
            Xd = Xad[:, self.meta['disc_idx']]
            lp_disc = self.disc.log_prob(Z, Xd)
            lp = lp + lp_disc
        return lp

    def log_g_at(self, a: int, x0: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        if x0.dim()==1: x0 = x0.unsqueeze(0)
        xfeat = build_xad_features_for_A_head(x0.expand(Z.size(0), -1), self.meta)
        logits = self.A_head(torch.cat([Z, xfeat], dim=1))
        lpA = F.log_softmax(logits, dim=1)[:, a]
        lp = lpA
        if self.has_cont:
            mu, logs = self.cont(Z)
            x0c = x0[:, self.meta['cont_idx']].expand_as(mu) if self.meta['cont_idx'] else None
            if x0c is not None:
                lp_cont = -0.5 * (((x0c - mu) * torch.exp(-logs))**2 + 2*logs + math.log(2*math.pi)).sum(dim=1)
                lp = lp + lp_cont
        if self.has_disc:
            x0d = x0[:, self.meta['disc_idx']]
            lp_disc = self.disc.log_prob_at_x0(Z, x0d.squeeze(0))
            lp = lp + lp_disc
        return lp

class PropensityA_General(nn.Module):
    """
    g(a | Z) = P(A=a | Z)
    """
    def __init__(self, z_dim: int, K_A: int):
        super().__init__()
        self.K_A = K_A
        self.A_head = SoftmaxA(in_dim=z_dim, K=K_A)

    def log_g(self, A: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        logits = self.A_head(Z)
        lpA = F.log_softmax(logits, dim=1).gather(1, A.long().unsqueeze(1)).squeeze(1)
        return lpA

    def log_g_at(self, a: int, Z: torch.Tensor) -> torch.Tensor:
        logits = self.A_head(Z)
        lpA = F.log_softmax(logits, dim=1)[:, a]
        return lpA

def auc_from_logits_sklearn(
    logits: torch.Tensor,
    y_true: torch.Tensor,
) -> float:
    """
    Binary AUC using scikit-learn's roc_auc_score.
    logits can be raw logits or any continuous score (sigmoid is not required).
    y_true must be 0/1.

    Returns nan if AUC is undefined (only one class present).
    """
    y = y_true.detach().view(-1).to("cpu").numpy().astype(int)
    s = logits.detach().view(-1).to("cpu").numpy()

    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))

def train_mlp(
    Xtr, Ytr, Xte, Yte, *, epochs, lr=2e-3, wd=0,
    is_binary=False, device=None, batch_size: int = None,
    Xval=None, Yval=None, patience: int = 15, min_delta: float = 1e-4,
):
    device = Xtr.device if device is None else device
    model = nn.Sequential(nn.Linear(Xtr.size(1), 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # === early stopping ===
    do_early_stopping = False
    use_val = do_early_stopping and (Xval is not None) and (Yval is not None)
    if use_val:
        Xval = Xval.to(device)
        Yval = Yval.to(device)
        best_obj = float("inf")
        best_state = None
        wait = 0
    # =======================================

    if not batch_size:  # full-batch
        for _ in range(epochs):
            sc = model(Xtr.to(device)).squeeze(1)
            loss = (torch.mean((sc - Ytr.to(device))**2) if not is_binary
                    else F.binary_cross_entropy_with_logits(sc, Ytr.to(device)))
            opt.zero_grad(); loss.backward(); opt.step()

            if use_val:  # do_early_stopping=False implies use_val=False
                with torch.no_grad():
                    scv = model(Xval).squeeze(1)
                    val_task = (torch.mean((scv - Yval)**2) if not is_binary
                                else F.binary_cross_entropy_with_logits(scv, Yval))
                    val_obj = float(val_task.item())
                if val_obj < best_obj - min_delta:
                    best_obj = val_obj
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
    else:  # mini-batch
        for _ in range(epochs):
            for Xb, Yb, _idx in _batches(Xtr.to(device), Ytr.to(device), batch_size, shuffle=True):
                sc = model(Xb).squeeze(1)
                loss = (torch.mean((sc - Yb)**2) if not is_binary
                        else F.binary_cross_entropy_with_logits(sc, Yb))
                opt.zero_grad(); loss.backward(); opt.step()

            if use_val:
                with torch.no_grad():
                    scv = model(Xval).squeeze(1)
                    val_task = (torch.mean((scv - Yval)**2) if not is_binary
                                else F.binary_cross_entropy_with_logits(scv, Yval))
                    val_obj = float(val_task.item())
                if val_obj < best_obj - min_delta:
                    best_obj = val_obj
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

    # === best state ===
    if use_val and (best_state is not None):
        model.load_state_dict(best_state)

    with torch.no_grad():
        sc_te = model(Xte.to(device)).squeeze(1)
        _rmse = (torch.sqrt(torch.mean((sc_te - Yte.to(device))**2)) if not is_binary
                else auc_from_logits_sklearn(sc_te, Yte.to(device)))
        rmse = _rmse.item() if torch.is_tensor(_rmse) else _rmse

    return float(rmse), model

# =========================
# Mini-batch utilities
# =========================
def _batches(X: torch.Tensor, Y: torch.Tensor, batch_size: int, *, shuffle: bool = True):
    """
    Yield (Xb, Yb, idx_sel) where idx_sel are the row indices of this batch.
    """
    N = X.size(0)
    dev = X.device
    if shuffle:
        perm = torch.randperm(N, device=dev)
    else:
        perm = torch.arange(N, device=dev)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        idx_sel = perm[s:e]
        yield X[idx_sel], Y[idx_sel], idx_sel

def slice_Z_models(Z_models: list, idx_sel: torch.Tensor):
    """
    Z_models: [(Sset, Z_tr, Z_te, model), ...]
    """
    out = []
    for (Sset, Z_tr, Z_te, model) in Z_models:
        out.append((Sset, Z_tr[idx_sel], Z_te, model))
    return out


# ---------------- Kernels on X_ad ----------------
def kernel_on_Xad(Xad: torch.Tensor, x0: torch.Tensor, meta: Dict, h_cont: Optional[torch.Tensor] = None) -> torch.Tensor:
    if Xad.dim()==1: Xad = Xad.unsqueeze(1)
    if x0.dim()==1: x0 = x0.unsqueeze(0)
    N = Xad.size(0)
    K = torch.ones(N, device=Xad.device, dtype=Xad.dtype)
    if meta['disc_idx']:
        Xd = Xad[:, meta['disc_idx']]
        x0d = x0[:, meta['disc_idx']].expand(N, -1)
        K = K * (Xd == x0d).all(dim=1).float()
    if meta['cont_idx']:
        Xc = Xad[:, meta['cont_idx']]
        x0c = x0[:, meta['cont_idx']].expand(N, -1)
        hs = median_bandwidth_per_dim(Xc) if h_cont is None else h_cont.to(Xad.device)
        diffs = (Xc - x0c) / hs.clamp(min=1e-6)
        K = K * torch.exp(-0.5 * (diffs**2).sum(dim=1))
    return K

# ======== 1D Gaussian-kernel MMD ========
def gaussian_mmd1d(y0: torch.Tensor, y1: torch.Tensor, sigma: Optional[float] = None) -> float:
    y0 = y0.detach().view(-1,1); y1 = y1.detach().view(-1,1)
    with torch.no_grad():
        if sigma is None:
            z = torch.cat([y0, y1], dim=0)
            pd = torch.cdist(z, z, p=2.0) ** 2
            tri = pd[torch.triu_indices(pd.size(0), pd.size(1), offset=1).unbind()]
            tri = tri[tri > 0]
            if tri.numel() == 0:
                sigma = torch.tensor(1.0, device=z.device, dtype=z.dtype)
            else:
                med = tri.median()
                sigma = torch.sqrt(med / 2.0).clamp(min=1e-6)
        def k(a, b): 
            return torch.exp(-torch.cdist(a,b,p=2.0)**2 / (2*sigma**2))
        m, n = y0.size(0), y1.size(0)
        Kxx = k(y0, y0); Kyy = k(y1, y1); Kxy = k(y0, y1)
        mmd2 = (Kxx.sum() - torch.trace(Kxx)) / max(m*(m-1),1) \
             + (Kyy.sum() - torch.trace(Kyy)) / max(n*(n-1),1) \
             - 2 * Kxy.mean()
        return float(mmd2.item())

# =========== Unfairness penalty ===========

def barycenter_mmd_for_x_given_z(z: torch.Tensor, weights_for_a, chunk: int = 8192) -> torch.Tensor:
    """
    z: (N, D) precomputed RFF features of the MMD input
    weights_for_a: list of (N,) normalized weights per A value
    Returns scalar tensor (barycenter MMD).
    """
    device, dtype = z.device, z.dtype
    N, D = z.shape
    K = len(weights_for_a)
    vs = torch.zeros(K, D, device=device, dtype=dtype)

    for k, w in enumerate(weights_for_a):
        w = w.to(device=device, dtype=dtype, non_blocking=True)
        acc = torch.zeros(D, device=device, dtype=dtype)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            acc = acc + (z[s:e] * w[s:e].unsqueeze(1)).sum(0)
        vs[k] = acc

    omega = torch.ones(K, device=device, dtype=dtype)
    vbar = (omega[:, None] * vs).sum(0) / omega.sum()
    return (omega[:, None] * ((vs - vbar) ** 2)).sum() / K

## Aggregation over possible adjustment sets
def aggregate_pens(pens_Z: torch.Tensor,
                   mode: str = 'mlm',
                   beta: float = 10.0,
                   alpha: float = 0.2,
                   p: float = 8.0) -> torch.Tensor:
    """
    Aggregate penalties across adjustment sets.
    """
    if mode == 'max':
        return pens_Z.max()
    elif mode == 'mean':
        return pens_Z.mean()
    elif mode == 'sum':
        return pens_Z.sum()
    elif mode == 'lse': # log-sum-exp
        return torch.logsumexp(beta * pens_Z, dim=0) / beta
    elif mode == 'cvar':
        k = max(1, int(math.ceil(alpha * pens_Z.numel())))
        topk, _ = torch.topk(pens_Z, k=k, largest=True, sorted=False)
        return topk.mean()
    elif mode == 'pnorm':
        return (torch.mean(torch.clamp(pens_Z, min=0)**p))**(1.0/p)
    elif mode == 'mlm':# mellowmax; normalized lse 
        n = pens_Z.numel()
        # log(mean(exp(beta*x))) / beta
        return (torch.logsumexp(beta * pens_Z, dim=0) - math.log(n)) / beta    
    else:
        raise ValueError(f"Unknown agg mode: {mode}")
    
def weight_diagnostics(w: torch.Tensor, top_p=0.01):
    w = w.detach().flatten()
    w = w[torch.isfinite(w)]
    if w.numel() == 0:
        return {"ESS": float("nan"), "max": float("nan"), "top_share": float("nan")}
    s1 = w.sum().item()
    s2 = w.pow(2).sum().item()
    ess = (s1 * s1) / (s2 + 1e-12)
    w_max = w.max().item()
    k = max(1, int(round(w.numel() * top_p)))
    top_sum = w.topk(k).values.sum().item()
    top_share = top_sum / (s1 + 1e-12)
    return {"ESS": ess, "max": w_max, "top_share": top_share, "k": k, "N": w.numel()}

def measure_unfairness(
    *,
    pred: torch.nn.Module,
    X_pred_tr: torch.Tensor,            
    Y_is_cont: bool,
    hard_pred: bool,
    tau: float,
    D_FEAT: int,
    sigma0: float,                      # kernel bandwidth used in RFF
    A_lab_tr: torch.Tensor,             # 0..K_A-1
    Z_models: list,                     # [(Sset, Z_tr, Z_te, model), ...]
    K_A: int,
    device: torch.device,
    HAS_XAD: bool = True,               
    Xad_tr: torch.Tensor = None,        
    meta_xad: dict = None,              
    x_targets: torch.Tensor = None      
) -> torch.Tensor:
    """
    - HAS_XAD=True  : do(x_ad) -> use Kx and log_g_at(a,x0,Z) 
    - HAS_XAD=False : no do(x_ad) -> use weight 1[A=a]/g(a|Z) and log_g_at(a,Z) 
    """
    if HAS_XAD:
        if Xad_tr is None:
            raise ValueError("HAS_XAD=True, but Xad_tr is None")
        if meta_xad is None:
            raise ValueError("HAS_XAD=True, but meta_xad is None")
        if x_targets is None:
            raise ValueError("HAS_XAD=True, but x_targets is None")

    score = pred(X_pred_tr).squeeze(1)

    # precompute z once per call
    mmd_input = torch.sigmoid(score) if not Y_is_cont else score
    use_autocast = (device.type == 'cuda')
    use_multi_sigma = True if Y_is_cont else False
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_autocast):
        if use_multi_sigma:
            mults = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
            sigmas = [m * sigma0 for m in mults]
            z = multi_rff_features_1d(mmd_input, D=D_FEAT, sigmas=sigmas, device=device) 
        else:
            z = rff_features(mmd_input, D=D_FEAT, sigma=sigma0, device=device)  # (N,D)
    z = z.float()
    
    pens_Z = []
    clip_percentile = 1.0 # 0.95, 0.975, 0.99, 1.0 (no clipping)
    for (_Sset, Z_tr, _Z_te, model) in Z_models:

        if HAS_XAD:
            # ==============================
            # do(x_ad) 
            # ==============================
            weights_per_x = []
            for r in range(x_targets.size(0)):
                x0 = x_targets[r].view(1, -1)
                Kx = kernel_on_Xad(Xad_tr.float(), x0.float(), meta_xad, h_cont=None)  # (N,)

                ws = []
                for a in range(K_A):
                    # PropensityAX_General: log_g_at(a, x0, Z)
                    logg = model.log_g_at(a, x0.float(), Z_tr.float())  # (N,)
                    g = torch.exp(logg).clamp(1e-12)
                    numer = (A_lab_tr == a).float() * Kx
                    w = numer / g
                    diag_raw = weight_diagnostics(w)
                    w_max = torch.quantile(w[w>0], clip_percentile) if w[w > 0].numel() > 0 else torch.tensor(float("inf"), device=w.device, dtype=w.dtype)
                    w = w.clamp(max=w_max) # weight clipping (if necessary)
                    w = w / (w.sum() + 1e-12) # self-normalization
                    diag = weight_diagnostics(w)
                    ws.append(w)
                weights_per_x.append(ws)

            pens_list = []
            for ws in weights_per_x:
                pens_list.append(barycenter_mmd_for_x_given_z(z, ws, chunk=4096))
            pens_Z.append(torch.stack(pens_list).mean())  # mean over x_targets

        else:
            # ==============================
            # w_i^a ∝ 1[A_i=a] / g(a|Z_i)
            # ==============================
            ws = []
            for a in range(K_A):
                # PropensityA_General：log_g_at(a, Z)
                try:
                    logg = model.log_g_at(a, Z_tr.float())  # (N,)
                except TypeError as e:
                    raise TypeError(
                        "HAS_XAD=False, but cannot call model.log_g_at(a, Z)"
                        "Put PropensityA_General P(A|Z) into Z_models."
                    ) from e

                g = torch.exp(logg).clamp(1e-12)
                numer = (A_lab_tr == a).float()
                w = numer / g
                diag_raw = weight_diagnostics(w)
                w_max = torch.quantile(w[w>0], clip_percentile) if w[w > 0].numel() > 0 else torch.tensor(float("inf"), device=w.device, dtype=w.dtype)
                w = w.clamp(max=w_max) # weight clipping (if necessary)
                w = w / (w.sum() + 1e-12) # self-normalization
                diag = weight_diagnostics(w)
                ws.append(w)

            pens = barycenter_mmd_for_x_given_z(z, ws, chunk=4096)  # scalar tensor
            pens_Z.append(pens)

    fair = aggregate_pens(torch.stack(pens_Z))
    return fair

