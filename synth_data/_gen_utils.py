"""
Shared data-generation utilities for synthetic ANM generators.

Provides:
  - get_nonlinear: unified nonlinearity factory (superset of both random and fixed variants)
  - _simulate_linear: linear ANM simulation loop shared by random and inadmissible generators;
      also used (via fixed_anm wrappers) by the fixed generator
  - _simulate_nonlinear: nonlinear ANM simulation loop shared similarly

Key behavioral differences between the three generators are expressed as parameters:

  Y_set / y_noise_std
      random & inadmissible: pass Y_set and y_noise_std so Y-cluster variables
      receive scaled noise (lin + y_noise_std * eps).
      fixed: pass Y_set=None; all variables use uniform noise_scale.

  apply_nonlin_to_binary  (nonlinear only)
      random: True — the nonlinearity f is applied unconditionally before the
      logistic link for binary nodes (logits = f(lin) + eps + b).
      fixed:  False — binary nodes always use a linear pre-activation
      (logits = lin + eps + b), and ftype="sigmoid" in params is only for
      downstream sample_do_from_scm compatibility.

  noise_scale / bin_noise_scale
      random & inadmissible: default 1.0 (eps = randn(N)).
      fixed: caller passes explicit scale values (eps = randn(N) * scale).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set
import torch
import torch.nn.functional as F


def get_nonlinear(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return an elementwise nonlinearity callable by name.

    Supports all functions used across random_anm and fixed_anm:
      sin, cos, tanh, sigmoid, relu, square, cube,
      signed_quadratic, mild_cubic, silu
    Falls back to identity for unknown names.
    """
    name = (name or "linear").lower()
    if name == "sin":            return torch.sin
    if name == "cos":            return torch.cos
    if name == "tanh":           return torch.tanh
    if name == "sigmoid":        return torch.sigmoid
    if name == "relu":           return torch.relu
    if name == "square":         return lambda x: x ** 2
    if name == "cube":           return lambda x: x ** 3
    if name == "signed_quadratic": return lambda x: x + 0.05 * x * torch.abs(x)
    if name == "mild_cubic":     return lambda x: x + 0.01 * x.pow(3)
    if name == "silu":           return lambda x: F.silu(x)
    return lambda x: x  # identity / "linear"


def _simulate_linear(
    N: int,
    V: int,
    order: List[int],
    parents: List[List[int]],
    W: List,
    b: torch.Tensor,
    var_type: List[str],
    *,
    Y_set: Optional[Set[int]] = None,
    y_noise_std: float = 1.0,
    noise_scale: float = 1.0,
    bin_noise_scale: float = 1.0,
) -> torch.Tensor:
    """Linear ANM forward simulation.

    Parameters
    ----------
    W[v]:
        1-D weight tensor for variable v's parents (aligned with parents[v]),
        or None / empty tensor for root nodes.
    var_type[v]:
        "binary" or "cont".
    Y_set:
        Indices of Y-cluster variables.  When not None, those variables receive
        ``lin + y_noise_std * eps`` regardless of their type (random / inadmissible
        behavior).  When None, all variables use the same noise scaling (fixed behavior).
    noise_scale, bin_noise_scale:
        Noise standard deviations for continuous and binary variables respectively.
        Default 1.0 matches the random / inadmissible generators.
    """
    X = torch.zeros((N, V))
    for v in order:
        is_bin = (var_type[v] == "binary")
        eps = torch.randn(N) * (bin_noise_scale if is_bin else noise_scale)
        ps = parents[v]
        if len(ps) == 0:
            if is_bin:
                X[:, v] = torch.bernoulli(torch.sigmoid(b[v] + eps))
            else:
                X[:, v] = eps
        else:
            Wv = W[v]
            lin = X[:, ps].matmul(Wv) if (Wv is not None and Wv.numel() > 0) else 0.0
            if Y_set is not None and v in Y_set:
                X[:, v] = lin + y_noise_std * eps
            elif is_bin:
                X[:, v] = torch.bernoulli(torch.sigmoid(lin + b[v] + eps))
            else:
                X[:, v] = lin + eps
    return X


def _simulate_nonlinear(
    N: int,
    V: int,
    order: List[int],
    parents: List[List[int]],
    W: List,
    b: torch.Tensor,
    var_type: List[str],
    ftype_per_var: List[str],
    *,
    apply_nonlin_to_binary: bool = True,
    Y_set: Optional[Set[int]] = None,
    y_noise_std: float = 1.0,
    noise_scale: float = 1.0,
    bin_noise_scale: float = 1.0,
) -> torch.Tensor:
    """Nonlinear ANM forward simulation.

    Parameters
    ----------
    ftype_per_var[v]:
        Name of the nonlinearity for variable v (e.g. "tanh", "sin", "linear").
        "linear" / None means no nonlinearity (identity).
    apply_nonlin_to_binary:
        True  (random generator)  — nonlinearity is applied unconditionally
        before the logistic link, i.e. logits = f(lin) + eps + b[v].
        False (fixed generator)   — binary nodes always receive linear pre-
        activation regardless of ftype_per_var[v].
    """
    X = torch.zeros((N, V))
    _fn_cache: Dict[str, Callable] = {}
    for v in order:
        is_bin = (var_type[v] == "binary")
        eps = torch.randn(N) * (bin_noise_scale if is_bin else noise_scale)
        ps = parents[v]
        if len(ps) == 0:
            if is_bin:
                X[:, v] = torch.bernoulli(torch.sigmoid(b[v] + eps))
            else:
                X[:, v] = eps
        else:
            Wv = W[v]
            z = X[:, ps].matmul(Wv) if (Wv is not None and Wv.numel() > 0) else 0.0
            ftype = ftype_per_var[v]
            if ftype not in ("linear", None) and (apply_nonlin_to_binary or not is_bin):
                if ftype not in _fn_cache:
                    _fn_cache[ftype] = get_nonlinear(ftype)
                z = _fn_cache[ftype](z)
            if Y_set is not None and v in Y_set:
                X[:, v] = z + y_noise_std * eps
            elif is_bin:
                X[:, v] = torch.bernoulli(torch.sigmoid(z + b[v] + eps))
            else:
                X[:, v] = z + eps
    return X
