# SPDX-License-Identifier: Apache-2.0
import random
from typing import List, Optional
import torch, torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger("ephapsys.ecm")

# ================= Ephaptic Coupling Matrix (ECM) ==================
# ECM, represented as Λ in our general activation function, is fully learnable, device-aware,
# and dynamically resizable across heterogeneous layer widths (such as Flan-T5).
# This architecture-neutral version can work for any artificial neural network (ANN), provided
# the model’s forward pass exposes intermediate activations as tensors (which nearly all PyTorch models do).
# It’s fully model-agnostic i.e., the injection works with MLPs, CNNs, RNNs, Transformers, etc.


# ============================================================
#  Matrix Initialization Helpers
# ============================================================
def init_ecm(h: int, strategy: str = "transpose", lambda0: float = 0.05,
             W_T: Optional[list] = None, k: Optional[int] = None) -> List[List[float]]:
    """
    Initialize the ephaptic coupling matrix (ECM) Λ for a given hidden size.
    """
    ECM = [[0.0 for _ in range(h)] for __ in range(h)]

    # --- Identity initialization: diagonal scaled by λ₀
    if strategy == "identity":
        for i in range(h):
            ECM[i][i] = lambda0
        return ECM

    # --- Uniform random initialization in [-λ₀, λ₀]
    if strategy == "random":
        for i in range(h):
            for j in range(h):
                ECM[i][j] = lambda0 * (random.random() * 2 - 1)
        return ECM

    # --- Transpose / top-k / behavioral mask variants
    if strategy in ("transpose", "topk_from_WT", "behavioral_mask") and W_T:
        m_out = len(W_T[0]) if W_T else h
        for i in range(h):
            for j in range(min(h, m_out)):
                ECM[i][j] = lambda0 * float(W_T[i][j])

        # Keep only top-k strongest couplings if specified
        if strategy == "topk_from_WT" and k:
            for i in range(h):
                row = ECM[i]
                idx = sorted(range(len(row)), key=lambda c: abs(row[c]), reverse=True)[:k]
                keep = set(idx)
                for c in range(len(row)):
                    if c not in keep:
                        row[c] = 0.0

        # Behavioral mask: remove negative interactions
        if strategy == "behavioral_mask":
            for i in range(h):
                for j in range(h):
                    if ECM[i][j] < 0:
                        ECM[i][j] = 0.0
        return ECM

    # Default fallback: identity matrix
    return init_ecm(h, "identity", lambda0)


# ============================================================
# Nonlinearities (legacy scalar functions)
# ============================================================
def _phi(x: float, kind: str) -> float:
    """Compute scalar nonlinearity for legacy (non-torch) use."""
    if kind == "relu":
        return x if x > 0 else 0.0
    if kind == "tanh":
        import math
        return math.tanh(x)
    if kind == "silu":
        import math
        return x / (1.0 + math.exp(-x))
    if kind == "gelu":
        import math
        return 0.5 * x * (1.0 + math.tanh((2.0 / math.pi) ** 0.5 * (x + 0.044715 * x ** 3)))
    return x  # identity fallback


# ============================================================
# Vector Utilities
# ============================================================
def _dot(a: list, b: list) -> float:
    """Compute dot product between two 1D vectors."""
    return sum((ai * bi for ai, bi in zip(a, b)), 0.0)

def _matvec(M: list, v: list) -> list:
    """Matrix-vector product using lists (legacy fallback)."""
    return [_dot(row, v) for row in M]


# ============================================================
# 🧠 Legacy Python (non-Torch) ECM application
# ============================================================
def apply_additive(S: list, x_layer: list, ECM: list, epsilon: float, phi: str = "silu") -> list:
    """Additive ephaptic update:  S' = S + ε * ECM * φ(x)"""
    x_mod = [_phi(xx, phi) for xx in x_layer]
    E = _matvec(ECM, x_mod)
    return [s + epsilon * e for s, e in zip(S, E)]

def apply_multiplicative(S: list, x_layer: list, ECM: list, epsilon: float,
                         gate: str = "sigmoid", phi: str = "silu") -> list:
    """Multiplicative ephaptic update:  S' = S * g(ε * ECM * φ(x))"""
    x_mod = [_phi(xx, phi) for xx in x_layer]
    import math
    E = _matvec(ECM, x_mod)
    def g(z: float) -> float:
        if gate == "sigmoid":
            return 1.0 / (1.0 + math.exp(-z))
        if gate == "tanh1":
            return 1.0 + math.tanh(z)
        if gate == "softplus":
            return math.log1p(math.exp(z))
        return 1.0
    return [s * g(epsilon * e) for s, e in zip(S, E)]


# ============================================================
# ⚡ Torch Integration — Dynamic ECM Injection
# ============================================================
def inject_ecm(module: nn.Module,
               epsilon: float = 0.05,
               lambda_init_mag: float = 0.01,
               phi: str = "gelu",
               ecm_init: str = "transpose",
               mu: int = 200,
               kappa: int = 32,
               variant: str = "additive",
               hidden_dim: Optional[int] = None):
    """
    Dynamically inject ephaptic coupling (Λ) into a PyTorch module’s forward pass.
    The ECM introduces column-space modulation (orthogonal to normal synaptic row ops).
    """

    # --- Infer hidden dimension from module or config ---
    if hidden_dim is None:
        hidden_dim = getattr(module, "hidden_size", None) or getattr(module, "d_model", None)
    if hidden_dim is None:
        config = getattr(module, "config", None)
        if config is not None:
            hidden_dim = getattr(config, "hidden_size", None)
    if hidden_dim is None:
        raise ValueError(
            "inject_ecm: could not infer hidden size. "
            "Pass hidden_dim explicitly (e.g., inject_ecm(module, hidden_dim=model.config.hidden_size))."
        )

    # ============================================================
    # Initialize Λ as a learnable nn.Parameter (trainable & device-aware)
    # ============================================================
    if ecm_init == "transpose":
        Lambda_np = np.eye(hidden_dim)[::-1].copy() * lambda_init_mag
    elif ecm_init == "random":
        Lambda_np = np.random.randn(hidden_dim, hidden_dim) * lambda_init_mag
    elif ecm_init == "identity":
        Lambda_np = np.eye(hidden_dim) * lambda_init_mag
    elif ecm_init == "topk_from_WT":
        W_T = np.random.randn(hidden_dim, hidden_dim)
        Lambda_np = lambda_init_mag * W_T
        for i in range(hidden_dim):
            row = Lambda_np[i]
            idx = np.argsort(np.abs(row))[-kappa:]
            mask = np.zeros_like(row)
            mask[idx] = 1.0
            Lambda_np[i] = row * mask
    else:
        raise ValueError(f"Unknown ecm_init: {ecm_init}")

    # Detect device of module and create Λ directly there
    device = next(module.parameters(), torch.tensor([])).device
    Lambda = nn.Parameter(torch.tensor(Lambda_np, dtype=torch.float32, device=device), requires_grad=True)
    module.register_parameter("lambda_ecm", Lambda)

    # --- Select nonlinearity φ(x) ---
    if phi == "gelu":
        act = torch.nn.functional.gelu
    elif phi == "relu":
        act = torch.relu
    elif phi == "silu":
        act = torch.nn.functional.silu
    elif phi == "tanh":
        act = torch.tanh
    elif phi == "identity":
        act = lambda x: x
    else:
        raise ValueError(f"Unknown phi: {phi}")

    # ============================================================
    #  Forward Hook: Applies ephaptic influence during inference/training
    # ============================================================
    def _hook(_, __, output):
        """
        Hook intercepts the forward pass, modifies hidden activations:
          x' = x + ε * φ(xΛ)   (additive)
          x' = x * (1 + ε * φ(xΛ))   (multiplicative)
        """

        # Extract tensor output (handles HF Transformers' dicts)
        if isinstance(output, dict) and "last_hidden_state" in output:
            x = output["last_hidden_state"]
        elif isinstance(output, (tuple, list)) and len(output) > 0:
            x = output[0]
        elif isinstance(output, torch.Tensor):
            x = output
        else:
            return output  # skip unknown types

        # ✅ Use registered Parameter (ensures gradient flow through Λ)
        Lambda_param = getattr(module, "lambda_ecm", None)
        if Lambda_param is None:
            return output  # skip if not found

        # ✅ Ensure Λ and x are on the same device
        if Lambda_param.device != x.device:
            Lambda_param = Lambda_param.to(x.device)

        # Compute ephaptic modulation (no dynamic resize anymore)
        coupling = torch.matmul(x, Lambda_param)
        ephaptic = epsilon * act(coupling)

        # Apply variant
        if variant == "additive":
            x_mod = x + ephaptic
        elif variant == "multiplicative":
            x_mod = x * (1 + ephaptic)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Return modified tensor preserving structure
        if isinstance(output, dict):
            output["last_hidden_state"] = x_mod
            return output
        elif isinstance(output, (tuple, list)):
            return (x_mod,) + tuple(output[1:])
        else:
            return x_mod

    # ============================================================
    #  Attach hook only to modules matching the main hidden_dim
    #  (prevents constant resizing like 384↔512↔1024 in T5)
    # ============================================================
    for name, sub in module.named_modules():
        if isinstance(sub, nn.Linear):
            out_features = getattr(sub, "out_features", None)
            if out_features == hidden_dim:
                sub.register_forward_hook(_hook)
                logger.debug("[ECM] Attached hook to %s (dim=%s)", name, out_features)

    # ✅ Return both module and Λ (trainable) for optimizer inclusion and saving
    return module, Lambda
