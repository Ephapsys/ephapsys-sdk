"""
Regression tests for inject_ecm()'s Λ initialization (platform#168).

ecm_init="random" and "topk_from_WT" used np.random directly, which is not
covered by torch.manual_seed() - a caller seeding torch for reproducibility
(e.g. per-trial AOC search seeding) still got a non-reproducible Λ. Fixed by
switching both paths to torch.randn(), which does honor torch.manual_seed().
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from ephapsys.ecm import inject_ecm


def _lambda_after_injection(seed, ecm_init, hidden_dim=16):
    torch.manual_seed(seed)
    model = nn.Linear(hidden_dim, hidden_dim)
    inject_ecm(model, epsilon=0.5, lambda_init_mag=0.01, phi="identity",
               ecm_init=ecm_init, hidden_dim=hidden_dim)
    return dict(model.named_parameters())["lambda_ecm"].detach().clone()


def test_random_init_reproducible_under_torch_seed():
    print("Testing ecm_init='random' is reproducible under torch.manual_seed()...")
    l1 = _lambda_after_injection(seed=123, ecm_init="random")
    l2 = _lambda_after_injection(seed=123, ecm_init="random")
    l3 = _lambda_after_injection(seed=456, ecm_init="random")
    assert torch.equal(l1, l2), "same torch seed should produce identical Λ for ecm_init='random'"
    assert not torch.equal(l1, l3), "different torch seeds should produce different Λ"
    print("  PASSED\n")


def test_topk_from_wt_init_reproducible_under_torch_seed():
    print("Testing ecm_init='topk_from_WT' is reproducible under torch.manual_seed()...")
    l1 = _lambda_after_injection(seed=123, ecm_init="topk_from_WT")
    l2 = _lambda_after_injection(seed=123, ecm_init="topk_from_WT")
    l3 = _lambda_after_injection(seed=456, ecm_init="topk_from_WT")
    assert torch.equal(l1, l2), "same torch seed should produce identical Λ for ecm_init='topk_from_WT'"
    assert not torch.equal(l1, l3), "different torch seeds should produce different Λ"
    print("  PASSED\n")


def test_random_init_independent_of_numpy_global_rng():
    """The fix must not depend on NumPy's global RNG state at all - only
    torch's. Perturbing np.random's global state should have zero effect."""
    print("Testing ecm_init='random' is unaffected by NumPy's global RNG state...")
    import numpy as np
    np.random.seed(1)
    l_numpy_seed_1 = _lambda_after_injection(seed=123, ecm_init="random")
    np.random.seed(2)
    l_numpy_seed_2 = _lambda_after_injection(seed=123, ecm_init="random")
    assert torch.equal(l_numpy_seed_1, l_numpy_seed_2), \
        "Λ init changed with NumPy's global RNG state despite the same torch seed - still coupled to np.random"
    print("  PASSED\n")


if __name__ == "__main__":
    test_random_init_reproducible_under_torch_seed()
    test_topk_from_wt_init_reproducible_under_torch_seed()
    test_random_init_independent_of_numpy_global_rng()
    print("All tests passed!")
