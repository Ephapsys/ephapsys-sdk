"""
Quick local test for compute_indispensability_loss() and run_ablation_probe().
Validates the functions work end-to-end with a tiny model.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from ephapsys.ecm import inject_ecm
from ephapsys.modulation import compute_indispensability_loss, run_ablation_probe


def _make_tiny_model():
    """Create a minimal causal LM-like model for testing."""
    class TinyLM(nn.Module):
        def __init__(self, vocab_size=100, hidden_dim=32, num_layers=2):
            super().__init__()
            self.config = type("Cfg", (), {"hidden_size": hidden_dim})()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
            self.head = nn.Linear(hidden_dim, vocab_size)
            self.loss_fn = nn.CrossEntropyLoss()

        def forward(self, input_ids, labels=None, output_hidden_states=False, **kwargs):
            h = self.embed(input_ids)
            hidden_states = [h] if output_hidden_states else None
            for layer in self.layers:
                h = torch.relu(layer(h))
                if output_hidden_states:
                    hidden_states.append(h)
            logits = self.head(h)
            loss = None
            if labels is not None:
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return type("Out", (), {
                "logits": logits,
                "loss": loss,
                "hidden_states": tuple(hidden_states) if hidden_states else None,
            })()

    return TinyLM()


def test_compute_indispensability_loss():
    print("Testing compute_indispensability_loss()...")
    model = _make_tiny_model()
    hidden_dim = model.config.hidden_size

    # Inject ECM
    inject_ecm(model, epsilon=0.5, lambda_init_mag=0.01, phi="identity",
               ecm_init="identity", variant="multiplicative", hidden_dim=hidden_dim)

    # Create dummy inputs
    input_ids = torch.randint(0, 100, (2, 10))
    labels = torch.randint(0, 100, (2, 10))
    inputs = {"input_ids": input_ids, "labels": labels}

    result = compute_indispensability_loss(model, inputs, alpha=10.0, beta=0.01)

    assert "task_loss" in result, "Missing task_loss"
    assert "indispensability_loss" in result, "Missing indispensability_loss"
    assert "stability_loss" in result, "Missing stability_loss"
    assert "total_loss" in result, "Missing total_loss"
    assert "separation" in result, "Missing separation"

    # All should be tensors
    for k, v in result.items():
        assert isinstance(v, torch.Tensor), f"{k} should be Tensor, got {type(v)}"

    # Indispensability loss should be >= 0
    assert result["indispensability_loss"].item() >= 0, "indispensability_loss should be >= 0"

    print(f"  task_loss={result['task_loss'].item():.4f}")
    print(f"  indispensability_loss={result['indispensability_loss'].item():.6f}")
    print(f"  stability_loss={result['stability_loss'].item():.6f}")
    print(f"  total_loss={result['total_loss'].item():.4f}")
    print(f"  separation={result['separation'].item():.6f}")
    print("  PASSED\n")


def test_run_ablation_probe():
    print("Testing run_ablation_probe()...")
    model = _make_tiny_model()
    hidden_dim = model.config.hidden_size

    # Inject ECM
    inject_ecm(model, epsilon=0.5, lambda_init_mag=0.01, phi="identity",
               ecm_init="identity", variant="multiplicative", hidden_dim=hidden_dim)

    # Create dummy inputs
    input_ids = torch.randint(0, 100, (2, 10))
    labels = torch.randint(0, 100, (2, 10))
    inputs = {"input_ids": input_ids, "labels": labels}

    result = run_ablation_probe(model, inputs)

    expected_keys = [
        "authorized_ppl", "unauthorized_ppl", "separation_ratio",
        "kl_divergence", "authorized_accuracy", "unauthorized_accuracy",
        "governance_strength",
    ]
    for k in expected_keys:
        assert k in result, f"Missing key: {k}"

    # Governance strength should be one of the valid levels
    valid_levels = {"none", "low", "moderate", "high", "critical"}
    assert result["governance_strength"] in valid_levels, \
        f"Invalid governance_strength: {result['governance_strength']}"

    # PPL values should be positive
    assert result["authorized_ppl"] > 0, "authorized_ppl should be > 0"
    assert result["unauthorized_ppl"] > 0, "unauthorized_ppl should be > 0"

    # Separation ratio should be >= 0
    assert result["separation_ratio"] >= 0, "separation_ratio should be >= 0"

    for k, v in result.items():
        print(f"  {k}={v}")
    print("  PASSED\n")


def test_hooks_restored_after_probe():
    """Verify ECM hooks are properly restored after ablation probe."""
    print("Testing hook restoration...")
    model = _make_tiny_model()
    hidden_dim = model.config.hidden_size

    inject_ecm(model, epsilon=0.5, lambda_init_mag=0.01, phi="identity",
               ecm_init="identity", variant="multiplicative", hidden_dim=hidden_dim)

    # Count hooks before
    hooks_before = sum(
        len(mod._forward_hooks) for _, mod in model.named_modules()
        if hasattr(mod, '_forward_hooks')
    )

    input_ids = torch.randint(0, 100, (2, 10))
    inputs = {"input_ids": input_ids, "labels": input_ids}

    # Run probe
    run_ablation_probe(model, inputs)

    # Count hooks after
    hooks_after = sum(
        len(mod._forward_hooks) for _, mod in model.named_modules()
        if hasattr(mod, '_forward_hooks')
    )

    assert hooks_before == hooks_after, \
        f"Hook count changed: {hooks_before} -> {hooks_after}"
    print(f"  Hooks before={hooks_before}, after={hooks_after}")
    print("  PASSED\n")


if __name__ == "__main__":
    test_compute_indispensability_loss()
    test_run_ablation_probe()
    test_hooks_restored_after_probe()
    print("All tests passed!")
