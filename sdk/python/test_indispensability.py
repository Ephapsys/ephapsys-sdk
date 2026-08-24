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


def _make_tiny_model(dropout=0.0, raise_on_call=None):
    """Create a minimal causal LM-like model for testing.

    ``raise_on_call``: if set, the N-th forward() call (1-indexed) raises
    RuntimeError instead of returning, to test exception-safety.
    """
    class TinyLM(nn.Module):
        def __init__(self, vocab_size=100, hidden_dim=32, num_layers=2):
            super().__init__()
            self.config = type("Cfg", (), {"hidden_size": hidden_dim})()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
            self.head = nn.Linear(hidden_dim, vocab_size)
            self.drop = nn.Dropout(dropout)
            self.loss_fn = nn.CrossEntropyLoss()
            self._call_count = 0

        def forward(self, input_ids, labels=None, output_hidden_states=False, **kwargs):
            self._call_count += 1
            if raise_on_call is not None and self._call_count == raise_on_call:
                raise RuntimeError("synthetic forward failure for exception-safety test")
            h = self.embed(input_ids)
            hidden_states = [h] if output_hidden_states else None
            for layer in self.layers:
                h = torch.relu(layer(h))
                h = self.drop(h)
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

    # All should be tensors, except `gap`, which is only populated in
    # objective="hinge" mode - the default "dispensability" objective legitimately
    # returns gap=None (kept as a dict key for consumers that expect it to exist).
    for k, v in result.items():
        if k == "gap":
            assert v is None, f"gap should be None for the default objective, got {type(v)}"
            continue
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


def test_dropout_disabled_and_mode_restored():
    """SDK-03: paired forwards must run with dropout off, and the caller's
    training mode must be restored afterward regardless of what it was."""
    print("Testing dropout disabled + training-mode restoration...")
    model = _make_tiny_model(dropout=0.9)  # aggressive, to make leaked noise obvious
    hidden_dim = model.config.hidden_size
    inject_ecm(model, epsilon=0.5, lambda_init_mag=0.01, phi="identity",
               ecm_init="identity", variant="multiplicative", hidden_dim=hidden_dim)

    input_ids = torch.randint(0, 100, (2, 10))
    labels = torch.randint(0, 100, (2, 10))
    inputs = {"input_ids": input_ids, "labels": labels}

    for was_training in (True, False):
        model.train(was_training)
        compute_indispensability_loss(model, inputs, alpha=10.0, beta=0.01)
        assert model.training == was_training, \
            f"training mode not restored: expected {was_training}, got {model.training}"

    # With dropout at 0.9, if it were still active during the paired forwards,
    # separation would vary run-to-run even with everything else fixed. With
    # eval() enforced internally, two calls must agree exactly.
    model.train()
    r1 = compute_indispensability_loss(model, inputs, alpha=10.0, beta=0.01)
    r2 = compute_indispensability_loss(model, inputs, alpha=10.0, beta=0.01)
    assert torch.allclose(r1["separation"], r2["separation"]), \
        "separation differs across calls - dropout is leaking into the paired forwards"
    print("  PASSED\n")


def test_unrelated_hooks_preserved_with_order():
    """SDK-04: non-ECM hooks on a module that also carries an ECM hook must
    survive the no-Lambda branch AND actually fire during it, in their
    original relative order. Checking hook presence only after completion is
    not enough - the old "clear everything, then restore" behavior would also
    leave the right keys present afterward, while still failing to fire the
    foreign hooks during the no-Lambda forward."""
    print("Testing unrelated-hook preservation, ordering, and firing...")
    model = _make_tiny_model()
    hidden_dim = model.config.hidden_size

    calls_before, calls_after = [], []

    def foreign_before(mod, inp, out):
        calls_before.append(1)
    handle_before = model.layers[0].register_forward_hook(foreign_before)

    inject_ecm(model, epsilon=0.5, lambda_init_mag=0.01, phi="identity",
               ecm_init="identity", variant="multiplicative", hidden_dim=hidden_dim)

    def foreign_after(mod, inp, out):
        calls_after.append(1)
    handle_after = model.layers[0].register_forward_hook(foreign_after)

    hooks_dict = model.layers[0]._forward_hooks
    keys_before = list(hooks_dict.keys())
    assert handle_before.id in keys_before and handle_after.id in keys_before, \
        "test setup sanity check failed: expected hooks not present"

    input_ids = torch.randint(0, 100, (2, 10))
    labels = torch.randint(0, 100, (2, 10))
    inputs = {"input_ids": input_ids, "labels": labels}
    compute_indispensability_loss(model, inputs, alpha=10.0, beta=0.01)

    keys_after = list(hooks_dict.keys())
    assert keys_after == keys_before, \
        f"hook order changed: {keys_before} -> {keys_after}"

    # Two forwards happen (no-Lambda, with-Lambda); a foreign hook that only
    # fires once means it was skipped during one of them - the exact bug the
    # old "clear every _forward_hooks dict" implementation had.
    assert len(calls_before) == 2, \
        f"foreign_before should fire during BOTH forwards, fired {len(calls_before)}x"
    assert len(calls_after) == 2, \
        f"foreign_after should fire during BOTH forwards, fired {len(calls_after)}x"
    print("  PASSED\n")


def test_zero_ecm_hooks_raises():
    """SDK-04: with zero ECM hooks attached, the with/without forwards would be
    byte-identical (gap silently reads 0.0) - must raise instead."""
    print("Testing zero-ECM-hooks guard...")
    model = _make_tiny_model()  # no inject_ecm() call - no ECM hooks anywhere

    input_ids = torch.randint(0, 100, (2, 10))
    labels = torch.randint(0, 100, (2, 10))
    inputs = {"input_ids": input_ids, "labels": labels}

    try:
        compute_indispensability_loss(model, inputs, alpha=10.0, beta=0.01)
        raise AssertionError("expected RuntimeError for zero ECM hooks, none raised")
    except RuntimeError as e:
        assert "No Ephapsys ECM forward hooks found" in str(e)
    print("  PASSED\n")


def test_restoration_after_exception():
    """Hooks and training mode must be restored even if the WITHOUT-ECM
    forward (inside the hook-disable context) raises."""
    print("Testing restoration after an exception in the no-Lambda forward...")
    model = _make_tiny_model(raise_on_call=1)  # first forward() call raises
    hidden_dim = model.config.hidden_size
    inject_ecm(model, epsilon=0.5, lambda_init_mag=0.01, phi="identity",
               ecm_init="identity", variant="multiplicative", hidden_dim=hidden_dim)

    hooks_before = {
        name: list(mod._forward_hooks.keys()) for name, mod in model.named_modules()
        if hasattr(mod, "_forward_hooks") and mod._forward_hooks
    }

    input_ids = torch.randint(0, 100, (2, 10))
    labels = torch.randint(0, 100, (2, 10))
    inputs = {"input_ids": input_ids, "labels": labels}

    model.train()
    raised = False
    try:
        compute_indispensability_loss(model, inputs, alpha=10.0, beta=0.01)
    except RuntimeError:
        raised = True
    assert raised, "expected the synthetic forward failure to propagate"

    assert model.training is True, "training mode not restored after exception"
    hooks_after = {
        name: list(mod._forward_hooks.keys()) for name, mod in model.named_modules()
        if hasattr(mod, "_forward_hooks") and mod._forward_hooks
    }
    assert hooks_after == hooks_before, \
        f"hooks not restored after exception: {hooks_before} -> {hooks_after}"
    print("  PASSED\n")


if __name__ == "__main__":
    test_compute_indispensability_loss()
    test_run_ablation_probe()
    test_hooks_restored_after_probe()
    test_dropout_disabled_and_mode_restored()
    test_unrelated_hooks_preserved_with_order()
    test_zero_ecm_hooks_raises()
    test_restoration_after_exception()
    print("All tests passed!")
