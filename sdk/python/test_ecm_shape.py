"""ECM application checks that need only torch (no PKCS#11 token)."""
import types

import pytest



def test_ecm_of_the_wrong_size_names_both_shapes(caplog):
    """#189: a 512x512 identity ECM on a 1024-wide model must say so, not surface a reshape error."""
    import io
    import torch
    import ephapsys.agent as agent_mod
    target = types.SimpleNamespace(_ephaptic_hook_installed=True, lambda_ecm=torch.nn.Parameter(torch.zeros(4, 4)))
    stub = types.SimpleNamespace(_resolve_ecm_target=lambda model, t: target, _device=lambda: "cpu")
    buf = io.BytesIO()
    torch.save(torch.eye(2), buf)
    runtime = {"kind": "language", "_ecm_required": True, "ecm_bytes": buf.getvalue()}
    with pytest.raises(agent_mod.SecureInferenceError, match=r"\(2, 2\) does not fit this model \(expects \(4, 4\)\).*re-publish"):
        agent_mod.TrustedAgent._apply_ecm_if_available(stub, object(), runtime)
    with caplog.at_level("WARNING", logger="ephapsys.sdk"):
        agent_mod.TrustedAgent._apply_ecm_if_available(stub, object(), {"kind": "language", "ecm_bytes": buf.getvalue()})
    assert "re-publish" in caplog.text and torch.count_nonzero(target.lambda_ecm) == 0
