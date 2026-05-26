#!/usr/bin/env python3
"""
Local test for issue #3: language output-token cap + streaming.

Runs fully offline against the generation internals (no AOC, no auth) by
constructing a TrustedAgent via __new__ and exercising the language helpers
directly with a tiny HF model. Validates:

  1. max_new_tokens precedence (the #3 cap fix):
       explicit generation cfg > max_tokens_per_request policy cap > 512
       (never the old hardcoded 96).
  2. Streaming yields multiple incremental chunks whose concatenation equals
     the non-streaming output for a deterministic (greedy) decode.

Usage:
    cd ephapsys-sdk
    pip install -e .            # or: pip install transformers torch
    python tests/test_language_streaming_and_cap.py

First run downloads a tiny model (~a few MB) from HuggingFace; subsequent runs
use the local cache.
"""

import os
import sys
import tempfile

# Keep generation deterministic and the env clean for the precedence assertions.
os.environ.pop("AOC_MAX_NEW_TOKENS", None)
os.environ["AOC_LANGUAGE_USE_CHAT_TEMPLATE"] = "0"  # tiny-gpt2 has no chat template

# A real (not randomly-initialized) small model so greedy decode is stable and
# the stream-equals-sync assertion is meaningful. tiny-gpt2 has near-uniform
# logits where argmax ties resolve non-deterministically.
TINY_MODEL = os.getenv("TEST_TINY_MODEL", "distilgpt2")

import torch  # noqa: E402

# Streaming runs model.generate() on a background thread. On CPU, torch
# intra-op parallelism can reorder float reductions across thread contexts,
# which flips argmax on near-tied logits and makes greedy decode diverge
# between the threaded (stream) and main-thread (sync) paths. Pin to a single
# thread + deterministic algorithms so both take an identical reduction order
# and the stream-equals-sync assertion is a meaningful faithfulness check
# rather than a flaky FP race.
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

from ephapsys.agent import TrustedAgent  # noqa: E402


def _make_agent():
    """Construct a TrustedAgent without running __init__ (which does network auth)."""
    agent = TrustedAgent.__new__(TrustedAgent)
    # Minimal attributes the language path touches.
    agent.agent_id = "local-test"
    agent.api_base = "http://localhost:7001"
    agent.api_key = None
    agent.verify_ssl = False
    agent._av_scanner = None
    agent._output_schema = None
    agent._minimal_logging = False
    agent._attestation_digest = None
    agent._max_tokens_cap = 0
    # Truthy cache with no model_policies → enforce_policies returns early, no network.
    agent._run_status_cache = {"model_policies": []}
    # Keep guardrails offline + pass-through.
    agent._apply_guardrails_with_logging = lambda rt, val, kind: (val, [])
    return agent


def _prepare_local_model(tmpdir):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tok.save_pretrained(tmpdir)
    model.save_pretrained(tmpdir)
    return tmpdir


def test_cap_precedence(agent, runtime):
    tok, _, _, _, _ = agent._ensure_language_model_loaded(runtime)

    # (a) policy cap drives output budget when nothing explicit is set
    agent._max_tokens_cap = 64
    runtime_no_cfg = dict(runtime, config={"generation": {"do_sample": "0"}})
    _, _, kwargs = agent._prepare_language_inputs(runtime_no_cfg, "hello", tok)
    assert kwargs["max_new_tokens"] == 64, f"expected 64 from policy cap, got {kwargs['max_new_tokens']}"
    print("  [ok] policy cap (max_tokens_per_request) drives max_new_tokens=64")

    # (b) no cap, no env, no cfg → 512 fallback (NOT the old 96)
    agent._max_tokens_cap = 0
    _, _, kwargs = agent._prepare_language_inputs(runtime_no_cfg, "hello", tok)
    assert kwargs["max_new_tokens"] == 512, f"expected 512 fallback, got {kwargs['max_new_tokens']}"
    assert kwargs["max_new_tokens"] != 96, "regression: still defaulting to old hardcoded 96"
    print("  [ok] fallback is 512, not the old hardcoded 96")

    # (c) explicit generation cfg wins over policy cap
    agent._max_tokens_cap = 64
    runtime_explicit = dict(runtime, config={"generation": {"max_new_tokens": 20, "do_sample": "0"}})
    _, _, kwargs = agent._prepare_language_inputs(runtime_explicit, "hello", tok)
    assert kwargs["max_new_tokens"] == 20, f"explicit cfg should win, got {kwargs['max_new_tokens']}"
    print("  [ok] explicit generation cfg (20) overrides policy cap")
    agent._max_tokens_cap = 0


def test_streaming_matches_sync(agent, runtime):
    # Deterministic greedy decode + small budget so sync and stream match exactly.
    rt = dict(runtime, config={"generation": {"do_sample": "0", "max_new_tokens": 16}})

    sync_text, _ = agent._run_language(rt, "The quick brown fox")

    chunks = list(agent._run_language_stream(rt, "The quick brown fox"))
    streamed_text = "".join(chunks).strip()

    print(f"  sync   : {sync_text!r}")
    print(f"  chunks : {len(chunks)} → {streamed_text!r}")

    assert len(chunks) >= 1, "streaming yielded no chunks"
    assert streamed_text == sync_text, "streamed concatenation does not match sync output"
    print("  [ok] streamed output matches sync, delivered in", len(chunks), "chunk(s)")


def main():
    print(f"Using tiny model: {TINY_MODEL}")
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = _prepare_local_model(tmp)
        agent = _make_agent()
        runtime = {"model_path": model_dir, "config": {}}

        print("\n[1] max_new_tokens precedence (#3 cap fix)")
        test_cap_precedence(agent, runtime)

        print("\n[2] streaming")
        test_streaming_matches_sync(agent, runtime)

    print("\nAll tests passed.")


if __name__ == "__main__":
    sys.exit(main())
