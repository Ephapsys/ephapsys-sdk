#!/usr/bin/env python3
"""
Decoding-constraint precedence for the language path.

HuggingFace applies RepetitionPenaltyLogitsProcessor and NoRepeatNGramLogitsProcessor
to the PROMPT as well as the output for decoder-only models. With the old hardcoded
defaults (repetition_penalty=1.1, no_repeat_ngram_size=3) a chat model was penalized
for copying names out of its own system prompt ("Bloxtel" -> "Bloxtell", "Graham" ->
"Harrison") and forbidden from repeating any prompt trigram ("5G" -> "4/5G").

Contract under test (shared by sync + streaming via _prepare_language_inputs):
  1. Nothing set              -> neutral: repetition_penalty=1.0, no_repeat_ngram_size=0
  2. Manifest config only     -> config value used
  3. Env set                  -> env OVERRIDES manifest config (operators cannot edit the
                                 manifest; env is the deployment-level switch)
  4. Both keys are ALWAYS present in generate kwargs, even when neutral, so a
     checkpoint's own generation_config.json cannot re-introduce them.
  5. Invalid values fail loudly (non-finite / non-positive penalty, fractional or
     negative ngram size) instead of silently changing decoding.

Runs offline like test_language_streaming_and_cap.py (TrustedAgent via __new__, tiny HF
model). Usage:
    TEST_TINY_MODEL=gpt2 python tests/test_language_decoding_constraints.py
"""

import os
import sys
import tempfile

for _k in ("AOC_LANGUAGE_REPETITION_PENALTY", "AOC_LANGUAGE_NO_REPEAT_NGRAM_SIZE"):
    os.environ.pop(_k, None)

sys.path.insert(0, os.path.dirname(__file__))
from test_language_streaming_and_cap import _make_agent, _prepare_local_model, TINY_MODEL  # noqa: E402

ENV_RP = "AOC_LANGUAGE_REPETITION_PENALTY"
ENV_NG = "AOC_LANGUAGE_NO_REPEAT_NGRAM_SIZE"


def _kwargs(agent, tok, runtime, generation=None, env=None):
    saved = {k: os.environ.get(k) for k in (ENV_RP, ENV_NG)}
    try:
        for k in (ENV_RP, ENV_NG):
            os.environ.pop(k, None)
        for k, v in (env or {}).items():
            os.environ[k] = v
        rt = dict(runtime, config={"generation": {"do_sample": "0", **(generation or {})}})
        _, _, kw = agent._prepare_language_inputs(rt, "hello", tok)
        return kw
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _expect_raises(fn, what):
    try:
        fn()
    except (ValueError, TypeError):
        print(f"  [ok] rejected {what}")
        return
    raise AssertionError(f"expected {what} to be rejected")


def test_neutral_defaults(agent, tok, runtime):
    kw = _kwargs(agent, tok, runtime)
    assert "repetition_penalty" in kw and "no_repeat_ngram_size" in kw, "both keys must always be present"
    assert kw["repetition_penalty"] == 1.0, f"default penalty must be neutral, got {kw['repetition_penalty']}"
    assert kw["no_repeat_ngram_size"] == 0, f"default ngram size must be 0, got {kw['no_repeat_ngram_size']}"
    print("  [ok] neutral defaults (1.0 / 0), both keys present")


def test_manifest_config_used(agent, tok, runtime):
    kw = _kwargs(agent, tok, runtime, generation={"repetition_penalty": 1.05, "no_repeat_ngram_size": 4})
    assert kw["repetition_penalty"] == 1.05 and kw["no_repeat_ngram_size"] == 4
    print("  [ok] manifest generation config honoured when env is absent")


def test_env_overrides_manifest(agent, tok, runtime):
    # The exact manifest values that caused the identity bug.
    kw = _kwargs(agent, tok, runtime,
                 generation={"repetition_penalty": 1.1, "no_repeat_ngram_size": 3},
                 env={ENV_RP: "1.0", ENV_NG: "0"})
    assert kw["repetition_penalty"] == 1.0, f"env must override manifest, got {kw['repetition_penalty']}"
    assert kw["no_repeat_ngram_size"] == 0, f"env must override manifest, got {kw['no_repeat_ngram_size']}"
    print("  [ok] env overrides a manifest carrying 1.1 / 3")


def test_env_can_enable(agent, tok, runtime):
    kw = _kwargs(agent, tok, runtime, env={ENV_RP: "1.2", ENV_NG: "6"})
    assert kw["repetition_penalty"] == 1.2 and kw["no_repeat_ngram_size"] == 6
    print("  [ok] env can turn the constraints on")


def test_validation(agent, tok, runtime):
    _expect_raises(lambda: _kwargs(agent, tok, runtime, env={ENV_RP: "0"}), "penalty 0")
    _expect_raises(lambda: _kwargs(agent, tok, runtime, env={ENV_RP: "-1.1"}), "negative penalty")
    _expect_raises(lambda: _kwargs(agent, tok, runtime, env={ENV_RP: "inf"}), "infinite penalty")
    _expect_raises(lambda: _kwargs(agent, tok, runtime, env={ENV_RP: "nan"}), "NaN penalty")
    _expect_raises(lambda: _kwargs(agent, tok, runtime, env={ENV_NG: "2.5"}), "fractional ngram size")
    _expect_raises(lambda: _kwargs(agent, tok, runtime, env={ENV_NG: "-1"}), "negative ngram size")
    _expect_raises(lambda: _kwargs(agent, tok, runtime, generation={"no_repeat_ngram_size": 2.5}), "fractional ngram size from manifest")


def main():
    agent = _make_agent()
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = _prepare_local_model(tmp)
        runtime = {"model_path": model_dir, "config": {}}
        tok, _, _, _, _ = agent._ensure_language_model_loaded(runtime)
        for t in (test_neutral_defaults, test_manifest_config_used, test_env_overrides_manifest,
                  test_env_can_enable, test_validation):
            print(f"\n== {t.__name__}")
            t(agent, tok, runtime)
    print("\nALL PASSED")



# ── Added after review ─────────────────────────────────────────────────────────

def test_validators_direct():
    from ephapsys.agent import _parse_repetition_penalty as rp, _parse_no_repeat_ngram_size as ng
    assert rp(1.0) == 1.0 and rp("1.05") == 1.05 and rp(2) == 2.0
    for bad in (True, False, 0, -1, "inf", "nan", "abc", None):
        _expect_raises(lambda: rp(bad), f"repetition_penalty {bad!r}")
    assert ng(0) == 0 and ng("3") == 3 and ng(4.0) == 4 and ng(" 5 ") == 5
    for bad in (True, False, -1, 2.5, "2.5", "abc", None, "-3"):
        _expect_raises(lambda: ng(bad), f"no_repeat_ngram_size {bad!r}")


def test_model_config_fallback_and_generation_precedence(agent, tok, runtime):
    # manifest config (model level) used when generation block lacks the key
    rt = dict(runtime, config={"repetition_penalty": 1.07, "no_repeat_ngram_size": 5,
                               "generation": {"do_sample": "0"}})
    _, _, kw = agent._prepare_language_inputs(rt, "hello", tok)
    assert kw["repetition_penalty"] == 1.07 and kw["no_repeat_ngram_size"] == 5
    print("  [ok] manifest model-level config used as fallback")
    # generation block beats model level
    rt = dict(runtime, config={"repetition_penalty": 1.07, "no_repeat_ngram_size": 5,
                               "generation": {"do_sample": "0", "repetition_penalty": 1.02, "no_repeat_ngram_size": 2}})
    _, _, kw = agent._prepare_language_inputs(rt, "hello", tok)
    assert kw["repetition_penalty"] == 1.02 and kw["no_repeat_ngram_size"] == 2
    print("  [ok] config.generation beats model-level config")


def test_source_logged_once_per_runtime(agent, tok, runtime, caplog_records):
    rt = dict(runtime, config={"generation": {"do_sample": "0", "repetition_penalty": 1.1}})
    rt.pop("_decoding_constraints_logged", None)
    caplog_records.clear()
    os.environ[ENV_NG] = "0"
    try:
        agent._prepare_language_inputs(rt, "hello", tok)
        agent._prepare_language_inputs(rt, "hello again", tok)
    finally:
        os.environ.pop(ENV_NG, None)
    msgs = [r.getMessage() for r in caplog_records if "decoding constraints" in r.getMessage()]
    assert len(msgs) == 1, f"expected exactly one log line per runtime, got {len(msgs)}"
    assert "repetition_penalty=1.1 (manifest:config.generation)" in msgs[0], msgs[0]
    assert "no_repeat_ngram_size=0 (env:AOC_LANGUAGE_NO_REPEAT_NGRAM_SIZE)" in msgs[0], msgs[0]
    print("  [ok] effective settings logged once with sources")


def test_neutral_values_reach_generate_sync_and_stream(agent):
    """Model saved WITH generation_config repetition_penalty=1.1 / no_repeat_ngram_size=3.
    The SDK must still hand neutral values to generate() on both paths."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    with tempfile.TemporaryDirectory() as tmp:
        tok = AutoTokenizer.from_pretrained(TINY_MODEL)
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        model.generation_config.repetition_penalty = 1.1
        model.generation_config.no_repeat_ngram_size = 3
        tok.save_pretrained(tmp); model.save_pretrained(tmp)
        import json
        saved = json.load(open(os.path.join(tmp, "generation_config.json")))
        assert saved["repetition_penalty"] == 1.1 and saved["no_repeat_ngram_size"] == 3

        rt = {"model_path": tmp, "config": {"generation": {"do_sample": "0", "max_new_tokens": 4}}}
        _, _, loaded, _, _ = agent._ensure_language_model_loaded(rt)
        seen = []
        real_generate = loaded.generate
        def spy(*a, **kw):
            seen.append(kw); return real_generate(*a, **kw)
        loaded.generate = spy

        agent._run_language(rt, "hello")
        list(agent._run_language_stream(rt, "hello"))
        assert len(seen) == 2, f"expected sync + stream generate calls, got {len(seen)}"
        for kw in seen:
            assert kw["repetition_penalty"] == 1.0, kw
            assert kw["no_repeat_ngram_size"] == 0, kw
        print("  [ok] neutral values reach generate() on sync and stream despite saved 1.1 / 3")


_orig_main = main
def main():  # noqa: F811
    import logging
    records = []
    class _H(logging.Handler):
        def emit(self, r): records.append(r)
    logging.getLogger("ephapsys.sdk").addHandler(_H()); logging.getLogger("ephapsys.sdk").setLevel(logging.INFO)
    logging.getLogger("ephapsys.agent").addHandler(_H()); logging.getLogger("ephapsys.agent").setLevel(logging.INFO)

    _orig_main()
    agent = _make_agent()
    print("\n== test_validators_direct"); test_validators_direct()
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = _prepare_local_model(tmp)
        runtime = {"model_path": model_dir, "config": {}}
        tok, _, _, _, _ = agent._ensure_language_model_loaded(runtime)
        print("\n== test_model_config_fallback_and_generation_precedence"); test_model_config_fallback_and_generation_precedence(agent, tok, runtime)
        print("\n== test_source_logged_once_per_runtime"); test_source_logged_once_per_runtime(agent, tok, runtime, records)
    print("\n== test_neutral_values_reach_generate_sync_and_stream"); test_neutral_values_reach_generate_sync_and_stream(agent)
    print("\nALL PASSED (review additions)")

if __name__ == "__main__":
    main()
