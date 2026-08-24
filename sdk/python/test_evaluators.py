"""
Regression tests for the streaming evaluators' dataset-name requirements.

compute_language_metrics_stream and compute_stt_metrics_stream used to
default ds_name/ds_config/ds_split to WikiText/LibriSpeech, so a caller that
omitted the dataset was silently scored on an unrelated corpus. Both are now
required (no default) and validated - a caller that omits them gets a clear
ValueError instead of a silent wrong-dataset eval.

These tests only exercise the validation guard, which runs before any of the
functions' heavy imports (evaluate, datasets) - so they don't need those
packages installed.
"""
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(__file__))

from ephapsys.modulation import ModulatorClient


def _client():
    return ModulatorClient(base_url="http://localhost:0", api_key="test-token")


def test_language_metrics_requires_ds_name():
    print("Testing compute_language_metrics_stream rejects missing ds_name...")
    mc = _client()
    for bad_name in (None, ""):
        try:
            next(mc.compute_language_metrics_stream(
                model=None, tokenizer=None, model_id="x", ds_name=bad_name))
            raise AssertionError(f"expected ValueError for ds_name={bad_name!r}")
        except ValueError as e:
            assert "ds_name" in str(e)
    print("  PASSED\n")


def test_language_metrics_no_default_for_ds_name():
    """A caller that omits ds_name entirely must get a TypeError (missing
    required argument), not a silent WikiText default."""
    print("Testing compute_language_metrics_stream has no ds_name default...")
    mc = _client()
    try:
        mc.compute_language_metrics_stream(model=None, tokenizer=None, model_id="x")
        raise AssertionError("expected TypeError for omitted ds_name, none raised")
    except TypeError as e:
        assert "ds_name" in str(e)
    print("  PASSED\n")


def test_language_metrics_requires_ds_split_for_remote_dataset():
    """A remote (non-local-file) ds_name without ds_split must be rejected -
    that's the "held-out eval was actually training data" failure mode."""
    print("Testing compute_language_metrics_stream requires ds_split for a remote dataset...")
    mc = _client()
    try:
        next(mc.compute_language_metrics_stream(
            model=None, tokenizer=None, model_id="x",
            ds_name="Salesforce/wikitext", ds_config="wikitext-103-raw-v1", ds_split=None))
        raise AssertionError("expected ValueError for a remote dataset with no ds_split")
    except ValueError as e:
        assert "ds_split" in str(e)
    print("  PASSED\n")


def test_language_metrics_allows_missing_ds_split_for_local_file():
    """ds_split may be omitted when ds_name resolves to a local file - the
    function's own local-file branch hardcodes split="train" and never reads
    ds_split, so there's nothing for an omitted split to silently default."""
    print("Testing compute_language_metrics_stream allows missing ds_split for a local file...")
    mc = _client()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        f.write('{"text": "hello world"}\n')
        local_path = f.name
    try:
        try:
            next(mc.compute_language_metrics_stream(
                model=None, tokenizer=None, model_id="x",
                ds_name=local_path, ds_config=None, ds_split=None))
        except ValueError as e:
            if "ds_split" in str(e):
                raise AssertionError(
                    "local-file ds_name should NOT require ds_split, but got: " + str(e))
            raise  # some other ValueError - not our concern here, let it propagate
        except Exception:
            # Expected: validation passed, execution reached the heavy imports
            # (evaluate/datasets) or beyond - that's what we're testing for.
            pass
    finally:
        os.unlink(local_path)
    print("  PASSED\n")


def test_stt_metrics_requires_all_three():
    print("Testing compute_stt_metrics_stream rejects missing ds_name/config/split...")
    mc = _client()
    cases = [
        dict(ds_name=None, ds_config="clean", ds_split="validation[:100]"),
        dict(ds_name="librispeech_asr", ds_config=None, ds_split="validation[:100]"),
        dict(ds_name="librispeech_asr", ds_config="clean", ds_split=None),
        dict(ds_name="", ds_config="", ds_split=""),
    ]
    for kwargs in cases:
        try:
            next(mc.compute_stt_metrics_stream(model=None, processor=None, model_id="x", **kwargs))
            raise AssertionError(f"expected ValueError for {kwargs}")
        except ValueError as e:
            assert "ds_name" in str(e) or "ds_config" in str(e) or "ds_split" in str(e)
    print("  PASSED\n")


def test_stt_metrics_no_defaults():
    """A caller that omits ds_name/ds_config/ds_split entirely must get a
    TypeError, not a silent LibriSpeech default."""
    print("Testing compute_stt_metrics_stream has no dataset defaults...")
    mc = _client()
    try:
        mc.compute_stt_metrics_stream(model=None, processor=None, model_id="x")
        raise AssertionError("expected TypeError for omitted dataset args, none raised")
    except TypeError:
        pass
    print("  PASSED\n")


if __name__ == "__main__":
    test_language_metrics_requires_ds_name()
    test_language_metrics_no_default_for_ds_name()
    test_language_metrics_requires_ds_split_for_remote_dataset()
    test_language_metrics_allows_missing_ds_split_for_local_file()
    test_stt_metrics_requires_all_three()
    test_stt_metrics_no_defaults()
    print("All tests passed!")
