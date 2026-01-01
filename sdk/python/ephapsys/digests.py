# SPDX-License-Identifier: Apache-2.0
import hashlib, json

def _hash_file(path: str, h=None):
    if h is None:
        h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h

def rms_hash(weights_path: str, ecm_path: str, hyperparams: dict) -> str:
    """Compute Root Modulation Signature (RMS).
    Combines binary weights + ECM + stable JSON of hyperparameters.
    Returns a 'sha256:<hex>' string.
    """
    h = hashlib.sha256()
    _hash_file(weights_path, h)
    _hash_file(ecm_path, h)
    h.update(json.dumps(hyperparams, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    return "sha256:" + h.hexdigest()
