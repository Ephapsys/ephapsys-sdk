# SPDX-License-Identifier: Apache-2.0
import os

def get_api_key(explicit: str = None) -> str:
    if explicit:
        return explicit
    k = os.getenv("AOC_API_KEY") or os.getenv("EPHAPSYS_API_KEY")
    if not k:
        raise RuntimeError("Missing API key. Set AOC_API_KEY or pass api_key=...")
    return k
