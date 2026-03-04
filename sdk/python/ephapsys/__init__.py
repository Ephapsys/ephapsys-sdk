# SPDX-License-Identifier: Apache-2.0
###########################
# Ephapsys SDK Definition
##########################


# ephapsys/__init__.py
from .agent import TrustedAgent

__all__ = ["TrustedAgent", "ModulatorClient", "A2AClient"]


def __getattr__(name):
    if name == "ModulatorClient":
        try:
            from .modulation import ModulatorClient as _ModulatorClient
        except ImportError as exc:
            raise ImportError(
                "ModulatorClient requires optional modulation dependencies. "
                "Install with: pip install 'ephapsys[modulation]'"
            ) from exc
        return _ModulatorClient
    if name == "A2AClient":
        from .a2a import A2AClient as _A2AClient
        return _A2AClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

