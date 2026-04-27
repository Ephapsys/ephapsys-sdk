# SPDX-License-Identifier: Apache-2.0
###########################
# Ephapsys SDK Definition
##########################


# ephapsys/__init__.py
from .agent import TrustedAgent

__all__ = [
    "TrustedAgent",
    "ModulatorClient",
    "A2AClient",
    "VerifiedMessage",
    "MessageJournal",
    "compute_indispensability_loss",
    "run_ablation_probe",
]


def __getattr__(name):
    if name == "ModulatorClient":
        try:
            from .modulation import ModulatorClient as _ModulatorClient
        except ImportError as exc:
            raise ImportError(
                "ModulatorClient requires optional modulation dependencies. "
                "Install with: pip install ephapsys"
            ) from exc
        return _ModulatorClient
    if name == "A2AClient":
        from .a2a import A2AClient as _A2AClient
        return _A2AClient
    if name == "VerifiedMessage":
        from .a2a import VerifiedMessage as _VerifiedMessage
        return _VerifiedMessage
    if name == "MessageJournal":
        from .journal import MessageJournal as _MessageJournal
        return _MessageJournal
    if name == "compute_indispensability_loss":
        from .modulation import compute_indispensability_loss as _fn
        return _fn
    if name == "run_ablation_probe":
        from .modulation import run_ablation_probe as _fn
        return _fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
