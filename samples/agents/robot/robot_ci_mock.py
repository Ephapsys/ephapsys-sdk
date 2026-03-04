#!/usr/bin/env python3
"""
CI mock path for Robot sample.

Purpose:
- Validate robot agent template provisioning/personalization/runtime path
  without requiring mic/camera/audio hardware dependencies.
"""

import os
import sys
import time
from ephapsys.agent import TrustedAgent


def main() -> int:
    agent = TrustedAgent.from_env()
    print(f"[Robot CI] agent_id={agent.agent_id}")

    try:
        ok, _ = agent.verify()
    except RuntimeError as e:
        if "404" in str(e):
            print(f"[Robot CI] template not found: {agent.agent_id}")
            return 1
        raise

    if not ok:
        status = agent.get_status()
        personalized = status.get("state", {}).get("personalized", False) or status.get("personalized", False)
        if not personalized:
            anchor = os.getenv("PERSONALIZE_ANCHOR", "none")
            print(f"[Robot CI] personalize(anchor={anchor})")
            agent.personalize(anchor=anchor)
            for _ in range(5):
                ok, _ = agent.verify()
                if ok:
                    break
                time.sleep(1)

    if not ok:
        print("[Robot CI] agent not ready after personalization")
        return 1

    rt = agent.prepare_runtime()
    print(f"[Robot CI] runtime keys={sorted(list(rt.keys()))}")

    prompt = os.getenv("ROBOT_CI_PROMPT", "robot ci health check")
    out = agent.run(prompt, model_kind="language")
    text = out.strip() if isinstance(out, str) else str(out)
    print(f"[Robot CI] language response={text[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

