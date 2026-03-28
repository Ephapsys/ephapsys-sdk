#!/usr/bin/env python3
"""Robot sample launcher for local body + brain + face demo."""

import asyncio
import faulthandler
import os
import sys
import threading
import time
import urllib.request

try:
    import torch
except ImportError:
    sys.exit("[ERROR] Missing torch; install torch>=2.6.0 for TTS audio playback.")
try:
    import transformers
except ImportError:
    sys.exit("[ERROR] Missing transformers; install transformers>=4.46.0 for TTS audio playback.")
try:
    import soundfile
except ImportError:
    sys.exit("[ERROR] Missing soundfile; install soundfile>=0.12.0 for TTS audio playback.")

import uvicorn

from robot_face import run_terminal_face

try:
    faulthandler.enable(all_threads=True)
except Exception:
    pass

try:
    torch_threads = int(os.getenv("TORCH_NUM_THREADS", "1"))
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(torch_threads)
except Exception:
    pass


def wait_for_health(url: str, timeout_s: float = 20.0):
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    payload = response.read().decode("utf-8")
                    if '"ready": true' in payload:
                        return
        except Exception as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"Robot brain did not become healthy in time: {last_error}")


async def main():
    port = int(os.getenv("ROBOT_BRAIN_PORT", "8765"))
    server = uvicorn.Server(
        uvicorn.Config(
            "robot_brain_server:app",
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
    )

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    wait_for_health(f"http://127.0.0.1:{port}/health")
    await run_terminal_face(f"ws://127.0.0.1:{port}/ws/state")
    server.should_exit = True
    server_thread.join(timeout=3)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down (Ctrl+C).", file=sys.stderr)
    sys.exit(0)
