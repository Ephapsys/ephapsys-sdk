#!/usr/bin/env python3
"""Robot sample launcher for local body + brain + face demo."""

import asyncio
import faulthandler
import os
import subprocess
import sys
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
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"Robot brain did not become healthy in time: {last_error}")


async def main():
    port = int(os.getenv("ROBOT_BRAIN_PORT", "8765"))
    server_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "robot_brain_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    try:
        wait_for_health(f"http://127.0.0.1:{port}/health")
        await run_terminal_face(f"ws://127.0.0.1:{port}/ws/state")
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down (Ctrl+C).", file=sys.stderr)
    sys.exit(0)
