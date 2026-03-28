#!/usr/bin/env python3
"""Robot sample launcher for local body + brain + face demo."""

import asyncio
import faulthandler
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

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


def stream_subprocess_output(pipe, sink):
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            sink.write(f"[brain] {line}")
            sink.flush()
    finally:
        try:
            pipe.close()
        except Exception:
            pass


async def main():
    port = int(os.getenv("ROBOT_BRAIN_PORT", "8765"))
    stream_logs = os.getenv("ROBOT_STREAM_BRAIN_LOGS", "").lower() in ("1", "true", "yes")
    log_path = Path(os.getenv("ROBOT_BRAIN_LOG_PATH", ".ephapsys_state/robot_brain.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_sink = sys.stderr if stream_logs else log_path.open("w", encoding="utf-8")
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
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_thread = threading.Thread(
        target=stream_subprocess_output,
        args=(server_proc.stdout, log_sink),
        daemon=True,
    )
    output_thread.start()
    try:
        wait_for_health(f"http://127.0.0.1:{port}/health")
        await run_terminal_face(f"ws://127.0.0.1:{port}/ws/state")
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        output_thread.join(timeout=1)
        if log_sink is not sys.stderr:
            log_sink.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down (Ctrl+C).", file=sys.stderr)
    sys.exit(0)
