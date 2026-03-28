#!/usr/bin/env python3
"""
Robot sample entrypoint.

This sample is now structured around three in-process layers:
- body: local sensors/actuators and device I/O
- brain: trusted runtime, memory, reasoning, orchestration
- face: terminal presentation for developers

The process still runs locally as a single demo, but the separation is explicit
so it can evolve toward Francisca-style body/brain/face architecture later.
"""

import asyncio
import faulthandler
import os
import sys

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

from robot_body import RobotBody
from robot_brain import RobotBrain
from robot_face import RobotFace

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


async def main():
    shutdown_event = asyncio.Event()
    face = RobotFace()
    body = RobotBody(face, shutdown_event)
    brain = RobotBrain(face, body, shutdown_event)

    greeting = await brain.startup()
    try:
        with face.live(greeting) as live:
            await asyncio.gather(
                body.mic_task(),
                body.cam_task(),
                brain.process_task(live),
                brain.periodic_verify(),
                body.tts_worker(brain.agent),
                return_exceptions=True,
            )
    finally:
        body.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down (Ctrl+C).", file=sys.stderr)
    sys.exit(0)
