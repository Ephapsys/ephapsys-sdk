#!/usr/bin/env python3
"""Run the robot with a remote brain and a local body + terminal face."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import suppress

import cv2
import numpy as np
import websockets

from robot_face import run_terminal_face

try:
    import pyaudio
except ImportError:
    pyaudio = None


def pcm_chunk_bytes(chunk: np.ndarray) -> bytes:
    pcm16 = np.clip(chunk, -32768, 32767).astype("<i2")
    return pcm16.tobytes()


async def stream_microphone(ws_url: str, shutdown_event: asyncio.Event):
    if pyaudio is None:
        raise RuntimeError("pyaudio is required for remote microphone streaming")

    sample_rate = int(os.getenv("ROBOT_REMOTE_AUDIO_SR", "16000"))
    frame_ms = int(os.getenv("ROBOT_REMOTE_AUDIO_FRAME_MS", "30"))
    chunk_size = int(sample_rate * frame_ms / 1000)
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )
    try:
        async with websockets.connect(ws_url, max_size=None) as websocket:
            while not shutdown_event.is_set():
                data = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: stream.read(chunk_size, exception_on_overflow=False),
                )
                if data:
                    await websocket.send(data)
    finally:
        with suppress(Exception):
            stream.stop_stream()
            stream.close()
        pa.terminate()


async def stream_camera(ws_url: str, shutdown_event: asyncio.Event):
    cap = cv2.VideoCapture(int(os.getenv("ROBOT_REMOTE_CAMERA_INDEX", "0")))
    interval_s = float(os.getenv("ROBOT_REMOTE_VIDEO_INTERVAL_S", "1.0"))
    jpeg_quality = int(os.getenv("ROBOT_REMOTE_VIDEO_JPEG_QUALITY", "80"))
    try:
        async with websockets.connect(ws_url, max_size=None) as websocket:
            while not shutdown_event.is_set():
                ok, frame = await asyncio.get_running_loop().run_in_executor(None, cap.read)
                if ok and frame is not None:
                    encode_ok, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                    )
                    if encode_ok:
                        await websocket.send(encoded.tobytes())
                await asyncio.sleep(interval_s)
    finally:
        with suppress(Exception):
            cap.release()


def speak_with_system_tts(text: str) -> float:
    started = time.perf_counter()
    player = None
    if shutil.which("say"):
        player = ["say", text]
    elif shutil.which("espeak"):
        player = ["espeak", text]
    elif shutil.which("spd-say"):
        player = ["spd-say", text]
    if player is not None:
        subprocess.run(player, check=False)
    return (time.perf_counter() - started) * 1000


async def handle_remote_control(ws_url: str, shutdown_event: asyncio.Event):
    async with websockets.connect(ws_url, max_size=None) as websocket:
        while not shutdown_event.is_set():
            raw = await websocket.recv()
            payload = json.loads(raw)
            if payload.get("type") != "command":
                continue
            kind = payload.get("kind")
            command_payload = payload.get("payload") or {}
            if kind == "speak":
                text = str(command_payload.get("text", "")).strip()
                duration_ms = await asyncio.get_running_loop().run_in_executor(None, speak_with_system_tts, text)
                await websocket.send(
                    json.dumps(
                        {
                            "type": "event",
                            "kind": "tts_done",
                            "payload": {"text": text, "duration_ms": duration_ms},
                        }
                    )
                )
            elif kind == "body_control":
                action = str(command_payload.get("action", "idle"))
                await asyncio.sleep(0.1)
                await websocket.send(
                    json.dumps(
                        {
                            "type": "event",
                            "kind": "body_control_done",
                            "payload": {"action": action},
                        }
                    )
                )


async def main():
    ws_state_url = os.getenv("ROBOT_BRAIN_WS_URL", "ws://127.0.0.1:8765/ws/state")
    ws_audio_url = os.getenv("ROBOT_BRAIN_AUDIO_WS_URL", "ws://127.0.0.1:8765/ws/body/audio")
    ws_video_url = os.getenv("ROBOT_BRAIN_VIDEO_WS_URL", "ws://127.0.0.1:8765/ws/body/video")
    ws_control_url = os.getenv("ROBOT_BRAIN_CONTROL_WS_URL", "ws://127.0.0.1:8765/ws/body/control")
    shutdown_event = asyncio.Event()

    face_task = asyncio.create_task(run_terminal_face(ws_state_url))
    audio_task = asyncio.create_task(stream_microphone(ws_audio_url, shutdown_event))
    video_task = asyncio.create_task(stream_camera(ws_video_url, shutdown_event))
    control_task = asyncio.create_task(handle_remote_control(ws_control_url, shutdown_event))
    tasks = [face_task, audio_task, video_task, control_task]
    try:
        await asyncio.gather(*tasks)
    finally:
        shutdown_event.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down (Ctrl+C).", file=sys.stderr)
