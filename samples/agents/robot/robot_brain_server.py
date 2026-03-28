#!/usr/bin/env python3

import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from robot_body import RobotBody
from robot_channel import RobotChannel
from robot_brain import RobotBrain
from robot_face import RobotStateFace

app = FastAPI(title="Robot Brain")

shutdown_event = asyncio.Event()
state_face = RobotStateFace()
channel = RobotChannel()
body = RobotBody(state_face, shutdown_event, channel)
brain = RobotBrain(state_face, body, channel, shutdown_event)
brain_task = None
brain_ready = asyncio.Event()


def _ensure_brain_task():
    global brain_task
    if brain_task is None:
        brain_task = asyncio.create_task(_run_brain())


async def _run_brain():
    try:
        mic_task = asyncio.create_task(body.mic_task())
        tts_task = asyncio.create_task(body.tts_worker(brain.agent))
        ingest_task = asyncio.create_task(brain.ingest_channel_events())
        output_task = asyncio.create_task(brain.output_arbiter())
        await brain.startup()
        brain_ready.set()
        await asyncio.gather(
            body.cam_task(),
            brain.process_task(None),
            brain.periodic_verify(),
            mic_task,
            tts_task,
            ingest_task,
            output_task,
            return_exceptions=True,
        )
    except Exception as exc:
        state_face.set_state(
            reasoning="Startup blocked",
            speaking="Unavailable",
            event=f"Startup failure: {exc}",
        )
        state_face.console_log.log(f"Robot brain startup failed: {exc}")
        raise
    finally:
        brain_ready.clear()


@app.on_event("shutdown")
async def shutdown_event_handler():
    shutdown_event.set()
    body.cleanup()
    if brain_task is not None:
        brain_task.cancel()


@app.get("/health")
async def health():
    _ensure_brain_task()
    return {"ok": True, "ready": brain_ready.is_set(), "state": state_face.snapshot()}


@app.websocket("/ws/state")
async def ws_state(ws: WebSocket):
    _ensure_brain_task()
    await ws.accept()
    last_snapshot = None
    try:
        while True:
            snapshot = state_face.snapshot()
            if snapshot != last_snapshot:
                await ws.send_text(json.dumps({"snapshot": snapshot}))
                last_snapshot = snapshot
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        return
