#!/usr/bin/env python3

import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from robot_body import RobotBody
from robot_brain import RobotBrain
from robot_face import RobotStateFace

app = FastAPI(title="Robot Brain")

shutdown_event = asyncio.Event()
state_face = RobotStateFace()
body = RobotBody(state_face, shutdown_event)
brain = RobotBrain(state_face, body, shutdown_event)
brain_task = None


async def _run_brain():
    await brain.startup()
    await asyncio.gather(
        body.mic_task(),
        body.cam_task(),
        brain.process_task(None),
        brain.periodic_verify(),
        body.tts_worker(brain.agent),
        return_exceptions=True,
    )


@app.on_event("startup")
async def startup_event():
    global brain_task
    if brain_task is None:
        brain_task = asyncio.create_task(_run_brain())


@app.on_event("shutdown")
async def shutdown_event_handler():
    shutdown_event.set()
    body.cleanup()
    if brain_task is not None:
        brain_task.cancel()


@app.get("/health")
async def health():
    return {"ok": True, "state": state_face.snapshot()}


@app.websocket("/ws/state")
async def ws_state(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_text(json.dumps({"snapshot": state_face.snapshot()}))
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return
