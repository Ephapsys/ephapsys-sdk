#!/usr/bin/env python3

import asyncio
import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from robot_body import RobotBody
from robot_channel import RobotChannel
from robot_brain import RobotBrain
from robot_contracts import ROBOT_PUBLIC_SCHEMAS, public_schema_bundle
from robot_face import RobotStateFace
from robot_remote_body import RemoteAudioSegmenter

app = FastAPI(title="Robot Brain")

shutdown_event = asyncio.Event()
state_face = RobotStateFace()
channel = RobotChannel()
body = RobotBody(state_face, shutdown_event, channel)
brain = RobotBrain(state_face, body, channel, shutdown_event)
remote_audio = RemoteAudioSegmenter(body, channel)
brain_task = None
brain_ready = asyncio.Event()
body_mode = os.getenv("ROBOT_BODY_MODE", "local").strip().lower()
remote_control_clients = set()
remote_control_lock = asyncio.Lock()


def _ensure_brain_task():
    global brain_task
    if brain_task is None:
        brain_task = asyncio.create_task(_run_brain())


async def _run_brain():
    try:
        mic_task = asyncio.create_task(body.mic_task()) if body_mode in {"local", "hybrid"} else None
        tts_task = asyncio.create_task(body.tts_worker(brain.agent)) if body_mode in {"local", "hybrid"} else None
        remote_tts_task = asyncio.create_task(remote_body_control_task()) if body_mode == "remote" else None
        ingest_task = asyncio.create_task(brain.ingest_channel_events())
        output_task = asyncio.create_task(brain.output_arbiter())
        await brain.startup()
        brain_ready.set()
        tasks = [
            brain.process_task(None),
            brain.periodic_verify(),
            ingest_task,
            output_task,
        ]
        if tts_task is not None:
            tasks.append(tts_task)
        if remote_tts_task is not None:
            tasks.append(remote_tts_task)
        if body_mode in {"local", "hybrid"}:
            tasks.extend([body.cam_task(), mic_task])
        await asyncio.gather(
            *tasks,
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
    return {"ok": True, "ready": brain_ready.is_set(), "body_mode": body_mode, "state": state_face.snapshot()}


async def _broadcast_remote_command(payload: dict) -> bool:
    async with remote_control_lock:
        clients = list(remote_control_clients)
    if not clients:
        return False
    stale = []
    delivered = False
    for ws in clients:
        try:
            await ws.send_text(json.dumps(payload))
            delivered = True
        except Exception:
            stale.append(ws)
    if stale:
        async with remote_control_lock:
            for ws in stale:
                remote_control_clients.discard(ws)
    return delivered


async def remote_body_control_task():
    while not shutdown_event.is_set():
        command = await channel.next_command()
        try:
            payload = {"type": "command", "kind": command.kind, "payload": command.payload}
            delivered = await _broadcast_remote_command(payload)
            if delivered:
                state_face.set_state(
                    speaking="Remote body active" if command.kind == "speak" else state_face.ui_state.get("speaking", "Idle"),
                    body="Remote body active" if command.kind == "body_control" else state_face.ui_state.get("body", "Idle"),
                    event=f"Remote command sent: {command.kind}",
                )
                continue
            if command.kind == "speak":
                await channel.emit_event("tts_done", text=command.payload.get("text", ""), duration_ms=0.0, error="No remote body client connected")
            elif command.kind == "body_control":
                await channel.emit_event("body_control_done", action=command.payload.get("action", "idle"), error="No remote body client connected")
            state_face.set_state(event=f"Remote body unavailable for {command.kind}")
        finally:
            channel.command_done()


@app.get("/schemas")
async def schemas():
    return public_schema_bundle()


@app.get("/schemas/{schema_name}")
async def schema_by_name(schema_name: str):
    schema = ROBOT_PUBLIC_SCHEMAS.get(schema_name)
    if schema is None:
        return {"ok": False, "error": f"unknown schema '{schema_name}'", "available": sorted(ROBOT_PUBLIC_SCHEMAS)}
    return {"ok": True, "schema_version": "robot-public-v1", "name": schema_name, "schema": schema}


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


@app.websocket("/ws/body/audio")
async def ws_body_audio(ws: WebSocket):
    _ensure_brain_task()
    await ws.accept()
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            if not data:
                continue
            await remote_audio.ingest(data)
            state_face.set_state(hearing="Remote microphone active", event="Remote body audio received")
    except WebSocketDisconnect:
        await remote_audio.flush()
        return


@app.websocket("/ws/body/video")
async def ws_body_video(ws: WebSocket):
    _ensure_brain_task()
    await ws.accept()
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            if not data:
                continue
            frame = body.decode_image_bytes(data)
            if frame is None:
                continue
            await channel.emit_event("camera", frame=frame, source="remote_ws")
            state_face.set_state(vision="Remote camera active", event="Remote body video received")
    except WebSocketDisconnect:
        return


@app.websocket("/ws/body/control")
async def ws_body_control(ws: WebSocket):
    _ensure_brain_task()
    await ws.accept()
    async with remote_control_lock:
        remote_control_clients.add(ws)
    state_face.set_state(body="Remote body connected", event="Remote body control connected")
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            text = message.get("text")
            if not text:
                continue
            payload = json.loads(text)
            if payload.get("type") != "event":
                continue
            kind = payload.get("kind")
            event_payload = payload.get("payload") or {}
            if kind == "tts_done":
                await channel.emit_event("tts_done", **event_payload)
            elif kind == "body_control_done":
                await channel.emit_event("body_control_done", **event_payload)
            else:
                await channel.emit_event(kind or "remote_event", **event_payload)
    except WebSocketDisconnect:
        return
    finally:
        async with remote_control_lock:
            remote_control_clients.discard(ws)
        state_face.set_state(body="Remote body disconnected", event="Remote body control disconnected")
