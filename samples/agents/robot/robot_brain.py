#!/usr/bin/env python3

import asyncio
import os
import sys
import time
import traceback
import cv2
import numpy as np

from PIL import Image
from ephapsys.agent import TrustedAgent
from robot_arch import (
    RobotGovernor,
    RobotIntent,
    RobotToolbox,
    body_intent_for_world,
    classify_tool_intent,
    face_intent_for_state,
)
from robot_state import RobotStateStore


class RobotBrain:
    def __init__(self, face, body, channel, shutdown_event):
        self.face = face
        self.body = body
        self.channel = channel
        self.shutdown_event = shutdown_event
        self.agent = TrustedAgent.from_env()
        self.governor = RobotGovernor(
            allow_body_control=os.getenv("ROBOT_ALLOW_BODY_CONTROL", "1").lower() not in ("0", "false", "no"),
            allow_tools=os.getenv("ROBOT_ALLOW_TOOLS", "1").lower() not in ("0", "false", "no"),
        )
        self.toolbox = RobotToolbox()
        self.state = RobotStateStore()
        self.language_warm_task = None
        self.reasoning_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        # Live per-turn vision is enabled by default again for the robot demo.
        # It can still be disabled explicitly via env when debugging other paths.
        self.live_vision_enabled = os.getenv("ROBOT_ENABLE_LIVE_VISION", "1").lower() not in ("0", "false", "no")
        self.world_enabled = os.getenv("ROBOT_ENABLE_WORLD_MODEL", "1").lower() not in ("0", "false", "no")

    def set_governor_state(self, decision):
        self.face.set_state(governor=f"{'ALLOW' if decision.allowed else 'BLOCK'}: {decision.reason}")

    async def dispatch_body_intent(self, world_summary: str):
        intent = body_intent_for_world(world_summary)
        decision = self.governor.approve(intent)
        self.set_governor_state(decision)
        if not decision.allowed:
            return
        action = intent.payload.get("action", "idle")
        self.face.set_state(body=str(action))
        await self.channel.send_command("body_control", action=action)

    async def maybe_use_tool(self, transcript: str):
        intent = classify_tool_intent(transcript)
        if intent is None:
            self.face.set_state(tools="Idle")
            return None
        decision = self.governor.approve(intent)
        self.set_governor_state(decision)
        if not decision.allowed:
            self.face.set_state(tools=f"Blocked: {intent.payload.get('tool', 'tool')}")
            return f"I am not allowed to use the {intent.payload.get('tool', 'requested')} tool right now."
        self.face.set_state(tools=f"Running {intent.payload.get('tool', 'tool')}")
        result = self.toolbox.execute(intent)
        self.face.set_state(tools=f"Completed {intent.payload.get('tool', 'tool')}")
        return result

    async def emit_reasoning_event(self, kind: str, **payload):
        await self.reasoning_queue.put({"kind": kind, "payload": payload})

    async def emit_output_event(self, kind: str, **payload):
        if kind in {"body_control", "face_control"}:
            self._coalesce_output_kind(kind)
        await self.output_queue.put({"kind": kind, "payload": payload})

    def _coalesce_output_kind(self, kind: str):
        queue_items = list(self.output_queue._queue)
        filtered = [item for item in queue_items if item.get("kind") != kind]
        if len(filtered) == len(queue_items):
            return
        self.output_queue._queue.clear()
        self.output_queue._queue.extend(filtered)

    async def emit_face_intent(self, *, world_summary: str = "", reasoning: str = "", speaking: str = "", event: str = ""):
        intent = face_intent_for_state(
            world_summary=world_summary,
            reasoning=reasoning,
            speaking=speaking,
            event=event,
        )
        decision = self.governor.approve(intent)
        self.set_governor_state(decision)
        if not decision.allowed:
            return
        self.state.latest_expression = str(intent.payload.get("expression", "neutral"))
        self.state.latest_gaze = str(intent.payload.get("gaze", "center"))
        await self.emit_output_event(
            "face_control",
            expression=self.state.latest_expression,
            gaze=self.state.latest_gaze,
        )

    def log_stage(self, label: str, started_at: float):
        self.face.console_log.log(f"[brain] {label} in {(time.perf_counter() - started_at):.2f}s")

    async def run_blocking(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    def build_startup_scene_observation(self):
        vision_label = None
        try:
            self.face.set_state(vision="Looking for a first impression", reasoning="Observing the scene")
            frame = self.body.capture_startup_frame()
            if frame is not None:
                self.state.latest_camera_frame = frame
                t0 = time.perf_counter()
                self.face.console_log.log("[brain] Loading startup vision model: Robot Vision Model (hustvl/yolos-base)")
                vision_input = Image.fromarray(frame)
                vision_raw = self.agent.run(vision_input, model_kind="vision")
                self.log_stage("Startup vision ready", t0)
                vision_label = str(vision_raw).strip() if vision_raw is not None else None
        except Exception as exc:
            self.face.console_log.log(f"Startup vision observation fallback: {exc}")
        return vision_label

    def compute_world_summary(self, frame, vision_label):
        movement_phrase = "scene steady"
        motion_score = None
        if self.state.latest_camera_frame is not None and frame is not self.state.latest_camera_frame:
            prev_small = cv2.resize(self.state.latest_camera_frame, (64, 64), interpolation=cv2.INTER_AREA)
            curr_small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            motion_score = float(np.mean(np.abs(curr_small.astype("float32") - prev_small.astype("float32"))) / 255.0)

        world_delta = None
        if self.world_enabled:
            try:
                current_embedding = np.asarray(
                    self.agent.run(Image.fromarray(frame), model_kind="world"),
                    dtype="float32",
                )
                if current_embedding is not None and self.state.prev_world_embedding is not None:
                    prev = self.state.prev_world_embedding
                    curr = current_embedding
                    denom = (np.linalg.norm(prev) * np.linalg.norm(curr)) or 1.0
                    world_delta = float(1.0 - np.dot(prev, curr) / denom)
                if current_embedding is not None:
                    self.state.prev_world_embedding = current_embedding
            except Exception as exc:
                self.face.console_log.log(f"World summary fallback: {exc}")

        activity_score = max(motion_score or 0.0, world_delta or 0.0)
        if activity_score >= 0.16:
            movement_phrase = "significant movement"
        elif activity_score >= 0.05:
            movement_phrase = "movement detected"

        vision_text = (vision_label or "").strip().lower()
        if vision_text and vision_text != "no objects detected":
            if "person" in vision_text and activity_score >= 0.05:
                return "person moving in view"
            if "person" in vision_text:
                return "person present"
            if activity_score >= 0.05:
                return f"{vision_label}; {movement_phrase}"
            return str(vision_label)
        if activity_score >= 0.05:
            return movement_phrase
        return "scene clear"

    @staticmethod
    def build_startup_greeting(vision_label):
        if vision_label and vision_label.strip() and vision_label.strip().lower() != "no objects detected":
            return f"Hello. I can see {vision_label}. I'm ready when you are."
        return "Hello. I'm ready when you are."

    async def warm_language_runtime(self):
        if self.state.language_warm_done:
            return
        started = time.perf_counter()
        try:
            self.face.console_log.log("[brain] Warming language model: Robot Language Model")
            await self.run_blocking(self.agent.run, "Hello.", model_kind="language")
            self.log_stage("Language model warmup ready", started)
            self.state.language_warm_done = True
        except Exception as exc:
            self.face.console_log.log(f"Language warmup skipped: {exc}")

    async def startup(self):
        self.face.startup()
        self.face.set_state(
            hearing="Stand by",
            vision="Stand by",
            reasoning="Verifying agent",
            expression=self.state.latest_expression,
            gaze=self.state.latest_gaze,
            body="Idle",
            tools="Idle",
            governor="Ready",
            speaking="Stand by",
            event="Starting brain",
        )
        try:
            self.face.set_state(event="Verifying agent", reasoning="Checking trusted state")
            ok, _ = await self.run_blocking(self.agent.verify)
        except RuntimeError as exc:
            if "404" in str(exc):
                self.face.console_live.print(f"[red]❌ Agent template '{self.agent.agent_id}' not found in backend.[/red]")
                self.face.console_live.print("[yellow]Please create it in the AOC before running this sample.[/yellow]")
                sys.exit(1)
            raise

        if not ok:
            status = await self.run_blocking(self.agent.get_status)
            is_personalized = status.get("state", {}).get("personalized", False) or status.get("personalized", False)
            if not is_personalized:
                anchor = os.getenv("PERSONALIZE_ANCHOR")
                self.face.set_state(event="Personalizing agent instance", reasoning="Binding device identity")
                self.face.console_live.print(
                    f"[yellow]Agent not personalized; running personalize(anchor={anchor})...[/yellow]"
                )
                await self.run_blocking(self.agent.personalize, anchor=anchor)
                self.face.console_live.print("[green]✅ Agent personalized (instance registered in AOC).[/green]")
                for _ in range(5):
                    ok, _ = await self.run_blocking(self.agent.verify)
                    if ok:
                        break
                    self.face.console_live.print("[yellow]...waiting for agent to become ready...[/yellow]")
                    await asyncio.sleep(1)
            if not ok:
                self.face.console_live.print("[red]❌ Agent not ready after personalization.[/red]")
                sys.exit(1)

        status = await self.run_blocking(self.agent.get_status)
        is_enabled = status.get("enabled", False) or (status.get("status", "").lower() == "enabled")
        is_revoked = status.get("state", {}).get("revoked", False)
        self.face.agent_status.update({"verified": ok, "enabled": is_enabled, "revoked": is_revoked})
        self.face.set_state(event="Agent verified", reasoning="Trusted runtime ready")
        self.face.console_live.print("[green]✅ Agent personalized and verified.[/green]")
        self.face.console_live.print(f"[dim]Instance DID: {self.agent.agent_id}[/dim]")

        self.face.set_state(event="Preparing runtime bundles", reasoning="Loading secure model runtimes")
        t0 = time.perf_counter()
        self.face.console_log.log("[brain] Preparing runtime bundles")
        runtimes = await self.run_blocking(self.agent.prepare_runtime)
        self.log_stage("Runtime bundles prepared", t0)
        if runtimes.get("world") is None:
            self.world_enabled = False
        tts_path = (runtimes.get("tts") or {}).get("model_path")
        self.body.tts_available = await self.run_blocking(self.body.ensure_preprocessor, tts_path) if tts_path else False
        self.face.set_state(
            hearing="Listening on microphone",
            vision="Scanning scene",
            world="Scanning scene dynamics",
            expression="warm",
            gaze="engage",
            body="Idle",
            tools="Idle",
            governor="Ready",
            reasoning="Preparing greeting",
            speaking="Preparing greeting" if self.body.tts_available else "Unavailable",
            memory="0 memories",
            latency={"turn": None, "stt": None, "vision": None, "language": None, "embedding": None, "tts": None},
            event=f"Runtime ready: {', '.join(sorted(runtimes.keys()))}",
        )
        self.face.set_latest(hearing="-", vision="-", world="-", reply="Preparing greeting...")
        self.face.console_live.print(
            f"[green]✅ Runtime prepared[/green] "
            f"(voice={'ready' if self.body.tts_available else 'unavailable'}, models={', '.join(sorted(runtimes.keys()))})"
        )

        self.face.set_state(event="Observing startup scene", reasoning="Waiting for first interaction")
        startup_vision = await self.run_blocking(self.build_startup_scene_observation)
        if self.state.latest_camera_frame is not None:
            self.state.latest_world_summary = self.compute_world_summary(self.state.latest_camera_frame, startup_vision)
        self.state.latest_scene_summary = (
            self.state.latest_world_summary if self.state.latest_world_summary != "-" else (startup_vision or "-")
        )
        greeting = self.build_startup_greeting(startup_vision)
        self.state.startup_vision_label = startup_vision or "-"
        self.face.set_latest(
            hearing="-",
            vision=startup_vision or "-",
            world=self.state.latest_world_summary,
            reply=greeting,
        )
        if startup_vision:
            self.face.set_state(
                vision=self.face.clip_text(startup_vision, 64),
                world=self.face.clip_text(self.state.latest_world_summary, 64),
            )
        if startup_vision:
            self.face.console_live.print(f"[cyan]👁️ Startup vision: {startup_vision}[/cyan]")
        if self.body.tts_available:
            self.body.speech_enabled = False
            self.face.set_state(reasoning="Greeting ready", speaking="Queued for startup greeting", event="Greeting")
            await self.emit_face_intent(
                world_summary=self.state.latest_world_summary,
                reasoning="Greeting ready",
                speaking="Queued for startup greeting",
                event="Greeting",
            )
            await self.channel.send_command("speak", text=greeting)
            while not self.shutdown_event.is_set():
                event = await self.channel.next_event(timeout=0.25)
                if event is None:
                    continue
                try:
                    if event.kind == "tts_done":
                        break
                finally:
                    self.channel.event_done()
        self.body.speech_enabled = True
        if self.language_warm_task is None:
            self.language_warm_task = asyncio.create_task(self.warm_language_runtime())

        self.face.set_state(
            hearing="Listening on microphone",
            vision="Scanning scene",
            world=self.face.clip_text(self.state.latest_world_summary or "Scanning scene dynamics", 64),
            expression="neutral",
            gaze="center",
            body="Idle",
            tools="Idle",
            reasoning="Waiting for speech",
            speaking="Idle" if self.body.tts_available else "Unavailable",
            event="Live interaction loop started",
        )
        self.face.console_live.print("[blue]Entering live interaction loop...[/blue]")
        return greeting

    async def periodic_verify(self):
        last_snapshot = None
        while not self.shutdown_event.is_set():
            await asyncio.sleep(5)
            try:
                ok, _ = await self.run_blocking(self.agent.verify)
                status = await self.run_blocking(self.agent.get_status)
                is_enabled = status.get("enabled", False) or (status.get("status", "").lower() == "enabled")
                is_revoked = status.get("state", {}).get("revoked", False)
                self.face.agent_status.update({"verified": ok, "enabled": is_enabled, "revoked": is_revoked})
                snapshot = (ok, is_enabled, is_revoked)
                if snapshot != last_snapshot:
                    self.face.set_state(event="Verification state updated")
                    self.face.console_log.log(f"Periodic verify={self.face.agent_status}")
                    last_snapshot = snapshot
            except Exception as exc:
                self.face.set_state(event=f"Verification failed: {exc}")
                self.face.console_log.log(f"⚠️ Verification failed: {exc}")
                self.face.agent_status.update({"enabled": False, "revoked": True})

    async def ingest_channel_events(self):
        while not self.shutdown_event.is_set():
            try:
                event = await self.channel.next_event(timeout=0.2)
            except Exception:
                event = None
            if event is None:
                continue
            try:
                await self.emit_reasoning_event(event.kind, **event.payload)
            finally:
                self.channel.event_done()

    async def output_arbiter(self):
        while not self.shutdown_event.is_set():
            item = await self.output_queue.get()
            try:
                kind = item.get("kind")
                payload = item.get("payload", {})
                if kind == "speak":
                    self.state.awaiting_tts_done = True
                    self.body.speech_enabled = False
                    self.face.set_state(speaking="Queued for playback", event="Reply ready")
                    await self.emit_face_intent(
                        world_summary=self.state.latest_world_summary,
                        reasoning=self.face.ui_state.get("reasoning", ""),
                        speaking="Queued for playback",
                        event="Reply ready",
                    )
                    await self.channel.send_command("speak", text=payload.get("text", ""))
                elif kind == "body_control":
                    await self.channel.send_command("body_control", action=payload.get("action", "idle"))
                elif kind == "face_control":
                    self.face.set_state(
                        expression=payload.get("expression", self.state.latest_expression),
                        gaze=payload.get("gaze", self.state.latest_gaze),
                    )
            finally:
                self.output_queue.task_done()

    async def publish_camera_fact(self, frame, latest_vision_label, awaiting_tts_done: bool):
        vision_label = latest_vision_label
        vision_ms = 0
        latest_world_summary = self.state.latest_world_summary or "-"
        if self.live_vision_enabled and frame is not None:
            camera_state = {
                "vision": "Analyzing scene",
                "event": "Camera update received",
            }
            if not awaiting_tts_done:
                camera_state["reasoning"] = "Waiting for speech"
            self.face.set_state(**camera_state)
            vision_started = time.perf_counter()
            vision_input = Image.fromarray(frame)
            vision_raw = await self.run_blocking(self.agent.run, vision_input, model_kind="vision")
            vision_ms = (time.perf_counter() - vision_started) * 1000
            vision_label = str(vision_raw).strip() if vision_raw is not None else None
            vision_label = vision_label or latest_vision_label
            latest_world_summary = self.compute_world_summary(frame, vision_label)
        await self.emit_reasoning_event(
            "vision_fact",
            frame=frame,
            vision_label=vision_label or "-",
            world_summary=latest_world_summary or "-",
            vision_ms=vision_ms,
        )

    async def publish_microphone_fact(self, mic_audio, heard_summary):
        self.face.set_latest(
            hearing=heard_summary,
            vision=self.face.latest.get("vision", "-"),
            world=self.face.latest.get("world", "-"),
            reply=self.face.latest.get("reply", "-"),
        )
        self.face.set_state(hearing="Transcribing speech", event="Processing microphone input")
        stt_started = time.perf_counter()
        text_input = await self.run_blocking(self.agent.run, mic_audio, model_kind="stt")
        stt_ms = (time.perf_counter() - stt_started) * 1000
        transcript = self.face.clip_text(text_input or heard_summary or "No speech detected", 64)
        await self.emit_reasoning_event(
            "speech_fact",
            transcript=transcript,
            stt_ms=stt_ms,
            heard_summary=heard_summary,
        )

    async def process_task(self, live=None):
        last_render_key = None
        latest_camera_frame = None
        latest_vision_label = self.state.startup_vision_label or "-"
        latest_world_summary = self.state.latest_world_summary or "-"
        latest_scene_summary = self.state.latest_scene_summary or latest_vision_label
        while not self.shutdown_event.is_set():
            if not self.face.agent_status.get("enabled", False) or self.face.agent_status.get("revoked", False):
                self.face.set_state(event="Agent disabled or revoked", reasoning="Paused", speaking="Muted")
                self.face.set_latest(hearing="-", vision="-", world="-", reply="-")
                if live is not None:
                    panel = self.face.render_status("-", "-", "-")
                    key = self.face.render_key("-", "-", "-")
                    if key != last_render_key:
                        live.update(panel, refresh=True)
                        last_render_key = key
                await asyncio.sleep(1)
                continue

            try:
                item = await asyncio.wait_for(self.reasoning_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                item = None

            if item is None:
                self.face.set_state(vision="Scanning", reasoning="Waiting for speech")
                if live is not None:
                    panel = self.face.render_status(
                        self.face.latest.get("hearing", "-"),
                        self.face.latest.get("vision", "-"),
                        self.face.latest.get("reply", "-"),
                    )
                    key = self.face.render_key(
                        self.face.latest.get("hearing", "-"),
                        self.face.latest.get("vision", "-"),
                        self.face.latest.get("reply", "-"),
                    )
                    if key != last_render_key:
                        live.update(panel)
                        last_render_key = key
                continue

            try:
                event_kind = item.get("kind")
                event_payload = item.get("payload", {})
                if event_kind == "camera":
                    latest_camera_frame = event_payload.get("frame")
                    if latest_camera_frame is not None:
                        await self.publish_camera_fact(
                            latest_camera_frame,
                            latest_vision_label,
                            self.state.awaiting_tts_done,
                        )
                    continue

                if event_kind == "tts_done":
                    self.state.awaiting_tts_done = False
                    self.body.speech_enabled = True
                    self.face.set_state(
                        expression="neutral",
                        gaze="center",
                        body=self.face.ui_state.get("body", "Idle"),
                        tools="Idle",
                        speaking="Idle" if self.body.tts_available else "Unavailable",
                        reasoning="Waiting for speech",
                        event="Ready for next interaction",
                    )
                    continue

                if event_kind == "body_control_done":
                    self.face.set_state(event=f"Body ready: {event_payload.get('action', 'idle')}")
                    continue

                if event_kind == "vision_fact":
                    latest_camera_frame = event_payload.get("frame")
                    latest_vision_label = event_payload.get("vision_label") or latest_vision_label
                    latest_world_summary = event_payload.get("world_summary") or latest_world_summary
                    latest_scene_summary = latest_world_summary or latest_vision_label or "-"
                    self.state.latest_camera_frame = latest_camera_frame
                    self.state.latest_world_summary = latest_world_summary
                    self.state.latest_scene_summary = latest_scene_summary
                    intent = body_intent_for_world(latest_world_summary)
                    decision = self.governor.approve(intent)
                    self.set_governor_state(decision)
                    if decision.allowed:
                        action = intent.payload.get("action", "idle")
                        self.face.set_state(body=str(action))
                        await self.emit_output_event("body_control", action=action)
                    await self.emit_face_intent(
                        world_summary=latest_world_summary,
                        reasoning=self.face.ui_state.get("reasoning", "Waiting for speech"),
                        speaking=self.face.ui_state.get("speaking", "Idle"),
                        event="Speaking reply" if self.state.awaiting_tts_done else "Waiting for speech",
                    )
                    self.face.set_state(
                        vision=self.face.clip_text(latest_vision_label or "No scene update", 64),
                        world=self.face.clip_text(latest_world_summary or "Scene clear", 64),
                        latency={"vision": event_payload.get("vision_ms") or None},
                        event="Speaking reply" if self.state.awaiting_tts_done else "Waiting for speech",
                    )
                    self.face.set_latest(
                        hearing=self.face.latest.get("hearing", "-"),
                        vision=latest_vision_label or "-",
                        world=latest_world_summary or "-",
                        reply=self.face.latest.get("reply", "-"),
                    )
                    if live is not None:
                        panel = self.face.render_status(
                            self.face.latest.get("hearing", "-"),
                            self.face.latest.get("vision", "-"),
                            self.face.latest.get("reply", "-"),
                        )
                        key = self.face.render_key(
                            self.face.latest.get("hearing", "-"),
                            self.face.latest.get("vision", "-"),
                            self.face.latest.get("reply", "-"),
                        )
                        if key != last_render_key:
                            live.update(panel)
                            last_render_key = key
                    continue

                if event_kind == "microphone":
                    mic_audio = event_payload.get("audio")
                    heard_summary = event_payload.get("summary") or "No speech"
                    await self.publish_microphone_fact(mic_audio, heard_summary)
                    continue

                if event_kind != "speech_fact":
                    continue

                turn_started = time.perf_counter()
                stt_ms = float(event_payload.get("stt_ms") or 0)
                vision_ms = 0
                language_ms = 0
                embedding_ms = 0

                text_input = event_payload.get("transcript") or "No speech detected"
                self.face.set_state(hearing=text_input)
                await self.emit_face_intent(
                    world_summary=latest_world_summary,
                    reasoning="Processing speech",
                    speaking=self.face.ui_state.get("speaking", "Idle"),
                    event="Processing microphone input",
                )
                if live is not None:
                    panel = self.face.render_status(
                        text_input,
                        latest_vision_label or "-",
                        self.face.latest.get("reply", "-"),
                    )
                    key = self.face.render_key(
                        text_input,
                        latest_vision_label or "-",
                        self.face.latest.get("reply", "-"),
                    )
                    if key != last_render_key:
                        live.update(panel)
                        last_render_key = key
                self.face.set_latest(
                    hearing=text_input,
                    vision=latest_vision_label or "-",
                    world=latest_world_summary or "-",
                    reply=self.face.latest.get("reply", "-"),
                )

                vision_label = latest_vision_label if latest_vision_label != "-" else None
                if self.live_vision_enabled and latest_camera_frame is not None:
                    self.face.set_state(vision="Analyzing scene", event="Running vision model")
                    vision_started = time.perf_counter()
                    vision_input = Image.fromarray(latest_camera_frame)
                    vision_raw = await self.run_blocking(self.agent.run, vision_input, model_kind="vision")
                    vision_ms = (time.perf_counter() - vision_started) * 1000
                    vision_label = str(vision_raw).strip() if vision_raw is not None else None
                    latest_vision_label = vision_label or latest_vision_label
                    latest_world_summary = self.compute_world_summary(latest_camera_frame, latest_vision_label)
                    latest_scene_summary = latest_world_summary or latest_vision_label or "-"
                    self.state.latest_camera_frame = latest_camera_frame
                    self.state.latest_world_summary = latest_world_summary
                    self.state.latest_scene_summary = latest_scene_summary
                    intent = body_intent_for_world(latest_world_summary)
                    decision = self.governor.approve(intent)
                    self.set_governor_state(decision)
                    if decision.allowed:
                        action = intent.payload.get("action", "idle")
                        self.face.set_state(body=str(action))
                        await self.emit_output_event("body_control", action=action)
                    await self.emit_face_intent(
                        world_summary=latest_world_summary,
                        reasoning=self.face.ui_state.get("reasoning", "Waiting for speech"),
                        speaking=self.face.ui_state.get("speaking", "Idle"),
                        event="Running vision model",
                    )
                    self.face.set_state(
                        vision=self.face.clip_text(latest_vision_label or "No scene update", 64),
                        world=self.face.clip_text(latest_world_summary or "Scene clear", 64),
                    )

                tool_response = await self.maybe_use_tool(text_input)
                if tool_response is not None:
                    response_text = str(tool_response).strip()
                    self.face.set_state(reasoning="Tool response ready")
                    language_ms = 0
                else:
                    context_parts = []
                    if vision_label:
                        context_parts.append(f"vision={vision_label}")
                    if latest_world_summary and latest_world_summary != "-":
                        context_parts.append(f"world={latest_world_summary}")
                    context = f" ({'; '.join(context_parts)})" if context_parts else ""
                    self.face.set_latest(
                        hearing=text_input,
                        vision=latest_vision_label or "-",
                        world=latest_world_summary or "-",
                        reply="Thinking...",
                    )
                    self.face.set_state(
                        tools="Idle",
                        reasoning="Composing response",
                        event="Running language model",
                        speaking="Thinking",
                    )
                    await self.emit_face_intent(
                        world_summary=latest_world_summary,
                        reasoning="Composing response",
                        speaking="Thinking",
                        event="Running language model",
                    )
                    if self.language_warm_task is not None and not self.language_warm_task.done():
                        self.face.set_state(event="Waiting for language model warmup")
                        await self.language_warm_task
                        self.state.language_warm_done = True
                    language_started = time.perf_counter()
                    response_text = str(await asyncio.wait_for(
                        self.run_blocking(self.agent.run, f"{text_input}{context}", model_kind="language"),
                        timeout=float(os.getenv("ROBOT_LANGUAGE_TIMEOUT_S", "45")),
                    )).strip()
                    language_ms = (time.perf_counter() - language_started) * 1000
                    self.face.set_state(reasoning=self.face.clip_text(response_text or "No response generated", 64))
                    await self.emit_face_intent(
                        world_summary=latest_world_summary,
                        reasoning=response_text or "No response generated",
                        speaking="Queued for playback" if self.body.tts_available else "Idle",
                        event="Reply ready",
                    )

                self.face.set_state(event="Updating memory")
                # Disabled for the live sample path for now: FAISS-based semantic memory
                # has been causing unstable shape/broadcast failures during interactive turns.
                #
                # embedding_started = time.perf_counter()
                # embedding_out = await self.run_blocking(self.agent.run, response_text, model_kind="embedding")
                # embedding_ms = (time.perf_counter() - embedding_started) * 1000
                # vec = np.array(embedding_out, dtype="float32").reshape(1, -1)
                # if vec.size == 0:
                #     self.face.set_state(event="Embedding unavailable")
                #     self.face.console_log.log("Empty embedding vector, skipping")
                #     continue
                # if self.index is None:
                #     self.index = faiss.IndexFlatL2(vec.shape[1])
                #     self.face.console_log.log(f"Initialized robot memory index with dim={vec.shape[1]}")
                # elif self.index.d != vec.shape[1]:
                #     self.face.console_log.log(
                #         f"Embedding dimension changed from {self.index.d} to {vec.shape[1]}; resetting memory index."
                #     )
                #     self.index = faiss.IndexFlatL2(vec.shape[1])
                #     self.stored_responses = []
                #
                # memory_context = ""
                # if self.index.ntotal > 0:
                #     _, ids = self.index.search(vec, k=1)
                #     memory_context = f" Previously: {self.stored_responses[ids[0][0]]}"
                # self.index.add(vec)
                # self.stored_responses.append(response_text)
                # self.face.set_state(memory=f"{self.index.ntotal} memories")

                memory_context = ""
                if self.state.stored_responses:
                    memory_context = self.state.latest_memory_context()
                self.state.append_response(response_text)
                self.face.set_state(memory=f"{len(self.state.stored_responses)} memories")

                augmented_text = response_text + memory_context
                turn_ms = (time.perf_counter() - turn_started) * 1000
                latency = {
                    "turn": turn_ms,
                    "stt": stt_ms,
                    "vision": vision_ms if vision_ms > 0 else None,
                    "language": language_ms,
                    "embedding": None,
                }
                self.face.set_state(latency=latency)
                self.face.console_log.log(
                    f"Latency turn={turn_ms:.0f} stt={stt_ms:.0f} vision={vision_ms:.0f} "
                    f"lang={language_ms:.0f}"
                )

                if self.body.tts_available:
                    await self.emit_output_event("speak", text=augmented_text)

                self.face.set_latest(
                    hearing=text_input,
                    vision=latest_vision_label or "-",
                    world=latest_world_summary or "-",
                    reply=augmented_text,
                )
                if live is not None:
                    panel = self.face.render_status(text_input, latest_vision_label or "-", augmented_text)
                    key = self.face.render_key(text_input, latest_vision_label or "-", augmented_text)
                    if key != last_render_key:
                        live.update(panel)
                        last_render_key = key
            except Exception as exc:
                detail = str(exc).strip() or exc.__class__.__name__
                self.face.set_state(event=f"Processing error: {detail}", reasoning="Error")
                await self.emit_face_intent(
                    world_summary=self.face.latest.get("world", "-"),
                    reasoning="Error",
                    speaking="Idle",
                    event=f"Processing error: {detail}",
                )
                self.face.set_latest(
                    hearing=self.face.latest.get("hearing", "-"),
                    vision=self.face.latest.get("vision", "-"),
                    world=self.face.latest.get("world", "-"),
                    reply=f"STT/turn failed: {detail}",
                )
                self.face.console_log.log(f"Processing error: {detail}")
                self.face.console_log.log(traceback.format_exc())
            finally:
                if item is not None:
                    self.reasoning_queue.task_done()

            await asyncio.sleep(0.1)
