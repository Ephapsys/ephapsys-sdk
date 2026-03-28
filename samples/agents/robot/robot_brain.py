#!/usr/bin/env python3

import asyncio
import os
import sys
import time

import faiss
import numpy as np
from ephapsys.agent import TrustedAgent


class RobotBrain:
    def __init__(self, face, body, shutdown_event):
        self.face = face
        self.body = body
        self.shutdown_event = shutdown_event
        self.agent = TrustedAgent.from_env()
        self.index = faiss.IndexFlatL2(768)
        self.stored_responses = []

    def build_startup_greeting(self):
        vision_label = None
        try:
            self.face.set_state(vision="Looking for a first impression", reasoning="Observing the scene")
            frame = self.body.capture_startup_frame()
            if frame is not None:
                vision_raw = self.agent.run(frame, model_kind="vision")
                vision_label = str(vision_raw).strip() if vision_raw is not None else None
        except Exception as exc:
            self.face.console_log.log(f"Startup vision greeting fallback: {exc}")

        prompt = (
            "You are Asimov, a trusted multimodal robot. "
            "Generate one short natural spoken greeting for the person in front of you. "
            "Keep it under 18 words, warm, intelligent, and do not mention model names."
        )
        if vision_label:
            prompt += f" You currently see: {vision_label}."
        else:
            prompt += " You do not yet have a clear visual classification."

        try:
            greeting = str(self.agent.run(prompt, model_kind='language')).strip()
        except Exception as exc:
            self.face.console_log.log(f"Startup language greeting fallback: {exc}")
            greeting = ""

        greeting = greeting or "Hello there."
        greeting = " ".join(greeting.split())
        return greeting, vision_label

    async def startup(self):
        self.face.startup()
        self.face.set_state(
            hearing="Stand by",
            vision="Stand by",
            reasoning="Verifying agent",
            speaking="Stand by",
            event="Starting brain",
        )
        try:
            self.face.set_state(event="Verifying agent", reasoning="Checking trusted state")
            ok, _ = self.agent.verify()
        except RuntimeError as exc:
            if "404" in str(exc):
                self.face.console_live.print(f"[red]❌ Agent template '{self.agent.agent_id}' not found in backend.[/red]")
                self.face.console_live.print("[yellow]Please create it in the AOC before running this sample.[/yellow]")
                sys.exit(1)
            raise

        if not ok:
            status = self.agent.get_status()
            is_personalized = status.get("state", {}).get("personalized", False) or status.get("personalized", False)
            if not is_personalized:
                anchor = os.getenv("PERSONALIZE_ANCHOR")
                self.face.set_state(event="Personalizing agent instance", reasoning="Binding device identity")
                self.face.console_live.print(
                    f"[yellow]Agent not personalized; running personalize(anchor={anchor})...[/yellow]"
                )
                self.agent.personalize(anchor=anchor)
                self.face.console_live.print("[green]✅ Agent personalized (instance registered in AOC).[/green]")
                for _ in range(5):
                    ok, _ = self.agent.verify()
                    if ok:
                        break
                    self.face.console_live.print("[yellow]...waiting for agent to become ready...[/yellow]")
                    time.sleep(1)
            if not ok:
                self.face.console_live.print("[red]❌ Agent not ready after personalization.[/red]")
                sys.exit(1)

        status = self.agent.get_status()
        is_enabled = status.get("enabled", False) or (status.get("status", "").lower() == "enabled")
        is_revoked = status.get("state", {}).get("revoked", False)
        self.face.agent_status.update({"verified": ok, "enabled": is_enabled, "revoked": is_revoked})
        self.face.set_state(event="Agent verified", reasoning="Trusted runtime ready")
        self.face.console_live.print("[green]✅ Agent personalized and verified.[/green]")
        self.face.console_live.print(f"[dim]Instance DID: {self.agent.agent_id}[/dim]")

        self.face.set_state(event="Preparing runtime bundles", reasoning="Loading secure model runtimes")
        runtimes = self.agent.prepare_runtime()
        tts_path = (runtimes.get("tts") or {}).get("model_path")
        self.body.tts_available = self.body.ensure_preprocessor(tts_path) if tts_path else False
        self.face.set_state(
            hearing="Listening on microphone",
            vision="Scanning scene",
            reasoning="Waiting for input",
            speaking="Ready" if self.body.tts_available else "Unavailable",
            memory="0 memories",
            latency={"turn": None, "stt": None, "vision": None, "language": None, "embedding": None, "tts": None},
            event=f"Runtime ready: {', '.join(sorted(runtimes.keys()))}",
        )
        self.face.console_live.print(
            f"[green]✅ Runtime prepared[/green] "
            f"(voice={'ready' if self.body.tts_available else 'unavailable'}, models={', '.join(sorted(runtimes.keys()))})"
        )

        self.face.set_state(event="Composing greeting", reasoning="Generating first response")
        greeting, startup_vision = self.build_startup_greeting()
        self.face.set_latest("-", startup_vision or "-", greeting)
        if startup_vision:
            self.face.set_state(vision=self.face.clip_text(startup_vision, 64))
        self.face.console_live.print(f"[cyan]🤖 Greeting: {greeting}[/cyan]")
        try:
            if self.face.agent_status.get("enabled", False) and not self.face.agent_status.get("revoked", False) and self.body.tts_available:
                self.face.console_live.print("[green]✅ GREETING TTS...")
                self.body.play_tts_sync(self.agent, greeting)
                self.face.console_live.print("[green]✅ GREETING TTS done.")
            else:
                self.face.console_live.print("[red]❌ Skipping greeting TTS (agent not enabled or TTS assets missing)")
        except Exception as exc:
            reason = str(exc)
            if "upgrade torch to at least v2.6" in reason:
                self.face.console_log.log(
                    "[red]⚠️ Greeting TTS unavailable: this environment needs torch>=2.6 for secure .pt loading.[/red]"
                )
                self.face.console_log.log(f"[dim]{reason}[/dim]")
            else:
                self.face.console_log.log(f"[red]⚠️ Greeting TTS failed:[/red] {reason}")

        self.face.set_state(
            hearing="Listening on microphone",
            vision="Scanning scene",
            reasoning="Waiting for input",
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
                ok, _ = self.agent.verify()
                status = self.agent.get_status()
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

    async def process_task(self, live=None):
        last_render_key = None
        while not self.shutdown_event.is_set():
            mic_audio = None
            cam_frame = None

            if not self.face.agent_status.get("enabled", False) or self.face.agent_status.get("revoked", False):
                self.face.set_state(event="Agent disabled or revoked", reasoning="Paused", speaking="Muted")
                self.face.set_latest("-", "-", "-")
                if live is not None:
                    panel = self.face.render_status("-", "-", "-")
                    key = self.face.render_key("-", "-", "-")
                    if key != last_render_key:
                        live.update(panel, refresh=True)
                        last_render_key = key
                await asyncio.sleep(1)
                continue

            try:
                mic_audio = self.body.mic_queue.get_nowait()
            except Exception:
                pass
            try:
                cam_frame = self.body.cam_queue.get_nowait()
            except Exception:
                pass

            if mic_audio is None and cam_frame is None:
                self.face.set_state(vision="Scanning", reasoning="Waiting for input")
                self.face.set_latest("-", "-", "-")
                if live is not None:
                    panel = self.face.render_status("-", "-", "-")
                    key = self.face.render_key("-", "-", "-")
                    if key != last_render_key:
                        live.update(panel)
                        last_render_key = key
                await asyncio.sleep(0.2)
                continue

            try:
                turn_started = time.perf_counter()
                stt_ms = 0
                vision_ms = 0
                language_ms = 0
                embedding_ms = 0

                self.face.set_state(hearing="Transcribing speech", event="Processing microphone input")
                stt_started = time.perf_counter()
                stt_audio = self.body.audio_to_wav_bytes(mic_audio)
                text_input = self.agent.run(stt_audio, model_kind="stt")
                stt_ms = (time.perf_counter() - stt_started) * 1000
                self.face.set_state(hearing=self.face.clip_text(text_input or "No speech detected", 64))

                vision_label = None
                if cam_frame is not None:
                    self.face.set_state(vision="Analyzing scene", event="Running vision model")
                    vision_started = time.perf_counter()
                    vision_raw = self.agent.run(cam_frame, model_kind="vision")
                    vision_ms = (time.perf_counter() - vision_started) * 1000
                    vision_label = str(vision_raw).strip() if vision_raw is not None else None
                    self.face.set_state(vision=self.face.clip_text(vision_label or "No scene update", 64))

                context = f"(vision={vision_label})" if vision_label else ""
                self.face.set_state(reasoning="Composing response", event="Running language model")
                language_started = time.perf_counter()
                response_text = str(self.agent.run(f"{text_input} {context}", model_kind="language")).strip()
                language_ms = (time.perf_counter() - language_started) * 1000
                self.face.set_state(reasoning=self.face.clip_text(response_text or "No response generated", 64))

                self.face.set_state(event="Updating memory")
                embedding_started = time.perf_counter()
                embedding_out = self.agent.run(response_text, model_kind="embedding")
                embedding_ms = (time.perf_counter() - embedding_started) * 1000
                vec = np.array(embedding_out, dtype="float32").reshape(1, -1)
                if vec.size == 0:
                    self.face.set_state(event="Embedding unavailable")
                    self.face.console_log.log("Empty embedding vector, skipping")
                    continue

                memory_context = ""
                if self.index.ntotal > 0:
                    _, ids = self.index.search(vec, k=1)
                    memory_context = f" Previously: {self.stored_responses[ids[0][0]]}"
                self.index.add(vec)
                self.stored_responses.append(response_text)
                self.face.set_state(memory=f"{self.index.ntotal} memories")

                augmented_text = response_text + memory_context
                turn_ms = (time.perf_counter() - turn_started) * 1000
                latency = {
                    "turn": turn_ms,
                    "stt": stt_ms,
                    "vision": vision_ms if vision_ms > 0 else None,
                    "language": language_ms,
                    "embedding": embedding_ms,
                }
                self.face.set_state(latency=latency)
                self.face.console_log.log(
                    f"Latency turn={turn_ms:.0f} stt={stt_ms:.0f} vision={vision_ms:.0f} "
                    f"lang={language_ms:.0f} embed={embedding_ms:.0f}"
                )

                if self.body.tts_available:
                    if self.body.tts_queue.qsize() < 3:
                        self.face.set_state(speaking="Queued for playback", event="Reply ready")
                        await self.body.tts_queue.put(augmented_text)
                    else:
                        self.face.set_state(speaking="Queue saturated", event="Dropping speech playback")
                        self.face.console_log.log("TTS queue full; dropping audio playback to stay responsive.")

                self.face.set_latest(text_input, vision_label or "-", augmented_text)
                if live is not None:
                    panel = self.face.render_status(text_input, vision_label or "-", augmented_text)
                    key = self.face.render_key(text_input, vision_label or "-", augmented_text)
                    if key != last_render_key:
                        live.update(panel)
                        last_render_key = key
            except Exception as exc:
                self.face.set_state(event=f"Processing error: {exc}", reasoning="Error")
                self.face.console_log.log(f"Processing error: {exc}")

            await asyncio.sleep(0.1)
