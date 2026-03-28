#!/usr/bin/env python3

import asyncio
import json
import re
import time

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


class SilentConsole:
    def print(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None


class RobotFaceBase:
    def __init__(self):
        self.console_log = Console(stderr=True)
        self.agent_status = {"verified": False, "enabled": False, "revoked": False}
        self.ui_state = {
            "hearing": "Stand by",
            "vision": "Stand by",
            "reasoning": "Starting brain",
            "speaking": "Stand by",
            "memory": "0 memories",
            "latency": {
                "turn": None,
                "stt": None,
                "vision": None,
                "language": None,
                "embedding": None,
                "tts": None,
            },
            "event": "Starting brain",
        }
        self.latest = {"hearing": "-", "vision": "-", "reply": "-"}
        self.activity_log = []

    def add_activity(self, label, value, limit=12):
        entry = f"{label}: {self.clip_text(value, 100)}"
        if self.activity_log and self.activity_log[-1] == entry:
            return
        self.activity_log.append(entry)
        if len(self.activity_log) > limit:
            self.activity_log = self.activity_log[-limit:]

    def set_state(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.ui_state and value is not None:
                previous = self.ui_state.get(key)
                if key == "latency" and isinstance(value, dict):
                    merged = dict(self.ui_state.get("latency", {}))
                    merged.update(value)
                    self.ui_state[key] = merged
                else:
                    self.ui_state[key] = str(value)
                if key == "event" and str(value) != str(previous):
                    self.add_activity("Event", value)

    def set_latest(self, hearing=None, vision=None, reply=None):
        if hearing is not None:
            hearing = str(hearing)
            if hearing != self.latest["hearing"] and hearing != "-":
                self.add_activity("Heard", hearing)
            self.latest["hearing"] = hearing
        if vision is not None:
            vision = str(vision)
            if vision != self.latest["vision"] and vision != "-":
                self.add_activity("Vision", vision)
            self.latest["vision"] = vision
        if reply is not None:
            reply = str(reply)
            if reply != self.latest["reply"] and reply != "-":
                self.add_activity("Reply", reply)
            self.latest["reply"] = reply

    @staticmethod
    def clip_text(value, limit=88):
        text = str(value or "-").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"

    @staticmethod
    def inline_text(value, width):
        return RobotFaceBase.clip_text(value, width).ljust(width)

    def format_status(self):
        if not self.agent_status.get("verified", False) and self.ui_state.get("event") in {
            "Starting brain",
            "Booting robot runtime",
            "Verifying agent",
            "Personalizing agent instance",
            "Preparing runtime bundles",
            "Agent verified",
        }:
            return ("STARTING", "cyan")
        if self.agent_status.get("revoked", False):
            return ("REVOKED", "red")
        if not self.agent_status.get("enabled", False):
            return ("DISABLED", "red")
        if not self.agent_status.get("verified", False):
            return ("VERIFYING", "yellow")
        return ("ENABLED", "green")

    def snapshot(self):
        return {
            "agent_status": dict(self.agent_status),
            "ui_state": {
                **self.ui_state,
                "latency": dict(self.ui_state.get("latency", {})),
            },
            "latest": dict(self.latest),
            "activity_log": list(self.activity_log),
        }

    def startup(self):
        return None

    def live(self, greeting):
        raise NotImplementedError("Only interactive robot faces implement live rendering")

    @staticmethod
    def pulse(frames, speed=6):
        idx = int(time.time() * speed) % len(frames)
        return frames[idx]

    def format_presence(self):
        hearing = self.ui_state.get("hearing", "").lower()
        speaking = self.ui_state.get("speaking", "").lower()
        reasoning = self.ui_state.get("reasoning", "").lower()
        if "playing" in speaking or "synthesizing" in speaking or "queued" in speaking:
            return f"[magenta]{self.pulse(['◜', '◠', '◝', '◞', '◡', '◟'], 10)} speaking[/magenta]"
        if "transcribing" in hearing or "speech detected" in hearing or "capturing" in self.ui_state.get("event", "").lower():
            return f"[cyan]{self.pulse(['▖', '▘', '▝', '▗'], 12)} listening[/cyan]"
        if "composing" in reasoning or "running language" in self.ui_state.get("event", "").lower():
            return f"[yellow]{self.pulse(['◴', '◷', '◶', '◵'], 8)} thinking[/yellow]"
        if self.agent_status.get("enabled", False):
            return "[green]● online[/green]"
        return "[cyan]◌ starting[/cyan]"

    def latency_text(self):
        latency = self.ui_state.get("latency", {})
        if not isinstance(latency, dict):
            return Text(str(latency), style="dim")
        text = Text()
        for label, key, style in (
            ("Turn", "turn", "bright_white"),
            ("STT", "stt", "cyan"),
            ("Vision", "vision", "green"),
            ("Language", "language", "yellow"),
            ("Embed", "embedding", "white"),
            ("TTS", "tts", "magenta"),
        ):
            value = latency.get(key)
            text.append(f"{label:<9}", style=f"bold {style}" if style != "white" else "bold white")
            if value is None:
                text.append("—\n", style="dim")
            else:
                text.append(f"{int(value)} ms\n", style=style)
        return text


class RobotFace(RobotFaceBase):
    def __init__(self):
        super().__init__()
        self.console_live = Console(force_terminal=True)

    def render_status(self, hearing_text, vision_text, response_text):
        status_width = 88
        latest_width = 100
        activity_width = 110

        header = Text("ASIMOV", style="bold bright_cyan")
        header.append("  trusted multimodal robot", style="dim")
        header.append("   ")
        header.append_text(Text.from_markup(self.format_presence()))

        state = Text()
        status_label, status_style = self.format_status()
        state.append("Status    ", style="bold white")
        state.append(status_label, style=status_style)
        state.append("\n")
        state.append("Listening ", style="bold cyan")
        state.append(f"{self.inline_text(self.ui_state['hearing'], status_width)}\n")
        state.append("Vision    ", style="bold green")
        state.append(f"{self.inline_text(self.ui_state['vision'], status_width)}\n")
        state.append("Reasoning ", style="bold yellow")
        state.append(f"{self.inline_text(self.ui_state['reasoning'], status_width)}\n")
        state.append("Speaking  ", style="bold magenta")
        state.append(f"{self.inline_text(self.ui_state['speaking'], status_width)}\n")
        state.append("Event     ", style="bold white")
        state.append(f"{self.inline_text(self.ui_state['event'], status_width)}\n")
        state.append("Memory    ", style="bold white")
        state.append(f"{self.inline_text(self.ui_state['memory'], 24)}\n")
        turn = self.ui_state.get("latency", {}).get("turn")
        stt = self.ui_state.get("latency", {}).get("stt")
        language = self.ui_state.get("latency", {}).get("language")
        vision = self.ui_state.get("latency", {}).get("vision")
        tts = self.ui_state.get("latency", {}).get("tts")
        state.append("Latency   ", style="bold bright_white")
        if all(v is None for v in (turn, stt, language, vision, tts)):
            state.append("No turns yet", style="dim")
        else:
            parts = []
            if turn is not None:
                parts.append(f"turn {int(turn)}ms")
            if stt is not None:
                parts.append(f"stt {int(stt)}")
            if language is not None:
                parts.append(f"lang {int(language)}")
            if vision is not None:
                parts.append(f"vision {int(vision)}")
            if tts is not None:
                parts.append(f"tts {int(tts)}")
            state.append(" | ".join(parts))

        latest = Text()
        latest.append("Heard  ", style="bold cyan")
        latest.append(f"{self.inline_text(hearing_text or '-', latest_width)}\n")
        latest.append("Vision ", style="bold green")
        latest.append(f"{self.inline_text(vision_text or '-', latest_width)}\n")
        latest.append("Reply  ", style="bold yellow")
        latest.append(f"{self.inline_text(response_text or '-', latest_width)}")

        activity = Text()
        if self.activity_log:
            for entry in self.activity_log[-8:]:
                activity.append("• ", style="dim")
                activity.append(f"{self.inline_text(entry, activity_width)}\n")
        else:
            activity.append(self.inline_text("No activity yet", activity_width), style="dim")

        footer = Text()
        footer.append("Ctrl+C", style="bold")
        footer.append(" to exit", style="dim")

        return Panel(
            Group(
                header,
                Text(""),
                Panel(state, title="Status", border_style="cyan", padding=(1, 1)),
                Text(""),
                Panel(latest, title="Latest", border_style="green", padding=(1, 1)),
                Text(""),
                Panel(activity, title="Recent Activity", border_style="yellow", padding=(1, 1)),
                Text(""),
                footer,
            ),
            title="Robot Console",
            border_style="bright_blue",
            padding=(1, 2),
        )

    def render_key(self, hearing_text, vision_text, response_text):
        return (
            hearing_text or "-",
            vision_text or "-",
            response_text or "-",
            self.format_status(),
            self.ui_state["hearing"],
            self.ui_state["vision"],
            self.ui_state["reasoning"],
            self.ui_state["speaking"],
            self.ui_state["memory"],
            self.ui_state["latency"],
            self.ui_state["event"],
        )

    def startup(self):
        self.console_live.print(Panel.fit("Asimov local runtime startup", border_style="bright_blue"))
        self.console_live.print("[bold cyan]Step 1[/bold cyan] Verify agent and prepare runtime")

    def live(self, greeting):
        return Live(
            self.render_status("-", "-", greeting),
            refresh_per_second=2,
            console=self.console_live,
            screen=False,
            auto_refresh=False,
        )


class RobotStateFace(RobotFaceBase):
    def __init__(self):
        super().__init__()
        self.console_live = SilentConsole()


async def run_terminal_face(ws_url: str):
    import websockets

    face = RobotFace()
    face.startup()
    last_key = None

    try:
        async with websockets.connect(ws_url, max_size=None) as websocket:
            first = json.loads(await websocket.recv())
            snapshot = first.get("snapshot", first)
            greeting = snapshot.get("latest", {}).get("reply") or "Asimov is starting..."
            with face.live(greeting) as live:
                while True:
                    payload = json.loads(await websocket.recv())
                    snapshot = payload.get("snapshot", payload)
                    face.agent_status.update(snapshot.get("agent_status", {}))
                    face.ui_state.update(snapshot.get("ui_state", {}))
                    face.latest.update(snapshot.get("latest", {}))
                    face.activity_log = snapshot.get("activity_log", [])
                    key = face.render_key(
                        face.latest.get("hearing", "-"),
                        face.latest.get("vision", "-"),
                        face.latest.get("reply", "-"),
                    )
                    if key != last_key:
                        live.update(
                            face.render_status(
                                face.latest.get("hearing", "-"),
                                face.latest.get("vision", "-"),
                                face.latest.get("reply", "-"),
                            ),
                            refresh=True,
                        )
                        last_key = key
    except (asyncio.CancelledError, KeyboardInterrupt):
        return
    except Exception:
        raise
