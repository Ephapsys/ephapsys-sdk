#!/usr/bin/env python3

import asyncio
import json
import re
import time

from rich.console import Console, Group
from rich.columns import Columns
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
            "latency": "No turns yet",
            "event": "Starting brain",
        }
        self.latest = {"hearing": "-", "vision": "-", "reply": "-"}

    def set_state(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.ui_state and value is not None:
                self.ui_state[key] = str(value)

    def set_latest(self, hearing=None, vision=None, reply=None):
        if hearing is not None:
            self.latest["hearing"] = str(hearing)
        if vision is not None:
            self.latest["vision"] = str(vision)
        if reply is not None:
            self.latest["reply"] = str(reply)

    @staticmethod
    def clip_text(value, limit=88):
        text = str(value or "-").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"

    def format_status(self):
        if not self.agent_status.get("verified", False) and self.ui_state.get("event") in {
            "Starting brain",
            "Booting robot runtime",
            "Verifying agent",
            "Personalizing agent instance",
            "Preparing runtime bundles",
            "Agent verified",
        }:
            return "[cyan]STARTING[/cyan]"
        if self.agent_status.get("revoked", False):
            return "[red]REVOKED[/red]"
        if not self.agent_status.get("enabled", False):
            return "[red]DISABLED[/red]"
        if not self.agent_status.get("verified", False):
            return "[yellow]VERIFYING[/yellow]"
        return "[green]ENABLED[/green]"

    def snapshot(self):
        return {
            "agent_status": dict(self.agent_status),
            "ui_state": dict(self.ui_state),
            "latest": dict(self.latest),
        }

    def startup(self):
        return None

    def live(self, greeting):
        raise NotImplementedError("Only interactive robot faces implement live rendering")

    @staticmethod
    def extract_level(text):
        match = re.search(r"level\s+([0-9.]+)", str(text or ""))
        return float(match.group(1)) if match else None

    @staticmethod
    def meter(level, width=12, color="cyan"):
        if level is None:
            return f"[dim]{'·' * width}[/dim]"
        normalized = max(0.0, min(level / 0.08, 1.0))
        filled = max(1, int(round(normalized * width))) if normalized > 0 else 0
        return f"[{color}]{'█' * filled}[/][dim]{'·' * (width - filled)}[/dim]"

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
            return "[green]● ready[/green]"
        return "[cyan]◌ starting[/cyan]"


class RobotFace(RobotFaceBase):
    def __init__(self):
        super().__init__()
        self.console_live = Console(force_terminal=True)

    def render_status(self, hearing_text, vision_text, response_text):
        header = Text("ASIMOV", style="bold bright_cyan")
        header.append("  trusted multimodal robot", style="dim")
        header.append("   ")
        header.append_text(Text.from_markup(self.format_presence()))

        hearing_level = self.extract_level(self.ui_state["hearing"])
        reply_has_content = response_text and response_text != "-"
        speaking_level = 0.065 if "playing" in self.ui_state["speaking"].lower() else (
            0.04 if "queued" in self.ui_state["speaking"].lower() or "synthesizing" in self.ui_state["speaking"].lower() else None
        )

        systems = Text()
        systems.append("Hearing   ", style="bold cyan")
        systems.append(f"{self.clip_text(self.ui_state['hearing'], 58)}\n")
        systems.append("           ", style="bold cyan")
        systems.append(f"{self.meter(hearing_level, color='cyan')}\n")
        systems.append("Vision    ", style="bold green")
        systems.append(f"{self.clip_text(self.ui_state['vision'], 58)}\n")
        systems.append("Reasoning ", style="bold yellow")
        systems.append(f"{self.clip_text(self.ui_state['reasoning'], 58)}\n")
        systems.append("Speaking  ", style="bold magenta")
        systems.append(f"{self.clip_text(self.ui_state['speaking'], 58)}\n")
        systems.append("           ", style="bold magenta")
        systems.append(f"{self.meter(speaking_level, color='magenta')}")

        state = Text()
        state.append("Status    ", style="bold white")
        state.append(f"{self.format_status()}\n")
        state.append("Event     ", style="bold white")
        state.append(f"{self.clip_text(self.ui_state['event'], 48)}\n")
        state.append("Memory    ", style="bold white")
        state.append(f"{self.clip_text(self.ui_state['memory'], 48)}\n")
        state.append("Latency   ", style="bold bright_white")
        state.append(f"{self.clip_text(self.ui_state['latency'], 48)}")

        input_panel = Text()
        input_panel.append("Input stream\n", style="bold cyan")
        input_panel.append("Heard     ", style="bold cyan")
        input_panel.append(f"{self.clip_text(hearing_text or '-', 44)}\n")
        input_panel.append("Vision    ", style="bold green")
        input_panel.append(f"{self.clip_text(vision_text or '-', 44)}")

        output_panel = Text()
        output_panel.append("Output stream\n", style="bold magenta")
        output_panel.append("Reply     ", style="bold yellow")
        output_panel.append(self.clip_text(response_text or "-", 44))
        if reply_has_content:
            output_panel.append("\nSignal    ", style="bold magenta")
            output_panel.append(self.meter(0.07 if "playing" in self.ui_state["speaking"].lower() else 0.04, color="magenta"))

        footer = Text()
        footer.append("Ctrl+C", style="bold")
        footer.append(" to exit", style="dim")

        return Panel(
            Group(
                header,
                Text(""),
                Columns(
                    [
                        Panel(systems, title="Body", border_style="cyan", padding=(1, 1)),
                        Panel(state, title="Brain", border_style="yellow", padding=(1, 1)),
                    ],
                    equal=True,
                    expand=True,
                ),
                Text(""),
                Columns(
                    [
                        Panel(input_panel, title="Perception", border_style="green", padding=(1, 1)),
                        Panel(output_panel, title="Expression", border_style="magenta", padding=(1, 1)),
                    ],
                    equal=True,
                    expand=True,
                ),
                Text(""),
                footer,
            ),
            title="Robot Console",
            border_style="bright_blue",
            padding=(1, 2),
        )

    def render_key(self, hearing_text, vision_text, response_text):
        return (
            int(time.time() * 4),
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
            refresh_per_second=4,
            console=self.console_live,
            screen=True,
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
