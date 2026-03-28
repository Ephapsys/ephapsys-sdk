#!/usr/bin/env python3

import asyncio
import json

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
            "hearing": "Idle",
            "vision": "Standing by",
            "reasoning": "Waiting for input",
            "speaking": "Silent",
            "memory": "0 memories",
            "latency": "No turns yet",
            "event": "Booting robot runtime",
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
            "Booting robot runtime",
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


class RobotFace(RobotFaceBase):
    def __init__(self):
        super().__init__()
        self.console_live = Console(force_terminal=True)

    def render_status(self, hearing_text, vision_text, response_text):
        header = Text("ASIMOV", style="bold bright_cyan")
        header.append("  trusted multimodal robot", style="dim")

        body = Text()
        body.append("Hearing   ", style="bold cyan")
        body.append(f"{self.clip_text(self.ui_state['hearing'], 72)}\n")
        body.append("Vision    ", style="bold green")
        body.append(f"{self.clip_text(self.ui_state['vision'], 72)}\n")
        body.append("Reasoning ", style="bold yellow")
        body.append(f"{self.clip_text(self.ui_state['reasoning'], 72)}\n")
        body.append("Speaking  ", style="bold magenta")
        body.append(f"{self.clip_text(self.ui_state['speaking'], 72)}\n")
        body.append("Memory    ", style="bold white")
        body.append(f"{self.clip_text(self.ui_state['memory'], 72)}\n")
        body.append("Latency   ", style="bold bright_white")
        body.append(f"{self.clip_text(self.ui_state['latency'], 72)}\n")
        body.append("Status    ", style="bold white")
        body.append(f"{self.format_status()}\n")
        body.append("Event     ", style="bold white")
        body.append(self.clip_text(self.ui_state["event"], 72), style="dim")

        latest = Text()
        latest.append("Latest Hearing  ", style="bold cyan")
        latest.append(f"{self.clip_text(hearing_text or '-', 84)}\n")
        latest.append("Latest Vision   ", style="bold green")
        latest.append(f"{self.clip_text(vision_text or '-', 84)}\n")
        latest.append("Latest Reply    ", style="bold yellow")
        latest.append(self.clip_text(response_text or "-", 84))

        footer = Text()
        footer.append("Ctrl+C", style="bold")
        footer.append(" to exit", style="dim")

        return Panel(
            Group(header, Text(""), body, Text(""), latest, Text(""), footer),
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
