#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RobotStateStore:
    startup_vision_label: str = "-"
    latest_world_summary: str = "-"
    latest_scene_summary: str = "-"
    latest_camera_frame: Optional[object] = None
    prev_world_embedding: Optional[object] = None
    stored_responses: List[str] = field(default_factory=list)
    language_warm_done: bool = False
    awaiting_tts_done: bool = False

    def append_response(self, text: str, limit: int = 8) -> None:
        self.stored_responses.append(text)
        self.stored_responses = self.stored_responses[-limit:]

    def latest_memory_context(self) -> str:
        if not self.stored_responses:
            return ""
        return f" Previously: {self.stored_responses[-1]}"
