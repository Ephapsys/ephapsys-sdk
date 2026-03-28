#!/usr/bin/env python3

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class RobotIntent:
    kind: str
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    requires_approval: bool = False


@dataclass
class GovernorDecision:
    allowed: bool
    reason: str


class RobotGovernor:
    def __init__(self, allow_body_control: bool = True, allow_tools: bool = True):
        self.allow_body_control = allow_body_control
        self.allow_tools = allow_tools

    def approve(self, intent: RobotIntent) -> GovernorDecision:
        if intent.kind == "face_control":
            return GovernorDecision(True, "Face control allowed")
        if intent.kind == "body_control":
            if self.allow_body_control:
                return GovernorDecision(True, "Body control allowed")
            return GovernorDecision(False, "Body control blocked")
        if intent.kind == "tool":
            tool_name = str(intent.payload.get("tool", ""))
            if not self.allow_tools:
                return GovernorDecision(False, "Tool use blocked")
            if tool_name in {"current_time"}:
                return GovernorDecision(True, "Low-risk tool allowed")
            if intent.requires_approval:
                return GovernorDecision(False, "Tool requires approval")
            return GovernorDecision(False, f"Tool '{tool_name}' not allowlisted")
        if intent.kind == "speak":
            return GovernorDecision(True, "Speech allowed")
        return GovernorDecision(True, "Allowed")

    def should_defer(self, intent: RobotIntent, *, speaking_active: bool = False) -> GovernorDecision:
        if not speaking_active:
            return GovernorDecision(False, "No active speech")
        if intent.kind in {"face_control", "body_control"}:
            return GovernorDecision(False, "Background control may continue while speaking")
        if intent.kind == "tool":
            return GovernorDecision(True, "Tool execution deferred while speaking")
        if intent.kind == "speak":
            return GovernorDecision(True, "Speech output serialized")
        return GovernorDecision(False, "No defer rule matched")


class RobotToolbox:
    def execute(self, intent: RobotIntent) -> Optional[str]:
        tool_name = str(intent.payload.get("tool", ""))
        if tool_name == "current_time":
            now = datetime.now().strftime("%I:%M %p")
            return f"The current local time is {now}."
        return None


def classify_tool_intent(text: str) -> Optional[RobotIntent]:
    lowered = (text or "").strip().lower()
    if not lowered:
        return None
    if "what time is it" in lowered or lowered in {"time", "current time"}:
        return RobotIntent(kind="tool", source="speech", payload={"tool": "current_time"})
    return None


def body_intent_for_world(world_summary: str) -> RobotIntent:
    summary = (world_summary or "").lower()
    if "person moving" in summary:
        return RobotIntent(kind="body_control", source="world", payload={"action": "track_person"})
    if "person present" in summary:
        return RobotIntent(kind="body_control", source="world", payload={"action": "look_at_person"})
    if "movement detected" in summary:
        return RobotIntent(kind="body_control", source="world", payload={"action": "orient_to_motion"})
    return RobotIntent(kind="body_control", source="world", payload={"action": "scan_scene"})


def face_intent_for_state(*, world_summary: str = "", reasoning: str = "", speaking: str = "", event: str = "") -> RobotIntent:
    world = (world_summary or "").lower()
    reasoning_text = (reasoning or "").lower()
    speaking_text = (speaking or "").lower()
    event_text = (event or "").lower()

    expression = "neutral"
    gaze = "center"

    if "greeting" in event_text:
        expression = "warm"
        gaze = "engage"
    elif "thinking" in speaking_text or "composing" in reasoning_text or "running language" in event_text:
        expression = "thinking"
        gaze = "steady"
    elif "transcribing" in reasoning_text or "processing microphone" in event_text:
        expression = "listening_focus"
        gaze = "attentive"
    elif "speaking" in event_text or "synthesizing" in speaking_text or "queued" in speaking_text:
        expression = "speaking"
        gaze = "engage"
    elif "person moving" in world:
        expression = "attentive"
        gaze = "track_person"
    elif "person present" in world:
        expression = "attentive"
        gaze = "look_at_person"
    elif "movement detected" in world:
        expression = "alert"
        gaze = "orient_to_motion"

    return RobotIntent(
        kind="face_control",
        source="brain",
        payload={"expression": expression, "gaze": gaze},
        priority=1,
    )
