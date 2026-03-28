#!/usr/bin/env python3

from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, Optional


@dataclass
class RobotFact:
    fact_type: str
    timestamp: float = field(default_factory=time)
    source: str = "brain"


@dataclass
class SpeechFact(RobotFact):
    transcript: str = ""
    stt_ms: float = 0.0
    heard_summary: str = ""
    source: str = "microphone"
    fact_type: str = "speech_fact"


@dataclass
class VisionFact(RobotFact):
    frame: Any = None
    vision_label: str = "-"
    world_summary: str = "-"
    vision_ms: float = 0.0
    source: str = "camera"
    fact_type: str = "vision_fact"


@dataclass
class SystemFact(RobotFact):
    name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    fact_type: str = "system_fact"


@dataclass
class RobotAction:
    action_type: str
    source: str = "brain"
    priority: int = 0
    interruptible: bool = True
    coalesce_key: Optional[str] = None


@dataclass
class SpeakAction(RobotAction):
    text: str = ""
    source: str = "reasoning"
    priority: int = 10
    interruptible: bool = False
    coalesce_key: Optional[str] = None
    action_type: str = "speak"


@dataclass
class BodyAction(RobotAction):
    action: str = "idle"
    source: str = "world"
    priority: int = 4
    interruptible: bool = True
    coalesce_key: Optional[str] = "body_control"
    action_type: str = "body_control"


@dataclass
class FaceAction(RobotAction):
    expression: str = "neutral"
    gaze: str = "center"
    source: str = "brain"
    priority: int = 3
    interruptible: bool = True
    coalesce_key: Optional[str] = "face_control"
    action_type: str = "face_control"


@dataclass
class ToolAction(RobotAction):
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    source: str = "reasoning"
    priority: int = 6
    interruptible: bool = True
    coalesce_key: Optional[str] = None
    action_type: str = "tool_action"
