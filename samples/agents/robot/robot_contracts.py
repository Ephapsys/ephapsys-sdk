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


ROBOT_PUBLIC_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "speech_fact": {
        "type": "object",
        "required": ["fact_type", "timestamp", "source", "transcript", "stt_ms", "heard_summary"],
        "properties": {
            "fact_type": {"const": "speech_fact"},
            "timestamp": {"type": "number"},
            "source": {"type": "string"},
            "transcript": {"type": "string"},
            "stt_ms": {"type": "number"},
            "heard_summary": {"type": "string"},
        },
    },
    "vision_fact": {
        "type": "object",
        "required": ["fact_type", "timestamp", "source", "vision_label", "world_summary", "vision_ms"],
        "properties": {
            "fact_type": {"const": "vision_fact"},
            "timestamp": {"type": "number"},
            "source": {"type": "string"},
            "vision_label": {"type": "string"},
            "world_summary": {"type": "string"},
            "vision_ms": {"type": "number"},
        },
    },
    "system_fact": {
        "type": "object",
        "required": ["fact_type", "timestamp", "source", "name", "payload"],
        "properties": {
            "fact_type": {"const": "system_fact"},
            "timestamp": {"type": "number"},
            "source": {"type": "string"},
            "name": {"type": "string"},
            "payload": {"type": "object"},
        },
    },
    "speak_action": {
        "type": "object",
        "required": ["action_type", "source", "priority", "interruptible", "text"],
        "properties": {
            "action_type": {"const": "speak"},
            "source": {"type": "string"},
            "priority": {"type": "integer"},
            "interruptible": {"type": "boolean"},
            "text": {"type": "string"},
        },
    },
    "body_action": {
        "type": "object",
        "required": ["action_type", "source", "priority", "interruptible", "action"],
        "properties": {
            "action_type": {"const": "body_control"},
            "source": {"type": "string"},
            "priority": {"type": "integer"},
            "interruptible": {"type": "boolean"},
            "action": {"type": "string"},
        },
    },
    "face_action": {
        "type": "object",
        "required": ["action_type", "source", "priority", "interruptible", "expression", "gaze"],
        "properties": {
            "action_type": {"const": "face_control"},
            "source": {"type": "string"},
            "priority": {"type": "integer"},
            "interruptible": {"type": "boolean"},
            "expression": {"type": "string"},
            "gaze": {"type": "string"},
        },
    },
    "tool_action": {
        "type": "object",
        "required": ["action_type", "source", "priority", "interruptible", "tool_name", "arguments"],
        "properties": {
            "action_type": {"const": "tool_action"},
            "source": {"type": "string"},
            "priority": {"type": "integer"},
            "interruptible": {"type": "boolean"},
            "tool_name": {"type": "string"},
            "arguments": {"type": "object"},
        },
    },
    "state_snapshot": {
        "type": "object",
        "required": ["agent_status", "ui_state", "latest", "activity_log"],
        "properties": {
            "agent_status": {
                "type": "object",
                "required": ["verified", "enabled", "revoked"],
                "properties": {
                    "verified": {"type": "boolean"},
                    "enabled": {"type": "boolean"},
                    "revoked": {"type": "boolean"},
                },
            },
            "ui_state": {
                "type": "object",
                "required": [
                    "hearing",
                    "vision",
                    "world",
                    "expression",
                    "gaze",
                    "body",
                    "tools",
                    "governor",
                    "reasoning",
                    "speaking",
                    "memory",
                    "latency",
                    "event",
                ],
                "properties": {
                    "hearing": {"type": "string"},
                    "vision": {"type": "string"},
                    "world": {"type": "string"},
                    "expression": {"type": "string"},
                    "gaze": {"type": "string"},
                    "body": {"type": "string"},
                    "tools": {"type": "string"},
                    "governor": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "speaking": {"type": "string"},
                    "memory": {"type": "string"},
                    "latency": {"type": "object"},
                    "event": {"type": "string"},
                },
            },
            "latest": {
                "type": "object",
                "required": ["hearing", "vision", "world", "reply"],
                "properties": {
                    "hearing": {"type": "string"},
                    "vision": {"type": "string"},
                    "world": {"type": "string"},
                    "reply": {"type": "string"},
                },
            },
            "activity_log": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    },
    "ws_state_envelope": {
        "type": "object",
        "required": ["snapshot"],
        "properties": {
            "snapshot": {"$ref": "#/definitions/state_snapshot"},
        },
        "definitions": {},
    },
}

ROBOT_PUBLIC_SCHEMAS["ws_state_envelope"]["definitions"]["state_snapshot"] = ROBOT_PUBLIC_SCHEMAS["state_snapshot"]


def public_schema_bundle() -> Dict[str, Any]:
    return {
        "schema_version": "robot-public-v1",
        "schemas": ROBOT_PUBLIC_SCHEMAS,
    }
