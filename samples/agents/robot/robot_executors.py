#!/usr/bin/env python3

from robot_contracts import BodyAction, FaceAction, SpeakAction, ToolAction


class ToolExecutor:
    def __init__(self, toolbox, face, governor, decision_callback):
        self.toolbox = toolbox
        self.face = face
        self.governor = governor
        self._decision_callback = decision_callback

    async def execute(self, intent):
        decision = self.governor.approve(intent)
        self._decision_callback(decision)
        tool_name = str(intent.payload.get("tool", "tool"))
        if not decision.allowed:
            self.face.set_state(tools=f"Blocked: {tool_name}")
            return f"I am not allowed to use the {tool_name} tool right now."
        self.face.set_state(tools=f"Running {tool_name}")
        result = self.toolbox.execute(intent)
        self.face.set_state(tools=f"Completed {tool_name}")
        return result


class BodyController:
    def __init__(self, channel, face):
        self.channel = channel
        self.face = face

    async def execute(self, action: BodyAction):
        self.face.set_state(body=str(action.action))
        await self.channel.send_command("body_control", action=action.action)


class FaceController:
    def __init__(self, face):
        self.face = face

    async def execute(self, action: FaceAction):
        self.face.set_state(expression=action.expression, gaze=action.gaze)


class SpeechController:
    def __init__(self, channel, face):
        self.channel = channel
        self.face = face

    async def execute(self, action: SpeakAction):
        self.face.set_state(speaking="Queued for playback", event="Reply ready")
        await self.channel.send_command("speak", text=action.text)


class ActionExecutor:
    def __init__(self, body_controller, face_controller, speech_controller):
        self.body = body_controller
        self.face = face_controller
        self.speech = speech_controller

    async def execute(self, action):
        if isinstance(action, SpeakAction):
            await self.speech.execute(action)
            return
        if isinstance(action, BodyAction):
            await self.body.execute(action)
            return
        if isinstance(action, FaceAction):
            await self.face.execute(action)
            return
        if isinstance(action, ToolAction):
            return
