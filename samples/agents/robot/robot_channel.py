#!/usr/bin/env python3

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RobotEvent:
    kind: str
    payload: Dict[str, Any]


@dataclass
class RobotCommand:
    kind: str
    payload: Dict[str, Any]


class RobotChannel:
    def __init__(self):
        self._events: asyncio.Queue[RobotEvent] = asyncio.Queue()
        self._commands: asyncio.Queue[RobotCommand] = asyncio.Queue()

    async def emit_event(self, kind: str, **payload: Any):
        await self._events.put(RobotEvent(kind=kind, payload=payload))

    async def next_event(self, timeout: Optional[float] = None) -> Optional[RobotEvent]:
        if timeout is None:
            return await self._events.get()
        try:
            return await asyncio.wait_for(self._events.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def event_done(self):
        self._events.task_done()

    async def send_command(self, kind: str, **payload: Any):
        await self._commands.put(RobotCommand(kind=kind, payload=payload))

    async def next_command(self) -> RobotCommand:
        return await self._commands.get()

    def command_done(self):
        self._commands.task_done()
