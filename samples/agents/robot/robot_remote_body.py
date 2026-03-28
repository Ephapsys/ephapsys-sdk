#!/usr/bin/env python3

import asyncio
import os
import time

import numpy as np
import webrtcvad


class RemoteAudioSegmenter:
    def __init__(self, body, channel):
        self.body = body
        self.channel = channel
        self.sample_rate = 16000
        self.frame_ms = 30
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)
        self.vad = webrtcvad.Vad(int(os.getenv("ROBOT_VAD_AGGRESSIVENESS", "1")))
        self.min_level = float(os.getenv("ROBOT_MIN_SPEECH_LEVEL", "0.002"))
        self.quiet_level = float(os.getenv("ROBOT_QUIET_SPEECH_LEVEL", str(self.min_level * 0.5)))
        self.speech_start_frames = int(os.getenv("ROBOT_SPEECH_START_FRAMES", "2"))
        self.silence_limit = int(0.5 * 1000 / self.frame_ms)
        self._pcm_buffer = bytearray()
        self._utterance = []
        self._speech_run = 0
        self._raw_speech_run = 0
        self._silence_count = 0
        self._heard_speech = False
        self._lock = asyncio.Lock()

    async def ingest(self, pcm16_bytes: bytes):
        if not pcm16_bytes:
            return
        async with self._lock:
            self._pcm_buffer.extend(pcm16_bytes)
            frame_bytes = self.frame_samples * 2
            while len(self._pcm_buffer) >= frame_bytes:
                chunk = bytes(self._pcm_buffer[:frame_bytes])
                del self._pcm_buffer[:frame_bytes]
                await self._process_frame(chunk)

    async def flush(self):
        async with self._lock:
            await self._emit_if_complete(force=True)
            self._pcm_buffer.clear()

    async def _process_frame(self, chunk: bytes):
        audio = np.frombuffer(chunk, dtype="<i2").astype("float32") / 32768.0
        level = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
        raw_speech = self.vad.is_speech(chunk, self.sample_rate)
        if raw_speech:
            self._raw_speech_run += 1
        else:
            self._raw_speech_run = 0
        is_speech = raw_speech and level >= self.min_level
        quiet_speech = raw_speech and self._raw_speech_run >= self.speech_start_frames and level >= self.quiet_level
        active_speech = is_speech or quiet_speech
        self._speech_run = self._speech_run + 1 if active_speech else 0

        if not self._heard_speech and self._speech_run >= self.speech_start_frames:
            self._heard_speech = True
            self._silence_count = 0
            self.body.set_hearing_state("Speech detected", "Capturing utterance", force=True)

        if self._heard_speech:
            self._utterance.append(chunk)
            if active_speech:
                self._silence_count = 0
                self.body.set_hearing_state("Capturing utterance", "Capturing utterance")
            else:
                self._silence_count += 1
                self.body.set_hearing_state("Finishing utterance", "Capturing utterance")
                if self._silence_count > self.silence_limit:
                    await self._emit_if_complete(force=False)
        else:
            self.body.set_hearing_state("Listening for speech", "Waiting for speech")

    async def _emit_if_complete(self, force: bool):
        if not self._utterance:
            return
        if not self._heard_speech and not force:
            return
        pcm16_bytes = b"".join(self._utterance)
        summary = self.body.summarize_pcm16_bytes(pcm16_bytes, sr=self.sample_rate)
        self.body.set_hearing_state(summary, "Speech captured", force=True)
        await self.channel.emit_event(
            "microphone",
            audio=self.body.pcm16_bytes_to_wav_bytes(pcm16_bytes, samplerate=self.sample_rate),
            summary=summary,
            source="remote_ws",
        )
        self._utterance = []
        self._speech_run = 0
        self._raw_speech_run = 0
        self._silence_count = 0
        self._heard_speech = False
        self.body.set_hearing_state("Listening for speech", "Waiting for speech", force=True)
