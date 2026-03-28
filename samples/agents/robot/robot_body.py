#!/usr/bin/env python3

import asyncio
import json
import os
import queue
import shutil
import subprocess
import tempfile
import time
import wave

import cv2
import numpy as np
import pyaudio
import sounddevice as sd
import webrtcvad


class RobotBody:
    def __init__(self, face, shutdown_event):
        self.face = face
        self.shutdown_event = shutdown_event
        self.mic_queue = queue.Queue()
        self.cam_queue = queue.Queue()
        self.tts_queue = asyncio.Queue()
        self.camera_cap = None
        self.tts_available = True
        self.disable_audio_output = os.getenv("DISABLE_AUDIO", "").lower() in ("1", "true", "yes")
        self.audio_debug = os.getenv("AUDIO_DEBUG", "").lower() in ("1", "true", "yes")
        self.last_tts_ms = 0.0

    def ensure_preprocessor(self, tts_path: str) -> bool:
        try:
            cfg_path = os.path.join(tts_path, "preprocessor_config.json")
            if os.path.exists(cfg_path):
                return True
            minimal = {
                "feature_extractor_type": "SpeechT5FeatureExtractor",
                "sampling_rate": 16000,
                "padding_value": 0,
                "do_normalize": True,
            }
            with open(cfg_path, "w") as f:
                json.dump(minimal, f)
            self.face.console_log.log("Synthesized preprocessor_config.json for TTS")
            return True
        except Exception as exc:
            self.face.console_log.log(f"Failed to synthesize preprocessor_config.json: {exc}")
            return False

    def capture_microphone(self, sr=16000, chunk_ms=30, max_duration=10, max_wait_for_speech=4):
        vad = webrtcvad.Vad(int(os.getenv("ROBOT_VAD_AGGRESSIVENESS", "1")))
        chunk_size = int(sr * chunk_ms / 1000)
        buffer = []
        silence_count = 0
        silence_limit = int(0.5 * 1000 / chunk_ms)
        heard_speech = False

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sr,
            input=True,
            frames_per_buffer=chunk_size,
        )
        try:
            start_time = time.time()
            while True:
                if self.shutdown_event.is_set():
                    break
                data = stream.read(chunk_size, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16)
                level = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32) / 32768.0)))) if len(chunk) else 0.0
                is_speech = vad.is_speech(chunk.tobytes(), sr)
                if is_speech:
                    heard_speech = True
                    self.face.set_state(hearing=f"Speech detected @ level {level:.3f}", event="Capturing utterance")
                    silence_count = 0
                    buffer.extend(chunk)
                elif heard_speech:
                    silence_count += 1
                    buffer.extend(chunk)
                    self.face.set_state(
                        hearing=f"Listening for end of speech @ level {level:.3f}",
                        event="Capturing utterance",
                    )
                    if silence_count > silence_limit:
                        break
                else:
                    self.face.set_state(
                        hearing=f"Listening on microphone @ level {level:.3f}",
                        event="Waiting for speech",
                    )
                    if time.time() - start_time > max_wait_for_speech:
                        return np.array([], dtype="float32")
                if time.time() - start_time > max_duration:
                    break
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            pa.terminate()

        return np.array(buffer, dtype="int16").astype("float32") / 32768.0

    @staticmethod
    def summarize_audio(audio: np.ndarray, sr: int = 16000) -> str:
        if audio is None or len(audio) == 0:
            return "No speech"
        duration_s = len(audio) / float(sr)
        rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
        return f"Heard {duration_s:.1f}s @ level {rms:.3f}"

    async def mic_task(self):
        while not self.shutdown_event.is_set():
            try:
                self.face.set_state(hearing="Listening on microphone", event="Waiting for speech")
                loop = asyncio.get_running_loop()
                audio = await loop.run_in_executor(None, self.capture_microphone)
                self.face.set_state(hearing=self.summarize_audio(audio), event="Speech captured")
                self.mic_queue.put(audio)
            except Exception as exc:
                self.face.set_state(hearing="Microphone error", event=f"Mic failure: {exc}")
                self.face.console_log.log(f"Mic capture error: {exc}")
            await asyncio.sleep(0.1)

    @staticmethod
    def capture_camera(cap, sample_interval=5, last_capture=0):
        now = time.time()
        if now - last_capture < sample_interval:
            return None, last_capture
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Camera capture failed")
        return frame[:, :, ::-1], now

    async def cam_task(self):
        self.camera_cap = cv2.VideoCapture(0)
        last = 0
        while not self.shutdown_event.is_set():
            try:
                frame, last = self.capture_camera(self.camera_cap, 5, last)
                if frame is not None:
                    self.cam_queue.put(frame)
            except Exception as exc:
                self.face.console_log.log(f"Cam capture error: {exc}")
            await asyncio.sleep(0.2)

    def play_audio(self, audio: np.ndarray, samplerate: int = 16000):
        tmp = None
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(samplerate)
                pcm16 = np.clip(audio * 32767, -32768, 32767).astype("<i2")
                wf.writeframes(pcm16.tobytes())
            tmp.close()

            player = None
            if shutil.which("afplay"):
                player = ["afplay", tmp.name]
            elif shutil.which("aplay"):
                player = ["aplay", tmp.name]

            if player:
                subprocess.run(player, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                sd.play(audio, samplerate=samplerate, blocking=True)
                sd.wait()
                sd.stop()
        finally:
            try:
                if tmp is not None:
                    os.unlink(tmp.name)
            except Exception:
                pass

    def play_tts_sync(self, agent, text):
        if self.disable_audio_output:
            self.face.set_state(speaking="Audio disabled")
            return
        if not self.tts_available:
            self.face.set_state(speaking="Unavailable")
            return
        self.face.set_state(speaking="Synthesizing reply", event="Generating speech")
        audio = agent.run(text, model_kind="tts")
        if audio is None:
            self.face.set_state(speaking="Skipped")
            return
        audio = np.array(audio, dtype="float32")
        if audio.size == 0:
            self.face.set_state(speaking="Skipped")
            return
        max_abs = np.max(np.abs(audio))
        if max_abs > 0:
            audio = audio / max_abs * 0.8
        self.face.set_state(speaking="Playing audio")
        self.play_audio(audio, samplerate=16000)
        self.face.set_state(speaking="Idle", event="Ready for next interaction")

    async def play_tts_async(self, agent, text):
        if self.disable_audio_output or not self.tts_available:
            self.face.set_state(speaking="Unavailable" if not self.tts_available else "Audio disabled")
            return
        try:
            loop = asyncio.get_running_loop()
            self.face.set_state(speaking="Synthesizing reply", event="Generating speech")
            tts_started = time.perf_counter()
            audio = await loop.run_in_executor(None, lambda: agent.run(text, model_kind="tts"))
            if audio is None:
                self.face.set_state(speaking="Skipped")
                return
            audio = np.array(audio, dtype="float32")
            if audio.size == 0:
                self.face.set_state(speaking="Skipped")
                return
            max_abs = np.max(np.abs(audio))
            if max_abs > 0:
                audio = audio / max_abs * 0.8
            self.face.set_state(speaking="Playing audio")
            self.play_audio(audio, samplerate=16000)
            self.last_tts_ms = (time.perf_counter() - tts_started) * 1000
            self.face.set_state(
                latency=f"{self.face.ui_state['latency']} | tts {self.last_tts_ms:.0f}",
                speaking="Idle",
                event="Ready for next interaction",
            )
        except Exception as exc:
            self.face.set_state(speaking="Error", event=f"TTS failed: {exc}")
            self.face.console_log.log(f"TTS error: {exc}")
            if "preprocessor_config" in str(exc):
                self.tts_available = False

    async def tts_worker(self, agent):
        while not self.shutdown_event.is_set():
            got_item = False
            try:
                text = await self.tts_queue.get()
                got_item = True
                await self.play_tts_async(agent, text)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.face.console_log.log(f"TTS worker error: {exc}")
            finally:
                if got_item:
                    self.tts_queue.task_done()

    def cleanup(self):
        self.face.console_log.log("Cleaning up resources...")
        if self.camera_cap is not None:
            try:
                self.camera_cap.release()
                self.face.console_log.log("Camera released")
            except Exception:
                pass
