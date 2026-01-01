#!/usr/bin/env python3
"""
Sample Robot Agent using Ephapsys SDK.

This robot (Asimov) agent can hear, see, think and speak using offline, self-hosted models.
Modulation in longer time horizons is required to get acceptable accuracies as ephaptic coupling modulation in just 100 steps is not sufficient for the network to synchronize.

Workflow:
1. Verify agent status at startup.
2. Prepare runtime (download/cache artifacts + decrypt ECM).
3. Enter main loop:
   - Mic → STT → Language → Embedding
   - Cam → Vision label
   - Store embeddings into FAISS memory, retrieve most similar past response.
   - Output action (TTS to audio) augmented with memory context.
4. A background verification loop periodically checks agent status (enabled/disabled/revoked).
"""

import os, sys, time, threading, asyncio, queue, tempfile, wave, subprocess, shutil, faulthandler
import numpy as np
import sounddevice as sd
import pyaudio
import cv2
import faiss
import webrtcvad
import json
try:
    import torch  # required for TTS model load
except ImportError:
    sys.exit("[ERROR] Missing torch; install torch>=2.6.0 for TTS audio playback.")
try:
    import transformers  # required for SpeechT5
except ImportError:
    sys.exit("[ERROR] Missing transformers; install transformers>=4.46.0 for TTS audio playback.")
try:
    import soundfile  # required for audio IO in transformers TTS
except ImportError:
    sys.exit("[ERROR] Missing soundfile; install soundfile>=0.12.0 for TTS audio playback.")

from ephapsys.agent import TrustedAgent
from rich.console import Console
from rich.live import Live
# from rich.table import Table
from rich.panel import Panel   

# Two consoles: one for live dashboard, one for logs
console_live = Console(force_terminal=True)
console_log = Console(stderr=True)

# Queues for async tasks
mic_queue = queue.Queue()
cam_queue = queue.Queue()
tts_queue: asyncio.Queue = asyncio.Queue()

# Global agent status tracker
agent_status = {"verified": False, "enabled": False, "revoked": False}

# Global shutdown flag
shutdown_event = asyncio.Event()

# Global camera handle
camera_cap = None

# Global TTS availability flag
tts_available = True
# Allow disabling audio playback (avoids PortAudio segfaults on some macOS setups)
disable_audio_output = os.getenv("DISABLE_AUDIO", "").lower() in ("1", "true", "yes")
if disable_audio_output:
    console_log.log("Audio output disabled via DISABLE_AUDIO")
# Debug flag to trace audio path/choices
audio_debug = os.getenv("AUDIO_DEBUG", "").lower() in ("1", "true", "yes")

# Enable faulthandler to capture native backtraces on crash
try:
    faulthandler.enable(all_threads=True)
except Exception:
    pass

# Reduce torch thread counts to avoid native crashes in TTS model init
try:
    torch_threads = int(os.getenv("TORCH_NUM_THREADS", "1"))
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(torch_threads)
    if audio_debug:
        console_log.log(f"torch threads set to {torch_threads}")
except Exception:
    pass

def _ensure_preprocessor(tts_path: str) -> bool:
    """
    Ensure preprocessor_config.json exists for TTS models.
    Some idempotent flows skip downloading full HF assets; we synthesize a minimal
    config so SpeechT5Processor can load.
    """
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
        console_log.log("Synthesized preprocessor_config.json for TTS")
        return True
    except Exception as e:
        console_log.log(f"Failed to synthesize preprocessor_config.json: {e}")
        return False


# -------------------------------------------------------------------
# Microphone
# -------------------------------------------------------------------
def capture_microphone(sr=16000, chunk_ms=30, max_duration=10):
    """Capture microphone audio until silence is detected using VAD (PyAudio backend)."""
    vad = webrtcvad.Vad(2)
    chunk_size = int(sr * chunk_ms / 1000)
    buffer, silence_count = [], 0
    silence_limit = int(0.5 * 1000 / chunk_ms)

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
            if shutdown_event.is_set():
                break
            data = stream.read(chunk_size, exception_on_overflow=False)
            chunk = np.frombuffer(data, dtype=np.int16)
            pcm = chunk.tobytes()
            is_speech = vad.is_speech(pcm, sr)
            buffer.extend(chunk)
            silence_count = 0 if is_speech else silence_count + 1
            if silence_count > silence_limit:
                break
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


async def mic_task():
    """Continuously capture mic audio and push into queue."""
    while not shutdown_event.is_set():
        try:
            audio = capture_microphone()
            mic_queue.put(audio)
        except Exception as e:
            console_log.log(f"Mic capture error: {e}")
        await asyncio.sleep(0.1)


# -------------------------------------------------------------------
# Camera
# -------------------------------------------------------------------
def capture_camera(cap, sample_interval=5, last_capture=0):
    """Capture a frame every `sample_interval` seconds."""
    now = time.time()
    if now - last_capture < sample_interval:
        return None, last_capture
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Camera capture failed")
    return frame[:, :, ::-1], now


async def cam_task():
    """Continuously capture camera frames and push into queue."""
    global camera_cap
    camera_cap = cv2.VideoCapture(0)
    last = 0
    while not shutdown_event.is_set():
        try:
            frame, last = capture_camera(camera_cap, 5, last)
            if frame is not None:
                cam_queue.put(frame)
        except Exception as e:
            console_log.log(f"Cam capture error: {e}")
        await asyncio.sleep(0.2)


# -------------------------------------------------------------------
# TTS
# -------------------------------------------------------------------
def _play_audio(audio: np.ndarray, samplerate: int = 16000):
    """
    Play audio via a temporary WAV + afplay/aplay to avoid PortAudio crashes.
    Falls back to sounddevice if no system player is found.
    """
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
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
            if audio_debug:
                console_log.log(f"Playing audio via {player[0]} (path={tmp.name})")
            subprocess.run(player, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Fallback to sounddevice if no system player exists
            if audio_debug:
                console_log.log("Playing audio via sounddevice fallback")
            sd.play(audio, samplerate=samplerate, blocking=True)
            sd.wait()
            sd.stop()
    finally:
        try:
            if tmp is not None:
                os.unlink(tmp.name)
        except Exception:
            pass


def play_tts(agent, text):
    """Deprecated synchronous TTS playback (kept for reference)."""
    return play_tts_sync(agent, text)


def play_tts_sync(agent, text):
    """Run TTS on text and play audio; used inside asyncio worker."""
    global tts_available
    if disable_audio_output:
        console_log.log("Audio output disabled; skipping TTS playback.")
        return
    if not tts_available:
        return
    audio = agent.run(text, model_kind="tts")
    if audio is None:
        console_log.log("TTS returned None, skipping")
        return
    audio = np.array(audio, dtype="float32")
    if audio.size == 0:
        console_log.log("TTS returned empty array, skipping")
        return
    # Normalize if values look tiny or large
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs * 0.8
    _play_audio(audio, samplerate=16000)


async def play_tts_async(agent, text):
    """Async wrapper that runs synthesis off the event loop and plays on the main thread."""
    global tts_available
    if disable_audio_output or not tts_available:
        return
    try:
        loop = asyncio.get_running_loop()
        # Run blocking agent.run in a thread executor
        audio = await loop.run_in_executor(None, lambda: agent.run(text, model_kind="tts"))
        if audio is None:
            console_log.log("TTS returned None, skipping")
            return
        audio = np.array(audio, dtype="float32")
        if audio.size == 0:
            console_log.log("TTS returned empty array, skipping")
            return
        max_abs = np.max(np.abs(audio))
        if max_abs > 0:
            audio = audio / max_abs * 0.8
        # Playback serialized on the main thread
        _play_audio(audio, samplerate=16000)
    except Exception as e:
        console_log.log(f"TTS error: {e}")
        # Disable further TTS attempts if the model assets are missing
        if "preprocessor_config" in str(e):
            console_log.log("TTS assets missing; disabling TTS for this session.")
            tts_available = False


# -------------------------------------------------------------------
# Status formatting
# -------------------------------------------------------------------
def format_status():
    """Return a color-coded status string for dashboard."""
    if agent_status.get("revoked", False):
        return "[red]REVOKED[/red]"
    if not agent_status.get("enabled", False):
        return "[red]DISABLED[/red]"
    if not agent_status.get("verified", False):
        return "[yellow]VERIFYING[/yellow]"
    return "[green]ENABLED[/green]"


# -------------------------------------------------------------------
# Main processing loop
# -------------------------------------------------------------------
async def process_task(agent, stored_responses, index, live):
    """Handle mic + cam → STT → Vision → Language → Embedding → TTS."""
    last_render_key = None
    while not shutdown_event.is_set():
        mic_audio, cam_frame = None, None

        if not agent_status.get("enabled", False) or agent_status.get("revoked", False):
            panel = render_status("-", "-", "-")
            key = render_key("-", "-", "-")
            if key != last_render_key:
                live.update(panel, refresh=True)
                last_render_key = key
            await asyncio.sleep(1)
            continue

        try:
            mic_audio = mic_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            cam_frame = cam_queue.get_nowait()
        except queue.Empty:
            pass

        if mic_audio is None and cam_frame is None:
            panel = render_status("-", "-", "-")
            key = render_key("-", "-", "-")
            if key != last_render_key:
                live.update(panel)
                last_render_key = key
            await asyncio.sleep(0.2)
            continue

        try:
            # STT
            text_input = agent.run(mic_audio, model_kind="stt")

            # Vision
            vision_label = None
            if cam_frame is not None:
                vision_raw = agent.run(cam_frame, model_kind="vision")
                vision_label = str(vision_raw).strip() if vision_raw is not None else None

            # Language
            context = f"(vision={vision_label})" if vision_label else ""
            response_text = str(agent.run(f"{text_input} {context}", model_kind="language")).strip()

            # Embedding + memory
            embedding_out = agent.run(response_text, model_kind="embedding")
            vec = np.array(embedding_out, dtype="float32").reshape(1, -1)
            if vec.size == 0:
                console_log.log("Empty embedding vector, skipping")
                continue

            memory_context = ""
            if index.ntotal > 0:
                D, I = index.search(vec, k=1)
                memory_context = f" Previously: {stored_responses[I[0][0]]}"
            index.add(vec)
            stored_responses.append(response_text)

            # TTS queued (serialized) to avoid PortAudio crashes
            augmented_text = response_text + memory_context
            if tts_available:
                if tts_queue.qsize() < 3:
                    await tts_queue.put(augmented_text)
                else:
                    console_log.log("TTS queue full; dropping audio playback to stay responsive.")

            # Update panel
            panel = render_status(text_input, vision_label or "-", augmented_text)
            key = render_key(text_input, vision_label or "-", augmented_text)
            if key != last_render_key:
                live.update(panel)
                last_render_key = key

        except Exception as e:
            console_log.log(f"Processing error: {e}")

        await asyncio.sleep(0.1)


# -------------------------------------------------------------------
# Periodic verification
# -------------------------------------------------------------------
async def periodic_verify(agent):
    """Background task to check agent status with backend every 5s."""
    global agent_status
    while not shutdown_event.is_set():
        await asyncio.sleep(5)
        try:
            ok, _ = agent.verify()
            status = agent.get_status()
            is_enabled = status.get("enabled", False) or (status.get("status", "").lower() == "enabled")
            is_revoked = status.get("state", {}).get("revoked", False)
            agent_status.update({"verified": ok, "enabled": is_enabled, "revoked": is_revoked})
            console_log.log(f"Periodic verify={agent_status}")
        except Exception as e:
            console_log.log(f"⚠️ Verification failed: {e}")
            agent_status.update({"enabled": False, "revoked": True})


async def tts_worker(agent):
    """Serialize TTS requests to avoid concurrent PortAudio calls."""
    while not shutdown_event.is_set():
        try:
            text = await tts_queue.get()
            if audio_debug:
                console_log.log(f"TTS worker dequeued: {text[:80]}{'...' if len(text)>80 else ''}")
            await play_tts_async(agent, text)
        except Exception as e:
            console_log.log(f"TTS worker error: {e}")
        finally:
            tts_queue.task_done()


# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------
def cleanup():
    """Release hardware resources safely."""
    global camera_cap
    console_log.log("Cleaning up resources...")
    if camera_cap is not None:
        try:
            camera_cap.release()
            console_log.log("Camera released")
        except Exception:
            pass

# -------------------------------------------------------------------
# Status rendering
# -------------------------------------------------------------------
def render_status(mic, vision, response):
    return Panel.fit(
        f"[bold cyan]Mic:[/bold cyan] {mic or '-'}\n"
        f"[bold green]Vision:[/bold green] {vision or '-'}\n"
        f"[bold yellow]Response:[/bold yellow] {response or '-'}\n"
        f"[bold white]Status:[/bold white] {format_status()}",
        title="🤖 Robot Agent",
        border_style="blue"
    )


def render_key(mic, vision, response):
    """Hashable key so we only redraw when content actually changes."""
    return (
        mic or "-",
        vision or "-",
        response or "-",
        format_status(),
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
async def main():
    agent = TrustedAgent.from_env()
    console_live.print("=== Step 1: Verify Agent ===")

    # Startup verify
    try:
        ok, report = agent.verify()
    except RuntimeError as e:
        if "404" in str(e):
            console_live.print(f"[red]❌ Agent template '{agent.agent_id}' not found in backend.[/red]")
            console_live.print("[yellow]Please create it in the AOC before running this sample.[/yellow]")
            sys.exit(1)
        else:
            raise

    # Personalize if needed
    if not ok:
        status = agent.get_status()
        is_personalized = status.get("state", {}).get("personalized", False) or status.get("personalized", False)
        if not is_personalized:
            anchor = os.getenv("PERSONALIZE_ANCHOR")
            console_live.print(f"[yellow]Agent not personalized; running personalize(anchor={anchor})...[/yellow]")
            agent.personalize(anchor=anchor)
            console_live.print(f"[green]✅ Agent personalized (instance registered in AOC).[/green]")

            # Re-verify with retries
            for _ in range(5):
                ok, report = agent.verify()
                if ok:
                    break
                console_live.print("[yellow]...waiting for agent to become ready...[/yellow]")
                time.sleep(1)

        if not ok:
            console_live.print("[red]❌ Agent not ready after personalization.[/red]")
            sys.exit(1)

    # At this point agent is ready
    status = agent.get_status()
    is_enabled = status.get("enabled", False) or (status.get("status", "").lower() == "enabled")
    is_revoked = status.get("state", {}).get("revoked", False)
    agent_status.update({"verified": ok, "enabled": is_enabled, "revoked": is_revoked})
    console_live.print("[green]✅ Agent personalized and verified.[/green]")

    # Runtime prep
    runtimes = agent.prepare_runtime()
    tts_runtime = runtimes.get("tts") or {}
    tts_path = tts_runtime.get("model_path")
    global tts_available
    if tts_path:
        tts_available = _ensure_preprocessor(tts_path)
    else:
        tts_available = False
    console_live.print(f"[green]✅ Runtime prepared[/green] (tts_ready={tts_available}, runtimes={list(runtimes.keys())})")

    # Greeting
    greeting = "Hi, my name is Asimov, at your service."
    console_live.print(f"[cyan]🤖 Greeting: {greeting}[/cyan]")

    # Speak greeting only if agent is enabled
    try:
        if agent_status.get("enabled", False) and not agent_status.get("revoked", False) and tts_available:
            console_live.print("[green]✅ GREETING TTS...")
            play_tts(agent, greeting)
            console_live.print("[green]✅ GREETING TTS done.")
        else:
            console_live.print("[red]❌ Skipping greeting TTS (agent not enabled or TTS assets missing)")
    except Exception as e:
        console_log.log(
            f"[red]⚠️ Greeting TTS failed "
            f"(MOST LIKELY DUE TO INSUFFICIENT MODULATION. "
            f"IT IS OKAY FOR NOW UNTIL WE COMPLETE MODULATION ON GCP): {e}"
        )

   
    dim = 768
    index = faiss.IndexFlatL2(dim)
    stored_responses = []

    console_live.print("[blue]Entering main loop... (Ctrl+C to exit)[/blue]")

    try:
        # Create live panel
        with Live(
            render_status("-", "-", greeting),
            refresh_per_second=2,
            console=console_live,
            screen=True,
        ) as live:
            await asyncio.gather(
                mic_task(),
                cam_task(),
                process_task(agent, stored_responses, index, live),  # ✅ pass live correctly
                periodic_verify(agent),
                tts_worker(agent),
            )
    finally:
        cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console_log.log("Shutting down (Ctrl+C).")
        shutdown_event.set()
        cleanup()
        sys.exit(0)
