from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from typing import Optional
import soundfile as sf
import numpy as np
from app.models.registry import ModelRegistry
import io

router = APIRouter()

SAMPLE_RATE = 16000

def load_audio_from_upload(file: UploadFile,) -> np.ndarray:
    """
    Decode uploaded audio into a mono float32 waveform @ 16 kHz
    """
    try:
        audio_bytes = file.file.read()
        data, sr = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
            always_2d=False
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    # Convert stereo → mono if needed
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)

    return data


def _decode_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """Decode raw audio file bytes into a mono float32 waveform @ 16kHz."""
    try:
        data, sr = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
            always_2d=False,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio bytes: {e}")

    if data.ndim == 2:
        data = data.mean(axis=1)

    if sr != SAMPLE_RATE:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)

    return data


def _pcm16le_to_float32_mono(pcm16le: bytes, channels: int = 1) -> np.ndarray:
    """Convert raw PCM16LE bytes to float32 mono waveform in [-1, 1].

    This is the preferred format for WebSocket streaming because it avoids container
    parsing on every chunk.
    """
    if channels < 1:
        channels = 1

    audio_i16 = np.frombuffer(pcm16le, dtype=np.int16)
    if audio_i16.size == 0:
        return np.zeros((0,), dtype=np.float32)

    if channels > 1:
        frames = audio_i16.size // channels
        audio_i16 = audio_i16[: frames * channels].reshape(frames, channels)
        audio_f32 = audio_i16.astype(np.float32).mean(axis=1)
    else:
        audio_f32 = audio_i16.astype(np.float32)

    return audio_f32 / 32768.0

@router.post("/audio_fb")
def asr_relevance_feedback(
    audio: UploadFile = File(...)
):
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Audio required."
        )
    waveform = load_audio_from_upload(audio)

    asr_model = ModelRegistry.get("asr")
    transcript = asr_model.transcribe(waveform)

    return {
        "transcript": transcript,
    }


@router.websocket("/ws/asr")
async def asr_websocket(ws: WebSocket):

    await ws.accept()

    fmt: str = "pcm16le"
    sample_rate: int = SAMPLE_RATE
    channels: int = 1
    partial_sec: float = 2.5  # how often to emit partials (best-effort)
    max_buffer_sec: float = 30.0

    audio_buf = np.zeros((0,), dtype=np.float32)
    next_partial_at = int(partial_sec * SAMPLE_RATE)

    asr_model = ModelRegistry.get("asr")

    try:
        try:
            first = await ws.receive()
        except WebSocketDisconnect:
            return

        if first["type"] == "websocket.receive" and "text" in first and first["text"]:
            import json

            cfg = json.loads(first["text"])
            if isinstance(cfg, dict) and cfg.get("type") == "start":
                fmt = str(cfg.get("format", fmt))
                sample_rate = int(cfg.get("sample_rate", sample_rate))
                channels = int(cfg.get("channels", channels))
                partial_sec = float(cfg.get("partial_sec", partial_sec))
                max_buffer_sec = float(cfg.get("max_buffer_sec", max_buffer_sec))
            else:
                pass
        else:
            pass

        if fmt not in {"pcm16le", "file"}:
            await ws.send_json({"type": "error", "detail": f"Unsupported format: {fmt}"})
            await ws.close(code=1003)
            return

        await ws.send_json({"type": "ready"})

        async def ingest_bytes(b: bytes) -> None:
            nonlocal audio_buf, next_partial_at

            if fmt == "file":
                audio_buf = _decode_audio_bytes(b)
                next_partial_at = audio_buf.size + 1
                return

            if sample_rate != SAMPLE_RATE:
                await ws.send_json({
                    "type": "error",
                    "detail": f"Only 16kHz supported for pcm16le streaming. Got {sample_rate}.",
                })
                await ws.close(code=1003)
                raise WebSocketDisconnect

            chunk = _pcm16le_to_float32_mono(b, channels=channels)
            if chunk.size == 0:
                return

            audio_buf = np.concatenate([audio_buf, chunk])

            max_samples = int(max_buffer_sec * SAMPLE_RATE)
            if audio_buf.size > max_samples:
                # Keep last window only
                audio_buf = audio_buf[-max_samples:]

            # Emit partial transcript best-effort
            if partial_sec > 0 and audio_buf.size >= next_partial_at:
                try:
                    text = asr_model.transcribe(audio_buf)
                    await ws.send_json({"type": "partial", "transcript": text})
                except Exception as e:
                    await ws.send_json({"type": "error", "detail": f"ASR failed: {e}"})
                next_partial_at = audio_buf.size + int(partial_sec * SAMPLE_RATE)

        # If we had received a binary message first (or a non-start text), process it.
        if first["type"] == "websocket.receive":
            if "bytes" in first and first["bytes"] is not None:
                await ingest_bytes(first["bytes"])
            elif "text" in first and first["text"]:
                import json
                msg = json.loads(first["text"])
                if isinstance(msg, dict) and msg.get("type") == "end":
                    pass

        while True:
            msg = await ws.receive()

            if msg["type"] != "websocket.receive":
                continue

            if "bytes" in msg and msg["bytes"] is not None:
                await ingest_bytes(msg["bytes"])
                if fmt == "file":
                    break
                continue

            if "text" in msg and msg["text"]:
                import json
                payload = json.loads(msg["text"])
                if isinstance(payload, dict) and payload.get("type") == "end":
                    break

        try:
            final_text = asr_model.transcribe(audio_buf)
            await ws.send_json({"type": "final", "transcript": final_text})
        except Exception as e:
            await ws.send_json({"type": "error", "detail": f"ASR failed: {e}"})

        await ws.close(code=1000)

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "detail": str(e)})
            await ws.close(code=1011)
        except Exception:
            return