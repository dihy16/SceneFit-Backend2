"""
ASR orchestration service.
Canonical location: app/services/asr/service.py
(Backward-compat shim remains at app/services/speech_to_text.py)
"""
import io

import numpy as np
import soundfile as sf
from fastapi import File, HTTPException, UploadFile

from app.models.registry import ModelRegistry

SAMPLE_RATE = 16000


def load_audio_from_upload(file: UploadFile) -> np.ndarray:
    """Decode an uploaded audio file into a mono float32 waveform at 16 kHz."""
    try:
        audio_bytes = file.file.read()
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    if data.ndim == 2:
        data = data.mean(axis=1)

    if sr != SAMPLE_RATE:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)

    return data


def convert_speech_to_text(audio: UploadFile = File(...)) -> str:
    """Transcribe an uploaded audio clip to text via the ASR model."""
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Audio required.")
    waveform = load_audio_from_upload(audio)
    asr_model = ModelRegistry.get("asr")
    return asr_model.transcribe(waveform)
