import unsloth
from unsloth import FastModel
from transformers import WhisperForConditionalGeneration, pipeline
import torch
from app.utils.device import resolve_device
import numpy as np


class ASRModel():
    def __init__(self, device: str | None = None):
        self.device = resolve_device(device)
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name = "unsloth/whisper-large-v3",
            dtype = None, 
            load_in_4bit = False,
            auto_model = WhisperForConditionalGeneration,
            whisper_language = "English",
            whisper_task = "transcribe",
        )
        FastModel.for_inference(self.model)
        self.model.eval()
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.tokenizer.tokenizer,
            feature_extractor=self.tokenizer.feature_extractor,
            processor=self.tokenizer,
            return_language=True,
            torch_dtype=torch.float16  # Remove the device parameter
        )
    def transcribe(self, waveform: np.ndarray) -> str:
        """
        waveform:
          - mono
          - 16 kHz
          - float32
          - range [-1.0, 1.0]
        """
        with torch.no_grad():
            result = self.pipeline(waveform)
        text = result.get("text", "")
        return text
