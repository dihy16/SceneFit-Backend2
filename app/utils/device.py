# app/utils/device.py

# app/utils/device.py

import torch

def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def resolve_autocast(device: torch.device) -> bool:
    return device.type in ("cuda", "mps")


def resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    if device.type == "mps":
        return torch.float16

    return torch.float32