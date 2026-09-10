"""Download and validate the Flux checkpoint without starting inference."""

import gc

import torch
from diffusers import Flux2KleinPipeline


MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"


def main() -> None:
    pipeline = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    del pipeline
    gc.collect()
    print(f"Flux checkpoint is cached and loadable: {MODEL_ID}", flush=True)


if __name__ == "__main__":
    main()
