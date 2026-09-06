"""FastAPI application startup and shutdown lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI

from app.core.vector_db import VectorDatabase


STARTUP_ATTEMPTS = 3
STARTUP_RETRY_SECONDS = 10
BENCHMARK_MANIFEST_ENV = "BENCHMARK_MANIFEST"


def _vector_database_options() -> dict:
    manifest_value = os.getenv(BENCHMARK_MANIFEST_ENV)
    if not manifest_value:
        return {}

    manifest_path = Path(manifest_value).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outfits = manifest.get("outfits")
    if not isinstance(outfits, list) or not outfits:
        raise ValueError(f"Benchmark manifest has no outfits: {manifest_path}")

    project_root = Path(__file__).resolve().parents[2]
    outfit_paths = [(project_root / item["path"]).resolve() for item in outfits]
    outfit_directories = {path.parent for path in outfit_paths}
    if len(outfit_directories) != 1:
        raise ValueError("All benchmark outfits must be in the same directory")

    print(
        f"[START] Benchmark index mode: {len(outfits)} outfits from {manifest_path}"
    )
    return {
        "data_dir": outfit_directories.pop(),
        "allowed_names": [item["id"] for item in outfits],
        "index_path": manifest_path.parent / "vector.index",
        "metadata_path": manifest_path.parent / "vector.index.meta.json",
    }


async def _initialize_vector_db() -> VectorDatabase:
    """Initialize the shared retrieval index, retrying transient failures."""
    last_error: Exception | None = None
    options = _vector_database_options()
    for attempt in range(1, STARTUP_ATTEMPTS + 1):
        try:
            return VectorDatabase(**options)
        except Exception as exc:
            last_error = exc
            print(
                f"[START] Vector database initialization failed "
                f"({attempt}/{STARTUP_ATTEMPTS}): {exc}"
            )
            if attempt < STARTUP_ATTEMPTS:
                await asyncio.sleep(STARTUP_RETRY_SECONDS)

    raise RuntimeError(
        f"Failed to initialize the vector database after {STARTUP_ATTEMPTS} attempts"
    ) from last_error


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare shared state while heavy endpoint models remain lazy-loaded."""
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[START] Initializing vector database...")
    app.state.vector_db = await _initialize_vector_db()
    print("[START] Backend ready; endpoint models will load on first use")

    try:
        yield
    finally:
        app.state.vector_db = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[SHUTDOWN] Backend stopped")
