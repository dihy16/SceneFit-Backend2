"""FastAPI application startup and shutdown lifecycle."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

from app.core.vector_db import VectorDatabase


STARTUP_ATTEMPTS = 3
STARTUP_RETRY_SECONDS = 10


async def _initialize_vector_db() -> VectorDatabase:
    """Initialize the shared retrieval index, retrying transient failures."""
    last_error: Exception | None = None
    for attempt in range(1, STARTUP_ATTEMPTS + 1):
        try:
            return VectorDatabase()
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
