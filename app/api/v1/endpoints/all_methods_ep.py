"""
Retrieval API endpoints
-----------------------
All retrieval methods are served through two routes:

  POST /api/v1/retrieval/all-methods
      Runs every registered strategy in parallel and returns a combined payload.

  POST /api/v1/retrieval/{method_name}
      Runs a single named strategy (clip | image_edit | vlm | aesthetic).
      Returns the same {name, score, image_url} list format.

Adding a new strategy: register it in `app/services/retrieval_strategies.py`.
No changes to this file are required.
"""

import asyncio

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.retrieval.strategies import REGISTRY

router = APIRouter()


# ---------------------------------------------------------------------------
# Helper: read image once and fan out to strategies
# ---------------------------------------------------------------------------

async def _run_strategy(name: str, image_content: bytes, filename: str, content_type: str, top_k: int):
    """Run a single named strategy in a thread pool and return its result."""
    strategy = REGISTRY.get(name)
    return await asyncio.to_thread(
        strategy.retrieve,
        image_content, filename, content_type, top_k, False,
    )


# ---------------------------------------------------------------------------
# POST /all-methods — aggregate all registered strategies in parallel
# ---------------------------------------------------------------------------

@router.post("/all-methods")
async def retrieve_all_methods(
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    Run every registered retrieval strategy in parallel.

    Returns partial results if some services fail — a failed strategy entry
    will be `{"error": true, "message": "<reason>"}` instead of a list.

    Response shape:
    ```json
    {
        "clip":       [{"name": str, "score": float, "image_url": str}, ...],
        "image_edit": [...] | {"error": true, "message": str},
        "vlm":        [...],
        "aesthetic":  [...]
    }
    ```
    """
    image_content = await image.read()
    filename = image.filename
    content_type = image.content_type

    strategy_names = REGISTRY.names()

    tasks = [
        _run_strategy(name, image_content, filename, content_type, top_k)
        for name in strategy_names
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    response = {}
    for name, result in zip(strategy_names, results):
        if isinstance(result, Exception):
            response[name] = {"error": True, "message": str(result)}
        else:
            response[name] = result

    success = sum(1 for v in response.values() if not (isinstance(v, dict) and v.get("error")))
    total = len(strategy_names)
    print(f"[ALL-METHODS] {success}/{total} strategies succeeded")

    return response


# ---------------------------------------------------------------------------
# POST /{method_name} — single dynamic endpoint for any registered strategy
# ---------------------------------------------------------------------------

@router.post("/{method_name}")
async def retrieve_by_method(
    method_name: str,
    image: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    Run a specific retrieval strategy by name.

    Available names are whatever is registered in the `RetrievalRegistry`
    (currently: **clip**, **image_edit**, **vlm**, **aesthetic**).

    Response:
    ```json
    [{"name": str, "score": float, "image_url": str}, ...]
    ```
    """
    # Let the registry raise a 404 HTTPException if the name is unknown.
    REGISTRY.get(method_name)

    image_content = await image.read()
    return await _run_strategy(method_name, image_content, image.filename, image.content_type, top_k)