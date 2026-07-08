"""
Lightweight mock test server
-----------------------------
Mounts ONLY the retrieval router (no heavy models, no vector DB).
This is enough to fully exercise the Strategy/Registry endpoints with mock data.

Run with:
    D:\\anaconda3\\envs\\perception_models\\python.exe -m uvicorn test_server:app --host 127.0.0.1 --port 8000
"""

from fastapi import FastAPI
from app.api.v1.endpoints.all_methods_ep import router as retrieval_router

app = FastAPI(title="SceneFit Mock Test Server")
app.include_router(retrieval_router, prefix="/api/v1/retrieval", tags=["retrieval"])


@app.get("/health")
def health():
    from app.services.retrieval.strategies import REGISTRY
    return {
        "status": "ok",
        "registered_strategies": REGISTRY.names(),
    }
