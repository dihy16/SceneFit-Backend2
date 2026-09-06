# app/api/v1/api.py

from fastapi import APIRouter
from app.api.v1.endpoints import (
    aesthetic_ep,
    all_methods_ep,
    experiment_ep,
    image_edit_ep,
    pe_clip_ep,
    vlm_ep,
)


api_router = APIRouter()


api_router.include_router(all_methods_ep.router, prefix="/retrieval", tags=["retrieval"])
api_router.include_router(pe_clip_ep.router, prefix="/workers", tags=["workers"])
api_router.include_router(image_edit_ep.router, prefix="/workers", tags=["workers"])
api_router.include_router(vlm_ep.router, prefix="/workers", tags=["workers"])
api_router.include_router(aesthetic_ep.router, prefix="/workers", tags=["workers"])
api_router.include_router(experiment_ep.router, tags=["study"])
