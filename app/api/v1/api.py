# app/api/v1/api.py

from fastapi import APIRouter
from app.api.v1.endpoints import all_methods_ep
from app.api.v1.endpoints import experiment_ep


api_router = APIRouter()


api_router.include_router(all_methods_ep.router, prefix="/retrieval", tags=["retrieval"])
api_router.include_router(experiment_ep.router, tags=["study"])
