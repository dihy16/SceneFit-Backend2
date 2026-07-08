from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.study.scorer import score_methods
from app.services.study.storage import read_all_payloads, write_response


router = APIRouter()


DEFAULT_STUDY_DIR = os.path.join(os.getcwd(), "user_study")


class UnityMethodResponse(BaseModel):
    methodName: str
    imgURLs: List[str] = Field(
        ...,
        description="top-k images of the currentmethod",
    )
    selectedURL: str = Field(..., description="URL of the image the participant selected for this method.")
    viewCounts: Optional[List[int]] = Field(
        None,
        description="Optional per-outfit counts of 'View' button clicks (length == num_outfits).",
    )


class UnityParticipantPayload(BaseModel):
    """Stored representation (server-generated fields included)."""
    participantId: str
    responses: List[UnityMethodResponse]
    winnerMethodName: str


class UnityParticipantSubmission(BaseModel):
    """Client submission format for /study/response (JSON body).

    The client sends the list of responses (each containing the method's
    image paths and the URL of the selected image), and the name of
    the winning method. The server generates participantId and writes
    a UnityParticipantPayload to storage.
    """

    responses: List[UnityMethodResponse]
    winnerMethodName: str


class StudyScoreQuery(BaseModel):
    alpha: float = Field(0.6, ge=0.0, le=1.0)
    num_outfits: int = Field(5, ge=1)


@router.post("/study/response")
def submit_study_response(
    submission: UnityParticipantSubmission,
) -> Dict[str, Any]:
    """Ingest a single participant response from Unity (application/json).

    Contract:
    - Client sends JSON body with {responses: [...], winnerMethodName: str}.
    - Server auto-generates a participantId (UUID4).
    - winnerMethodName is the client's chosen overall winner.
    """
    if not submission.responses:
        raise HTTPException(status_code=400, detail="responses list must not be empty")

    # Basic validation: ensure methodNames are unique per participant
    method_names = [r.methodName for r in submission.responses]
    if len(set(method_names)) != len(method_names):
        raise HTTPException(status_code=400, detail="Duplicate methodName in responses")

    data = UnityParticipantPayload(
        participantId=str(uuid4()),
        responses=submission.responses,
        winnerMethodName=submission.winnerMethodName,
    )

    # Store the payload exactly (as dict) so we can re-score later.
    meta = write_response(data.model_dump(), directory_path=DEFAULT_STUDY_DIR)

    return {
        "ok": True,
        "participantId": data.participantId,
        "stored": meta,
    }


@router.post("/study/score")
def get_study_score(query: StudyScoreQuery) -> Dict[str, Any]:
    """Compute aggregated scores from all stored participant payloads.

    Method names are derived automatically from the stored responses —
    no hard-coded list required.
    """
    payloads = read_all_payloads(directory_path=DEFAULT_STUDY_DIR)

    if not payloads:
        return {
            "methods": {},
            "summary": {
                "total_participants": 0,
                "ranked_methods": [],
                "alpha": query.alpha,
                "num_outfits": query.num_outfits,
                "storage_path": DEFAULT_STUDY_DIR,
            },
        }

    # Collect every unique methodName that appears across all payloads
    methods: List[str] = sorted(
        {
            r["methodName"]
            for p in payloads
            for r in p.get("responses", [])
        }
    )

    result = score_methods(methods, payloads, alpha=query.alpha, num_outfits=query.num_outfits)

    result.setdefault("summary", {})
    result["summary"].update({
        "alpha": query.alpha,
        "num_outfits": query.num_outfits,
        "storage_path": DEFAULT_STUDY_DIR,
    })

    return result
