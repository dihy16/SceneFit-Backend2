from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from uuid import uuid4
import os
import time

app = FastAPI(title="SceneFit Mock API")

# Serve local images
app.mount("/images", StaticFiles(directory="images", check_dir=False), name="images")


# -----------------
# Study endpoint mocks (match app/api/v1/endpoints/experiment_ep.py)
# -----------------


class UnityMethodResponse(BaseModel):
    methodName: str
    imgURLs: List[str] = Field(..., description="top-k images of the current method")
    selectedURL: str = Field(..., description="URL of the image the participant selected for this method.")
    viewCounts: Optional[List[int]] = Field(
        None,
        description="Optional per-outfit counts of 'View' button clicks (length == num_outfits).",
    )


class UnityParticipantPayload(BaseModel):
    participantId: str
    responses: List[UnityMethodResponse]
    winnerMethodName: str


class UnityParticipantSubmission(BaseModel):
    responses: List[UnityMethodResponse]
    winnerMethodName: str


class StudyScoreQuery(BaseModel):
    alpha: float = Field(0.6, ge=0.0, le=1.0)
    num_outfits: int = Field(5, ge=1)


# If true, behave like the real study API: append JSONL on /study/response and
# compute score from stored payloads on /study/score.
# Toggle via env var: SCENEFIT_MOCK_PERSIST_STUDY=1
PERSIST_STUDY_DATA = os.getenv("SCENEFIT_MOCK_PERSIST_STUDY", "0") in {"1", "true", "True", "yes", "YES"}

DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data")
DEFAULT_JSON_PATH = os.path.join(DEFAULT_DATA_DIR, "user_study_responses.json")


@app.post("/study/response")
def submit_study_response(
    payload: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    time.sleep(1)

    participant_id = str(uuid4())

    if PERSIST_STUDY_DATA:
        from app.services.study.storage import append_response

        append_payload = {
            "participantId": participant_id,
            **payload,
        }

        meta = append_response(append_payload, file_path=DEFAULT_JSON_PATH)
        return {"ok": True, "participantId": participant_id, "stored": meta}

    return {
        "ok": True,
        "participantId": participant_id,
        "stored": {
            "path": "mock://static",
            "bytes": len(str(payload).encode("utf-8")),
            "appended": True,
        },
    }

@app.post("/mock-api")
async def all_methods(
    image: UploadFile = File(...),
    top_k: int = Form(5)
):
    time.sleep(10)

    def item(name, score):
        return {
            "name": name,
            "score": score,
            "image_url": f"http://localhost:8000/images/{name}.jpg"
        }

    return {
        "imageEdit": [
            item("avatars_0a1a3c69395646c88d765d3d03161400", 0.84),
            item("avatars_0a4a89d3ac42468d8c59d9b5f7014e92", 0.83),
            item("avatars_7ef08152199347c1b47a758a9b48b5f6", 0.82),
            item("avatars_7f2d3860795a4e35901c0154362a8424", 0.81),
            item("avatars_7fb3a21fae0f429a8baf8ebec38cd18f", 0.80),
        ],
        "vlm": [
            item("avatars_0a6ef917235d491f827871a5618b5bbf", 0.84),
            item("avatars_0a4a89d3ac42468d8c59d9b5f7014e92", 0.83),
            item("avatars_7ef08152199347c1b47a758a9b48b5f6", 0.82),
            item("avatars_7f2d3860795a4e35901c0154362a8424", 0.81),
            item("avatars_7fb3a21fae0f429a8baf8ebec38cd18f", 0.80),
            
        ],
        "clip": [
            item("avatars_0a5d20607a3b47d5b5ccf73f40529b6d", 0.84),
            item("avatars_0a4a89d3ac42468d8c59d9b5f7014e92", 0.83),
            item("avatars_7ef08152199347c1b47a758a9b48b5f6", 0.82),
            item("avatars_7f2d3860795a4e35901c0154362a8424", 0.81),
            item("avatars_8b28b9b622c8497d9751ad11f97b5773", 0.80),
        ],
        "aes": [
            item("avatars_0a4a89d3ac42468d8c59d9b5f7014e92", 0.84),
            item("avatars_8b28b9b622c8497d9751ad11f97b5773", 0.83),
            item("avatars_7fb3a21fae0f429a8baf8ebec38cd18f", 0.82),
            item("avatars_7f2d3860795a4e35901c0154362a8424", 0.81),
            item("avatars_7ef08152199347c1b47a758a9b48b5f6", 0.80),
        ],
    }
    
# avatars_7ef08152199347c1b47a758a9b48b5f6
# avatars_7f2d3860795a4e35901c0154362a8424
# avatars_7fb3a21fae0f429a8baf8ebec38cd18f
# avatars_8a0c81e649a8442cbc4d60a78997821e
# avatars_8b28b9b622c8497d9751ad11f97b5773

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "mock_app:app",  # make sure filename is mock_app.py
        host="0.0.0.0",
        port=8000,
        reload=True
    )