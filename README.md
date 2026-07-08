# SceneFit-Backend
A scene-aware system that retrieves suitable clothing based on environmental context.

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

**API Endpoints (Retrieval - Common Format)**
- Base path: `/api/v1/retrieval`
- Request: `multipart/form-data` with required `image` (file). Common optional fields: `top_k` (int, default 5);
- Response envelope (always): `method` (str), `count` (int), `results` (list).
- In `results`, it must always have `outfit_name`, `score`.

This is the required parameters
```jsonc
{
  "image": "<file>",          // required
  "top_k": 5,                  // optional (default 5), int
}
```

**Method: image-edit**
- Endpoint: `POST /api/v1/retrieval/image-edit`
- Request extras: `gender` ("male"|"female"), `crop_clothes` (bool): for croping image or not before retrieval, `return_metadata` (bool; when false, omit `outfit_path`).
- Response extras: `gender` (str), `edited_image_path` (string). 
- Request extras:
  - `gender` ("male"|"female")
  - `crop_clothes` (bool)
  - `return_metadata` (bool; when false, omit `outfit_path`)
  - `preference_text` (string, optional) — preferred style or intent
  - `preference_audio` (file, optional) — speech alternative to text; backend transcribes if text is absent
- Response extras: `gender` (str), `edited_image_path` (string), optional `ref_image_path` (string), `session_id` (string for follow-up). Bulk variant (`POST /api/v1/retrieval/image-edit/retrieve-all`) also returns `bg_path`.

Audio feedback loop:
- Endpoint: `POST /api/v1/retrieval/image-edit/apply-feedback`
- Request fields:
  - `session_id` (string, required; from the initial image-edit response)
  - `top_k` (int, optional)
  - `crop_clothes` (bool, optional)
  - `feedback_text` (string, optional)
  - `feedback_audio` (file, optional; used if text is absent)
- Response: same envelope (`method`, `count`, `results`) plus `session_id`, `edited_image_path`.

Example response (image-edit):

```json
{
  "method": "image-edit",
  "count": 3,
  "gender": "male",                  // optional
  "edited_image_path": "data/edited_image/12.jpg", // optional
  "ref_image_path": "data/man.png",                // optional
  "session_id": "5f0c8c2e8c5d4f1c8e7b3a2f1d6e4c9a",    // for apply-feedback
  "results": [
    { "outfit_name": "item17", "score": 0.83 },
    { "outfit_name": "item04", "score": 0.79 },
    { "outfit_name": "item09", "score": 0.76 }
  ]
}
```

Example request (image-edit) — pseudo JSON for multipart form:

```jsonc
{
  "image": "<file>",          // required
  "top_k": 5,
  "gender": "male",
  "crop_clothes": true,
  "return_metadata": true,
  "preference_text": "sporty winter outfit",
  "preference_audio": "<audio file>"  // optional; used if preference_text is empty
}
```

Example request (apply-feedback) — pseudo JSON for multipart form:

```jsonc
{
  "session_id": "5f0c8c2e8c5d4f1c8e7b3a2f1d6e4c9a", // from initial response
  "top_k": 5,
  "crop_clothes": true,
  "feedback_text": "make the jacket darker",
  "feedback_audio": "<audio file>" // optional; used if feedback_text is empty
}
```

**VLM Faiss Composed Fashion Retrieval**
- Endpoint: `POST /api/v1/retrieval/vlm-faiss-composed-retrieval`

Example response (vlm-faiss-composed-retrieval):

```json
{
  "method": "vlm-faiss-composed-retrieval",
  "count": 10,
  "count": 10,
  "scene_caption": "This vibrant, sunlit outdoor scene features a lush purple floral meadow under a bright blue sky, with cascading purple trees and rocky terrain, suggesting a casual, festive spring or summer outing where bold, complementary colors like soft pastels or earthy tones would harmonize with the vivid purple palette.",
  "results": [
    {
      "name_clothes": "m1_light_22.png",
      "similarity": 0.30995649099349976,
      "rerank_score": 0.3404726982116699
    },
    {
      "name_clothes": "m6_brown_5.png",
      "similarity": 0.3073599338531494,
      "rerank_score": 0.32914280891418457
    },
    {
      "name_clothes": "m1_light_13.png",
      "similarity": 0.287604421377182,
      "rerank_score": 0.3193117380142212
    }    
  ],
  "best": {
    "name_clothes": "m1_light_22.png",
    "similarity": 0.30995649099349976,
    "rerank_score": 0.3404726982116699
  }
}
```

**All methods API**
- Endpoint: `POST /api/v1/retrieval/all-methods`

- Request: `multipart/form-data`

```jsonc
{
  "image": "<file>",          // required
  "top_k": 5,                  // optional (default 5), int
}
```

Example response:

```jsonc
{
   "imageEdit": [{
      "name": "name1",
      "score": 0.24,
      "image_url": "http://localhost:8000/images/{name1}.jpg",
   }],
   "vlm": [],
   "clip": [],
   "aesthetic": []
}
```


## User Study API

### Submit a participant response

- Endpoint: `POST /study/response`
- Content-Type: `application/json`
- Purpose: Append a single participant payload to an on-disk JSONL file (append-only).

Request body:

The client sends the list of per-method responses (each containing the method's top-k image URLs and the URL of the selected image), and the name of the overall winning method.
The backend auto-generates a `participantId` (UUID4). Method names are free-form strings; they are **not** hard-coded on the backend.

```json
{
  "responses": [
    {
      "methodName": "Image Editing",
      "imgURLs": ["http://host/img/ie0.jpg", "http://host/img/ie1.jpg", "http://host/img/ie2.jpg", "http://host/img/ie3.jpg", "http://host/img/ie4.jpg"],
      "selectedURL": "http://host/img/ie2.jpg",
      "viewCounts": [1, 0, 3, 0, 1]
    },
    {
      "methodName": "Vision Language Model",
      "imgURLs": ["http://host/img/vlm0.jpg", "http://host/img/vlm1.jpg", "http://host/img/vlm2.jpg", "http://host/img/vlm3.jpg", "http://host/img/vlm4.jpg"],
      "selectedURL": "http://host/img/vlm0.jpg",
      "viewCounts": [2, 1, 0, 1, 0]
    },
    {
      "methodName": "CLIP Model",
      "imgURLs": ["http://host/img/clip0.jpg", "http://host/img/clip1.jpg", "http://host/img/clip2.jpg", "http://host/img/clip3.jpg", "http://host/img/clip4.jpg"],
      "selectedURL": "http://host/img/clip1.jpg"
    },
    {
      "methodName": "Asthetic Model",
      "imgURLs": ["http://host/img/aes0.jpg", "http://host/img/aes1.jpg", "http://host/img/aes2.jpg", "http://host/img/aes3.jpg", "http://host/img/aes4.jpg"],
      "selectedURL": "http://host/img/aes3.jpg"
    }
  ],
  "winnerMethodName": "Vision Language Model"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `responses` | `List[Object]` | ✅ | One entry per method the participant evaluated. |
| `responses[].methodName` | `string` | ✅ | Display name / identifier of the method. |
| `responses[].imgURLs` | `List[string]` | ✅ | Top-k image URLs that were shown for this method. |
| `responses[].selectedURL` | `string` | ✅ | URL of the image the participant selected (must appear in `imgURLs`). |
| `responses[].viewCounts` | `List[int]` | ❌ | Per-outfit "View" button click counts. Length should equal the number of outfits. |
| `winnerMethodName` | `string` | ✅ | The client's chosen overall winning method. |

Response:

```json
{
  "ok": true,
  "participantId": "<generated-uuid>",
  "stored": {
    "file_path": "data/responses.json",
    "entry_index": 0,
    "receivedAt": "2026-03-06T12:00:00Z"
  }
}
```

Notes:

- `selectedURL` must be one of the URLs listed in the same response's `imgURLs`. The backend resolves it to a 0-based index for scoring.
- `winnerMethodName` is the participant's overall preferred method, sent by the client.
- `viewCounts` is optional. If provided, it must be a list of length `num_outfits` where each entry is the number of times the participant clicked the "View" button for that outfit index.
- Duplicate `methodName` entries in `responses` return HTTP 400.
- An empty `responses` list returns HTTP 400.

### Compute aggregated scores

- Endpoint: `POST /study/score`
- Content-Type: `application/json`
- Purpose: Score and rank all methods across stored participant responses.

Request body:

```json
{
  "alpha": 0.6,
  "num_outfits": 5
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `alpha` | `float` | `0.6` | Weighting factor between rank score (MRR) and win rate (0.0 – 1.0). |
| `num_outfits` | `int` | `5` | Number of outfits per method (used for rank normalisation). |

> Method names are **derived automatically** from the stored responses — no need to supply them.

Storage path behavior:

- The JSON file is stored at `<cwd>/data/user_study_responses.json` (relative to where you start the server).
- If no payloads exist yet, `/study/score` returns `methods: {}` and `total_participants: 0`.



**Contributor Guide (add a new retrieval endpoint)**
- Expose as `POST /api/v1/retrieval/<method-name>`; accept `image` plus `top_k` and method-specific knobs.
- Return the shared envelope `{ method, count, results }` with required result keys `outfit_name`, `outfit_path`, `score`. Add optional fields clearly (e.g., captions, paths) and gate them with flags like `return_metadata`.
- Avoid embedding images in responses; return file paths. Persist uploads under `app/uploads/` and method outputs in `app/data/` or `app/outputs/`.


## Data
[Google Drive Quan](https://drive.google.com/drive/folders/1Vii6WOEMJgGIVk5DmnA9ciK4llKHuDlQ?fbclid=IwY2xjawPuqwFleHRuA2FlbQIxMABicmlkETE4Wk5weUw1a2JObU9VODU3c3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHulNNSWtU4sttSuIjrjC0R8HTWYoUpwc8azMm8M6m0sLGfT4hw9tLZewWPx9_aem_6tlkSC-HOwumuWwDc2Tk2A)
[Google Drive Main](https://drive.google.com/drive/folders/1AAHqvWLGTxXsRxc85inFpjssEHwWWQsy?dmr=1&ec=wgc-drive-globalnav-goto)