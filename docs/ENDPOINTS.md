# SceneFit-Backend API Endpoints

The SceneFit-Backend application acts as a central facade that routes retrieval requests to remote worker services and aggregates their results. It also hosts endpoints for the User Study application.

### Base URL
All endpoints are served under:
```
/api/v1
```

## Retrieval API

The retrieval system is modular, with methods registered in `app/services/retrieval/strategies.py` and configured in `config/retrieval_methods.yaml`.

### `POST /retrieval/all-methods`
Runs all registered retrieval strategies (`clip`, `image_edit`, `vlm`, `aesthetic`) in parallel.

**Request:** `multipart/form-data`
- `image` (file, required): The background/scene image.
- `top_k` (int, optional, default: `5`): Number of results to retrieve per method.

**Response:** `application/json`
A dictionary mapping each strategy name to its array of retrieved outfits.
```json
{
  "clip": [
    { "name": "item1", "score": 0.95, "image_url": "http://localhost:8000/images/item1.png" }
  ],
  "image_edit": [
    { "name": "item2", "score": 0.88, "image_url": "http://localhost:8000/images/item2.png" }
  ],
  "vlm": [],
  "aesthetic": []
}
```

### `POST /retrieval/{method_name}`
Runs a specific retrieval strategy. 
*Valid method names:* `clip`, `image_edit`, `vlm`, `aesthetic`.

**Request:** `multipart/form-data`
- `image` (file, required): The background/scene image.
- `top_k` (int, optional, default: `5`): Number of results to retrieve.

**Response:** `application/json`
An array of retrieved outfits.
```json
[
  { "name": "item1", "score": 0.95, "image_url": "http://localhost:8000/images/item1.png" },
  { "name": "item2", "score": 0.88, "image_url": "http://localhost:8000/images/item2.png" }
]
```

## User Study API

### `POST /study/response`
Submits a participant's evaluation response from the study UI.

**Request:** `application/json`
```json
{
  "responses": [
    {
      "methodName": "Vision Language Model",
      "imgURLs": ["http://host/images/item1.png", "http://host/images/item2.png"],
      "selectedURL": "http://host/images/item1.png",
      "viewCounts": [2, 1]
    }
  ],
  "winnerMethodName": "Vision Language Model"
}
```

**Response:** `application/json`
```json
{
  "ok": true,
  "participantId": "uuid-string",
  "stored": {
    "file_path": "user_study/responses.json",
    "entry_index": 0,
    "receivedAt": "2026-03-06T12:00:00Z"
  }
}
```

### `POST /study/score`
Computes aggregated scores across all stored participant responses.

**Request:** `application/json`
- `alpha` (float, default: `0.6`): Weighting factor between rank score and win rate.
- `num_outfits` (int, default: `5`): Number of outfits per method.
```json
{
  "alpha": 0.6,
  "num_outfits": 5
}
```

**Response:** `application/json`
Returns aggregated scoring and statistics.
