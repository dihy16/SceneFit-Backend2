# SceneFit-Backend
A scene-aware system that retrieves suitable clothing based on environmental context.

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

## Main Method: SaMaG-R Architecture

The architecture of SaMaG-R operates in four main stages, orchestrated by the `SaMaGPipeline` builder in `app/services/retrieval/samag_pipeline.py`:

| Pipeline Stage | Corresponding Function |
| --- | --- |
| **Input Stage** | `SaMaGPipeline` initialization |
| **Visual Encoding** (Path A) | `extract_visual_features()` |
| **Generative Encoding** (Path B) | `extract_semantic_features()` |
| **Orthogonal Rejection** (Query Formulation) | `formulate_query()` |
| **Faiss Vector Search** | `execute_faiss_search()` |
| **MMR Diversification** | `rerank_with_soft_penalty()` in `app/services/retrieval/post_processing.py` |
| **Reranking Stage** (Qwen VLM) | `rerank_candidates()` |

## API Endpoints (Retrieval)

This backend acts as a facade/router that delegates retrieval tasks to specialized remote model workers configured in `config/retrieval_methods.yaml`.

### Common Request Format
- **Request Type:** `multipart/form-data`
- **Fields:**
  - `image` (file, required): The scene/background image.
  - `top_k` (int, optional, default: 5): The number of clothing items to retrieve.

### 1. All Methods
- **Endpoint:** `POST /api/v1/retrieval/all-methods`
- **Description:** Runs every registered strategy (e.g., `clip`, `image_edit`, `vlm`, `aesthetic`) in parallel.
- **Example Response:**
```json
{
  "clip": [
    { "name": "m1_light_22", "score": 0.83, "image_url": "http://localhost:8000/images/m1_light_22.png" }
  ],
  "image_edit": [
    { "name": "item04", "score": 0.79, "image_url": "http://localhost:8000/images/item04.png" }
  ],
  "vlm": [],
  "aesthetic": []
}
```

### 2. Single Method
- **Endpoint:** `POST /api/v1/retrieval/{method_name}`
- **Description:** Runs a specific retrieval strategy by name. Current available methods: `clip`, `image_edit`, `vlm`, `aesthetic`.
- **Example Response:**
```json
[
  { "name": "m1_light_22", "score": 0.83, "image_url": "http://localhost:8000/images/m1_light_22.png" },
  { "name": "m6_brown_5", "score": 0.71, "image_url": "http://localhost:8000/images/m6_brown_5.png" }
]
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

## Running on Google Colab (via Google Colab CLI)

You can run the backend server on a Google Colab VM (with GPU hardware acceleration) using the official `google-colab-cli` tool.

### Prerequisites

1. Install the official `google-colab-cli` tool on your local machine (Linux, macOS, or WSL):
   ```bash
   uv tool install google-colab-cli
   # or
   pip install google-colab-cli
   ```
2. Log in and authenticate the CLI:
   ```bash
   colab auth login
   ```
3. (Optional) Create a `.env` file in the root of the project with your secrets, which will be uploaded to Colab automatically:
   ```env
   NGROK_TOKEN="your_ngrok_token_here"
   HF_TOKEN="your_hugging_face_token_here"
   ```

### Execution

Run the provided script:
```bash
chmod +x run_colab.sh
./run_colab.sh [session_name]
```
By default, the script:
- Verifies/creates a session named `scenefit-backend` with a `T4` GPU.
- Mounts Google Drive (requires a one-time permission prompt/confirmation).
- Uploads your local `.env` file to the remote environment securely.
- Executes setup via `setup_colab.py` (clones repo, extracts dataset, builds search index).
- Initializes the Ngrok tunnel via `start_ngrok.py`.
- Starts the FastAPI Uvicorn server in the foreground, streaming server logs directly to your local terminal using `run.sh`.