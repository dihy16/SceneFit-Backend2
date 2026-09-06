# Testing the 4 Retrieval Methods

The SceneFit Backend exposes individual endpoints for each of the four main retrieval methods:
1. **CLIP** (`/api/v1/retrieval/clip`)
2. **Image Editing** (`/api/v1/retrieval/image_edit`)
3. **Vision-Language Model (VLM)** (`/api/v1/retrieval/vlm`)
4. **Aesthetic Predictor** (`/api/v1/retrieval/aesthetic`)

This guide explains how to test these methods both locally and against a remote server (like Google Colab via Ngrok).

---

## 1. Using the Automated Test Script

We have provided a Python script in the `tests/` directory that automatically tests all 4 methods sequentially using a sample image.

### Prerequisites
Make sure you have `requests` installed locally:
```bash
pip install requests
```

### Running the Script
Run the `test_endpoints.py` script and pass your backend's base URL (e.g., your Ngrok URL or `http://localhost:8000`).

```bash
python tests/test_endpoints.py <YOUR_BASE_URL>
```

**Example:**
```bash
python tests/test_endpoints.py https://synonymic-knowledgeable-edgardo.ngrok-free.dev
```

The script will automatically upload `data/2d/m1_light_22.png` to all 4 endpoints and print the top 2 matching clothing items and their similarity scores for each method.

---

## 2. Using cURL (Command Line)

You can manually test any individual method using the `curl` command. This is useful for quick debugging without python.

### Format:
```bash
curl -X POST "<BASE_URL>/api/v1/retrieval/<METHOD_NAME>" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@<PATH_TO_IMAGE>" \
  -F "top_k=5"
```

### Example (Testing the CLIP method):
```bash
curl -X POST "https://synonymic-knowledgeable-edgardo.ngrok-free.dev/api/v1/retrieval/clip" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@data/2d/m1_light_22.png" \
  -F "top_k=5"
```

---

## 3. Using FastAPI Swagger UI (Browser)

FastAPI provides an interactive, visual interface to test all endpoints.

1. Open your browser and go to `<BASE_URL>/docs` (e.g., `https://synonymic-knowledgeable-edgardo.ngrok-free.dev/docs`).
2. Scroll down to the **retrieval** section.
3. Click on the endpoint you want to test (e.g., `POST /api/v1/retrieval/{method_name}`).
4. Click **"Try it out"**.
5. Type the name of the method in the `method_name` box (e.g., `clip`, `image_edit`, `vlm`, or `aesthetic`).
6. Click **"Choose File"** and upload an image from your computer.
7. Click **"Execute"** and view the results in the window below!

---

## Important Note on Mock Data
As explained in the `README.md`, if the remote model workers for these 4 methods are offline (or if their URLs in `config/retrieval_methods.yaml` are incorrect), the backend will elegantly catch the `404 Not Found` error and generate **mock data** (e.g., `outfit_7.png`) so the frontend doesn't crash. 

If you see randomly generated mock names instead of real clothing items, check that your model workers are online!
