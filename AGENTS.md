# Repository Guidelines

## Project Structure & Module Organization

`app/main.py` creates the FastAPI application. API routing lives under `app/api/v1/`, with endpoint modules in `app/api/v1/endpoints/`. Put orchestration and domain logic in `app/services/`, model wrappers in `app/models/`, shared infrastructure in `app/core/`, and reusable helpers in `app/utils/`. Runtime settings and prompts belong in `config/`. Utility and index-building programs live in `scripts/`; executable smoke and integration checks live in `tests/`. Treat `data/` as local/runtime input and `results/` as generated output; do not commit large model artifacts or datasets.

## Build, Test, and Development Commands

Create and activate a virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The second command runs the local API with reload; interactive docs are available at `/docs`. Run lightweight checks directly, for example `python tests/test_mock_data.py`. For HTTP integration checks, start the server first, then run `python tests/test_endpoints.py http://127.0.0.1:8000`. Some checks require model weights and images under `data/`. `./run_colab.sh` provisions and starts the GPU-backed Colab workflow; always finish with `./stop_colab.sh`.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_CASE` for constants. Endpoint modules use the existing `*_ep.py` suffix. Keep route handlers thin: validate HTTP input there, then delegate model or retrieval behavior to services. Add type hints to public functions and concise docstrings where behavior is not obvious. No formatter or linter is configured, so match adjacent code and keep imports grouped and focused.

## Testing Guidelines

Name new checks `tests/test_<feature>.py` and make failures return a nonzero exit status. Prefer FastAPI `TestClient` for deterministic endpoint tests; isolate remote workers and heavyweight models with mocks. Cover success responses, validation failures, and shared response keys. Note any required server, token, dataset, or GPU in the test module docstring.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, lowercase subjects such as `fix endpoint path` and `update README`. Keep each commit focused and describe the user-visible change. Pull requests should summarize behavior, list commands run, link relevant issues, and include sample requests/responses for API changes. Call out configuration, data, model, or GPU requirements explicitly.

## Security & Configuration

Copy `.env.example` to `.env` for local secrets. Never commit API keys, Hugging Face tokens, Ngrok credentials, private dataset URLs, or generated participant data. Keep worker URLs in `config/retrieval_methods.yaml` current without embedding credentials.
