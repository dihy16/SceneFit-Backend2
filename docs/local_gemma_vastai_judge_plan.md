# Local Gemma 4 26B A4B Judge for Vast.ai

## Summary

Add a second pairwise-judge backend that sends multimodal requests to a locally
hosted `llama.cpp` server instead of Google’s API. Keep the existing Swiss
tournament, mirrored comparisons, Elo calculation, checkpointing, and
evaluation unchanged. Start the local Q4 run in a new ground-truth directory so
hosted and quantized judgments are never mixed.

Use Google’s official multimodal, instruction-tuned QAT Q4_0 checkpoint:
`google/gemma-4-26B-A4B-it-qat-q4_0-gguf:Q4_0`. It includes a 14.4 GB model
file and a 1.19 GB vision projector. A 48 GB Vast.ai GPU is the recommended
profile for four concurrent requests.

References:

- [Official Q4 checkpoint](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/tree/main)
- [Gemma 4 memory and QAT guidance](https://ai.google.dev/gemma/docs/core)
- [llama.cpp multimodal server documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)

## Implementation Changes

- Add a `llama-cpp` implementation of the existing pairwise judge interface.
  - POST to `http://127.0.0.1:8080/v1/chat/completions`.
  - Send the scene, left outfit, and right outfit as base64 image content in the
    existing prompt order.
  - Reuse the current five criteria, balanced few-shot rubric, one-pair
    default, and completion checklist.
  - Request schema-constrained JSON using `PairDecisionBatch.model_json_schema()`.
  - Disable thinking, use temperature `0`, cap output tokens, validate with
    Pydantic, and retain bounded retry/backoff.
  - Run a startup health/model check before the pilot or tournament.

- Extend benchmark commands with:
  - `--judge-backend {gemini,llama-cpp}`, defaulting to `gemini` for backward
    compatibility.
  - `--judge-base-url`, defaulting to `http://127.0.0.1:8080/v1`.
  - `--request-timeout`, with a local-model-safe default of 300 seconds.
  - Reuse `--model`; set the local model to
    `google/gemma-4-26B-A4B-it-qat-q4_0-gguf:Q4_0`.
  - Apply these options consistently to `pilot`, `judge`, `scene`, `run`,
    `validate-judge`, and runtime helpers.

- Record `judge_backend`, model identity, quantization, and prompt version in
  `judge.meta.json`, response rows, and progress.
  - Existing Gemini artifacts without `judge_backend` remain readable as legacy
    Gemini artifacts.
  - Local runs require matching local metadata when resumed or validated.
  - A local run cannot resume into the existing API ground-truth directory.

- Add a Vast.ai runbook covering server installation, the model launch,
  health check, pilot, resumable judging, validation, evaluation, download, and
  shutdown. No `GEMINI_API_KEY` is required for this backend.

## Intended Run

Run the local server on the Vast.ai machine first. Use a 48 GB GPU with four
server slots and benchmark concurrency 4. For a 24 GB GPU, use one or two
slots and the matching benchmark concurrency.

```powershell
python scripts/benchmark.py pilot `
  --judge-backend llama-cpp `
  --judge-base-url http://127.0.0.1:8080/v1 `
  --model "google/gemma-4-26B-A4B-it-qat-q4_0-gguf:Q4_0" `
  --manifest results/benchmark/all-methods-evaluation/manifest.json

python scripts/benchmark.py judge `
  --judge-backend llama-cpp `
  --judge-base-url http://127.0.0.1:8080/v1 `
  --model "google/gemma-4-26B-A4B-it-qat-q4_0-gguf:Q4_0" `
  --manifest results/benchmark/all-methods-evaluation/manifest.json `
  --judge-dir results/benchmark/pairwise_ground_truth_gemma4_26b_q4_local `
  --pairs-per-request 1 `
  --concurrency 4
```

The clean local run will retain the existing protocol: seven Swiss rounds,
mirrored orientation, 3,500 reconciled comparisons, and 7,000 local inference
requests. Once validation succeeds, evaluate the existing `clip`, `aesthetic`,
`vlm`, and `image_edit` rankings without rerunning retrieval.

## Test Plan

- Unit-test multimodal request ordering, MIME/base64 encoding, JSON schema,
  response parsing, retry behavior, timeout handling, and missing pair IDs.
- Verify selecting `llama-cpp` never reads `GEMINI_API_KEY`.
- Verify legacy Gemini checkpoints still resume and validate.
- Verify API and local metadata cannot be mixed.
- Run the existing pairwise tournament tests plus an HTTP-mocked local
  end-to-end test.
- On Vast.ai, require a successful health check and pilot before the full
  tournament.
- After completion, run `validate-judge`, then score the four cached ranking
  files.
