# Mock Functions for Testing

This directory contains mock versions of the retrieval functions that return test data without making remote API calls.

## Files

- **`all_methods_mock.py`**: Mock implementations of all retrieval methods
- **`test_mock_functions.py`**: Example script demonstrating usage

## Mock Functions

The following mock functions are available:

1. `get_clip_results_mock(image, top_k)` - Mock CLIP retrieval
2. `get_image_edit_results_mock(image, top_k)` - Mock image edit retrieval
3. `get_vlm_results_mock(image, top_k)` - Mock VLM retrieval
4. `get_aes_results_mock(image, top_k)` - Mock aesthetic predictor retrieval

## Features

### Automatic Outfit Name Loading
- Mock functions automatically load outfit names from `app/data/data/2d/`
- Outfit names are cached for performance
- Falls back to default outfit names if directory is not accessible

### Random Selection
- Randomly selects outfit names from available data
- Each call returns different results (randomized)
- Ensures no duplicates in a single result set

### Realistic Scores
- Generates realistic similarity scores (0.0 to 1.0)
- Different methods have different score ranges
- Scores decrease as ranking position increases
- Small random variations added for realism

## Usage

### Basic Usage

```python
from io import BytesIO
from fastapi import UploadFile
from app.services.all_methods_mock import get_clip_results_mock

# Create a dummy image file
content = b"dummy image content"
file = BytesIO(content)
image = UploadFile(filename="test.jpg", file=file)

# Get mock results
results = get_clip_results_mock(image, top_k=10)

# Results format: [{"name": "outfit_name", "score": 0.95}, ...]
for result in results:
    print(f"{result['name']}: {result['score']}")
```

### Using in API Endpoints

To use mock functions in your API during testing, you can conditionally import them:

```python
import os
from fastapi import UploadFile

# Check if we're in mock mode
USE_MOCK = os.getenv("USE_MOCK_DATA", "false").lower() == "true"

if USE_MOCK:
    from app.services.all_methods_mock import (
        get_clip_results_mock as get_clip_results,
        get_image_edit_results_mock as get_image_edit_results,
        get_vlm_results_mock as get_vlm_results,
        get_aes_results_mock as get_aes_results,
    )
else:
    from app.services.all_methods import (
        get_clip_results,
        get_image_edit_results,
        get_vlm_results,
        get_aes_results,
    )

# Use the functions normally
def some_endpoint(image: UploadFile):
    results = get_clip_results(image, top_k=5)
    return results
```

Then set the environment variable:
```bash
# Use mock data
export USE_MOCK_DATA=true

# Use real API calls
export USE_MOCK_DATA=false
```

### Running the Test Script

To test all mock functions:

```bash
python test_mock_functions.py
```

Expected output:
```
============================================================
Testing Mock Functions
============================================================

Requesting top_5 results from each method

------------------------------------------------------------
Testing CLIP Mock:
------------------------------------------------------------
[CLIP] MOCK - Simulating retrieval with top_k=5
[CLIP] MOCK - Generated 5 mock results
  1. m1_light_5: 0.9712
  2. avatars_00320df985504a278f628f6b168d2495: 0.9201
  3. m5_brown_3: 0.8754
  4. m6_light_8: 0.8289
  5. m1_dark_1: 0.7756

... (similar for other methods)
```

## Clearing the Cache

If you modify the files in `app/data/data/2d/` and want to reload the outfit names:

```python
from app.services.all_methods_mock import clear_outfit_cache

clear_outfit_cache()
# Next call will reload outfit names from disk
```

## Testing with Different Top-K Values

```python
# Get different numbers of results
results_5 = get_clip_results_mock(image, top_k=5)   # 5 results
results_10 = get_clip_results_mock(image, top_k=10) # 10 results
results_20 = get_clip_results_mock(image, top_k=20) # 20 results
```

## Score Ranges by Method

Different methods have different base score ranges:

- **CLIP**: 0.95 base, 0.05 decay per rank
- **Image Edit**: 0.92 base, 0.04 decay per rank
- **VLM**: 0.88 base, 0.06 decay per rank
- **Aesthetic**: 0.90 base, 0.03 decay per rank

All scores include small random variations (±0.02) for realism.

## Notes

- Mock functions don't actually process the image - they ignore the image parameter
- Results are randomized on each call
- Outfit names always match actual files in `app/data/data/2d/`
- No network calls are made
- No models are loaded
- Fast execution for rapid testing
