"""
Mock API integration test
--------------------------
Sends real HTTP requests to the running FastAPI server using a small dummy
PNG image. All retrieval strategies return mock data (no upstream calls are
made) because mock=True is the default when the strategy cannot reach the
configured ngrok URLs.

Run with:
    python test_mock_api.py

The server must be running on http://127.0.0.1:8000 first.
"""

import io
import json
import sys
import struct
import zlib

import requests

BASE_URL = "http://127.0.0.1:8000/api/v1/retrieval"
TOP_K = 3

# ---------------------------------------------------------------------------
# Minimal valid 1x1 red PNG (no external dependency)
# ---------------------------------------------------------------------------

def _make_png() -> bytes:
    """Build a minimal but valid 1x1 red PNG in pure Python."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr   = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw    = b"\x00\xff\x00\x00"          # filter byte + R G B
    idat   = chunk(b"IDAT", zlib.compress(raw))
    iend   = chunk(b"IEND", b"")
    return header + ihdr + idat + iend

DUMMY_PNG = _make_png()


def _post(endpoint: str, label: str) -> dict | list | None:
    url = f"{BASE_URL}/{endpoint}"
    files = {"image": ("test.png", io.BytesIO(DUMMY_PNG), "image/png")}
    data  = {"top_k": TOP_K}

    try:
        resp = requests.post(url, files=files, data=data, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        print(f"[ERROR] Cannot connect to {url}. Is the server running?")
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"[FAIL] {label} — HTTP {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        print(f"[FAIL] {label} — {e}")
        return None


def _print_results(label: str, data: dict | list | None):
    if data is None:
        return
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(json.dumps(data, indent=2)[:1200])   # cap output length


def _assert_result_list(name: str, results) -> bool:
    """Validate a strategy result list has the expected shape."""
    if not isinstance(results, list):
        print(f"  [FAIL] {name}: expected list, got {type(results).__name__}")
        return False
    if len(results) == 0:
        print(f"  [WARN] {name}: empty result list")
        return True
    item = results[0]
    missing = [k for k in ("name", "score", "image_url") if k not in item]
    if missing:
        print(f"  [FAIL] {name}: missing keys {missing} in first result")
        return False
    print(f"  [PASS] {name}: {len(results)} result(s), top = {item['name']} (score={item['score']:.4f})")
    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_individual_methods():
    print("\n" + "#"*60)
    print("# Test 1 — individual /{method_name} endpoints")
    print("#"*60)

    methods = ["clip", "image_edit", "vlm", "aesthetic"]
    all_pass = True

    for method in methods:
        result = _post(method, method)
        _print_results(method.upper(), result)
        ok = _assert_result_list(method, result)
        all_pass = all_pass and ok

    return all_pass


def test_all_methods():
    print("\n" + "#"*60)
    print("# Test 2 — /all-methods aggregator")
    print("#"*60)

    result = _post("all-methods", "all-methods")
    _print_results("ALL-METHODS", result)

    if not isinstance(result, dict):
        print("  [FAIL] all-methods: expected dict response")
        return False

    all_pass = True
    for method, data in result.items():
        if isinstance(data, dict) and data.get("error"):
            print(f"  [WARN] {method}: service returned error — {data.get('message', '')[:100]}")
        else:
            ok = _assert_result_list(method, data)
            all_pass = all_pass and ok

    return all_pass


def test_unknown_method():
    print("\n" + "#"*60)
    print("# Test 3 — unknown method returns 404")
    print("#"*60)

    url = f"{BASE_URL}/nonexistent_method"
    files = {"image": ("test.png", io.BytesIO(DUMMY_PNG), "image/png")}
    data  = {"top_k": TOP_K}

    resp = requests.post(url, files=files, data=data, timeout=10)
    if resp.status_code == 404:
        print(f"  [PASS] nonexistent_method: correctly returned 404")
        return True
    else:
        print(f"  [FAIL] nonexistent_method: expected 404, got {resp.status_code}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Target: {BASE_URL}")
    print(f"top_k:  {TOP_K}")

    results = [
        test_individual_methods(),
        test_all_methods(),
        test_unknown_method(),
    ]

    print("\n" + "="*60)
    if all(results):
        print("ALL TESTS PASSED")
    else:
        failed = results.count(False)
        print(f"FAILED — {failed} test group(s) had failures")
        sys.exit(1)
