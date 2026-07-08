"""Verification script — run with the perception_models conda env."""
import sys

errors = []

# 1. Core strategy module
try:
    from app.services.retrieval_strategies import REGISTRY, BaseRetrievalStrategy
    names = REGISTRY.names()
    assert names == sorted(["clip", "image_edit", "vlm", "aesthetic"]), f"Unexpected strategy names: {names}"
    print(f"[PASS] retrieval_strategies: registered = {names}")
except Exception as e:
    errors.append(f"retrieval_strategies: {e}")

# 2. Backward-compat shim
try:
    from app.services.all_methods import (
        get_clip_results,
        get_image_edit_results,
        get_vlm_results,
        get_aes_results,
        generate_mock_results,
    )
    print("[PASS] all_methods shim imports OK")
except Exception as e:
    errors.append(f"all_methods shim: {e}")

# 3. Safe wrappers
try:
    from app.services.all_methods_safe import (
        get_clip_results_safe,
        get_image_edit_results_safe,
        get_vlm_results_safe,
        get_aes_results_safe,
    )
    print("[PASS] all_methods_safe imports OK")
except Exception as e:
    errors.append(f"all_methods_safe: {e}")

# 4. Adapter
try:
    from app.services.retrieval_adapter import (
        get_clip_results as adapt_clip,
        USE_MOCK_DATA,
    )
    print(f"[PASS] retrieval_adapter imports OK  (USE_MOCK_DATA={USE_MOCK_DATA})")
except Exception as e:
    errors.append(f"retrieval_adapter: {e}")

# 5. Endpoint router
try:
    from app.api.v1.endpoints.all_methods_ep import router
    routes = [r.path for r in router.routes]
    assert "/all-methods" in routes, f"/all-methods not found in {routes}"
    assert "/{method_name}" in routes, f"dynamic route not found in {routes}"
    print(f"[PASS] all_methods_ep router routes: {routes}")
except Exception as e:
    errors.append(f"all_methods_ep router: {e}")

# 6. Mock data generation smoke test
try:
    from app.services.retrieval_strategies import REGISTRY
    mock_results = REGISTRY.get("clip")._generate_mock(5)
    assert len(mock_results) <= 5
    for r in mock_results:
        assert "name" in r and "score" in r and "image_url" in r
    print(f"[PASS] mock generation returned {len(mock_results)} results, sample: {mock_results[0]}")
except Exception as e:
    errors.append(f"mock generation: {e}")

# Summary
print()
if errors:
    print(f"FAILED -- {len(errors)} error(s):")
    for err in errors:
        print(f"  FAIL: {err}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
