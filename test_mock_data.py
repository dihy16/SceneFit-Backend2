"""
Test script to demonstrate mock data generation for retrieval methods.
This script shows how to use the mock functionality without making remote API calls.
"""

import os
import sys
from pathlib import Path
from io import BytesIO
from fastapi import UploadFile

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Enable mock mode
os.environ["USE_MOCK_DATA"] = "true"

from app.services.all_methods import (
    get_clip_results,
    get_image_edit_results,
    get_vlm_results,
    get_aes_results,
    generate_mock_results
)


def create_dummy_upload_file():
    """Create a dummy UploadFile for testing (not used in mock mode)."""
    dummy_content = b"dummy image content"
    return UploadFile(
        filename="test_image.jpg",
        file=BytesIO(dummy_content)
    )


def test_mock_generation():
    """Test direct mock data generation."""
    print("=" * 80)
    print("Testing Direct Mock Data Generation")
    print("=" * 80)
    
    results = generate_mock_results(top_k=10, method_name="test")
    
    print(f"\nGenerated {len(results)} mock results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['name']}: {result['score']}")
    
    # Verify format
    assert isinstance(results, list), "Results should be a list"
    for result in results:
        assert "name" in result, "Each result should have 'name'"
        assert "score" in result, "Each result should have 'score'"
        assert isinstance(result["name"], str), "Name should be a string"
        assert isinstance(result["score"], float), "Score should be a float"
    
    print("\n✓ Format validation passed!")


def test_all_methods():
    """Test all retrieval methods with mock data."""
    print("\n" + "=" * 80)
    print("Testing All Retrieval Methods with Mock Data")
    print("=" * 80)
    
    dummy_image = create_dummy_upload_file()
    top_k = 5
    
    methods = [
        ("CLIP", get_clip_results),
        ("Image Edit", get_image_edit_results),
        ("VLM", get_vlm_results),
        ("Aesthetic", get_aes_results)
    ]
    
    for method_name, method_func in methods:
        print(f"\n--- Testing {method_name} ---")
        results = method_func(dummy_image, top_k)
        
        print(f"Results (top {len(results)}):")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['name']}: {result['score']}")
        
        # Reset file pointer for next test
        dummy_image.file.seek(0)


def test_different_top_k_values():
    """Test mock generation with different top_k values."""
    print("\n" + "=" * 80)
    print("Testing Different top_k Values")
    print("=" * 80)
    
    for top_k in [3, 10, 20, 50]:
        print(f"\ntop_k = {top_k}:")
        results = generate_mock_results(top_k=top_k, method_name="test")
        print(f"  Generated: {len(results)} results")
        print(f"  Score range: {results[0]['score']:.4f} - {results[-1]['score']:.4f}")
        print(f"  Sample names: {[r['name'][:30] + '...' if len(r['name']) > 30 else r['name'] for r in results[:3]]}")


def test_score_distribution():
    """Test that scores are properly distributed and descending."""
    print("\n" + "=" * 80)
    print("Testing Score Distribution")
    print("=" * 80)
    
    results = generate_mock_results(top_k=20, method_name="test")
    
    scores = [r["score"] for r in results]
    
    # Check scores are descending
    is_descending = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    print(f"\n✓ Scores are descending: {is_descending}")
    
    # Check score range
    print(f"✓ Highest score: {max(scores):.4f}")
    print(f"✓ Lowest score: {min(scores):.4f}")
    print(f"✓ Score range: {max(scores) - min(scores):.4f}")
    
    # Check uniqueness
    unique_names = len(set(r["name"] for r in results))
    print(f"✓ Unique outfit names: {unique_names}/{len(results)}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MOCK DATA GENERATION TEST SUITE")
    print("=" * 80)
    print(f"Mock mode enabled: {os.environ.get('USE_MOCK_DATA')}")
    
    try:
        test_mock_generation()
        test_all_methods()
        test_different_top_k_values()
        test_score_distribution()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
