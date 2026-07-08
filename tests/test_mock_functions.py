"""
Example script demonstrating the use of mock functions for testing.
"""

from io import BytesIO
from fastapi import UploadFile
from app.services.all_methods_mock import (
    get_clip_results_mock,
    get_image_edit_results_mock,
    get_vlm_results_mock,
    get_aes_results_mock
)


def create_dummy_upload_file():
    """Create a dummy UploadFile for testing."""
    content = b"dummy image content"
    file = BytesIO(content)
    return UploadFile(filename="test_image.jpg", file=file)


def test_mock_functions():
    """Test all mock functions."""
    
    print("=" * 60)
    print("Testing Mock Functions")
    print("=" * 60)
    
    # Create a dummy upload file
    dummy_image = create_dummy_upload_file()
    top_k = 5
    
    print(f"\nRequesting top_{top_k} results from each method\n")
    
    # Test CLIP mock
    print("\n" + "-" * 60)
    print("Testing CLIP Mock:")
    print("-" * 60)
    clip_results = get_clip_results_mock(dummy_image, top_k)
    for i, result in enumerate(clip_results, 1):
        print(f"  {i}. {result['name']}: {result['score']}")
    
    # Reset file pointer
    dummy_image.file.seek(0)
    
    # Test Image Edit mock
    print("\n" + "-" * 60)
    print("Testing Image Edit Mock:")
    print("-" * 60)
    image_edit_results = get_image_edit_results_mock(dummy_image, top_k)
    for i, result in enumerate(image_edit_results, 1):
        print(f"  {i}. {result['name']}: {result['score']}")
    
    # Reset file pointer
    dummy_image.file.seek(0)
    
    # Test VLM mock
    print("\n" + "-" * 60)
    print("Testing VLM Mock:")
    print("-" * 60)
    vlm_results = get_vlm_results_mock(dummy_image, top_k)
    for i, result in enumerate(vlm_results, 1):
        print(f"  {i}. {result['name']}: {result['score']}")
    
    # Reset file pointer
    dummy_image.file.seek(0)
    
    # Test Aesthetic mock
    print("\n" + "-" * 60)
    print("Testing Aesthetic Mock:")
    print("-" * 60)
    aes_results = get_aes_results_mock(dummy_image, top_k)
    for i, result in enumerate(aes_results, 1):
        print(f"  {i}. {result['name']}: {result['score']}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_mock_functions()
