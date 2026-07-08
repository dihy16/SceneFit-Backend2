import requests
import json
import sys

def test_single_method(base_url, method_name, image_path):
    url = f"{base_url}/api/v1/retrieval/{method_name}"
    print(f"\n[{method_name.upper()}] Testing endpoint: {url}")
    
    try:
        with open(image_path, "rb") as f:
            response = requests.post(
                url, 
                files={"image": ("test_image.png", f, "image/png")},
                data={"top_k": 5}
            )
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print("Response JSON snippet (top 2 results):")
                results = response.json()
                if isinstance(results, dict) and "results" in results:
                    items = results["results"][:2]
                elif isinstance(results, list):
                    items = results[:2]
                else:
                    items = results
                print(json.dumps(items, indent=2))
            else:
                print("Error:", response.text)
    except Exception as e:
        print(f"Failed to test {method_name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_endpoints.py <NGROK_URL>")
        sys.exit(1)
        
    base_url = sys.argv[1].rstrip("/")
    image_path = "data/2d/m1_light_22.png" # Existing test image from test_all_methods.py
    
    methods = ["clip", "image_edit", "vlm", "aesthetic"]
    
    print(f"Testing 4 main retrieval methods using base URL: {base_url}")
    
    for method in methods:
        test_single_method(base_url, method, image_path)
