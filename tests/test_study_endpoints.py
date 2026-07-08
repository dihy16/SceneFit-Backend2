import requests
import json
import sys

def test_study_endpoints(base_url):
    print(f"Testing User Study endpoints using base URL: {base_url}\n")
    
    # 1. Test POST /api/v1/study/response
    response_url = f"{base_url}/api/v1/study/response"
    print(f"Testing POST {response_url}...")
    
    payload = {
      "responses": [
        {
          "methodName": "Image Editing",
          "imgURLs": [
            "http://host/img/ie0.jpg", 
            "http://host/img/ie1.jpg", 
            "http://host/img/ie2.jpg", 
            "http://host/img/ie3.jpg", 
            "http://host/img/ie4.jpg"
          ],
          "selectedURL": "http://host/img/ie2.jpg",
          "viewCounts": [1, 0, 3, 0, 1]
        },
        {
          "methodName": "CLIP Model",
          "imgURLs": [
            "http://host/img/clip0.jpg", 
            "http://host/img/clip1.jpg", 
            "http://host/img/clip2.jpg"
          ],
          "selectedURL": "http://host/img/clip1.jpg",
          "viewCounts": [0, 1, 0]
        }
      ],
      "winnerMethodName": "Image Editing"
    }
    
    try:
        resp = requests.post(response_url, json=payload)
        print(f"Status Code: {resp.status_code}")
        print("Response JSON:")
        print(json.dumps(resp.json(), indent=2))
        print("-" * 50 + "\n")
    except Exception as e:
        print(f"Failed to hit /api/v1/study/response: {e}")
        return

    # 2. Test POST /api/v1/study/score
    score_url = f"{base_url}/api/v1/study/score"
    print(f"Testing POST {score_url}...")
    
    score_payload = {
        "alpha": 0.6,
        "num_outfits": 5
    }
    
    try:
        resp = requests.post(score_url, json=score_payload)
        print(f"Status Code: {resp.status_code}")
        print("Response JSON:")
        print(json.dumps(resp.json(), indent=2))
        print("-" * 50 + "\n")
    except Exception as e:
        print(f"Failed to hit /study/score: {e}")

if __name__ == "__main__":
    url = "https://synonymic-knowledgeable-edgardo.ngrok-free.dev"
    if len(sys.argv) > 1:
        url = sys.argv[1].rstrip("/")
    test_study_endpoints(url)
