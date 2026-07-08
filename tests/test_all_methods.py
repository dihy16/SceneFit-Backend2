from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)

with open("data/2d/m1_light_22.png", "rb") as f:
    response = client.post("/api/v1/retrieval/all-methods", files={"image": ("m1_light_22.png", f, "image/png")})

print(f"Status Code: {response.status_code}")
print(json.dumps(response.json(), indent=2))
