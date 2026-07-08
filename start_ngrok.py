import os
from pyngrok import ngrok
from dotenv import load_dotenv

# Load uploaded .env file to get NGROK_TOKEN and other secrets
# This ensures os.environ is updated for the Jupyter kernel,
# and subsequent commands will inherit these variables.
load_dotenv("/content/.env")

ngrok_token = os.environ.get("NGROK_TOKEN", "37hWdRD7mFcrgvepVbBpRWWU6z4_2BRnCfQfJhL9e3vHyjAW2")
if ngrok_token:
    ngrok.set_auth_token(ngrok_token)

ngrok.kill()
public_url = ngrok.connect(8000)

print("\n" + "="*70)
print("FastAPI Backend is live on Google Colab!")
print(f"Public URL:  {public_url}")
print(f"Swagger UI:  {public_url}/docs")
print("="*70 + "\n")
