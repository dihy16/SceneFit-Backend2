from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "Multi-Model AI Backend"
    API_VERSION: str = "v1"
    DEVICE: str = "cuda"
    LOG_LEVEL: str = "INFO"

    IMAGEROUTER_API_KEY: str = Field(..., description="ImageRouter API key")
    HF_TOKEN: Optional[str] = Field(None, description="Hugging Face authentication token")
    MAIN_PREFIX: str = "https://wifelier-melita-soapiest.ngrok-free.dev/"

    model_config = {
        "env_file": ".env",
        "extra": "forbid",
    }

settings = Settings()

# Automatically authenticate with Hugging Face if token is available
if settings.HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=settings.HF_TOKEN, add_to_git_credential=False)
        print(f"[INFO] Logged in to Hugging Face Hub.")
    except ImportError:
        pass  # huggingface_hub not installed yet
