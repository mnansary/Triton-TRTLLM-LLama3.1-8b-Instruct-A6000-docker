# MODIFY this file

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application configuration settings.
    Settings are loaded from the .env file, with defaults provided here.
    """
    # --- Triton Settings ---
    TRITON_URL: str = "localhost:8001"
    MODEL_NAME: str = "ensemble"
    DEFAULT_MODEL_ID: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # --- Generation Parameters ---
    # ADD THIS LINE to make max_tokens configurable
    MAX_TOKENS: int = 32768

    # --- Logging Settings ---
    # These paths assume the script is run from the project root
    LOG_FILE: str = "logs/app.log"
    LOG_LEVEL: str = "INFO"
    LOG_ROTATION: str = "1 day"
    
    # --- Gunicorn/Uvicorn Production Server Settings ---
    WORKERS: int = 4
    HOST: str = "0.0.0.0"
    PORT: int = 24434

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()