# llm_service/config.py

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application configuration settings.
    Settings are loaded from the .env file, with sensible defaults provided here.
    """
    # --- Triton and Model Settings ---
    TRITON_URL: str = "localhost:8001"
    MODEL_NAME: str = "ensemble"
    DEFAULT_MODEL_ID: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # --- Server Settings ---
    WORKERS: int = 4
    HOST: str = "0.0.0.0"
    PORT: int = 24434

    # --- Logging Settings ---
    LOG_FILE: str = "logs/app.log"
    LOG_LEVEL: str = "INFO"
    LOG_ROTATION: str = "1 day"

    # ===================================================================
    # --- Default Generation Parameters ---
    # This section defines the default behavior of the model.
    # These values can be overridden by the user in each API request
    # OR by setting them in the .env file for a system-wide override.
    # ===================================================================
    
    # --- Core Defaults ---
    DEFAULT_MAX_TOKENS: int = 2048 # A safe default for maximum output length.

    # --- Sampling & Creativity Defaults ---
    DEFAULT_TEMPERATURE: float = 0.6  # Balances coherence and creativity.
    DEFAULT_TOP_K: int = 0            # Disabled by default to favor Top-P.
    DEFAULT_TOP_P: float = 0.9        # Nucleus sampling for high-quality output.
    
    # --- Content & Penalty Defaults ---
    DEFAULT_REPETITION_PENALTY: float = 1.1 # Gently discourages repeating the same words.
    DEFAULT_PRESENCE_PENALTY: float = 0.0  # No penalty by default.
    DEFAULT_FREQUENCY_PENALTY: float = 0.0 # No penalty by default.
    DEFAULT_LENGTH_PENALTY: float = 1.0    # No preference for length by default.

    # --- Advanced Defaults ---
    DEFAULT_BEAM_WIDTH: int = 1 # Use standard sampling, not beam search.

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()