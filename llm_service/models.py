# llm_service/models.py

from pydantic import BaseModel, Field
import uuid
from typing import Optional, List
from config import settings # Import the single source of truth

class GenerationRequest(BaseModel):
    """
    The complete request model. Defaults are now pulled from the central settings,
    making this model purely a definition of the API's structure.
    """
    # --- Core Parameters ---
    prompt: str = Field(
        ..., 
        description="The initial text prompt to start the generation."
    )
    # This now reads the default from your .env file, falling back to config.py
    max_tokens: int = Field(
        default=settings.DEFAULT_MAX_TOKENS, 
        gt=0, 
        description="Maximum number of new tokens to generate."
    )
    min_tokens: Optional[int] = Field(
        default=None, 
        gt=0, 
        description="Minimum number of new tokens to generate."
    )
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), 
        description="Unique identifier for the request."
    )

    # --- Sampling & Creativity Parameters ---
    temperature: Optional[float] = Field(
        default=settings.DEFAULT_TEMPERATURE, 
        ge=0.0, 
        le=2.0,
        description="Controls randomness. Lower is more deterministic."
    )
    top_k: Optional[int] = Field(
        default=settings.DEFAULT_TOP_K, 
        ge=0, 
        description="Restricts sampling to the k most likely tokens. 0 disables it."
    )
    top_p: Optional[float] = Field(
        default=settings.DEFAULT_TOP_P, 
        ge=0.0, le=1.0, 
        description="Restricts sampling to a cumulative probability mass. 1.0 disables it."
    )
    seed: Optional[int] = Field(
        default=None, 
        description="A seed for the random number generator for reproducible outputs."
    )

    # --- Content & Penalty Parameters ---
    repetition_penalty: Optional[float] = Field(
        default=settings.DEFAULT_REPETITION_PENALTY, 
        ge=0.0,
        description="Penalizes repeated tokens. 1.0 means no penalty."
    )
    presence_penalty: Optional[float] = Field(
        default=0.0,
        description="Applies a flat penalty to any token that has appeared at least once."
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0,
        description="Applies a penalty that increases based on a token's frequency."
    )
    length_penalty: Optional[float] = Field(
        default=settings.DEFAULT_LENGTH_PENALTY,
        description="Adjusts preference for longer sequences. >1.0 encourages length."
    )

    # --- Stopping & Filtering Parameters ---
    stop_words: Optional[List[str]] = Field(
        default=None, 
        description="A list of words or phrases that will immediately stop generation."
    )
    bad_words: Optional[List[str]] = Field(
        default=None, 
        description="A list of words that are forbidden from being generated."
    )
    
    # --- Advanced Parameters ---
    beam_width: Optional[int] = Field(
        default=settings.DEFAULT_BEAM_WIDTH,
        ge=1,
        description="Number of beams for beam search. 1 means standard sampling."
    )
    return_log_probs: Optional[bool] = Field(
        default=False,
        description="If true, returns the log probabilities of the generated tokens."
    )


class GenerationResponse(BaseModel):
    """
    The standard response model for non-streaming generation.
    """
    text: str
    request_id: str
    model_name: str