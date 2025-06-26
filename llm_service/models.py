from pydantic import BaseModel, Field
import uuid
from config import settings

class GenerationRequest(BaseModel):
    """
    Request model for the text generation endpoint.
    """
    prompt: str = Field(..., description="The input prompt for the language model.")
    temperature: float = Field(
        0.0, ge=0.0, le=2.0, description="Controls randomness. 0.0 for deterministic output."
    )
    # MODIFY THIS LINE to use the configurable default value
    max_tokens: int = Field(
        default=settings.MAX_TOKENS, 
        gt=0, 
        description="Maximum number of tokens to generate."
    )
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the request."
    )

class GenerationResponse(BaseModel):
    """
    Response model for the non-streaming text generation endpoint.
    """
    text: str
    request_id: str
    model_name: str