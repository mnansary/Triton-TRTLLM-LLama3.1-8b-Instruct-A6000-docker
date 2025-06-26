from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from loguru import logger
import asyncio

from llm_service.logging_config import setup_logging
from llm_service.triton_client import TritonInferenceClient
from llm_service.models import GenerationRequest, GenerationResponse
from config import settings

# Setup logging as the very first step
setup_logging()

# A dictionary to hold application state, including the Triton client instance.
# This is the recommended way to manage shared resources in FastAPI.
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan events.
    On startup: Initializes the singleton Triton client.
    On shutdown: Gracefully closes the client connection.
    """
    logger.info("Application startup sequence initiated...")
    client = await TritonInferenceClient.get_instance()
    app_state["triton_client"] = client
    yield
    logger.info("Application shutdown sequence initiated...")
    await client.close()
    app_state.clear()


app = FastAPI(
    title="Production LLM Inference Service",
    description="A high-performance API for serving LLM models via the Triton Inference Server.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Simple health check endpoint to confirm the service is running."""
    return {"status": "ok", "service": "LLM Inference Service"}


@app.post("/generate", response_model=GenerationResponse, tags=["Inference"])
async def generate_text(request: GenerationRequest):
    """
    Generates text based on a prompt (non-streaming).
    This endpoint collects the full response from the model before sending it back,
    which is suitable for tasks requiring the complete output at once.
    """
    logger.info(f"Received non-streaming request: {request.request_id}")
    client: TritonInferenceClient = app_state["triton_client"]
    
    full_response = "".join([chunk async for chunk in client.generate_stream(request)])

    if full_response.startswith("Error:"):
         logger.error(f"Failed to process request {request.request_id}: {full_response}")
         # Consider returning a different HTTP status code for errors.
         # For simplicity, we return it in the response body here.
         
    logger.info(f"Completed non-streaming request: {request.request_id}")
    return GenerationResponse(
        text=full_response.strip(),
        request_id=request.request_id,
        model_name=settings.MODEL_NAME
    )


@app.post("/generate_stream", tags=["Inference"])
async def generate_text_stream(request: GenerationRequest):
    """
    Generates text using a streaming response.
    This is ideal for interactive applications (like chatbots) where text
    should be displayed to the user as it becomes available.
    """
    logger.info(f"Received streaming request: {request.request_id}")
    client: TritonInferenceClient = app_state["triton_client"]

    async def stream_generator():
        try:
            async for text_chunk in client.generate_stream(request):
                yield text_chunk
            logger.info(f"Stream completed for request: {request.request_id}")
        except asyncio.CancelledError:
            # This occurs if the client disconnects before the stream is finished.
            logger.warning(f"Client disconnected for request: {request.request_id}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in stream for {request.request_id}: {e}")


    return StreamingResponse(stream_generator(), media_type="text/event-stream")