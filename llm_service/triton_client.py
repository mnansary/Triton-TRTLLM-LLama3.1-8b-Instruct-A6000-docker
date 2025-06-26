import asyncio
from loguru import logger
import numpy as np
import tritonclient.grpc.aio as async_grpcclient
from transformers import AutoTokenizer
from config import settings
from llm_service.models import GenerationRequest

class TritonInferenceClient:
    """
    An asynchronous, singleton client to interact with the Triton Inference Server.
    """
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self):
        # This is a singleton, so the constructor should not be called directly.
        raise RuntimeError("Call get_instance() instead")

    @classmethod
    async def get_instance(cls):
        """
        Get the singleton instance of the client.
        This pattern ensures only one client connection pool is created per application process.
        """
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    await cls._instance._initialize()
        return cls._instance

    async def _initialize(self):
        """Initializes the tokenizer and the asynchronous Triton client."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.DEFAULT_MODEL_ID)
            self.stop_token_str = "<|eot_id|>"
            self.end_id = self.tokenizer.eos_token_id
            logger.info(f"Tokenizer for '{settings.DEFAULT_MODEL_ID}' loaded successfully.")
            
            self.client = async_grpcclient.InferenceServerClient(url=settings.TRITON_URL, verbose=False)
            await self.client.is_server_live() # Check connection on startup
            logger.info(f"Successfully connected to Triton Inference Server at {settings.TRITON_URL}")
        except Exception as e:
            logger.critical(f"Failed to initialize Triton client or tokenizer: {e}")
            raise

    def _format_prompt(self, prompt: str) -> str:
        """Applies the model-specific chat template to the raw prompt."""
        messages = [{"role": "user", "content": prompt}]
        # The tokenizer correctly formats the prompt for the model.
        # We don't add special tokens like BOS/EOS, as the backend (NVIDIA NIM / TRT-LLM) handles this.
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    async def generate_stream(self, params: GenerationRequest):
        """
        Sends a generation request to Triton and yields the text chunks as they arrive.
        """
        formatted_prompt = self._format_prompt(params.prompt)
        logger.debug(f"Request ID {params.request_id}: Sending formatted prompt to Triton.")

        try:
            inputs = [
                self._create_input("text_input", [[formatted_prompt.encode("utf-8")]], "BYTES"),
                self._create_input("max_tokens", [[params.max_tokens]], "INT32"),
                self._create_input("temperature", [[params.temperature]], "FP32"),
                self._create_input("stream", [[True]], "BOOL"),
                self._create_input("stop_words", [[self.stop_token_str.encode("utf-8")]], "BYTES"),
                self._create_input("end_id", [[self.end_id]], "INT32"),
            ]

            async for result in self.client.stream_infer(
                model_name=settings.MODEL_NAME,
                inputs=inputs,
                request_id=params.request_id,
            ):
                if result.has_error():
                    error_message = f"Inference stream error for request {params.request_id}: {result.get_error()}"
                    logger.error(error_message)
                    yield f"Error: {error_message}"
                    break
                
                output = result.as_numpy('text_output')
                if output is not None:
                    yield output[0].decode('utf-8')
        
        except Exception as e:
            error_message = f"An exception occurred during stream generation for request {params.request_id}: {e}"
            logger.error(error_message)
            yield f"Error: Could not process the request. {error_message}"

    def _create_input(self, name: str, data: list, dtype: str):
        """Helper function to create a Triton InferInput object."""
        np_dtype = np.object_ if dtype == "BYTES" else np.dtype(dtype.lower())
        infer_input = async_grpcclient.InferInput(name, [1, 1], dtype)
        infer_input.set_data_from_numpy(np.array(data, dtype=np_dtype))
        return infer_input

    async def close(self):
        """Gracefully closes the connection to the Triton server."""
        if self.client:
            await self.client.close()
            logger.info("Triton client connection has been closed.")