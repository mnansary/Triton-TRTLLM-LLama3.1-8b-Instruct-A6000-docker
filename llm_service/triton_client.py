import asyncio
from loguru import logger
import numpy as np
import tritonclient.grpc.aio as async_grpcclient
from tritonclient.grpc import InferenceServerException
from transformers import AutoTokenizer
from typing import List
import random

from config import settings
from llm_service.models import GenerationRequest

class TritonInferenceClient:
    """
    An asynchronous, singleton client to interact with the Triton Inference Server.

    This class manages the connection to Triton and handles the logic for formatting
    requests and processing responses, including all optional generation parameters.
    """
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self):
        raise RuntimeError("Call get_instance() instead. This is a singleton.")

    @classmethod
    async def get_instance(cls):
        """Get the singleton instance of the client, creating it if necessary."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    await cls._instance._initialize()
        return cls._instance

    async def _initialize(self):
        """Initializes the tokenizer and the asynchronous Triton client."""
        logger.info("Initializing TritonInferenceClient...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.DEFAULT_MODEL_ID)
            self.end_id = self.tokenizer.eos_token_id
            logger.info(f"Tokenizer for '{settings.DEFAULT_MODEL_ID}' loaded successfully.")
            
            self.client = async_grpcclient.InferenceServerClient(url=settings.TRITON_URL, verbose=False)
            await self.client.is_server_live()
            logger.info(f"Successfully connected to Triton Inference Server at {settings.TRITON_URL}")
        except Exception as e:
            logger.critical(f"Fatal error during TritonInferenceClient initialization: {e}")
            raise

    def _format_prompt(self, prompt: str) -> str:
        """Applies the model-specific chat template to the raw prompt."""
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _create_input(self, name: str, data: np.ndarray, dtype: str):
        """A simple helper to create a Triton InferInput object from a numpy array."""
        infer_input = async_grpcclient.InferInput(name, list(data.shape), dtype)
        infer_input.set_data_from_numpy(data)
        return infer_input

    def _build_triton_inputs(self, params: GenerationRequest) -> List[async_grpcclient.InferInput]:
        """
        Constructs the list of InferInput objects based on the request parameters.
        This isolates the complex input creation logic.
        """
        formatted_prompt = self._format_prompt(params.prompt)
        
        # Start with required inputs
        prompt_data = np.array([[formatted_prompt.encode("utf-8")]], dtype=np.object_)
        inputs = [
            self._create_input("text_input", prompt_data, "BYTES"),
            self._create_input("max_tokens", np.array([[params.max_tokens]], dtype=np.int32), "INT32"),
            self._create_input("stream", np.array([[True]], dtype=np.bool_), "BOOL"),
            self._create_input("end_id", np.array([[self.end_id]], dtype=np.int32), "INT32"),
        ]

        # Use a copy of the params to avoid modifying the original request object
        request_params = params.model_copy()

        # If temperature is high and no seed is given, generate a random one
        # to ensure non-deterministic output, which is what users expect.
        if request_params.temperature > 0.0 and request_params.seed is None:
            request_params.seed = random.randint(0, 2**32 - 1)
            logger.debug(f"Generated random seed {request_params.seed} for high-temperature request.")

        scalar_param_map = {
            "temperature": np.float32, "top_k": np.int32, "top_p": np.float32,
            "repetition_penalty": np.float32, "presence_penalty": np.float32,
            "frequency_penalty": np.float32, "length_penalty": np.float32,
            "beam_width": np.int32, "min_tokens": np.int32, "seed": np.uint64,
            "return_log_probs": np.bool_
        }
        
        # The logic is simplified: if the parameter exists (is not None), send it.
        # Don't try to be clever about filtering defaults. Let the backend decide.
        for name, np_dtype in scalar_param_map.items():
            value = getattr(request_params, name, None)
            if value is not None:
                triton_dtype = str(np_dtype.__name__).upper().replace("FLOAT", "FP")
                data_array = np.array([[value]], dtype=np_dtype)
                inputs.append(self._create_input(name, data_array, triton_dtype))
        # Safely add optional list-based parameters
        if params.stop_words:
            # Re-introduce reshape! Triton expects a 2D tensor of shape [1, N]
            # to match the batch size of 1 from the other inputs.
            stop_words_data = np.array([s.encode("utf-8") for s in params.stop_words], dtype=np.object_).reshape((1, -1))
            inputs.append(self._create_input("stop_words", stop_words_data, "BYTES"))
        
        if params.bad_words:
            # Re-introduce reshape! This must also be a 2D tensor.
            bad_words_data = np.array([s.encode("utf-8") for s in params.bad_words], dtype=np.object_).reshape((1, -1))
            inputs.append(self._create_input("bad_words", bad_words_data, "BYTES"))
    
        return inputs

    async def _process_stream_response(self, response_iterator, request_id: str):
        """
        Processes the gRPC stream from Triton, handling errors and yielding text chunks.
        This isolates the response handling logic.
        """
        try:
            # The iterator yields a TUPLE of (result, error). We must unpack it.
            async for result, error in response_iterator:
                if error:
                    error_message = f"Inference stream returned an error for request {request_id}: {error}"
                    logger.error(error_message)
                    yield f"Error: {error_message}"
                    return # Stop generation on error
                
                # If there's no error, 'result' is the InferResult object.
                output = result.as_numpy('text_output')
                if output is not None:
                    yield output[0].decode('utf-8')
        except InferenceServerException as e:
            # This handles errors in setting up the stream itself (e.g., connection issues).
            error_message = f"InferenceServerException during stream processing for request {request_id}: {e}"
            logger.error(error_message, exc_info=True)
            yield f"Error: Could not process the request. {e}"

    async def generate_stream(self, params: GenerationRequest):
        """
        Orchestrates the text generation stream.
        1. Builds the Triton inputs.
        2. Sets up the stream request.
        3. Processes the stream response.
        """
        try:
            # Step 1: Build the list of input tensors.
            inputs = self._build_triton_inputs(params)

            # Step 2: Define the asynchronous iterator that stream_infer expects.
            async def input_iterator():
                yield {
                    "model_name": settings.MODEL_NAME,
                    "inputs": inputs,
                    "request_id": params.request_id,
                }
            
            # Step 3: Call the Triton client and process the response stream.
            response_iterator = self.client.stream_infer(inputs_iterator=input_iterator())
            async for chunk in self._process_stream_response(response_iterator, params.request_id):
                yield chunk

        except Exception as e:
            # This is a final catch-all for any unexpected errors during orchestration.
            error_message = f"A fatal exception occurred in generate_stream orchestration for request {params.request_id}: {e}"
            logger.error(error_message, exc_info=True)
            yield f"Error: Could not process the request. {e}"
    
    async def close(self):
        """Gracefully closes the connection to the Triton server."""
        if self.client:
            await self.client.close()
            logger.info("Triton client connection closed.")