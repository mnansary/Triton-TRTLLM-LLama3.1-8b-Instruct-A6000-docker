# trtllm-triton-repo/postprocessing/1/model.py

import json
import threading
import traceback

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

class TritonPythonModel:
    """
    V6: Definitive Word-by-Word Streaming.

    Combines the correct triggering mechanism of V3 with safe data handling.
    It inspects each new token individually to detect word boundaries,
    ensuring robust streaming for all languages with modern tokenizers.
    """

    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        self.logger = pb_utils.Logger
        self.request_states = {}
        self.lock = threading.Lock()

        tokenizer_dir = model_config['parameters']['tokenizer_dir']['string_value']
        skip_param = model_config['parameters'].get('skip_special_tokens', {})
        skip_str = skip_param.get('string_value', 'true').lower()
        self.skip_special_tokens = skip_str in ['true', '1', 't', 'y', 'yes']

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, legacy=False, padding_side='left', trust_remote_code=True
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config['data_type'])
        
        # The special character used by SentencePiece models to mark a new word.
        self.new_word_char = " " # (U+2581)

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                request_id = request.request_id()
                
                tokens_tensor = pb_utils.get_input_tensor_by_name(request, "TOKENS_BATCH")
                if tokens_tensor is None:
                    responses.append(pb_utils.InferenceResponse(output_tensors=[]))
                    continue
                
                new_token_ids = tokens_tensor.as_numpy()[0, 0].tolist()

                is_final_tensor = pb_utils.get_input_tensor_by_name(request, "IS_FINAL")
                is_final = is_final_tensor.as_numpy()[0] if is_final_tensor is not None else False

                text_to_send = ""
                with self.lock:
                    token_buffer = self.request_states.get(request_id, [])
                    
                    # Process each new token individually
                    for token_id in new_token_ids:
                        # Safely decode just this one token to check its properties
                        decoded_token = self.tokenizer.decode([token_id])

                        # Check if this token starts a new word.
                        is_boundary = decoded_token.startswith(' ') or decoded_token.startswith(self.new_word_char)

                        if is_boundary and token_buffer:
                            # This token starts a new word, so the current buffer
                            # holds a complete word. Decode and send it.
                            text_to_send += self.tokenizer.decode(
                                token_buffer, skip_special_tokens=self.skip_special_tokens
                            )
                            # The new buffer starts with just the new boundary token.
                            token_buffer = [token_id]
                        else:
                            # This is a mid-word token, so just add it to the buffer.
                            token_buffer.append(token_id)
                    
                    if is_final:
                        # If this is the final call, send whatever is left.
                        if token_buffer:
                            text_to_send += self.tokenizer.decode(
                                token_buffer, skip_special_tokens=self.skip_special_tokens
                            )
                        # Clean up state.
                        if request_id in self.request_states:
                            del self.request_states[request_id]
                    else:
                        # It's not the final call, so store the current buffer.
                        self.request_states[request_id] = token_buffer

                output_tensor = pb_utils.Tensor(
                    'OUTPUT', np.array([text_to_send.encode('utf-8')]).astype(self.output_dtype)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

            except Exception:
                error_message = f"Error in postprocessing: {traceback.format_exc()}"
                self.logger.log_error(error_message)
                error = pb_utils.TritonError(error_message)
                responses.append(pb_utils.InferenceResponse(output_tensors=[], error=error))

        return responses

    def finalize(self):
        self.logger.log_info('Cleaning up postprocessing model...')