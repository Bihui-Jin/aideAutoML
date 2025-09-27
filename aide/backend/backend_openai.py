"""Backend for OpenAI API."""

import json
import logging
import time

from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
from funcy import notnone, once, select_values
import openai

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openai_client(model: str):
    global _client
    if model == "qwen3-max":
        _client = openai.OpenAI(base_url='https://dashscope-intl.aliyuncs.com/compatible-mode/v1', max_retries=0)
    else:
        _client = openai.OpenAI(max_retries=0)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    model = model_kwargs.get("model", "")
    _setup_openai_client(model)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=convert_system_to_user)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model the use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:

            function_args = choice.message.tool_calls[0].function.arguments
            logger.debug(f"Raw function arguments: {repr(function_args)}")
            
            # Clean up the function arguments string
            function_args_cleaned = function_args.strip()
            
            # Try to parse the JSON
            output = json.loads(function_args_cleaned)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Error position: line {e.lineno}, column {e.colno}")
            
            # Try to extract partial JSON or provide a fallback
            function_args = choice.message.tool_calls[0].function.arguments
            try:
                # Try to find the JSON object boundaries and extract it
                start_idx = function_args.find('{')
                if start_idx != -1:
                    # Find the matching closing brace
                    brace_count = 0
                    end_idx = start_idx
                    for i, char in enumerate(function_args[start_idx:], start_idx):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    
                    if brace_count == 0:
                        clean_json = function_args[start_idx:end_idx]
                        logger.info(f"Attempting to parse extracted JSON: {clean_json}")
                        output = json.loads(clean_json)
                    else:
                        raise e
                else:
                    raise e
            except json.JSONDecodeError:
                logger.error("Failed to extract valid JSON, falling back to error response")
                # Provide a fallback response that matches the expected schema
                output = {
                    "is_bug": True,
                    "has_csv_submission": False,
                    "summary": f"JSON parsing error in function response: {str(e)}",
                    "metric": None,
                    "lower_is_better": True
                }

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
