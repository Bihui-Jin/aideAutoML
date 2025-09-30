"""Backend for Anthropic API."""

import json
import time
import logging

import anthropic
from .utils import FunctionSpec, OutputType, backoff_create, opt_messages_to_list
from funcy import notnone, once, select_values

logger = logging.getLogger("aide")

_client: anthropic.Anthropic = None  # type: ignore

ANTHROPIC_TIMEOUT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)

@once
def _setup_anthropic_client():
    global _client
    _client = anthropic.Anthropic(max_retries=0) 

def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_anthropic_client()

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 16_384  # default for Claude models

    # if func_spec is not None:
    #     raise NotImplementedError(
    #         "Anthropic does not support function calling for now."
    #     )
    if func_spec is not None:
        logger.info(f"Using function spec: {func_spec.as_anthropic_tool_dict}")
        logger.info(f"Function spec schema: {json.dumps(func_spec.parameters, indent=2)}")
        logger.info(f"Function tool name: {func_spec.name}")
        
        filtered_kwargs["tools"] = [func_spec.as_anthropic_tool_dict]
        # Force the model to use the tool
        filtered_kwargs["tool_choice"] = {"type": "tool", "name": func_spec.name}

    # Anthropic doesn't allow not having a user messages
    # if we only have system msg -> use it as user msg
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    # Anthropic passes the system messages as a separate argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_message)

    max_retries = 2
    retries = 0
    req_time = 0.0
    message = None
    while retries < max_retries:
        try:
            t0 = time.time()
            message = backoff_create(
                _client.messages.create,
                ANTHROPIC_TIMEOUT_EXCEPTIONS,
                messages=messages,
                **filtered_kwargs,
            )
            logger.info(f"Anthropic API call successful: {message.content[0].text}")
            req_time = time.time() - t0
            break
        except Exception as e:
            retries += 1
            logger.info(e)
            if retries <= max_retries:
                time.sleep(61)
    
    if func_spec is None:
        assert len(message.content) == 1 and message.content[0].type == "text"
        output: str = message.content[0].text
    else:
        logger.info(f"Function call response: {message.content}")
        # Find the tool use block in the response
        tool_use_block = None
        for content in message.content:
            if content.type in ["tool_use", "tool_call"]:
                tool_use_block = content
                break
        
        assert tool_use_block is not None, f"No tool_use block found in response: {message.content}"
        assert tool_use_block.name == func_spec.name, f"Tool name mismatch: expected {func_spec.name}, got {tool_use_block.name}"
        
        try:
            # tool_use_block.input is already a dict, no need to parse JSON
            output = tool_use_block.input
            logger.debug(f"Tool use input: {output}")
        except Exception as e:
            logger.error(f"Error processing tool use input: {tool_use_block.input}")
            logger.error(f"Error: {str(e)}")
            # Provide a fallback response that matches the expected schema
            output = {
                "is_bug": True,
                "has_csv_submission": False,
                "summary": f"Error processing tool response: {str(e)}",
                "metric": None,
                "lower_is_better": True
            }
    # output: str = message.content[0].text
    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
    }

    return output, req_time, in_tokens, out_tokens, info
