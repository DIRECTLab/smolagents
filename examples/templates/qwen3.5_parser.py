import requests
from smolagents import ToolCallingAgent, tool, TransformersModel, ParsedToolCall
import re

import ast

from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

@tool
def get_weather(location: str, celsius: bool | None = False) -> str:
    """
    Get the current weather at the given location using the WeatherStack API.

    Args:
        location: The location (city name).
        celsius: Whether to return the temperature in Celsius (default is False, which returns Fahrenheit).

    Returns:
        A string describing the current weather at the location.
    """
    return f"The current weather in {location} is sunny, with a temperature of 70 degrees."


def parse_qwen(message: str) -> ParsedToolCall:
    if "<tool_call>" in message:
        message += "</tool_call>"
        xml_match = re.search(r"<tool_call>\s*<function=([^>]+)>\s*([\s\S]*?)\s*</function>\s*</tool_call>", message, re.DOTALL)
        tool_name = xml_match.group(1).strip()
        params_text = xml_match.group(2)
        tool_arguments = {}
        
        # Extract all parameters within the function block
        param_matches = re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", params_text, re.DOTALL)
        for param_match in param_matches:
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2).strip()
            tool_arguments[param_name] = param_value
            
        return ParsedToolCall(tool_name=tool_name, arguments=tool_arguments)
    raise ValueError('Message contained no "<tool_call>" sequence.')


# def parse_gemma_4(message: str) -> ParsedToolCall:
#     print(f"\x1b[33mParsing message:\n{message}\033[0m")
#     if "<|tool_call>" in message:
#         message += "<|tool_response>"
#         xml_match = re.search(r"<\|tool_call>call:([^{]+).*<tool_call\|>", message, re.DOTALL)
#         tool_name = xml_match.group(1).strip()
#         params_text = xml_match.group(2)
#         tool_arguments = {}
        
#         # Extract all parameters within the function block
#         param_matches = re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", params_text, re.DOTALL)
#         for param_match in param_matches:
#             param_name = param_match.group(1).strip()
#             param_value = param_match.group(2).strip()
#             tool_arguments[param_name] = param_value
            
#         return ParsedToolCall(tool_name=tool_name, arguments=tool_arguments)
#     raise ValueError('Message contained no "<tool_call>" sequence.')

def parse_gemma_4(message: str) -> ParsedToolCall:
    # Example tool call:
    #   call:get_weather{location:<|"|>New York City<|"|>}
    print(f"\x1b[33mParsing message:\n{message}\033[0m")
    xml_match = re.search(r"call:([^{]+){(.*)}", message, re.DOTALL)
    tool_name = xml_match.group(1).strip()
    params_text = xml_match.group(2)
    tool_arguments = {}
    
    # Extract all parameters within the function block
    param_matches = re.finditer(r"([a-zA-Z0-9_]+)\s*:\s*(<\|\"\|>.*?<\|\"\|>|\d+|\[.*?\])", params_text, re.DOTALL)
    for param_match in param_matches:
        param_name = param_match.group(1).strip()
        param_value = param_match.group(2).strip()

            
        # Clean up the custom quotes if they exist
        if param_value.startswith('<|"|>') and param_value.endswith('<|"|>'):
            param_value = param_value[5:-5]
        elif param_value.startswith('[') and param_value.endswith(']'):
            # If value is a list, clean up custom quotes and convert to list
            param_value = param_value.replace("<|\"|>", "\"")
            param_value = ast.literal_eval(param_value)
        else:
            # If value is not a string or list, try to convert it to a number
            # NOTE: This breaks list values
            try:
                param_value = int(param_value)
            except ValueError:
                try:
                    param_value = float(param_value)
                except ValueError:
                    # Give a guiding error message for models that forget to wrap strings in quotes
                    raise ValueError(f"Value of {param_name} (\"{param_value}\") is not a string, but cannot be converted to a number.")

        tool_arguments[param_name] = param_value
    print(ParsedToolCall(tool_name=tool_name, arguments=tool_arguments))
        
    return ParsedToolCall(tool_name=tool_name, arguments=tool_arguments)

# custom_stop_sequences = ["</tool_call>"]
custom_stop_sequences = ["<|tool_response>"]

# model = OpenAIModel(
#     model_id="local",
#     api_base="http://127.0.0.1:8000/v1",
#     api_key=""
# )

# message = "call:FUNCTION_NAME{ARG_1:111,ARG_2:222,ARG_3:333,LIST_ARG:[111,<|\"|>str<|\"|>,333]}"

# print(parse_gemma_4(message))
register(project_name="Parser Experiments")
SmolagentsInstrumentor().instrument()

model = TransformersModel(
    model_id="google/gemma-4-E2B-it",
    device_map="cuda"
)

agent = ToolCallingAgent(
    tools=[get_weather],
    model=model,
    custom_tool_call_parser=parse_gemma_4,
    custom_tool_call_stop_sequences=custom_stop_sequences
)

agent.run("What's the weather like in New York City?")