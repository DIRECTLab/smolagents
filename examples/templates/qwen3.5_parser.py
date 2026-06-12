import requests

from smolagents import ToolCallingAgent, tool, OpenAIModel
import re

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
    api_key = "your_api_key"  # Replace with your API key from https://weatherstack.com/
    units = "m" if celsius else "f"  # 'm' for Celsius, 'f' for Fahrenheit

    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={location}&units={units}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        data = response.json()

        if data.get("error"):  # Check if there's an error in the response
            return f"Error: {data['error'].get('info', 'Unable to fetch weather data.')}"

        weather = data["current"]["weather_descriptions"][0]
        temp = data["current"]["temperature"]
        temp_unit = "°C" if celsius else "°F"

        return f"The current weather in {location} is {weather} with a temperature of {temp} {temp_unit}."

    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"


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
            
        return ChatMessageToolCallFunction(name=tool_name, arguments=tool_arguments)
    raise ValueError('Message contained no "<tool_call>" sequence.')

custom_stop_sequences = ["</tool_call>"]

model = OpenAIModel(
    model_id="local",
    api_base="http://127.0.0.1:8000/v1",
    api_key=""
)

agent = ToolCallingAgent(
    tools=[get_weather],
    model=model,
    custom_tool_call_parser=parse_qwen,
    custom_stop_sequences=["</tool_call>"]
)

agent.run("What's the weather like in New York City?")