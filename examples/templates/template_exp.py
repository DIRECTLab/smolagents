from jinja2 import Environment, FileSystemLoader, Template
from smolagents.models import ChatMessage, MessageRole, ChatMessageToolCall

msg = ChatMessage(
        role=MessageRole.USER,
        content=[
            {
                "type": "text",
                "text": "This is the user query!"
            }
        ]
    )

user_msg = {
    "role": "user",
    "content": "This is the user query!"
}

tool_call_msg = {
    "role": "assistant",
    "tool_calls": [
        {
        "type": "function",
        "function": {
            "name": "multiply",
            "arguments": {
            "a": 5,
            "b": 6
            }
        }
        }
    ]
}

tool_response_msg = {
    "role": "tool",
    "name": "multiply",
    "content": "30"
}

messages = [user_msg, tool_call_msg, tool_response_msg]

file_loader = FileSystemLoader('chat_templates')
env = Environment(loader=file_loader)

# template = env.get_template('experiment.jinja')
# template = env.get_template('gemma4-31B-it.jinja')
template = env.get_template('gpt-oss-120b.jinja')
# template = env.get_template('Qwen3.5-9B.jinja')

def format_msgs(template: Template, messages: list[dict]) -> list[str]:
    def strftime_now(format: str):
        return "NOWNOWNOW"


    formatted_msgs = []
    msgs_to_format = []
    prev_output = ""
    for msg in messages:
        msgs_to_format.append(msg)
        variables = {
            "messages": msgs_to_format,
            "strftime_now": strftime_now,
            "tools": []
        }
        output = template.render(**variables)
        formatted_msgs.append(output[prev_output.__len__():])
        prev_output = output

    return formatted_msgs




# output = template.render(**variables)
# template.render(**variables)
formatted_output = format_msgs(template, [user_msg, tool_call_msg, tool_response_msg])

print(f"TOOL CALL: \n{formatted_output[1]}\nTOOL RESPONSE:\n{formatted_output[2]}")