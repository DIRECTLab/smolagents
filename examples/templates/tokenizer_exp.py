# from transformers import AutoTokenizer
from jinja2 import Template
import sys
import re
import json
from dataclasses import dataclass
from smolagents.models import ChatMessageToolCallFunction


def get_chat_template(model_id, variant=None):
    # Adapted from llama.cpp get_chat_template.py
    # Copyright (c) 2025 Olivier Chafik
    # Licensed under MIT license
    # Source URL: https://github.com/ggml-org/llama.cpp/blob/master/scripts/get_chat_template.py

    # Use huggingface_hub library to download chat template.
    # Allows access to gated models if the user has access and ran `huggingface-cli login`.
    from huggingface_hub import hf_hub_download
    try:
        with open(hf_hub_download(repo_id=model_id, filename="chat_template.jinja"), encoding="utf-8") as f:
            return f.read()
    except Exception:
        # If repo has no chat_template.jinja, see if tokenizer_config.json defines a template
        with open(hf_hub_download(repo_id=model_id, filename="tokenizer_config.json"), encoding="utf-8") as f:
            config_str = f.read()

    try:
        config = json.loads(config_str)
    except json.JSONDecodeError:
        # Fix https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json
        # (Remove extra '}' near the end of the file)
        config = json.loads(re.sub(r'\}([\n\s]*\}[\n\s]*\],[\n\s]*"clean_up_tokenization_spaces")', r'\1', config_str))

    chat_template = config['chat_template']
    if isinstance(chat_template, str):
        return chat_template
    else:
        variants = {
            ct['name']: ct['template']
            for ct in chat_template
        }

        def format_variants():
            return ', '.join(f'"{v}"' for v in variants.keys())

        if variant is None:
            if 'default' not in variants:
                raise Exception(f'Please specify a chat template variant (one of {format_variants()})')
            variant = 'default'
            sys.stderr.write(f'Note: picked "default" chat template variant (out of {format_variants()})\n')
        elif variant not in variants:
            raise Exception(f"Variant {variant} not found in chat template (found {format_variants()})")

        return variants[variant]


def format_msgs(template: Template, messages: list[dict]) -> list[str]:
    def strftime_now(format: str):
        return "FAKE STRFTIME NOW"


    formatted_msgs = []
    msgs_to_format = []
    prev_output = ""
    for msg in messages:
        msgs_to_format.append(msg)
        variables = {
            "messages": msgs_to_format,
            "strftime_now": strftime_now,
        }

        output = template.render(**variables)
        formatted_msgs.append(output[prev_output.__len__():])
        prev_output = output

    return formatted_msgs



def extract_spans(message: str, inserts: list) -> dict[str, dict[str, str]]:
    '''
    Returns a dict where the keys are inserts and values are a dict containing 'pre', 'post'.
    'pre' is the text before the insert, and 'post' is the text after the insert.
    '''
    output = {}
    prev_item_end = 0

    # Sort inserts by their position in the message
    sorted_inserts = sorted(
        [item for item in inserts if item in message], 
        key=lambda x: message.find(x)
    )

    for i, item in enumerate(sorted_inserts):
        item_start = message.find(item)
        item_end = item_start + len(item)
        if i == len(sorted_inserts) - 1:
            # Last item
            next_item_start = None
        else:
            next_item_start = message.find(sorted_inserts[i + 1])

        output[item] = {
            'pre': message[prev_item_end:item_start],
            'post': message[item_end:next_item_start]
        }
        prev_item_end = item_end

    return output

def find_longest_substring(str1: str, str2: str) -> str:
    from difflib import SequenceMatcher
    match = SequenceMatcher(None, str1, str2, True).find_longest_match()
    return str1[match.a : match.a + match.size]

def find_suffix_prefix_overlap(str1, str2):
    # Start with the maximum possible overlap length
    max_possible = min(len(str1), len(str2))
    
    # Check smaller sizes until a match is found
    for i in range(max_possible, 0, -1):
        if str1.endswith(str2[:i]):
            return str2[:i]
    return ""

# model_id = "google/gemma-4-31B-it"
model_id = "Qwen/Qwen3.5-9B"
# model_id = "Qwen/Qwen3.6-27B"
# model_id = "openai/gpt-oss-120b"
# model_id = "deepseek-ai/DeepSeek-V4-Pro"
# model_id = "unsloth/Qwen3.6-27B-MTP-GGUF"
# model_id = "moonshotai/Kimi-K2.6"
# model_id = ""

system_msg = {
    "role": "system",
    "content": "SYSTEM_MSG_CONTENT"
}

user_msg = {
    "role": "user",
    "content": "USER_MSG_CONTENT"
}

tool_call_msg = {
    "role": "assistant",
    "tool_calls": [
        {
        "type": "function",
        "function": {
            "name": "FUNCTION_NAME",
        }
        }
    ]
}

num_tool_call_msg = {
    "role": "assistant",
    "tool_calls": [
        {
        "type": "function",
        "function": {
            "name": "FUNCTION_NAME",
            "arguments": {
            "ARG_1": 111,
            "ARG_2": 222,
            "ARG_3": 333,
            }
        }
        }
    ]
}


function = ChatMessageToolCallFunction(
    name="test_function",
    arguments={
        'test_arg': "test_val",
        'num_arg': 111,
    }
)

function_message = {
    "role": "assistant",
    "tool_calls": [
        {
        "type": "function",
        "function": {
            "name": function.name,
            "arguments": function.arguments
        }
        }
    ]
}


tool_response_msg = {
    "role": "tool",
    "name": "FUNCTION_NAME",
    "content": "RETURN_VALUE"
}

assistant_msg = {
    "role": "assistant",
    "content": "ASSISTANT_MSG_CONTENT"   
}

# multi_tool_call_msg = {
#     "role": "assistant",
#     "tool_calls": [
#         {
#             "type": "function",
#             "function": {
#                 "name": "FUNCTION1_NAME",
#                 "arguments": {
#                     "ARG_2": "STRING_VAL",
#                     "ARG_1": 99.9,
#                     "ARG_3": ["LIST_STRING_VAL", 100]
#                 }
#             }
#         },
#         {
#             "type": "function",
#             "function": {
#                 "name": "FUNCTION2_NAME",
#             }
#         }
#     ]
# }

chat_template = Template(get_chat_template(model_id=model_id))


formatted_output = format_msgs(chat_template, [user_msg, num_tool_call_msg, tool_response_msg, function_message])

for i, msg in enumerate(formatted_output):
    # print(f"Delta {i}:\n{msg}")
    pass


[user, tool_call, tool_res, str_tool_call] = formatted_output

print(tool_call)
print(str_tool_call)

import json
# print("\n==================== spans ================\n")
# # print(json.dumps(extract_spans(system, [system_msg["content"]]), indent=4))
# print(json.dumps(extract_spans(user, [user_msg['content']]), indent=4))
# print(json.dumps(extract_spans(tool_res, [tool_response_msg["name"], tool_response_msg["content"]]), indent=4))
# print()

func_dict = num_tool_call_msg['tool_calls'][0]['function']

tool_inserts = [
    func_dict['name'],
    'ARG_1',
    str(func_dict['arguments']['ARG_1']),
    'ARG_2',
    str(func_dict['arguments']['ARG_2']),
    'ARG_3',
    str(func_dict['arguments']['ARG_3'])
]

tool_spans = extract_spans(tool_call, tool_inserts)
# print(json.dumps(tool_spans, indent=4))
# print(extract_spans(tool_call, tool_inserts))

# quote_pre = tool_spans['STRING_VAL']['pre']
# quote_post = tool_spans['STRING_VAL']['post']

# print(f"{repr(quote_pre)}\n{repr(quote_post)}\nmatch: {repr(find_suffix_prefix_overlap(quote_pre, quote_post))}")

# string_arg_wrapper = find_suffix_prefix_overlap(tool_spans['STRING_VAL']['pre'], tool_spans['STRING_VAL']['post'])
# print(f"string arg wrapper: '{string_arg_wrapper}'")
arg_name_wrapper = find_suffix_prefix_overlap(tool_spans['ARG_1']['pre'], tool_spans['ARG_1']['post'])
# print(f"arg name wrapper: '{arg_name_wrapper}'")

# if arg_name_wrapper != "" and tool_spans['111']['post'].endswith(arg_name_wrapper):
#     val_arg_separator = tool_spans['111']['post'].replace(arg_name_wrapper, "")
# else:
#     val_arg_separator = tool_spans['111']['post']
    
val_arg_separator = tool_spans['111']['post']

# print(f"arg separator: '{val_arg_separator}'")

arg_val_separator = tool_spans['ARG_1']['post'].replace(arg_name_wrapper, "")
# print(f"arg, value separator: '{arg_val_separator}'")

# test_func_msgs = [
#     {
#         "role": "assistant",
#         "tool_calls": [
#             {
#             "type": "function",
#             "function": {
#                 "name": "FUNCTION_NAME",
#                 "arguments": {
#                 "ARG_1": "VAL_1",
#                 "ARG_2": "VAL_2",
#                 "ARG_3": "VAL_3",
#                 }
#             }
#             }
#         ]
#     }
# ]

@dataclass
class ModelGrammar():
    func_block_start: str # Characters preceding the name of the function (start of the message)
    func_block_end_no_args: str # Character(s) that end the function block if no args are present

    arg_block_pre: str # Character(s) between function name and first arg name

    # arg_name_wrapper: str # Character(s) surrounding the name of the argument
    arg_val_separator: str # Character(s) separating the name of an argument from its value
    string_val_wrapper: str # Character(s) wrapping string values in arguments (e.g. ", <|"|>)
    args_separator: str # Character(s) separating the value of an argument from the name of the next one
    # TODO: Need to handle list argument values
    # TODO: Need to find other types of argument values

    arg_block_post: str # Character(s) after last arg value (end of function call message)
    # TODO: May need something to handle models that do/don't append the opening of the tool response


def format_tool_call(grammar: ModelGrammar, tool_call: ChatMessageToolCallFunction) -> str:
    output = grammar.func_block_start
    output += tool_call.name
    if not tool_call.arguments:
        # if tool call has no arguments
        output += grammar.func_block_end_no_args
        return output
    
    output += grammar.arg_block_pre

    if not isinstance(tool_call.arguments, dict):
        raise TypeError("tool_call.arguments must be a dict")
    
    for i, (arg, value) in enumerate(tool_call.arguments.items()):
        output += arg + arg_val_separator
        if isinstance(value, str):
            output += grammar.string_val_wrapper + value + grammar.string_val_wrapper
        else:
            output += str(value)
        if i < len(tool_call.arguments) - 1:
            output += grammar.args_separator
    output += grammar.arg_block_post

    return output

gemma_grammar = ModelGrammar(
    func_block_start="<|turn>model\n<|tool_call>call:",
    func_block_end_no_args="{}<tool_call|><|tool_response>",

    arg_block_pre="{",
    arg_val_separator=":",
    string_val_wrapper="<|\"|>",
    args_separator=",",

    arg_block_post="}<tool_call|><|tool_response>"
)

qwen3_5_grammar = ModelGrammar(
    func_block_start="<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=",
    func_block_end_no_args="</function>\n</tool_call><|im_end|>",

    arg_block_pre=">\n<parameter=",
    arg_val_separator=">\n",
    string_val_wrapper="",
    args_separator="\n</parameter>\n<parameter=",

    arg_block_post="\n</parameter>\n</function>\n</tool_call><|im_end|>"
)

model_grammar_registry = {
    "google/gemma-4-31B-it": gemma_grammar,
    "Qwen/Qwen3.5": qwen3_5_grammar
}