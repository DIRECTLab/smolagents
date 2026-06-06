# from transformers import AutoTokenizer
from jinja2 import Template
import sys
import re
import json


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
        # output = tokenizer.apply_chat_template(
        #     msgs_to_format,
        #     add_generation_prompt=False,
        #     tokenize=False,
        #     enable_thinking=False
        # )
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

    for i, item in enumerate(inserts):
        item_start = message.find(item)
        item_end = item_start + len(item)
        if i == len(inserts) - 1:
            # Last item
            next_item_start = None
        else:
            next_item_start = message.find(inserts[i + 1])

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

# def find_suffix_prefix_overlap(str1, str2):
#     import re
#     str1_clean = re.sub(r'\s+', '', str1)
    
#     for i in range(len(str2), 0, -1):
#         prefix_clean = re.sub(r'\s+', '', str2[:i])
#         if prefix_clean and str1_clean.endswith(prefix_clean):
#             return str2[:i]
#     return ""

# model_id = "google/gemma-4-31B-it"
# model_id = "Qwen/Qwen3.5-9B"
# model_id = "Qwen/Qwen3.6-27B"
model_id = "openai/gpt-oss-120b"
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
            "arguments": {
            "ARG_1": 111,
            "ARG_2": "STRING_VAL",
            "ARG_3": "FILLER_VAL",
            # "ARG_3": ["LIST_STRING_VAL", 100]
            }
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


formatted_output = format_msgs(chat_template, [system_msg, user_msg, tool_call_msg, tool_response_msg, assistant_msg])

for i, msg in enumerate(formatted_output):
    print(f"Delta {i}:\n{msg}")


[system, user, tool_call, tool_res, assistant] = formatted_output
# user = formatted_output[0]
# tool_call = formatted_output[1]
# tool_res = formatted_output[2]
# assistant = formatted_output[3]


import json
print("\n==================== spans ================\n")
print(json.dumps(extract_spans(system, [system_msg["content"]]), indent=4))
print(json.dumps(extract_spans(user, [user_msg['content']]), indent=4))
print(json.dumps(extract_spans(tool_res, [tool_response_msg["name"], tool_response_msg["content"]]), indent=4))
print()

func_dict = tool_call_msg['tool_calls'][0]['function']

tool_inserts = [
    func_dict['name'],
    'ARG_1',
    str(func_dict['arguments']['ARG_1']),
    'ARG_2',
    func_dict['arguments']['ARG_2'],
    'ARG_3',
    func_dict['arguments']['ARG_3']
]

tool_spans = extract_spans(tool_call, tool_inserts)
print(tool_call)
print(json.dumps(tool_spans, indent=4))
# print(extract_spans(tool_call, tool_inserts))

quote_pre = tool_spans['STRING_VAL']['pre']
# quote_pre = tool_spans['']
quote_post = tool_spans['STRING_VAL']['post']

print(f"{repr(quote_pre)}\n{repr(quote_post)}\nmatch: {repr(find_suffix_prefix_overlap(quote_pre, quote_post))}")
