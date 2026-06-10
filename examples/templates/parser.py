# oss_schema = {
#     "type": "object",
#     "properties": {
#         "role": {"const": "assistant"},
#         # "content" and "thinking" are both similar to the previous example, and just extract a single string
#         # However, rather than using a single regex with named groups to extract both, we use a regex in each subkey.
#         # When an object node has no parser/regex, the entire input string is passed to all of its children, so 
#         # parsing can either be done with named groups at the object level, or with separate regexes at the property level.
#         "content": {"type": "string", "x-regex": r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)"},
#         "thinking": {"type": "string", "x-regex": r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>"},
#         "tool_calls": {
#             # "x-regex-iterator" uses re.finditer to find multiple possible manages, and returns them as an
#             # array/list. You don't need to worry about array handling, though - each item in the array will be
#             # parsed by the `items` schema, so just write the schema for a single item.
#             "x-regex-iterator": r"<\|channel\|>commentary (to=functions\..*?<\|message\|>.*?)(?:<\|call\|>|$)",
#             "type": "array",
#             "items": {
#                 "type": "object",
#                 "properties": {
#                     # A const property is a fixed value, and the input has no effect on it.
#                     "type": {"const": "function"},
#                     # Here, we wrap the entire tool call dict in a `{"function": ...}` block. The input string is passed through to it unchanged.
#                     "function": {
#                         "type": "object",
#                         "properties": {
#                             "name": {"type": "string", "x-regex": r"^to=functions\.(\w+)"},
#                             "arguments": {
#                                 "type": "object",
#                                 "x-regex": "<\|message\|>(.*)",
#                                 # The "x-parser" field indicates that the extracted string should be parsed as JSON.
#                                 # The output is then passed to the schema nodes below and recursive parsing continues.
#                                 "x-parser": "json",
#                                 # additionalProperties: True allows the parser to accept arbitrary keys 
#                                 # that are not specified in the schema.
#                                 "additionalProperties": True,
#                             },
#                         },
#                     },
#                 },
#             },
#         },
#     },
# }

# gemma_schema = {
#     "properties": {
#         "content": {
#             "type": "string"
#         },
#         "role": {
#           "const": "assistant"
#         },
#         "thinking": {
#           "type": "string"
#         },
#         "tool_calls": {
#             "items": {
#                 "properties": {
#                     "function": {
#                         "properties": {
#                             "arguments": {
#                                 "additionalProperties": {},
#                                 "type": "object",
#                                 "x-parser": "gemma4-tool-call"
#                             },
#                             "name": {
#                                 "type": "string"
#                             }
#                         },
#                         "type": "object",
#                         "x-regex": "call\\:(?P<name>\\w+)(?P<arguments>\\{.*\\})"
#                     },
#                     "type": {
#                         "const": "function"
#                     }
#                 },
#                 "type": "object"
#             },
#             "type": "array",
#             "x-regex-iterator": "<\\|tool_call>(.*?)<tool_call\\|>"
#         }
#     },
#     "type": "object",
#     "x-regex": "(\\<\\|channel\\>thought\\n(?P<thinking>.*?)\\<channel\\|\\>)?(?P<tool_calls>\\<\\|tool_call\\>.*\\<tool_call\\|\\>)?(?P<content>(?:(?!\\<turn\\|\\>)(?!\\<\\|tool_response\\>).)+)?(?:\\<turn\\|\\>|\\<\\|tool_response\\>)?"
# }

msg = """<|channel|>analysis<|message|>
The user asks: "What is the weather like in SF?" So we need to get the current weather in San Francisco, CA. 
We need to call get_current_weather function. So we should call get_current_weather with location "San Francisco, CA".
<|end|>
<|start|>assistant<|channel|>commentary 
to=functions.get_current_weather <|constrain|>json<|message|>
{
  "location": "San Francisco, CA"
}
<|call|>"""

import re

iter = re.finditer(r"<\|channel\|>commentary\s*(to=functions\..*?<\|message\|>.*?)(?:<\|call\|>|$)", msg, flags=re.S)

