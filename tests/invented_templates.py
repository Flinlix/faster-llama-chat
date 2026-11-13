"""Small chat templates in an invented format, each a fixture as a file of ``tests/templates/`` holds one.

Every tag is invented, not any real model's, so nothing can pass by matching a
format hard-coded somewhere. Each template is the one Jinja text below, with
the switches at its top set as that template needs them, and is rendered by
the renderer a model is loaded with.
"""

from llama_chat.template import CallBody

# The text every invented template shares. A call leaves its turn open, and the
# result and the prose after it stay inside that same turn. A past reply's thought
# is written only in a turn that called a tool, unless the template keeps thoughts.
_TEXT = r"""
{%- set ns = namespace(open=false, previous=none, newest=none) -%}
{%- set reasons = can_reason and not (enable_thinking is defined and not enable_thinking) -%}
{%- macro call_blocks(calls) -%}
  {%- if calls_json and calls -%}
    {{- '<call>[' -}}
    {%- for call in calls -%}
      {{- (', ' if not loop.first else '') ~ '{"name": "' ~ call.function.name ~ '", "arguments": ' -}}
      {{- call.function.arguments | tojson ~ '}' -}}
    {%- endfor -%}
    {{- ']</call>' -}}
  {%- else -%}
  {%- for call in calls -%}
    {{- '<call>fn:' ~ call.function.name ~ '{' -}}
    {%- for key, value in call.function.arguments.items() -%}
      {{- (',' if not loop.first else '') ~ (key ~ ':' if call_keys else '') -}}
      {{- "'" ~ value ~ "'" if value is string else value | string -}}
    {%- endfor -%}
    {{- '}</call>' -}}
  {%- endfor -%}
  {%- endif -%}
{%- endmacro -%}
{%- macro declarations() -%}
  {%- for tool in tools or [] -%}{{- '<decl>' ~ tool.function.name ~ '</decl>' -}}{%- endfor -%}
{%- endmacro -%}
{%- for m in messages if writes_results or m.role != 'tool' -%}
  {%- if m.role == 'user' -%}{%- set ns.newest = loop.index0 -%}{%- endif -%}
{%- endfor -%}
{%- for m in messages if writes_results or m.role != 'tool' -%}
  {%- if m.content is not string or m.role not in ('system', 'user', 'assistant', 'tool') -%}
    {{- raise_exception('this template writes text in four roles') -}}
  {%- endif -%}
  {%- if m.content == refused_content -%}{{- raise_exception('this text is refused') -}}{%- endif -%}
  {%- set role = 'model' if m.role == 'assistant' else m.role -%}
  {%- if role == 'user' and ns.previous == 'user' -%}
    {%- if user_join is none -%}{{- raise_exception('two user turns in a row') -}}{%- endif -%}
    {{- user_join -}}
  {%- endif -%}
  {%- set ns.previous = role -%}
  {%- if ns.open and role not in ('tool', 'model') and closes_turn_before_next -%}
    {{- '<turn]>\n' -}}{%- set ns.open = false -%}
  {%- endif -%}
  {%- set calls = (m.tool_calls or []) if calls_tools else [] -%}
  {%- if role == 'tool' -%}
    {{- '<result>' ~ m.name ~ '(' ~ m.content ~ ')</result>' -}}
    {%- if result_lines -%}
      {{- '\n' if loop.nextitem is defined and loop.nextitem.role == 'tool' else '<done>' -}}
    {%- endif -%}
  {%- elif role == 'model' and ns.open -%}
    {{- m.content | trim -}}{{- call_blocks(calls) -}}
    {%- if not calls -%}{{- '<turn]>\n' -}}{%- set ns.open = false -%}{%- endif -%}
  {%- else -%}
    {%- if declares == 'in the newest turn' and loop.index0 == ns.newest -%}{{- declarations() -}}{%- endif -%}
    {{- '<[turn>' ~ role ~ '\n' -}}
    {%- if role == 'system' and reasons -%}{{- '<think_on>\n' -}}{%- endif -%}
    {%- if (calls or keeps_thought) and can_reason and m.reasoning_content -%}
      {{- '<thought>' ~ m.reasoning_content ~ '</thought>' -}}
    {%- endif -%}
    {{- call_blocks(calls) -}}{{- m.content | trim -}}
    {%- if role == 'system' and declares == 'in the system turn' -%}{{- declarations() -}}{%- endif -%}
    {%- if calls -%}{%- set ns.open = true -%}{%- else -%}{{- '<turn]>\n' -}}{%- endif -%}
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt and not ns.open -%}
  {{- '<[turn>model\n' -}}
  {%- if can_reason and not reasons -%}{{- '<thought></thought>' -}}{%- endif -%}
{%- endif -%}
"""

# The switches at the top of every invented template, each with the value it takes
# where a template sets none.
_SWITCHES = {
    "calls_tools": False,               # a call is written; without, it is left out
    "call_keys": True,                  # a call's values are written after their keys
    "calls_json": False,                # a turn's calls are written as one JSON array
    "declares": "in the system turn",   # where tools are declared: or "in the newest turn", or None
    "writes_results": True,             # a result is written; without, it is left out
    "result_lines": False,              # results one per line, their block closed once
    "closes_turn_before_next": False,   # a turn open after a result closes before a question
    "can_reason": False,                # a reasoning switch, and a thought's tags
    "keeps_thought": False,             # a past reply keeps its thought
    "user_join": "",                    # between two user turns in a row; None refuses them
    "refused_content": None,            # a text refused wherever it stands
}


def _jinja(value) -> str:
    """``value`` as a Jinja literal."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value)


def _invented(**switches) -> dict:
    """The invented template with ``switches`` set, as a fixture holds it."""
    assert set(switches) <= set(_SWITCHES), set(switches) - set(_SWITCHES)
    set_to = {**_SWITCHES, **switches}
    header = "".join(f"{{%- set {name} = {_jinja(value)} -%}}\n" for name, value in set_to.items())
    can_reason = set_to["can_reason"]
    return {
        "source": "invented for the tests", "eos_token": "<turn]>", "adds_bos": True,
        "marks_thought": can_reason, "can_reason": can_reason,
        "calls_tools": set_to["calls_tools"],
        "render_variables": ({"off": {"enable_thinking": False}, "on": {"enable_thinking": True}}
                             if can_reason else {"off": {}}),
        "chat_template": header + _TEXT.lstrip("\n"),
    }


PLAIN = _invented()
JOINS_USER_TURNS = _invented(user_join="<sep>")
REFUSES_TWO_USER_TURNS = _invented(user_join=None)
# Two user turns in a row are refused, and so is the text "refused".
REFUSES_A_TEXT = _invented(user_join=None, refused_content="refused")
TOOLS = _invented(calls_tools=True)
JSON_TOOLS = _invented(calls_tools=True, calls_json=True)
TOOLS_REFUSING_TWO_USER_TURNS = _invented(calls_tools=True, user_join=None)
TOOLS_CLOSING_OPEN_TURNS = _invented(calls_tools=True, closes_turn_before_next=True)
TOOLS_DECLARED_IN_THE_NEWEST_TURN = _invented(calls_tools=True, declares="in the newest turn")
KEYLESS_CALLS = _invented(calls_tools=True, call_keys=False)
NO_DECLARATIONS = _invented(calls_tools=True, declares=None)
NO_RESULT_BLOCK = _invented(calls_tools=True, writes_results=False)
RESULT_LINES = _invented(calls_tools=True, result_lines=True)
JSON_RESULT_LINES = _invented(calls_tools=True, result_lines=True, calls_json=True)
REASONING = _invented(calls_tools=True, can_reason=True)
REASONING_KEEPING_THOUGHTS = _invented(calls_tools=True, can_reason=True, keeps_thought=True)
# A model whose default template knows no tools, and which ships a template of its own for a
# conversation that declares tools, in another format, so that a switch shows in every turn.
TOOLS_IN_A_TEMPLATE_OF_THEIR_OWN = {
    **TOOLS, "calls_tools": True,
    "chat_template": "{%- for m in messages -%}<[say>{{ m.role }}\n{{ m.content }}<said]>\n"
                     "{%- endfor -%}{%- if add_generation_prompt -%}<[say>assistant\n{%- endif -%}",
    "tool_use_template": TOOLS["chat_template"]}

# Every invented template, by the name it goes by here.
INVENTED = {name: value for name, value in globals().items()
            if isinstance(value, dict) and "chat_template" in value}

# What tests read off a render of the invented format.
THINK_ON = "<think_on>\n"
EMPTY_THOUGHT = "<thought></thought>"
CALL_TAGS = ("<call>", "</call>")
# How the invented templates spell a call between its tags, after its lead-in
# ``fn:``, written out so the parser tests read without a round trip through extraction.
CALL_BODY = CallBody(name_close="{", key_open="", key_close=":", pair_sep=",",
                     body_close="}", empty_close="}", quote_open="'", quote_close="'")
