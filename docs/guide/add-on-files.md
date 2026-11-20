# Add-on files

What a model needs beyond its chat template comes from a JSON add-on file,
never from code: the one next to the GGUF with its name and `.json`, else one
shipped in `llama_cpp_conversation/addons/`, each used only where it names the
model's template by its SHA-256. A file next to the GGUF that names no
template, or another one, or is no JSON object, is not used, and a warning on
the logger `llama_cpp_conversation` says why; one that is not JSON at all is
refused when the model loads, naming the file. It fills in only what the
template cannot tell; what the template states, the template decides, except a
call format the file names, which wins. The file used is checked when the model
loads, each value's type included and a null value refused, and anything else
in it is refused with an error naming the file.

```json
{"templates": ["<sha256 of the chat template>"],
 "source": "where the values come from",
 "sampling": {"temperature": 0.6, "top_k": 20, "top_p": 0.95},
 "call": {"opening": "...", "name_close": "...", "closing": "..."},
 "reasoning_on": {"reasoning_effort": "high"},
 "reasoning_levels": ["low", "medium", "high"]}
```

`sampling` sets any of the eight model sampling fields, `presence_penalty`,
`frequency_penalty` and `penalty_window_tokens` among them, and its values win over the GGUF's (see
[Configuration](configuration.md)). Since a shipped file is matched by the
template, it also applies to a fine-tuned model that keeps the same template,
and its values win over those that fine-tune's GGUF states; a file next to the
GGUF, or values set in `Config`, take precedence for such a model. `call` says
how the model writes a call: the text a call opens with, the text after the
function's name, and the text that closes it. It is taken over whatever the
template shows of a call, and refused where the template never writes that
closing.
`reasoning_on` names the template variables, with their values, that turn the
model's reasoning on, in place of what the library reads off the template. The
library reads only the switches templates share: `enable_thinking`, and a
`reasoning_effort` whose `"none"` turns reasoning off, the template's default
then being on. Any other switch is named here, such as an effort whose
`"high"` turns reasoning on, and so is the case of a template whose default
does not reason although it writes something of its own for `"none"`. A
template whose prompt with reasoning on closes the thought at once is read as
unable to reason. `reasoning_on` is refused at load where it writes the prompt
as reasoning off does.
`reasoning_levels` names the effort levels `begin(reasoning_level=...)` takes, for a
template that writes any `reasoning_effort` it is given into the prompt as it
is, so that the template itself tells no level apart. Each one is checked when
the model loads, as a level read from a template is; a file that names a level
the template does not take as one, or names levels for a template that tells
its own apart, is refused at load with the cause.
