# A model without a usable template

A model that ships no chat template is refused when it loads: the template is
what every conversation is written with. For one whose template writes no
tool call it can be read from, its [add-on file](add-on-files.md) says how the
model writes one. The closing after a reply must begin with one of the model's own tags,
or construction raises: a model could never stop on plain text. A template
that writes nothing after a reply is used only where the tag that opens the
next turn ends a generation for the model, since the reply ends where the
model draws that tag.

## Special tokens in message content

A caller's text (a message, a thought, a
call's name and arguments, and each description of a declared tool) is
tokenized with special-token parsing off, and the template's tags with it on,
so a literal tag typed into a message or a tool's description stays text and
can never forge a turn boundary. The rest of a declaration (a tool's name, a
parameter's name, a choice value, a title) is what a template reads to decide
what it writes, so it is written as it stands: `begin` refuses one that holds
a tag with an `InvalidArgumentError` naming `tools`. A conversation is refused where its
template writes anything derived from a text besides the text itself, such as
the text's length, or something it writes only when the text holds a certain
word. The plain text between two tags, the template's and the caller's alike,
is tokenized as one run, as a render of the whole conversation tokenizes it.

## A template that writes the date

A template that writes the date follows the clock: a conversation that runs
past midnight is decoded again once, from the date on.
