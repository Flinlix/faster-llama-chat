# Security policy

## Supported versions

Security fixes are made for the newest release only.

## Reporting a vulnerability

Report it privately through GitHub: on the repository's **Security** tab,
choose **Report a vulnerability**. Do not open a public issue for it.

A report helps most when it holds the version, the model file or template
that shows the problem, and the shortest script that reproduces it.

## What the library trusts

- **A model file's chat template** is code written by whoever made the file.
  The library renders it in Jinja's immutable sandbox. A way past that
  sandbox is a vulnerability.
- **Add-on files** are read as data and never run.
- **Tools** a caller registers run with the caller's own rights, on the
  arguments the model wrote. For a tool in a `ToolRegistry`, the arguments
  are checked against the tool's signature before it runs; what a tool does
  with them is the caller's responsibility.
- **Message text** never becomes a tag: a tag typed into a message stays text.
  A message that the cache holds as a template's control token is a
  vulnerability.
- **Saved turns** (`generated_turns` in what `conversation()` returns) are
  trusted input: their tokens go into the cache as they stand, tags included,
  so whoever can change them can write any turn before the model. A
  conversation that passed through hands you do not trust is begun again from
  its messages alone.
- **The model file** is read by llama.cpp, which parses it and runs its
  weights. Load only model files you trust; a fault in llama.cpp's own parser
  is reported to llama.cpp.
- **The add-on file next to a model file** is read as data, but what it
  states (sampling, the call format, how reasoning is switched)
  changes what the model is given and how its calls are read. It is trusted as
  the model file is.
