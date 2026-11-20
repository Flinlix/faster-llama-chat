# Contributing

The rules every change keeps, and the commands that test them, are in
[AGENTS.md](AGENTS.md). Each rule names the tests that try to break it; a
change that needs a rule changed says so, with the reason.

Set up an environment with the test and development tools:

```bash
./install.sh --dev
```

This builds the engine for the machine, installs the package editable with
its `test`, `dev` and `docs` extras, and switches on the check that runs the tests
before every push.

A change that makes or revises a design decision comes with its decision
record in [docs/decisions/](docs/decisions/README.md), in the same commit.

Report a bug or ask for a feature through the issue forms. Report a
vulnerability privately, as [SECURITY.md](SECURITY.md) describes.
