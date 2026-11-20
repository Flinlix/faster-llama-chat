"""A stand-in model, for tests of code that drives a :class:`~llama_cpp_conversation.wrapper.ChatWrapper` without loading one.

:mod:`llama_cpp_conversation.testing.fake_context` holds :class:`FakeContext`, a cache
context over a list of ints whose model is a script, and
:mod:`llama_cpp_conversation.testing.invented_templates` the small chat templates it
renders, each in a format no real model uses.
"""
