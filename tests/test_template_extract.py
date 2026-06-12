"""Fragment-extraction tests driven by canned chat-template renderers (no model)."""

from llama_chat.template_extract import extract_fragments


def _gemma_render(messages):
    """Gemma-style template: trims content, rejects a system role, uses ``model``."""
    out = []
    for m in messages:
        role = m["role"]
        if role == "system":
            return None  # Gemma has no system role
        role = "model" if role == "assistant" else role
        out.append(f"<start_of_turn>{role}\n{m['content'].strip()}<end_of_turn>\n")
    return "".join(out)


def _chatml_render(messages):
    """ChatML-style template: verbatim content, explicit system role."""
    return "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    )


def test_gemma_user_assistant_fragments():
    f = extract_fragments(_gemma_render)
    assert f.user_prefix == "<start_of_turn>user\n"
    assert f.user_suffix == "<end_of_turn>\n"
    assert f.assistant_prefix == "<start_of_turn>model\n"
    assert f.assistant_suffix == "<end_of_turn>\n"


def test_gemma_trims_content():
    assert extract_fragments(_gemma_render).trim_content is True


def test_gemma_system_falls_back_to_user_tags():
    f = extract_fragments(_gemma_render)
    assert f.system_prefix == f.user_prefix
    assert f.system_suffix == f.user_suffix


def test_chatml_extracts_explicit_system():
    f = extract_fragments(_chatml_render)
    assert f.system_prefix == "<|im_start|>system\n"
    assert f.system_suffix == "<|im_end|>\n"
    assert f.user_prefix == "<|im_start|>user\n"
    assert f.assistant_prefix == "<|im_start|>assistant\n"


def test_chatml_does_not_trim_content():
    assert extract_fragments(_chatml_render).trim_content is False


def test_no_template_returns_none():
    assert extract_fragments(lambda messages: None) is None
