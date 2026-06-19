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


def _folding_render(messages):
    """Llama-2-style template: folds the system message into the first user turn."""
    out = []
    sys_text = ""
    for m in messages:
        if m["role"] == "system":
            sys_text = m["content"] + "\n\n"
        elif m["role"] == "user":
            out.append(f"<u>{sys_text}{m['content']}</u>")
            sys_text = ""
        else:
            out.append(f"<a>{m['content']}</a>")
    return "".join(out)


def test_folded_system_falls_back_to_user_tags():
    # No standalone system turn exists, so there is no clean fragment to
    # recover - the extractor must fall back instead of capturing the user
    # sentinel inside the system suffix.
    f = extract_fragments(_folding_render)
    assert f.system_prefix == f.user_prefix == "<u>"
    assert f.system_suffix == f.user_suffix == "</u>"


def _inconsistent_render(messages):
    """Degenerate template that restructures the user turn in multi-message
    renders, so the single-message user fragment never reappears."""
    if len(messages) == 1:
        return f"<u>{messages[0]['content']}</u>"
    return "".join(f"[{m['role']}]{m['content']}" for m in messages)


def test_inconsistent_multi_message_render_falls_back_to_user_tags():
    # The user fragment recovered from the single-message probe does not occur
    # in the two-message render; the extractor must fall back instead of
    # crashing on the missing split point.
    f = extract_fragments(_inconsistent_render)
    assert f.user_prefix == "<u>"
    assert f.assistant_prefix == f.user_prefix
    assert f.assistant_suffix == f.user_suffix


def _prefix_only_render(messages):
    """Degenerate template whose tags only open a turn (empty suffixes)."""
    return "".join(f"<{m['role']}>{m['content']}" for m in messages)


def test_empty_suffix_template_extracts_without_crashing():
    f = extract_fragments(_prefix_only_render)
    assert f.user_prefix == "<user>"
    assert f.user_suffix == ""
    assert f.assistant_prefix == "<assistant>"
    assert f.assistant_suffix == ""
    assert f.system_prefix == "<system>"
    assert f.system_suffix == ""
