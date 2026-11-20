"""Build hooks of the documentation site: a docstring's reference to a member is shown as the member's name in code."""

import re

# A reference as the API reference renders it from a docstring, such as :meth:`~a.b.c`, its
# name inside the element the code span became.
_ROLE = re.compile(r":(?:meth|attr|class|data|func|mod|exc|obj):<(\w+)>~?([\w.]+)</\1>")


def on_page_content(html, **_):
    """Show each reference as the code of the name it ends with."""
    return _ROLE.sub(lambda found: f"<{found.group(1)}>{found.group(2).rsplit('.', 1)[-1]}"
                                   f"</{found.group(1)}>", html)
