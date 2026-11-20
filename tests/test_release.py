"""A release is the tag of the version ``pyproject.toml`` states, and the documentation installs that release."""

import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def test_every_release_the_readme_installs_is_the_version_pyproject_states():
    stated = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]["version"]
    named = re.findall(r"@v(\d+\.\d+\.\d+)", (_ROOT / "README.md").read_text())
    assert named, "the README installs no release"
    assert set(named) == {stated}, f"the README installs {sorted(set(named))}, pyproject.toml states {stated}"
