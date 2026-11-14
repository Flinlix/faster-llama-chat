"""How much of their beginning or end two sequences share."""

from __future__ import annotations

from collections.abc import Sequence


def common_prefix_length(one: Sequence, other: Sequence) -> int:
    """How long a beginning two sequences of the same type share."""
    low, high = 0, min(len(one), len(other))
    while low < high:
        middle = (low + high + 1) // 2
        if one[:middle] == other[:middle]:
            low = middle
        else:
            high = middle - 1
    return low


def common_suffix_length(one: Sequence, other: Sequence) -> int:
    """How long an ending two sequences of the same type share."""
    return common_prefix_length(one[::-1], other[::-1])
