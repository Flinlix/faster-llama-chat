"""How much of their beginning or end two sequences share."""

from __future__ import annotations

from collections.abc import Sequence
from .loop_variant import LoopVariant


def common_prefix_length(one: Sequence, other: Sequence) -> int:
    """How long a beginning two sequences of the same type share."""
    low, high = 0, min(len(one), len(other))
    searching_variant = LoopVariant("measuring a shared beginning")
    while low < high:
        searching_variant.passes(high - low)
        middle = (low + high + 1) // 2
        if one[:middle] == other[:middle]:
            low = middle
        else:
            high = middle - 1
    return low


def common_suffix_length(one: Sequence, other: Sequence) -> int:
    """How long an ending two sequences of the same type share."""
    return common_prefix_length(one[::-1], other[::-1])
