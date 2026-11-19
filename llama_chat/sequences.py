"""How much of their beginning or end two sequences share."""

from __future__ import annotations

from collections.abc import Sequence
from .loop_variant import LoopVariant


def common_prefix_length(one: Sequence, other: Sequence) -> int:
    """How long a beginning two sequences of the same type share."""
    shared = min(len(one), len(other))
    # Compared a block at a time, so that the search copies each item about once, then halved
    # inside the first block that differs.
    low = next((at for at in range(0, shared, _BLOCK)
                if one[at:min(at + _BLOCK, shared)] != other[at:min(at + _BLOCK, shared)]), shared)
    if low == shared:
        return shared
    high = min(low + _BLOCK, shared) - 1
    searching_variant = LoopVariant("measuring a shared beginning")
    while low < high:
        searching_variant.passes(high - low)
        middle = (low + high + 1) // 2
        if one[low:middle] == other[low:middle]:
            low = middle
        else:
            high = middle - 1
    return low


# How many items a shared beginning is compared in at a time.
_BLOCK = 1024


def common_suffix_length(one: Sequence, other: Sequence) -> int:
    """How long an ending two sequences of the same type share."""
    return common_prefix_length(one[::-1], other[::-1])
