"""Domain-decomposition sizing rules for RHSMode=1 Schwarz preconditioners.

These helpers are deliberately small and deterministic. They encode the
angular-patch heuristics used by the driver without depending on the full
SFINCS operator, which makes the multi-device Schwarz policy easy to test in
isolation.
"""

from __future__ import annotations

import os


def _dd_core_patch_ranges(n: int, block: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Return ``(core_start, core_end, patch_start, patch_end)`` DD/RAS blocks."""

    n = int(n)
    block = max(1, int(block))
    overlap = max(0, int(overlap))
    ranges: list[tuple[int, int, int, int]] = []
    for core_start in range(0, n, block):
        core_end = min(n, core_start + block)
        patch_start = max(0, core_start - overlap)
        patch_end = min(n, core_end + overlap)
        ranges.append((core_start, core_end, patch_start, patch_end))
    return ranges


def _rhs1_dd_auto_block_size(
    *,
    n: int,
    n_dev: int,
    sum_nxi: int,
    dof_target: int,
) -> int:
    """Choose a shard-local Schwarz block that spans more than one local shard."""

    n = max(1, int(n))
    n_dev = max(1, int(n_dev))
    sum_nxi = max(1, int(sum_nxi))
    dof_target = max(128, int(dof_target))
    local_n = max(1, (n + n_dev - 1) // n_dev)
    dof_cap = max(2, int(dof_target) // int(sum_nxi))
    block = max(dof_cap, local_n + dof_cap)
    return max(1, min(n, int(block)))


def _rhs1_dd_coarse_block_size(*, n: int, block: int, overlap: int) -> int:
    """Choose a wider coarse theta/zeta block for two-level Schwarz correction."""

    n = max(1, int(n))
    block = max(1, min(n, int(block)))
    overlap = max(0, int(overlap))
    coarse = block + max(8, block // 2, 2 * overlap)
    return max(1, min(n, int(coarse)))


def _rhs1_dd_coarse_level_count(*, n_dev: int) -> int:
    """Choose how many coarse residual-correction levels to apply."""

    n_dev = max(1, int(n_dev))
    coarse_levels_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS", "").strip()
    if coarse_levels_env:
        try:
            return max(0, int(coarse_levels_env))
        except ValueError:
            return 1
    if n_dev >= 8:
        return 2
    if n_dev >= 4:
        return 1
    return 0


def _rhs1_dd_coarse_block_sizes(*, n: int, block: int, overlap: int, levels: int) -> tuple[int, ...]:
    """Choose one or more increasingly wider coarse theta/zeta block sizes."""

    n = max(1, int(n))
    current = max(1, min(n, int(block)))
    out: list[int] = []
    for _ in range(max(0, int(levels))):
        next_block = _rhs1_dd_coarse_block_size(n=n, block=current, overlap=overlap)
        if next_block <= current:
            break
        out.append(int(next_block))
        current = int(next_block)
        if current >= n:
            break
    return tuple(out)


__all__ = [
    "_dd_core_patch_ranges",
    "_rhs1_dd_auto_block_size",
    "_rhs1_dd_coarse_block_size",
    "_rhs1_dd_coarse_block_sizes",
    "_rhs1_dd_coarse_level_count",
]
