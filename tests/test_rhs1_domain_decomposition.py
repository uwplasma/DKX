from __future__ import annotations

from sfincs_jax.rhs1_domain_decomposition import (
    _dd_core_patch_ranges,
    _rhs1_dd_auto_block_size,
    _rhs1_dd_coarse_block_size,
    _rhs1_dd_coarse_block_sizes,
    _rhs1_dd_coarse_level_count,
)


def test_dd_core_patch_ranges_cover_domain_with_overlap() -> None:
    assert _dd_core_patch_ranges(n=10, block=4, overlap=1) == [
        (0, 4, 0, 5),
        (4, 8, 3, 9),
        (8, 10, 7, 10),
    ]


def test_dd_core_patch_ranges_clamp_to_domain_edges() -> None:
    assert _dd_core_patch_ranges(n=5, block=2, overlap=10) == [
        (0, 2, 0, 5),
        (2, 4, 0, 5),
        (4, 5, 0, 5),
    ]


def test_rhs1_dd_auto_block_size_spans_more_than_one_local_shard() -> None:
    block = _rhs1_dd_auto_block_size(n=31, n_dev=8, sum_nxi=144, dof_target=1200)

    assert block == 12
    assert block > 4


def test_rhs1_dd_auto_block_size_respects_global_extent() -> None:
    block = _rhs1_dd_auto_block_size(n=31, n_dev=2, sum_nxi=144, dof_target=1200)

    assert block == 24
    assert block <= 31


def test_rhs1_dd_coarse_block_size_widens_local_patch() -> None:
    coarse = _rhs1_dd_coarse_block_size(n=31, block=12, overlap=1)

    assert coarse == 20
    assert coarse > 12


def test_rhs1_dd_coarse_level_count_auto_and_env(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS", raising=False)
    assert _rhs1_dd_coarse_level_count(n_dev=2) == 0
    assert _rhs1_dd_coarse_level_count(n_dev=4) == 1
    assert _rhs1_dd_coarse_level_count(n_dev=8) == 2

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS", "3")
    assert _rhs1_dd_coarse_level_count(n_dev=2) == 3

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS", "bad")
    assert _rhs1_dd_coarse_level_count(n_dev=8) == 1


def test_rhs1_dd_coarse_block_sizes_build_and_stop_at_global_extent() -> None:
    assert _rhs1_dd_coarse_block_sizes(n=63, block=12, overlap=1, levels=2) == (20, 30)
    assert _rhs1_dd_coarse_block_sizes(n=25, block=12, overlap=4, levels=3) == (20, 25)
