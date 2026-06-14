from __future__ import annotations

from sfincs_jax.explicit_sparse_factor_policy import (
    canonical_explicit_sparse_factor_kind,
    explicit_sparse_factor_kind_from_env,
    explicit_sparse_monolithic_guard_enabled,
    explicit_sparse_monolithic_max_size,
    parse_explicit_sparse_bool,
    parse_explicit_sparse_float,
    parse_explicit_sparse_int,
)


def test_parse_explicit_sparse_numeric_options_are_fail_closed() -> None:
    assert parse_explicit_sparse_int("7", 3, minimum=1) == 7
    assert parse_explicit_sparse_int("bad", 3, minimum=1) == 3
    assert parse_explicit_sparse_int("-4", 3, minimum=0) == 0

    assert parse_explicit_sparse_float("2.5", 1.0, minimum=0.0) == 2.5
    assert parse_explicit_sparse_float("bad", 1.25, minimum=0.0) == 1.25
    assert parse_explicit_sparse_float("-4", 1.25, minimum=0.5) == 0.5


def test_parse_explicit_sparse_bool_accepts_python_and_fortran_forms() -> None:
    assert parse_explicit_sparse_bool("yes", False) is True
    assert parse_explicit_sparse_bool(".true.", False) is True
    assert parse_explicit_sparse_bool("off", True) is False
    assert parse_explicit_sparse_bool(".false.", True) is False
    assert parse_explicit_sparse_bool("not-a-bool", True) is True


def test_canonical_explicit_sparse_factor_kind_aliases() -> None:
    assert canonical_explicit_sparse_factor_kind("splu", default="ilu") == "lu"
    assert canonical_explicit_sparse_factor_kind("spilu", default="lu") == "ilu"
    assert canonical_explicit_sparse_factor_kind("native_block_schur_lu", default="lu") == "symbolic_block_schur_lu"
    assert canonical_explicit_sparse_factor_kind("multifrontal_schur_lu", default="lu") == "symbolic_frontal_schur_lu"
    assert canonical_explicit_sparse_factor_kind("compressed_frontal_schur_lu", default="lu") == (
        "symbolic_blr_frontal_schur_lu"
    )
    assert canonical_explicit_sparse_factor_kind("nested_dissection_frontal_schur_lu", default="lu") == (
        "symbolic_nd_frontal_schur_lu"
    )
    assert canonical_explicit_sparse_factor_kind("block_edge_lu", default="lu") == "symbolic_superblock_lu"
    assert canonical_explicit_sparse_factor_kind("unknown", default="native_block_lu_coarse") == (
        "symbolic_block_lu_coarse"
    )
    assert canonical_explicit_sparse_factor_kind("unknown", default="unknown") == "lu"


def test_explicit_sparse_factor_kind_env_override_precedes_default() -> None:
    env = {"SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND": "jacobi"}
    assert explicit_sparse_factor_kind_from_env("ilu", env=env) == "jacobi"

    assert explicit_sparse_factor_kind_from_env("spilu", env={}) == "ilu"
    assert explicit_sparse_factor_kind_from_env("unknown", env={}) == "lu"


def test_explicit_sparse_monolithic_guard_policy() -> None:
    assert explicit_sparse_monolithic_guard_enabled(True, env={}) is True
    assert explicit_sparse_monolithic_guard_enabled(False, env={}) is False
    assert (
        explicit_sparse_monolithic_guard_enabled(
            True,
            env={"SFINCS_JAX_EXPLICIT_SPARSE_MONOLITHIC_GUARD": "0"},
        )
        is False
    )
    assert (
        explicit_sparse_monolithic_guard_enabled(
            False,
            env={"SFINCS_JAX_EXPLICIT_SPARSE_MONOLITHIC_GUARD": "on"},
        )
        is True
    )


def test_explicit_sparse_monolithic_max_size_factor_specific_precedence() -> None:
    env = {
        "SFINCS_JAX_EXPLICIT_SPARSE_MONOLITHIC_MAX_SIZE": "300",
        "SFINCS_JAX_EXPLICIT_SPARSE_MONOLITHIC_LU_MAX_SIZE": "100",
        "SFINCS_JAX_EXPLICIT_SPARSE_MONOLITHIC_ILU_MAX_SIZE": "200",
    }

    assert explicit_sparse_monolithic_max_size("lu", env=env) == 100
    assert explicit_sparse_monolithic_max_size("ilu", env=env) == 200
    assert explicit_sparse_monolithic_max_size(
        "lu",
        env={"SFINCS_JAX_EXPLICIT_SPARSE_MONOLITHIC_MAX_SIZE": "bad"},
    ) == (
        250_000
    )
