from __future__ import annotations

from sfincs_jax.solvers.explicit_sparse import (
    ExplicitSparseFactorSettings,
    canonical_explicit_sparse_factor_kind,
    explicit_sparse_factor_kind_from_env,
    explicit_sparse_factor_settings_from_env,
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
    assert canonical_explicit_sparse_factor_kind("compressed_frontal_schur_lu", default="lu") == "lu"
    assert canonical_explicit_sparse_factor_kind("nested_dissection_frontal_schur_lu", default="lu") == "lu"
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


def test_explicit_sparse_factor_settings_defaults_are_driver_compatible() -> None:
    settings = explicit_sparse_factor_settings_from_env(default_factor_kind="spilu", env={})

    assert isinstance(settings, ExplicitSparseFactorSettings)
    assert settings.factor_kind == "ilu"
    assert settings.block_cols == 32
    assert settings.dense_max_mb == 128.0
    assert settings.csr_max_mb == 512.0
    assert settings.drop_tol == 0.0
    assert settings.pattern_color_batch == 1
    assert settings.permc_spec == "COLAMD"
    assert settings.diag_pivot_thresh == 1.0
    assert settings.monolithic_guard_enabled is True


def test_explicit_sparse_factor_settings_env_overrides() -> None:
    env = {
        "SFINCS_JAX_EXPLICIT_SPARSE_BLOCK_COLS": "64",
        "SFINCS_JAX_EXPLICIT_SPARSE_DENSE_MAX_MB": "256",
        "SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB": "1024",
        "SFINCS_JAX_EXPLICIT_SPARSE_DROP_TOL": "1e-12",
        "SFINCS_JAX_EXPLICIT_SPARSE_PATTERN_COLOR_BATCH": "5",
        "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_FRONTAL_MAX_SEPARATOR_COLS": "99",
        "SFINCS_JAX_EXPLICIT_SPARSE_SYMBOLIC_SUPERBLOCK_MIN_RETAINED_CROSS_FRACTION": "0.25",
        "SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND": "native_frontal_schur_lu",
        "SFINCS_JAX_EXPLICIT_SPARSE_MONOLITHIC_GUARD": "off",
        "SFINCS_JAX_EXPLICIT_SPARSE_ILU_FILL_FACTOR": "7",
        "SFINCS_JAX_EXPLICIT_SPARSE_ILU_DROP_TOL": "1e-6",
        "SFINCS_JAX_EXPLICIT_SPARSE_PERMC_SPEC": "mmd_ata",
        "SFINCS_JAX_EXPLICIT_SPARSE_DIAG_PIVOT_THRESH": "0.1",
    }

    settings = explicit_sparse_factor_settings_from_env(env=env)

    assert settings.block_cols == 64
    assert settings.dense_max_mb == 256.0
    assert settings.csr_max_mb == 1024.0
    assert settings.drop_tol == 1.0e-12
    assert settings.pattern_color_batch == 5
    assert settings.symbolic_frontal_max_separator_cols == 99
    assert settings.symbolic_superblock_min_retained_cross_fraction == 0.25
    assert settings.factor_kind == "symbolic_frontal_schur_lu"
    assert settings.monolithic_guard_enabled is False
    assert settings.ilu_fill_factor == 7.0
    assert settings.ilu_drop_tol == 1.0e-6
    assert settings.permc_spec == "MMD_ATA"
    assert settings.diag_pivot_thresh == 0.1


def test_explicit_sparse_factor_settings_bounds_and_invalid_permc_fallback() -> None:
    settings = explicit_sparse_factor_settings_from_env(
        env={
            "SFINCS_JAX_EXPLICIT_SPARSE_PATTERN_COLOR_BATCH": "-5",
            "SFINCS_JAX_EXPLICIT_SPARSE_PERMC_SPEC": "not-a-permc",
        },
        default_permc_spec="also-bad",
    )

    assert settings.pattern_color_batch == 1
    assert settings.permc_spec == "COLAMD"
