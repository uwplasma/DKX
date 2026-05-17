from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.io import write_sfincs_jax_output_h5
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.solver import GMRESSolveResult
import sfincs_jax.v3_driver as v3_driver
from sfincs_jax.v3_system import full_system_operator_from_namelist, rhs_v3_full_system


def _patch_block_value(block: str, key: str, value: str) -> str:
    pat = re.compile(rf"(?im)^[ \t]*{re.escape(key)}[ \t]*=[ \t]*([^!\n\r]+)[ \t]*$")
    new_line = f"  {key} = {value}"
    if pat.search(block):
        return pat.sub(new_line, block)
    if not block.endswith("\n"):
        block += "\n"
    return block + new_line + "\n"


def _patch_resolution_block(txt: str, *, ntheta: int, nzeta: int, nxi: int, nx: int, solver_tol: float) -> str:
    start = re.search(r"(?im)^\s*&resolutionParameters\s*$", txt)
    if start is None:
        raise ValueError("Missing &resolutionParameters")
    end = re.search(r"(?m)^\s*/\s*$", txt[start.end() :])
    if end is None:
        raise ValueError("Missing / terminator for &resolutionParameters")
    end_pos = start.end() + end.start()
    block = txt[start.end() : end_pos]
    block = _patch_block_value(block, "Ntheta", str(int(ntheta)))
    block = _patch_block_value(block, "Nzeta", str(int(nzeta)))
    block = _patch_block_value(block, "Nxi", str(int(nxi)))
    block = _patch_block_value(block, "Nx", str(int(nx)))
    block = _patch_block_value(block, "solverTolerance", f"{solver_tol:.16g}")
    return txt[: start.end()] + block + txt[end_pos:]


def _patch_export_block(txt: str) -> str:
    start = re.search(r"(?im)^\s*&export_f\s*$", txt)
    if start is None:
        return txt
    end = re.search(r"(?m)^\s*/\s*$", txt[start.end() :])
    if end is None:
        raise ValueError("Missing / terminator for &export_f")
    end_pos = start.end() + end.start()
    block = txt[start.end() : end_pos]
    block = _patch_block_value(block, "export_full_f", ".false.")
    block = _patch_block_value(block, "export_delta_f", ".false.")
    block = _patch_block_value(block, "export_f_theta_option", "0")
    block = _patch_block_value(block, "export_f_zeta_option", "0")
    block = _patch_block_value(block, "export_f_xi_option", "0")
    block = _patch_block_value(block, "export_f_x_option", "0")
    return txt[: start.end()] + block + txt[end_pos:]


def test_full_precond_uses_schur_for_constraint_scheme2(tmp_path: Path, monkeypatch) -> None:
    input_path = Path(__file__).parent / "reduced_inputs" / "tokamak_1species_PASCollisions_noEr_Nx1.input.namelist"
    out_path = tmp_path / "sfincsOutput_jax.h5"

    txt = input_path.read_text()
    txt = _patch_resolution_block(txt, ntheta=7, nzeta=1, nxi=4, nx=1, solver_tol=1e-6)
    txt = _patch_export_block(txt)
    patched = tmp_path / "input_schur_tiny.namelist"
    patched.write_text(txt)

    logs: list[str] = []

    def emit(_level: int, msg: str) -> None:
        logs.append(msg)

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SOLVE_METHOD", "incremental")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ACTIVE_CUTOFF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHUR_TOKAMAK", "1")

    write_sfincs_jax_output_h5(
        input_namelist=patched,
        output_path=out_path,
        compute_solution=True,
        emit=emit,
        verbose=True,
    )

    joined = "\n".join(logs)
    assert "building RHSMode=1 preconditioner=schur" in joined


def test_full_precond_tokamak_defaults_to_xblock_tz(tmp_path: Path, monkeypatch) -> None:
    input_path = Path(__file__).parent / "reduced_inputs" / "tokamak_1species_PASCollisions_noEr_Nx1.input.namelist"
    out_path = tmp_path / "sfincsOutput_jax.h5"

    txt = input_path.read_text()
    txt = _patch_resolution_block(txt, ntheta=7, nzeta=1, nxi=4, nx=1, solver_tol=1e-6)
    txt = _patch_export_block(txt)
    patched = tmp_path / "input_xblock_tz_tiny.namelist"
    patched.write_text(txt)

    logs: list[str] = []

    def emit(_level: int, msg: str) -> None:
        logs.append(msg)

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SOLVE_METHOD", "incremental")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ACTIVE_CUTOFF", "0")

    write_sfincs_jax_output_h5(
        input_namelist=patched,
        output_path=out_path,
        compute_solution=True,
        emit=emit,
        verbose=True,
    )

    joined = "\n".join(logs)
    assert "building RHSMode=1 preconditioner=xblock_tz" in joined


def test_pas_tokamak_structured_tail_matches_legacy(tmp_path: Path, monkeypatch) -> None:
    input_path = Path(__file__).parent / "reduced_inputs" / "tokamak_1species_PASCollisions_noEr_Nx1.input.namelist"

    txt = input_path.read_text()
    txt = _patch_resolution_block(txt, ntheta=7, nzeta=1, nxi=5, nx=2, solver_tol=1e-6)
    txt = _patch_export_block(txt)
    patched = tmp_path / "input_pas_tokamak_structured_tiny.namelist"
    patched.write_text(txt)

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_PAS_TOKAMAK_LMAX", "5")

    nml = read_sfincs_input(patched)
    op = full_system_operator_from_namelist(nml=nml)
    rhs = rhs_v3_full_system(op)

    v3_driver._RHSMODE1_PAS_TOKAMAK_THETA_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_PAS_TOKAMAK_STRUCTURED", "0")
    legacy_precond = v3_driver._build_rhsmode1_pas_tokamak_theta_preconditioner(op=op)
    legacy_state = legacy_precond(rhs)
    legacy_cache = next(iter(v3_driver._RHSMODE1_PAS_TOKAMAK_THETA_CACHE.values()))
    assert legacy_cache.tail_factors is None

    v3_driver._RHSMODE1_PAS_TOKAMAK_THETA_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_PAS_TOKAMAK_STRUCTURED", "1")
    structured_precond = v3_driver._build_rhsmode1_pas_tokamak_theta_preconditioner(op=op)
    structured_state = structured_precond(rhs)
    structured_cache = next(iter(v3_driver._RHSMODE1_PAS_TOKAMAK_THETA_CACHE.values()))
    assert structured_cache.tail_factors is not None
    assert structured_cache.tail_factors[0][0] is not None

    structured_state = np.asarray(structured_state, dtype=np.float64)
    legacy_state = np.asarray(legacy_state, dtype=np.float64)
    diff = structured_state - legacy_state
    rel_norm = np.linalg.norm(diff) / np.linalg.norm(legacy_state)
    assert rel_norm < 6.0e-2
    assert float(np.max(np.abs(diff))) < 2.0e-4


def test_pas_tokamak_structured_tail_is_opt_in_by_default(tmp_path: Path, monkeypatch) -> None:
    input_path = Path(__file__).parent / "reduced_inputs" / "tokamak_1species_PASCollisions_noEr_Nx1.input.namelist"

    txt = input_path.read_text()
    txt = _patch_resolution_block(txt, ntheta=7, nzeta=1, nxi=5, nx=2, solver_tol=1e-6)
    txt = _patch_export_block(txt)
    patched = tmp_path / "input_pas_tokamak_structured_default.namelist"
    patched.write_text(txt)

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_PAS_TOKAMAK_LMAX", "5")
    monkeypatch.delenv("SFINCS_JAX_PAS_TOKAMAK_STRUCTURED", raising=False)

    nml = read_sfincs_input(patched)
    op = full_system_operator_from_namelist(nml=nml)

    v3_driver._RHSMODE1_PAS_TOKAMAK_THETA_CACHE.clear()
    _ = v3_driver._build_rhsmode1_pas_tokamak_theta_preconditioner(op=op)
    cache = next(iter(v3_driver._RHSMODE1_PAS_TOKAMAK_THETA_CACHE.values()))
    assert cache.tail_factors is None


def test_schur_base_prefers_pas_tokamak_theta_for_tokamak_pas_noer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = Path(__file__).parent / "reduced_inputs" / "tokamak_1species_PASCollisions_noEr_Nx1.input.namelist"

    txt = input_path.read_text()
    txt = _patch_resolution_block(txt, ntheta=7, nzeta=1, nxi=5, nx=2, solver_tol=1e-6)
    txt = _patch_export_block(txt)
    patched = tmp_path / "input_schur_base_tokamak_theta.namelist"
    patched.write_text(txt)

    op = full_system_operator_from_namelist(nml=read_sfincs_input(patched))
    called: list[str] = []

    def _tokamak_theta_builder(**_kwargs):
        called.append("pas_tokamak_theta")
        return lambda v: v

    def _unexpected_xblock(**_kwargs):
        raise AssertionError("xblock_tz base should not be selected before pas_tokamak_theta")

    monkeypatch.setattr(v3_driver, "_build_rhsmode1_pas_tokamak_theta_preconditioner", _tokamak_theta_builder)
    monkeypatch.setattr(v3_driver, "_build_rhsmode1_xblock_tz_preconditioner", _unexpected_xblock)

    _ = v3_driver._build_rhsmode1_schur_preconditioner(op=op)

    assert called == ["pas_tokamak_theta"]


def test_schur_base_prefers_pas_tz_for_geometry4_pas_offender(monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = Path(__file__).parent / "reduced_inputs" / "geometryScheme4_2species_PAS_noEr.input.namelist"
    op = full_system_operator_from_namelist(nml=read_sfincs_input(input_path))
    called: list[str] = []

    def _pas_tz_builder(**_kwargs):
        called.append("pas_tz")
        return lambda v: v

    def _unexpected_builder(**_kwargs):
        raise AssertionError("geometry4 PAS Schur base should select pas_tz for the pinned offender block")

    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SCHUR_BASE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_MIN", raising=False)
    monkeypatch.setattr(v3_driver, "_build_rhsmode1_pas_tz_preconditioner", _pas_tz_builder)
    monkeypatch.setattr(v3_driver, "_build_rhsmode1_xblock_tz_preconditioner", _unexpected_builder)
    monkeypatch.setattr(v3_driver, "_build_rhsmode1_species_block_preconditioner", _unexpected_builder)
    monkeypatch.setattr(v3_driver, "_build_rhsmode1_pas_schur_preconditioner", _unexpected_builder)
    monkeypatch.setattr(v3_driver, "_build_rhsmode1_theta_zeta_preconditioner", _unexpected_builder)

    _ = v3_driver._build_rhsmode1_schur_preconditioner(op=op)

    assert called == ["pas_tz"]


def test_schur_base_small_pas_fallback_uses_geom_hint_without_nameerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = Path(__file__).parent / "reduced_inputs" / "geometryScheme4_2species_PAS_noEr.input.namelist"

    txt = input_path.read_text()
    txt = _patch_resolution_block(txt, ntheta=4, nzeta=4, nxi=4, nx=2, solver_tol=1e-6)
    txt = _patch_export_block(txt)
    patched = tmp_path / "input_small_pas_schur_base.namelist"
    patched.write_text(txt)

    op = full_system_operator_from_namelist(nml=read_sfincs_input(patched))
    called: list[str] = []

    def _pas_schur_builder(**_kwargs):
        called.append("pas_schur")
        return lambda v: v

    def _unexpected_builder(**_kwargs):
        raise AssertionError("small PAS fallback should reach pas_schur after cheaper block bases are disabled")

    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SCHUR_BASE", raising=False)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "0")
    monkeypatch.setattr(v3_driver, "_build_rhsmode1_pas_schur_preconditioner", _pas_schur_builder)
    monkeypatch.setattr(v3_driver, "_build_rhsmode1_species_block_preconditioner", _unexpected_builder)
    monkeypatch.setattr(v3_driver, "_build_rhsmode1_xblock_tz_preconditioner", _unexpected_builder)
    monkeypatch.setattr(v3_driver, "_build_rhsmode1_theta_zeta_preconditioner", _unexpected_builder)
    v3_driver._set_precond_policy_hints(geom_scheme=4)
    try:
        _ = v3_driver._build_rhsmode1_schur_preconditioner(op=op)
    finally:
        v3_driver._set_precond_policy_hints()

    assert called == ["pas_schur"]


def test_schur_auto_min_for_pas(tmp_path: Path, monkeypatch) -> None:
    """Auto Schur selection should respect SFINCS_JAX_RHSMODE1_SCHUR_AUTO_MIN."""
    input_path = Path(__file__).parent / "reduced_inputs" / "geometryScheme4_2species_PAS_noEr.input.namelist"
    out_path = tmp_path / "sfincsOutput_jax.h5"

    txt = input_path.read_text()
    txt = _patch_resolution_block(txt, ntheta=7, nzeta=7, nxi=4, nx=2, solver_tol=1e-6)
    txt = _patch_export_block(txt)
    patched = tmp_path / "input_schur_auto_tiny.namelist"
    patched.write_text(txt)

    logs: list[str] = []

    def emit(_level: int, msg: str) -> None:
        logs.append(msg)

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SOLVE_METHOD", "incremental")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ACTIVE_CUTOFF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHUR_AUTO_MIN", "0")

    write_sfincs_jax_output_h5(
        input_namelist=patched,
        output_path=out_path,
        compute_solution=True,
        emit=emit,
        verbose=True,
    )

    joined = "\n".join(logs)
    assert "building RHSMode=1 preconditioner=schur" in joined


def test_pas_auto_strong_retry_skips_after_strong_base(tmp_path: Path, monkeypatch) -> None:
    """PAS auto strong retry should skip after already-strong base preconditioners."""
    assert v3_driver._pas_auto_skip_strong_retry(
        has_pas=True,
        strong_precond_env="auto",
        rhs1_precond_kind="schur",
        residual_norm=5.0,
        target=1.0,
        ratio=10.0,
    )
    assert not v3_driver._pas_auto_skip_strong_retry(
        has_pas=True,
        strong_precond_env="auto",
        rhs1_precond_kind="schur",
        residual_norm=15.0,
        target=1.0,
        ratio=10.0,
    )
    assert not v3_driver._pas_auto_skip_strong_retry(
        has_pas=True,
        strong_precond_env="auto",
        rhs1_precond_kind="theta_line",
        residual_norm=5.0,
        target=1.0,
        ratio=10.0,
    )
    assert not v3_driver._pas_auto_skip_strong_retry(
        has_pas=False,
        strong_precond_env="auto",
        rhs1_precond_kind="schur",
        residual_norm=5.0,
        target=1.0,
        ratio=10.0,
    )


def test_precond_dtype_auto_promotes_geom4_pas_schur_cpu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PRECOND_DTYPE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4_MIN_SIZE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4_ER_MAX", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    v3_driver._set_precond_policy_hints(
        geom_scheme=4,
        use_dkes=False,
        rhs1_precond_kind="schur",
        has_pas=True,
        has_fp=False,
        include_phi1=False,
        rhs_mode=1,
        er_abs=0.0,
    )
    v3_driver._set_precond_size_hint(30000)
    assert v3_driver._precond_dtype() == v3_driver.jnp.float32
    v3_driver._set_precond_policy_hints()
    v3_driver._set_precond_size_hint(None)


def test_precond_dtype_auto_keeps_fp64_for_pas_dkes(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PRECOND_DTYPE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    v3_driver._set_precond_policy_hints(
        geom_scheme=11,
        use_dkes=True,
        rhs1_precond_kind="schur",
        has_pas=True,
        has_fp=False,
        include_phi1=False,
        rhs_mode=1,
        er_abs=0.0,
    )
    v3_driver._set_precond_size_hint(30000)
    assert v3_driver._precond_dtype() == v3_driver.jnp.float64
    v3_driver._set_precond_policy_hints()
    v3_driver._set_precond_size_hint(None)


@pytest.mark.parametrize("forced_precond", ["schur", "xblock_tz"])
def test_forced_rhs1_preconditioner_does_not_crash_before_er_is_computed(
    tmp_path: Path,
    monkeypatch,
    forced_precond: str,
) -> None:
    input_path = Path(__file__).parent / "reduced_inputs" / "tokamak_1species_PASCollisions_noEr_Nx1.input.namelist"
    txt = input_path.read_text()
    txt = _patch_resolution_block(txt, ntheta=5, nzeta=1, nxi=3, nx=1, solver_tol=1e-6)
    txt = _patch_export_block(txt)
    patched = tmp_path / f"input_forced_{forced_precond}.namelist"
    patched.write_text(txt)

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SOLVE_METHOD", "incremental")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PRECONDITIONER", forced_precond)
    monkeypatch.setenv("SFINCS_JAX_GMRES_MAXITER", "1")
    monkeypatch.setenv("SFINCS_JAX_LINEAR_STAGE2", "0")
    if forced_precond == "xblock_tz":
        monkeypatch.setattr(
            v3_driver,
            "_build_rhsmode1_xblock_tz_preconditioner",
            lambda **_kwargs: (lambda v: v),
        )
    else:
        monkeypatch.setattr(
            v3_driver,
            "_build_rhsmode1_schur_preconditioner",
            lambda **_kwargs: (lambda v: v),
        )

    def _fake_linear_with_residual(**kwargs):
        b = jnp.asarray(kwargs["b"])
        x = jnp.zeros_like(b)
        return GMRESSolveResult(x=x, residual_norm=jnp.asarray(0.0, dtype=b.dtype)), jnp.zeros_like(b)

    monkeypatch.setattr(v3_driver, "_gmres_solve_with_residual_dispatch", _fake_linear_with_residual)

    result = v3_driver.solve_v3_full_system_linear_gmres(
        nml=read_sfincs_input(patched),
        tol=1.0e-6,
        emit=lambda _level, _msg: None,
    )

    assert int(result.x.size) > 0


def test_pas_dkes_xblock_allowed_only_for_moderate_blocks() -> None:
    assert v3_driver._rhs1_pas_dkes_xblock_allowed(
        has_pas=True,
        use_dkes=True,
        backend="gpu",
        n_theta=9,
        n_zeta=11,
        max_l=21,
        xblock_tz_limit=2500,
    )
    assert v3_driver._rhs1_pas_dkes_xblock_allowed(
        has_pas=True,
        use_dkes=True,
        backend="cpu",
        n_theta=9,
        n_zeta=11,
        max_l=21,
        xblock_tz_limit=2500,
    )
    assert not v3_driver._rhs1_pas_dkes_xblock_allowed(
        has_pas=True,
        use_dkes=True,
        backend="gpu",
        n_theta=17,
        n_zeta=23,
        max_l=36,
        xblock_tz_limit=2500,
    )
    assert not v3_driver._rhs1_pas_dkes_xblock_allowed(
        has_pas=False,
        use_dkes=True,
        backend="gpu",
        n_theta=9,
        n_zeta=11,
        max_l=21,
        xblock_tz_limit=2500,
    )


def test_pas_tokamak_gpu_theta_allowed_only_for_small_er_tokamak_pas() -> None:
    assert v3_driver._rhs1_pas_tokamak_gpu_theta_allowed(
        has_pas=True,
        has_fp=False,
        backend="gpu",
        tokamak_like=True,
        active_size=245,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
    )


def test_pas_tokamak_gpu_xblock_preferred_only_for_small_tokamak_blocks(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_XBLOCK_ACTIVE_MAX", raising=False)
    assert not v3_driver._rhs1_pas_tokamak_gpu_xblock_preferred(
        has_pas=True,
        has_fp=False,
        backend="gpu",
        tokamak_like=True,
        active_size=245,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=1200,
    )
    assert v3_driver._rhs1_pas_tokamak_gpu_xblock_preferred(
        has_pas=True,
        has_fp=False,
        backend="gpu",
        tokamak_like=True,
        active_size=2650,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
        n_theta=15,
        n_zeta=1,
        max_l=31,
        xblock_tz_limit=1200,
    )
    assert not v3_driver._rhs1_pas_tokamak_gpu_xblock_preferred(
        has_pas=True,
        has_fp=False,
        backend="cpu",
        tokamak_like=True,
        active_size=245,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=1200,
    )
    assert not v3_driver._rhs1_pas_tokamak_gpu_xblock_preferred(
        has_pas=True,
        has_fp=False,
        backend="gpu",
        tokamak_like=True,
        active_size=15001,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=1200,
    )
    assert not v3_driver._rhs1_pas_tokamak_gpu_xblock_preferred(
        has_pas=True,
        has_fp=False,
        backend="gpu",
        tokamak_like=True,
        active_size=245,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
        n_theta=10,
        n_zeta=1,
        max_l=140,
        xblock_tz_limit=1200,
    )


def test_resource_exhausted_error_detection() -> None:
    assert v3_driver._is_resource_exhausted_error(RuntimeError("RESOURCE_EXHAUSTED: Out of memory"))
    assert v3_driver._is_resource_exhausted_error(RuntimeError("Allocator ran out of memory"))
    assert not v3_driver._is_resource_exhausted_error(RuntimeError("shape mismatch"))
    assert not v3_driver._rhs1_pas_tokamak_gpu_theta_allowed(
        has_pas=True,
        has_fp=False,
        backend="cpu",
        tokamak_like=True,
        active_size=245,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
    )
    assert not v3_driver._rhs1_pas_tokamak_gpu_theta_allowed(
        has_pas=True,
        has_fp=True,
        backend="gpu",
        tokamak_like=True,
        active_size=245,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
    )
    assert not v3_driver._rhs1_pas_tokamak_gpu_theta_allowed(
        has_pas=True,
        has_fp=False,
        backend="gpu",
        tokamak_like=True,
        active_size=12000,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
    )


def test_gpu_pas_tokamak_er_path_does_not_promote_to_pas_schur(tmp_path: Path, monkeypatch) -> None:
    input_path = (
        Path(__file__).parent
        / "reduced_inputs"
        / "tokamak_1species_PASCollisions_withEr_fullTrajectories.input.namelist"
    )
    out_path = tmp_path / "sfincsOutput_jax.h5"

    txt = input_path.read_text()
    txt = _patch_resolution_block(txt, ntheta=9, nzeta=1, nxi=11, nx=2, solver_tol=1e-6)
    txt = _patch_export_block(txt)
    patched = tmp_path / "input_tokamak_pas_gpu.namelist"
    patched.write_text(txt)

    logs: list[str] = []

    def emit(_level: int, msg: str) -> None:
        logs.append(msg)

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SOLVE_METHOD", "incremental")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ACTIVE_CUTOFF", "0")
    monkeypatch.setattr(v3_driver.jax, "default_backend", lambda: "gpu")

    write_sfincs_jax_output_h5(
        input_namelist=patched,
        output_path=out_path,
        compute_solution=True,
        emit=emit,
        verbose=True,
    )

    joined = "\n".join(logs)
    assert "GPU PAS tokamak auto -> tight unpreconditioned GMRES" in joined
    assert "GPU PAS tokamak tol tightened" in joined
    assert "building RHSMode=1 preconditioner=pas_tokamak_theta" not in joined
    assert "building RHSMode=1 preconditioner=xblock_tz" not in joined
    assert "building RHSMode=1 preconditioner=pas_schur" not in joined
    assert not v3_driver._rhs1_pas_tokamak_gpu_theta_allowed(
        has_pas=True,
        has_fp=False,
        backend="gpu",
        tokamak_like=True,
        active_size=245,
        er_abs=0.0,
        schur_er_min=1.0e-12,
        has_magdrift=False,
        has_collisionless=True,
    )


def test_cpu_pas_tokamak_er_path_prefers_xblock_tz(tmp_path: Path, monkeypatch) -> None:
    input_path = (
        Path(__file__).parent
        / "reduced_inputs"
        / "tokamak_1species_PASCollisions_withEr_fullTrajectories.input.namelist"
    )
    out_path = tmp_path / "sfincsOutput_jax.h5"

    txt = input_path.read_text()
    txt = _patch_resolution_block(txt, ntheta=9, nzeta=1, nxi=11, nx=2, solver_tol=1e-6)
    txt = _patch_export_block(txt)
    patched = tmp_path / "input_tokamak_pas_cpu.namelist"
    patched.write_text(txt)

    logs: list[str] = []

    def emit(_level: int, msg: str) -> None:
        logs.append(msg)

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SOLVE_METHOD", "incremental")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ACTIVE_CUTOFF", "0")
    monkeypatch.setattr(v3_driver.jax, "default_backend", lambda: "cpu")

    write_sfincs_jax_output_h5(
        input_namelist=patched,
        output_path=out_path,
        compute_solution=True,
        emit=emit,
        verbose=True,
    )

    joined = "\n".join(logs)
    assert "CPU PAS tokamak auto -> xblock_tz preconditioner" in joined
    assert "building RHSMode=1 preconditioner=xblock_tz" in joined
    assert "building RHSMode=1 preconditioner=pas_schur" not in joined


def test_pas_tokamak_cpu_xblock_allowed_only_for_bounded_cases() -> None:
    assert v3_driver._rhs1_pas_tokamak_cpu_xblock_preferred(
        has_pas=True,
        has_fp=False,
        backend="cpu",
        tokamak_like=True,
        active_size=245,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=True,
        has_collisionless=True,
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=6000,
    )
    assert not v3_driver._rhs1_pas_tokamak_cpu_xblock_preferred(
        has_pas=True,
        has_fp=False,
        backend="cpu",
        tokamak_like=True,
        active_size=6001,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=True,
        has_collisionless=True,
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=6000,
    )
    assert not v3_driver._rhs1_pas_tokamak_cpu_xblock_preferred(
        has_pas=True,
        has_fp=False,
        backend="gpu",
        tokamak_like=True,
        active_size=245,
        er_abs=1.0e-2,
        schur_er_min=1.0e-12,
        has_magdrift=True,
        has_collisionless=True,
        n_theta=10,
        n_zeta=1,
        max_l=14,
        xblock_tz_limit=6000,
    )


def test_gpu_sparse_fallback_skip_allowed_for_bounded_pas_schur_accept(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_GPU_SPARSE_SKIP_RATIO", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    op = SimpleNamespace(
        rhs_mode=1,
        include_phi1=False,
        fblock=SimpleNamespace(pas=object()),
    )
    assert v3_driver._rhs1_gpu_sparse_fallback_skip_allowed(
        op=op,
        rhs1_precond_kind="schur",
        use_active_dof_mode=True,
        residual_norm=4.0e-10,
        target=1.0e-10,
    )
    assert not v3_driver._rhs1_gpu_sparse_fallback_skip_allowed(
        op=op,
        rhs1_precond_kind="xblock_tz",
        use_active_dof_mode=True,
        residual_norm=4.0e-10,
        target=1.0e-10,
    )


def test_gpu_sparse_fallback_skip_rejects_cpu_or_large_residual(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_GPU_SPARSE_SKIP_RATIO", raising=False)
    op = SimpleNamespace(
        rhs_mode=1,
        include_phi1=False,
        fblock=SimpleNamespace(pas=object()),
    )
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not v3_driver._rhs1_gpu_sparse_fallback_skip_allowed(
        op=op,
        rhs1_precond_kind="schur",
        use_active_dof_mode=True,
        residual_norm=4.0e-10,
        target=1.0e-10,
    )
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not v3_driver._rhs1_gpu_sparse_fallback_skip_allowed(
        op=op,
        rhs1_precond_kind="schur",
        use_active_dof_mode=False,
        residual_norm=4.0e-10,
        target=1.0e-10,
    )
    assert not v3_driver._rhs1_gpu_sparse_fallback_skip_allowed(
        op=op,
        rhs1_precond_kind="schur",
        use_active_dof_mode=True,
        residual_norm=2.0e-9,
        target=1.0e-10,
    )


def test_sharded_line_override_preserves_dedicated_pas_preconditioners() -> None:
    assert v3_driver._rhs1_sharded_line_override_allowed("zeta_line")
    assert v3_driver._rhs1_sharded_line_override_allowed("pas_hybrid")
    assert not v3_driver._rhs1_sharded_line_override_allowed("pas_tz")
    assert not v3_driver._rhs1_sharded_line_override_allowed("pas_tokamak_theta")
    assert not v3_driver._rhs1_sharded_line_override_allowed("pas_ilu")
