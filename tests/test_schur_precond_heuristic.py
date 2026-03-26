from __future__ import annotations

from pathlib import Path
import re

from sfincs_jax.io import write_sfincs_jax_output_h5
import sfincs_jax.v3_driver as v3_driver


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
    assert v3_driver._rhs1_pas_tokamak_gpu_xblock_preferred(
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
    assert "GPU PAS tokamak auto -> xblock_tz preconditioner" in joined
    assert "building RHSMode=1 preconditioner=xblock_tz" in joined
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
