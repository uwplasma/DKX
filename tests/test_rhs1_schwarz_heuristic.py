from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.namelist import read_sfincs_input
import sfincs_jax.v3_driver as vd


def test_rhs1_auto_prefers_theta_schwarz_when_sharded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto RHSMode=1 preconditioner should pick Schwarz on sharded large-system path."""
    input_path = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    assert input_path.exists()
    nml = read_sfincs_input(input_path)

    logs: list[str] = []

    def emit(level: int, msg: str) -> None:
        logs.append(msg)

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "theta")
    monkeypatch.setenv("SFINCS_JAX_AUTO_SHARD", "0")
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHWARZ_AUTO_MIN", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_MIN", "1000000000")
    monkeypatch.setattr(vd.jax, "device_count", lambda: 2)

    res = vd.solve_v3_full_system_linear_gmres(nml=nml, tol=1e-8, emit=emit)
    assert np.isfinite(float(res.residual_norm))
    assert any("building RHSMode=1 preconditioner=theta_schwarz" in msg for msg in logs)


def test_pas_tz_memory_estimate_flags_large_case() -> None:
    class _Collisionless:
        n_xi_for_x = np.full((20,), 14, dtype=np.int32)

    class _FBlock:
        collisionless = _Collisionless()
        pas = object()
        fp = None
        fp_phi1 = None
        exb_theta = None
        exb_zeta = None
        magdrift_theta = None
        magdrift_zeta = None
        magdrift_xidot = None
        er_xdot = None
        er_xidot = None

    class _Op:
        rhs_mode = 1
        n_species = 1
        n_x = 20
        n_xi = 14
        n_theta = 127
        n_zeta = 127
        fblock = _FBlock()

    estimate = vd._estimate_rhs1_pas_tz_build_bytes(_Op())
    assert estimate > 100 * 2**30
    assert not vd._pas_tz_preconditioner_memory_safe(_Op())


def test_pas_tz_builder_falls_back_to_theta_schwarz_when_memory_unsafe(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Collisionless:
        n_xi_for_x = np.full((20,), 14, dtype=np.int32)

    class _FBlock:
        collisionless = _Collisionless()
        pas = object()
        fp = None
        fp_phi1 = None
        exb_theta = None
        exb_zeta = None
        magdrift_theta = None
        magdrift_zeta = None
        magdrift_xidot = None
        er_xdot = None
        er_xidot = None

    class _Op:
        rhs_mode = 1
        n_species = 1
        n_x = 20
        n_xi = 14
        n_theta = 127
        n_zeta = 127
        fblock = _FBlock()

    sentinel = object()
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "theta")
    monkeypatch.setenv("SFINCS_JAX_AUTO_SHARD", "0")
    monkeypatch.setattr(vd.jax, "device_count", lambda: 2)
    monkeypatch.setattr(vd, "_estimate_rhs1_pas_tz_build_bytes", lambda _op: 10 * 2**30)
    monkeypatch.setattr(vd, "_rhs1_pas_tz_max_bytes", lambda: 2 * 2**30)
    monkeypatch.setattr(vd, "_build_rhsmode1_theta_schwarz_preconditioner", lambda **_kwargs: sentinel)

    assert vd._build_rhsmode1_pas_tz_preconditioner(op=_Op()) is sentinel
