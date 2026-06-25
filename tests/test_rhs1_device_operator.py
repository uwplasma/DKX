from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import sfincs_jax.v3_driver as v3_driver_module
from sfincs_jax.explicit_sparse import build_operator_from_pattern
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.operators.profile_response.device_sparse import device_csr_from_matrix, validate_device_csr_matvec
from sfincs_jax.v3_sparse_pattern import v3_full_system_conservative_sparsity_pattern_for_indices
from sfincs_jax.v3_system import apply_v3_full_system_operator, full_system_operator_from_namelist


def _tiny_full_fp_namelist():
    nml = read_sfincs_input(Path(__file__).parent / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    resolution = nml.group("resolutionParameters")
    resolution["NTHETA"] = 5
    resolution["NZETA"] = 5
    resolution["NXI"] = 8
    resolution["NX"] = 1
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    return nml


def _active_tiny_full_fp_operator():
    nml = _tiny_full_fp_namelist()
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    active_idx = np.asarray(v3_driver_module._transport_active_dof_indices(op), dtype=np.int32)
    active_idx_jnp = jnp.asarray(active_idx, dtype=jnp.int32)

    def active_matvec(x):
        x_np = np.asarray(x, dtype=np.float64).reshape((-1,))
        full = np.zeros((int(op.total_size),), dtype=np.float64)
        full[active_idx] = x_np
        y_full = apply_v3_full_system_operator(op, jnp.asarray(full, dtype=jnp.float64))
        return np.asarray(jax.device_get(y_full[active_idx_jnp]), dtype=np.float64)

    return op, active_idx, active_matvec


def test_device_sparse_matvec_matches_host_and_matrix_free_on_tiny_active_system() -> None:
    op, active_idx, active_matvec = _active_tiny_full_fp_operator()
    pattern = v3_full_system_conservative_sparsity_pattern_for_indices(
        op,
        active_idx,
        fp_dense_velocity_block=False,
    )
    host_bundle = build_operator_from_pattern(
        active_matvec,
        pattern=pattern,
        dtype=np.float64,
        backend="cpu",
        csr_max_mb=8.0,
        max_colors=4096,
    )

    device_operator = device_csr_from_matrix(host_bundle.matrix, dtype=np.float64, max_nbytes=1_000_000)
    probe = np.linspace(-0.5, 0.75, int(active_idx.size), dtype=np.float64)

    matrix_free = active_matvec(probe)
    host = host_bundle.matvec(probe)
    device = np.asarray(jax.device_get(device_operator.jitted_matvec()(jnp.asarray(probe, dtype=jnp.float64))))

    assert device_operator.shape == (int(active_idx.size), int(active_idx.size))
    assert device_operator.nnz == host_bundle.matrix.nnz
    assert device_operator.nbytes_estimate < 1_000_000
    np.testing.assert_allclose(host, matrix_free, rtol=0.0, atol=3.0e-12)
    np.testing.assert_allclose(device, matrix_free, rtol=0.0, atol=3.0e-12)

    rel_errors = validate_device_csr_matvec(
        device_operator,
        host_bundle.matvec,
        probes=probe,
        samples=1,
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    assert max(rel_errors, default=0.0) < 1.0e-11


def test_xblock_side_probe_switch_keeps_physical_left_probe_seed_for_right_pc(monkeypatch) -> None:
    nml = _tiny_full_fp_namelist()
    physics = nml.group("physicsParameters")
    physics["includeXDotTerm"] = False
    physics["includeElectricFieldTermInXiDot"] = False

    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_SIDE_PROBE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_LGMRES_RESCUE", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_INITIAL_SEED", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_MINRES_STEPS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_COARSE", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE", "0")
    monkeypatch.delenv("SFINCS_JAX_GMRES_PRECONDITION_SIDE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", raising=False)

    monkeypatch.setattr(
        v3_driver_module,
        "_build_rhsmode1_xblock_tz_sparse_preconditioner",
        lambda **_kwargs: (lambda v: jnp.asarray(v, dtype=jnp.float64)),
    )
    monkeypatch.setattr(
        v3_driver_module._rhs1_xblock_policy,
        "rhs1_xblock_side_probe_should_switch",
        lambda *, residual_ratio, switch_ratio_env_value: True,
    )

    calls: list[dict[str, object]] = []
    left_probe_seed: np.ndarray | None = None

    def fake_gmres(*, b, x0, precondition_side, **_kwargs):
        nonlocal left_probe_seed
        calls.append(
            {
                "side": str(precondition_side),
                "x0": None if x0 is None else np.asarray(x0, dtype=np.float64).copy(),
            }
        )
        size = int(np.asarray(b).size)
        if len(calls) == 1:
            left_probe_seed = np.linspace(1.0, 2.0, size, dtype=np.float64)
            return left_probe_seed, 1.0e6, [1.0e6]
        return np.linspace(3.0, 4.0, size, dtype=np.float64), 0.0, [0.0]

    monkeypatch.setattr(v3_driver_module, "gmres_solve_with_history_scipy", fake_gmres)

    result = v3_driver_module.solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=4,
    )

    assert [call["side"] for call in calls] == ["left", "right"]
    assert calls[0]["x0"] is None
    assert left_probe_seed is not None
    np.testing.assert_allclose(calls[1]["x0"], left_probe_seed, rtol=0.0, atol=0.0)
    assert result.metadata["xblock_side_probe_initial_side"] == "left"
    assert result.metadata["xblock_side_probe_selected_side"] == "right"
    assert result.metadata["xblock_side_probe_switched"] is True
    assert result.metadata["xblock_side_probe_physical_seed_preserved_after_switch"] is True


def test_qi_device_krylov_nonconverged_rejection_evidence_is_documented() -> None:
    payload = json.loads(
        Path("docs/_static/qi_seed_robustness_scale060_device_krylov_rejected_2026_05_13.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["artifact_kind"] == "qi_seed_rejected_solver_probe_summary"
    assert payload["lane"] == "qi_seed_robustness"
    assert payload["active_size"] == 81377
    assert payload["conclusion"]["defaults_changed"] is False
    assert payload["conclusion"]["hard_seed_closed"] is False

    probes = payload["probes"]
    assert {probe["name"] for probe in probes} == {
        "gpu_device_fgmres_right_gc24",
        "gpu_device_gmresjax_left_gc24",
    }
    for probe in probes:
        assert probe["backend"] == "gpu"
        assert probe["process_returncode"] == 2
        assert probe["timed_out"] is False
        assert probe["accepted_converged"] is False
        assert probe["output_written"] is False
        assert probe["solver_trace_written"] is False
        assert probe["promotion_decision"] == "rejected_nonconverged_true_residual"
        assert any(
            "Refusing to write nonconverged RHSMode=1 diagnostics" in event
            for event in probe["observed_progress"]
        )
