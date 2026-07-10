from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

import sfincs_jax.outputs.transport as transport
from sfincs_jax.namelist import Namelist


def _fake_op(*, constraint_scheme: int = 2) -> SimpleNamespace:
    theta_weights = np.asarray([0.4, 0.6], dtype=np.float64)
    zeta_weights = np.asarray([0.25, 0.75], dtype=np.float64)
    b_hat = np.asarray([[1.0, 1.2], [0.9, 1.1]], dtype=np.float64)
    d_hat = np.asarray([[2.0, 2.2], [2.1, 2.3]], dtype=np.float64)
    return SimpleNamespace(
        rhs_mode=3,
        constraint_scheme=constraint_scheme,
        n_species=2,
        n_theta=2,
        n_zeta=2,
        n_x=2,
        n_xi=3,
        f_size=0,
        phi1_size=0,
        theta_weights=theta_weights,
        zeta_weights=zeta_weights,
        b_hat=jnp.asarray(b_hat),
        b_hat_sub_zeta=jnp.zeros_like(jnp.asarray(b_hat)),
        b_hat_sub_theta=jnp.zeros_like(jnp.asarray(b_hat)),
        db_hat_dtheta=jnp.zeros_like(jnp.asarray(b_hat)),
        db_hat_dzeta=jnp.zeros_like(jnp.asarray(b_hat)),
        d_hat=jnp.asarray(d_hat),
        fsab_hat2=jnp.asarray(1.44),
        x=jnp.asarray([0.5, 1.5]),
        x_weights=jnp.asarray([0.7, 0.3]),
        z_s=jnp.asarray([1.0, -1.0]),
        n_hat=jnp.asarray([2.0, 4.0]),
        t_hat=jnp.asarray([3.0, 5.0]),
        m_hat=jnp.asarray([2.0, 6.0]),
        dn_hat_dpsi_hat=jnp.asarray([0.1, 0.2]),
        dt_hat_dpsi_hat=jnp.asarray([0.3, 0.4]),
    )


def _state_for_constraint(op: SimpleNamespace, *, which_rhs: int) -> jnp.ndarray:
    if int(op.constraint_scheme) == 2:
        extra = np.asarray(
            [
                10.0 + which_rhs,
                20.0 + which_rhs,
                30.0 + which_rhs,
                40.0 + which_rhs,
            ],
            dtype=np.float64,
        )
    else:
        extra = np.asarray(
            [
                1.0 + which_rhs,
                2.0 + which_rhs,
                3.0 + which_rhs,
                4.0 + which_rhs,
            ],
            dtype=np.float64,
        )
    return jnp.asarray(extra)


def _diagnostic_for_rhs(which_rhs: int) -> SimpleNamespace:
    s, t, z, x = 2, 2, 2, 2
    base = float(which_rhs)
    stz = np.arange(s * t * z, dtype=np.float64).reshape(s, t, z) + base
    return SimpleNamespace(
        vprime_hat=jnp.asarray(1.0),
        particle_flux_vm_psi_hat=jnp.asarray([base, base + 1.0]),
        heat_flux_vm_psi_hat=jnp.asarray([2.0 * base, 2.0 * base + 1.0]),
        fsab_flow=jnp.asarray([3.0 * base, 3.0 * base + 1.0]),
        particle_flux_before_surface_integral_vm=jnp.asarray(stz),
        heat_flux_before_surface_integral_vm=jnp.asarray(stz + 10.0),
        particle_flux_before_surface_integral_vm0=jnp.asarray(stz + 20.0),
        heat_flux_before_surface_integral_vm0=jnp.asarray(stz + 30.0),
        particle_flux_vm_psi_hat_vs_x=jnp.asarray(np.full((x, s), base)),
        heat_flux_vm_psi_hat_vs_x=jnp.asarray(np.full((x, s), base + 2.0)),
        fsab_flow_vs_x=jnp.asarray(np.full((x, s), base + 4.0)),
    )


def _field_dict_for_rhs(which_rhs: int) -> dict[str, jnp.ndarray]:
    s, t, z = 2, 2, 2
    base = float(which_rhs)
    stz = np.arange(s * t * z, dtype=np.float64).reshape(s, t, z) + base
    return {
        "densityPerturbation": jnp.asarray(stz),
        "pressurePerturbation": jnp.asarray(stz + 1.0),
        "pressureAnisotropy": jnp.asarray(stz + 2.0),
        "flow": jnp.asarray(stz + 3.0),
        "totalDensity": jnp.asarray(stz + 4.0),
        "totalPressure": jnp.asarray(stz + 5.0),
        "velocityUsingFSADensity": jnp.asarray(stz + 6.0),
        "velocityUsingTotalDensity": jnp.asarray(stz + 7.0),
        "MachUsingFSAThermalSpeed": jnp.asarray(stz + 8.0),
        "jHat": jnp.asarray(np.arange(t * z, dtype=np.float64).reshape(t, z) + base),
        "FSADensityPerturbation": jnp.asarray([base, base + 1.0]),
        "FSAPressurePerturbation": jnp.asarray([base + 2.0, base + 3.0]),
        "momentumFluxBeforeSurfaceIntegral_vm": jnp.asarray(stz + 9.0),
        "momentumFluxBeforeSurfaceIntegral_vm0": jnp.asarray(stz + 10.0),
        "momentumFluxBeforeSurfaceIntegral_vE": jnp.asarray(stz + 11.0),
        "momentumFluxBeforeSurfaceIntegral_vE0": jnp.asarray(stz + 12.0),
        "momentumFlux_vm_psiHat": jnp.asarray([base + 4.0, base + 5.0]),
        "momentumFlux_vm0_psiHat": jnp.asarray([base + 6.0, base + 7.0]),
    }


@pytest.fixture
def patched_transport_kernels(monkeypatch: pytest.MonkeyPatch):
    def with_rhs(op: SimpleNamespace, *, which_rhs: int) -> SimpleNamespace:
        op.which_rhs = int(which_rhs)
        op.dn_hat_dpsi_hat = jnp.asarray([0.1, 0.2]) * float(which_rhs)
        op.dt_hat_dpsi_hat = jnp.asarray([0.3, 0.4]) * float(which_rhs)
        return op

    monkeypatch.setattr(transport, "with_transport_rhs_settings", with_rhs)
    monkeypatch.setattr(
        transport,
        "v3_transport_diagnostics_vm_only",
        lambda op, x_full: _diagnostic_for_rhs(int(op.which_rhs)),
    )
    monkeypatch.setattr(
        transport,
        "v3_rhsmode1_output_fields_vm_only_jit",
        lambda op, x_full: _field_dict_for_rhs(int(op.which_rhs)),
    )

    from sfincs_jax.physics import classical_transport

    monkeypatch.setattr(
        classical_transport,
        "classical_flux_v3",
        lambda **kwargs: (
            jnp.asarray([0.01, 0.02]) * float(np.asarray(kwargs["dn_hat_dpsi_hat"])[0] / 0.1),
            jnp.asarray([0.03, 0.04]) * float(np.asarray(kwargs["dt_hat_dpsi_hat"])[0] / 0.3),
        ),
    )


def test_transport_solver_diagnostic_arrays_reports_missing_rhs_as_nan() -> None:
    result = SimpleNamespace(residual_norms_by_rhs={1: 2.0}, rhs_norms_by_rhs={1: 4.0, 2: 0.0})

    arrays = transport.transport_solver_diagnostic_arrays(result, n_rhs=3)

    np.testing.assert_allclose(arrays["transportResidualNorms"][:1], [2.0])
    assert np.isnan(arrays["transportResidualNorms"][1])
    np.testing.assert_allclose(arrays["transportRelativeResidualNorms"][:1], [0.5])
    assert np.isnan(arrays["transportRelativeResidualNorms"][1])
    assert arrays["transportMaxResidualNorm"] == pytest.approx(2.0)
    assert arrays["transportMaxRelativeResidualNorm"] == pytest.approx(0.5)


def test_transport_coordinate_conversion_matches_radial_chain_rule() -> None:
    factors = transport.conversion_factors_to_from_dpsi_hat(psi_a_hat=2.0, a_hat=4.0, r_n=0.5)

    assert factors["ddpsiN2ddpsiHat"] == pytest.approx(0.5)
    assert factors["ddrHat2ddpsiHat"] == pytest.approx(2.0)
    assert factors["ddrN2ddpsiHat"] == pytest.approx(0.5)
    assert factors["ddpsiHat2ddpsiN"] == pytest.approx(2.0)
    assert factors["ddpsiHat2ddrHat"] == pytest.approx(0.5)
    assert factors["ddpsiHat2ddrN"] == pytest.approx(2.0)


def test_streaming_accumulator_collects_full_output_fields(patched_transport_kernels) -> None:
    op = _fake_op()
    nml = Namelist(
        groups={"geometryparameters": {"geometryScheme": 5}},
        indexed={},
        source_text="&geometryParameters\n geometryScheme = 5\n/\n",
    )
    accumulator = transport.TransportStreamingOutputAccumulator.create(
        nml=nml,
        grids=SimpleNamespace(),
        geom=SimpleNamespace(),
        op0=op,
        n_rhs=2,
        collect_full_output_fields=True,
    )

    accumulator.collect(1, _state_for_constraint(op, which_rhs=1))
    accumulator.collect(2, _state_for_constraint(op, which_rhs=2))
    particle_flux, heat_flux, flow = accumulator.diagnostic_flux_arrays()
    fields = accumulator.output_fields()

    np.testing.assert_allclose(np.asarray(particle_flux), [[1.0, 2.0], [2.0, 3.0]])
    np.testing.assert_allclose(np.asarray(heat_flux), [[2.0, 4.0], [3.0, 5.0]])
    np.testing.assert_allclose(np.asarray(flow), [[3.0, 6.0], [4.0, 7.0]])
    assert fields is not None
    assert fields["densityPerturbation"].shape == (2, 2, 2, 2)
    assert fields["sources"].shape == (2, 2, 2)
    np.testing.assert_allclose(fields["sources"][:, :, 0], [[11.0, 31.0], [21.0, 41.0]])
    np.testing.assert_allclose(fields["FSABjHat"], [-1.0, -1.0])
    np.testing.assert_allclose(fields["FSABVelocityUsingFSADensity"], [[1.5, 3.0], [1.0, 1.75]])


def test_streaming_accumulator_can_skip_full_fields(patched_transport_kernels) -> None:
    op = _fake_op()
    nml = Namelist(groups={"geometryparameters": {"geometryScheme": 5}}, indexed={})
    accumulator = transport.TransportStreamingOutputAccumulator.create(
        nml=nml,
        grids=SimpleNamespace(),
        geom=SimpleNamespace(),
        op0=op,
        n_rhs=1,
        collect_full_output_fields=False,
    )

    accumulator.collect(1, _state_for_constraint(op, which_rhs=1))

    assert accumulator.output_fields() is None
    np.testing.assert_allclose(np.asarray(accumulator.diagnostic_flux_arrays()[0]), [[1.0], [2.0]])


def test_write_transport_h5_streaming_writes_schema_and_solver_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patched_transport_kernels,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_WRITE_SOLVER_DIAGNOSTICS", "1")
    op = _fake_op(constraint_scheme=1)
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n RHSMode = 3\n/\n", encoding="utf-8")
    output_path = tmp_path / "sfincsOutput.h5"
    data = {
        "constraintScheme": np.asarray(1),
        "geometryScheme": np.asarray(5),
        "BHat": np.asarray(op.b_hat),
        "gpsiHatpsiHat": np.ones((2, 2), dtype=np.float64),
        "VPrimeHat": np.asarray(1.0),
        "alpha": np.asarray(1.0),
        "Delta": np.asarray(0.01),
        "nu_n": np.asarray(0.02),
        "Zs": np.asarray(op.z_s),
        "mHats": np.asarray(op.m_hat),
        "THats": np.asarray(op.t_hat),
        "nHats": np.asarray(op.n_hat),
        "psiAHat": np.asarray(2.0),
        "aHat": np.asarray(4.0),
        "rN": np.asarray(0.5),
        "elapsed time (s)": np.asarray(999.0),
        "transportMatrix": np.zeros((2, 2), dtype=np.float64),
    }
    result = SimpleNamespace(
        op0=op,
        state_vectors_by_rhs={
            1: _state_for_constraint(op, which_rhs=1),
            2: _state_for_constraint(op, which_rhs=2),
        },
        transport_matrix=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        elapsed_time_s=1.25,
        residual_norms_by_rhs={1: 1.0e-9, 2: 2.0e-9},
        rhs_norms_by_rhs={1: 1.0, 2: 2.0},
    )
    nml = Namelist(groups={}, indexed={}, source_text="&general\n RHSMode = 3\n/\n")

    written = transport.write_transport_h5_streaming(
        output_path=output_path,
        data=data,
        input_namelist=input_path,
        result=result,
        nml=nml,
        fortran_layout=True,
        overwrite=False,
    )

    assert written == output_path.resolve()
    with h5py.File(output_path, "r") as h5:
        assert h5["densityPerturbation"].shape == (2, 2, 2, 2)
        assert h5["sources"].shape == (2, 2, 2)
        np.testing.assert_allclose(h5["transportMatrix"][...], [[1.0, 3.0], [2.0, 4.0]])
        np.testing.assert_allclose(h5["FSABjHat"][...], [-1.0, -1.0])
        np.testing.assert_allclose(h5["particleFlux_vm_rN"][...], [[0.5, 1.0], [1.0, 1.5]])
        np.testing.assert_allclose(h5["transportRelativeResidualNorms"][...], [1.0e-9, 1.0e-9])
        assert h5["input.namelist"][()].decode() == nml.source_text

    with pytest.raises(FileExistsError):
        transport.write_transport_h5_streaming(
            output_path=output_path,
            data=data,
            input_namelist=input_path,
            result=result,
            nml=nml,
            fortran_layout=True,
            overwrite=False,
        )


def test_write_transport_h5_streaming_requires_all_rhs(tmp_path: Path, patched_transport_kernels) -> None:
    op = _fake_op()
    result = SimpleNamespace(op0=op, state_vectors_by_rhs={1: _state_for_constraint(op, which_rhs=1)})
    nml = Namelist(groups={}, indexed={}, source_text="&general\n/\n")

    with pytest.raises(ValueError, match="state vectors for every whichRHS"):
        transport.write_transport_h5_streaming(
            output_path=tmp_path / "sfincsOutput.h5",
            data={"constraintScheme": np.asarray(2)},
            input_namelist=tmp_path / "input.namelist",
            result=result,
            nml=nml,
            fortran_layout=True,
            overwrite=False,
        )
