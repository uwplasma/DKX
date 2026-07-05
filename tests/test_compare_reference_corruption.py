from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from sfincs_jax.compare import compare_h5_outputs, compare_sfincs_outputs


def _write_minimal_compare_h5(path: Path, *, gpsi: np.ndarray, dtheta: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f["geometryScheme"] = np.asarray(5, dtype=np.int32)
        f["RHSMode"] = np.asarray(3, dtype=np.int32)
        f["constraintScheme"] = np.asarray(2, dtype=np.int32)
        f["gpsiHatpsiHat"] = np.asarray(gpsi, dtype=np.float64)
        f["dBHat_sup_theta_dzeta"] = np.asarray(dtheta, dtype=np.float64)
        f["NTVBeforeSurfaceIntegral"] = np.asarray(gpsi, dtype=np.float64)


def _write_compare_case_h5(
    path: Path,
    *,
    rhs_mode: int,
    constraint_scheme: int,
    geometry_scheme: int = 11,
    collision_operator: int = 1,
    fields: dict[str, np.ndarray],
) -> None:
    with h5py.File(path, "w") as f:
        f["geometryScheme"] = np.asarray(geometry_scheme, dtype=np.int32)
        f["RHSMode"] = np.asarray(rhs_mode, dtype=np.int32)
        f["constraintScheme"] = np.asarray(constraint_scheme, dtype=np.int32)
        f["collisionOperator"] = np.asarray(collision_operator, dtype=np.int32)
        for key, value in fields.items():
            f[key] = np.asarray(value, dtype=np.float64)


def _write_rhs1_compare_h5(path: Path, *, density: np.ndarray, constraint_scheme: int) -> None:
    _write_compare_case_h5(
        path,
        rhs_mode=1,
        constraint_scheme=constraint_scheme,
        fields={"densityPerturbation": density},
    )


def _write_strict_h5(path: Path, datasets: dict[str, object]) -> None:
    with h5py.File(path, "w") as f:
        for key, value in datasets.items():
            if isinstance(value, str):
                f[key] = value
            else:
                f[key] = np.asarray(value)


def test_strict_h5_compare_reports_output_contract_failures(tmp_path: Path) -> None:
    reference = tmp_path / "reference.h5"
    candidate = tmp_path / "candidate.h5"
    _write_strict_h5(
        reference,
        {
            "same": np.asarray([1.0, 2.0]),
            "mismatch": np.asarray([1.0, 2.0]),
            "shape": np.asarray([[1.0, 2.0]]),
            "missing_in_candidate": np.asarray([5.0]),
            "nonnumeric": "reference metadata",
        },
    )
    _write_strict_h5(
        candidate,
        {
            "same": np.asarray([1.0, 2.0]),
            "mismatch": np.asarray([1.0, 2.2]),
            "shape": np.asarray([1.0, 2.0]),
            "extra_in_candidate": np.asarray([9.0]),
            "nonnumeric": "candidate metadata",
        },
    )

    payload = compare_h5_outputs(
        reference_path=reference,
        candidate_path=candidate,
        atol=1.0e-12,
        rtol=0.0,
    )

    assert payload["overall_status"] == "fail"
    assert payload["numeric_reference_dataset_count"] == 4
    assert payload["numeric_candidate_dataset_count"] == 4
    assert payload["compared_dataset_count"] == 2
    assert payload["failing_dataset_count"] == 3
    assert payload["status_counts"] == {
        "extra_in_candidate": 1,
        "missing_in_candidate": 1,
        "ok": 1,
        "shape_mismatch": 1,
        "value_mismatch": 1,
    }
    by_key = {row["key"]: row for row in payload["datasets"]}
    assert by_key["same"]["status"] == "ok"
    assert by_key["mismatch"]["status"] == "value_mismatch"
    assert by_key["mismatch"]["max_abs"] == pytest.approx(0.2)
    assert by_key["shape"]["status"] == "shape_mismatch"
    assert by_key["shape"]["reference_shape"] == [1, 2]
    assert by_key["shape"]["candidate_shape"] == [2]
    assert by_key["missing_in_candidate"]["status"] == "missing_in_candidate"
    assert by_key["extra_in_candidate"]["status"] == "extra_in_candidate"
    assert "nonnumeric" not in by_key


def test_strict_h5_compare_honors_key_selection_tolerances_and_ignore_list(tmp_path: Path) -> None:
    reference = tmp_path / "reference.h5"
    candidate = tmp_path / "candidate.h5"
    _write_strict_h5(
        reference,
        {
            "close": np.asarray([1.0, 2.0]),
            "ignored": np.asarray([0.0]),
        },
    )
    _write_strict_h5(
        candidate,
        {
            "close": np.asarray([1.0, 2.05]),
            "candidate_only": np.asarray([3.0]),
            "ignored": np.asarray([99.0]),
        },
    )

    payload = compare_h5_outputs(
        reference_path=reference,
        candidate_path=candidate,
        keys=["close", "candidate_only", "ignored"],
        ignore_keys=["ignored"],
        include_extra=False,
        atol=1.0e-12,
        rtol=0.0,
        tolerances={"close": {"atol": 0.1, "rtol": 0.0}},
    )

    assert payload["overall_status"] == "fail"
    assert payload["status_counts"] == {"missing_in_reference": 1, "ok": 1}
    by_key = {row["key"]: row for row in payload["datasets"]}
    assert by_key["close"]["status"] == "ok"
    assert by_key["close"]["atol"] == 0.1
    assert by_key["candidate_only"]["status"] == "missing_in_reference"
    assert "ignored" not in by_key


def test_compare_masks_vmec_reference_corruption_outliers(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran.h5"
    jax_path = tmp_path / "jax.h5"

    finite = np.asarray([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    corrupted_gpsi = finite.copy()
    corrupted_gpsi[0, 1] = 1.125899906842624e15
    corrupted_dtheta = finite.copy()
    corrupted_dtheta[1, 0] = np.nan

    _write_minimal_compare_h5(ref_path, gpsi=corrupted_gpsi, dtheta=corrupted_dtheta)
    _write_minimal_compare_h5(jax_path, gpsi=finite, dtheta=finite)

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["gpsiHatpsiHat", "dBHat_sup_theta_dzeta", "NTVBeforeSurfaceIntegral"],
        rtol=0.0,
        atol=1.0e-12,
    )

    assert all(result.ok for result in results), results


def test_compare_ignores_undefined_analytic_classical_flux_reference(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_analytic.h5"
    jax_path = tmp_path / "jax_analytic.h5"

    undefined_fields = {
        "gpsiHatpsiHat": np.asarray([[7.0]], dtype=np.float64),
        "classicalHeatFluxNoPhi1_psiHat": np.asarray([2.7e-6], dtype=np.float64),
        "classicalHeatFlux_psiHat": np.asarray([[2.7e-6]], dtype=np.float64),
        "FSABFlow": np.asarray([[1.0]], dtype=np.float64),
    }
    zeroed_fields = {
        "gpsiHatpsiHat": np.asarray([[0.0]], dtype=np.float64),
        "classicalHeatFluxNoPhi1_psiHat": np.asarray([0.0], dtype=np.float64),
        "classicalHeatFlux_psiHat": np.asarray([[0.0]], dtype=np.float64),
        "FSABFlow": np.asarray([[1.0 + 1.0e-3]], dtype=np.float64),
    }

    _write_compare_case_h5(
        ref_path,
        rhs_mode=1,
        constraint_scheme=2,
        geometry_scheme=1,
        fields=undefined_fields,
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=1,
        constraint_scheme=2,
        geometry_scheme=1,
        fields=zeroed_fields,
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=[
            "gpsiHatpsiHat",
            "classicalHeatFluxNoPhi1_psiHat",
            "classicalHeatFlux_psiHat",
            "FSABFlow",
        ],
        rtol=1.0e-6,
        atol=1.0e-12,
    )

    assert [result.key for result in results] == ["FSABFlow"]
    assert not results[0].ok


def test_compare_preserves_rhs1_model_floor_over_tighter_case_tolerance(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_rhs1.h5"
    jax_path = tmp_path / "jax_rhs1.h5"

    ref_density = np.asarray([[[[-1.1250897588471625e-03]]]], dtype=np.float64)
    jax_density = np.asarray([[[[-1.1237663486148910e-03]]]], dtype=np.float64)

    _write_rhs1_compare_h5(ref_path, density=ref_density, constraint_scheme=2)
    _write_rhs1_compare_h5(jax_path, density=jax_density, constraint_scheme=2)

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["densityPerturbation"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances={"densityPerturbation": {"atol": 1.0e-7}},
    )

    assert len(results) == 1
    assert results[0].ok, results


def test_compare_applies_rhs1_constraint0_center_fsa_and_flux_floors(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_cs0.h5"
    jax_path = tmp_path / "jax_cs0.h5"

    ref_density = np.asarray([[[[1.0, 2.0]]]], dtype=np.float64)
    jax_density = ref_density + 1.0e-3
    ref_heat = np.asarray([1.0e-8], dtype=np.float64)
    jax_heat = np.asarray([1.9e-6], dtype=np.float64)
    ref_aniso = np.asarray([0.0, -1.5e-6], dtype=np.float64)
    jax_aniso = np.asarray([0.0, -1.985e-4], dtype=np.float64)

    _write_compare_case_h5(
        ref_path,
        rhs_mode=1,
        constraint_scheme=0,
        fields={
            "densityPerturbation": ref_density,
            "heatFlux_vm_psiHat": ref_heat,
            "pressureAnisotropy": ref_aniso,
            "FSADensityPerturbation": np.asarray([1.0e-2], dtype=np.float64),
        },
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=1,
        constraint_scheme=0,
        fields={
            "densityPerturbation": jax_density,
            "heatFlux_vm_psiHat": jax_heat,
            "pressureAnisotropy": jax_aniso,
            "FSADensityPerturbation": np.asarray([0.0], dtype=np.float64),
        },
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["densityPerturbation", "heatFlux_vm_psiHat", "pressureAnisotropy", "FSADensityPerturbation"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    assert all(result.ok for result in results), results


def test_compare_applies_rhs1_constraint2_pressure_floors(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_cs2.h5"
    jax_path = tmp_path / "jax_cs2.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=1,
        constraint_scheme=2,
        fields={
            "FSAPressurePerturbation": np.asarray([1.32401224e-05], dtype=np.float64),
            "pressurePerturbation": np.asarray([355.43718933], dtype=np.float64),
            "heatFlux_vm_psiHat_vs_x": np.asarray([1.88506326e-05], dtype=np.float64),
        },
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=1,
        constraint_scheme=2,
        fields={
            "FSAPressurePerturbation": np.asarray([0.0], dtype=np.float64),
            "pressurePerturbation": np.asarray([355.42837155], dtype=np.float64),
            "heatFlux_vm_psiHat_vs_x": np.asarray([2.03554974e-05], dtype=np.float64),
        },
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["FSAPressurePerturbation", "pressurePerturbation", "heatFlux_vm_psiHat_vs_x"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    assert all(result.ok for result in results), results


def test_compare_applies_transport_momentum_flux_floor(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_transport.h5"
    jax_path = tmp_path / "jax_transport.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=3,
        constraint_scheme=2,
        geometry_scheme=1,
        fields={"momentumFlux_vm_psiHat": np.asarray([-3.94630491e-08], dtype=np.float64)},
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=3,
        constraint_scheme=2,
        geometry_scheme=1,
        fields={"momentumFlux_vm_psiHat": np.asarray([0.0], dtype=np.float64)},
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["momentumFlux_vm_psiHat"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    assert len(results) == 1
    assert results[0].ok, results


def test_compare_applies_rhsmode3_constraint2_sources_floor(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_transport_sources.h5"
    jax_path = tmp_path / "jax_transport_sources.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=3,
        constraint_scheme=2,
        geometry_scheme=1,
        fields={"sources": np.asarray([[[-8.11042861e-15, 1.85079290e-09]]], dtype=np.float64)},
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=3,
        constraint_scheme=2,
        geometry_scheme=1,
        fields={"sources": np.asarray([[[1.57713320e-19, -1.53522989e-14]]], dtype=np.float64)},
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["sources"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    assert len(results) == 1
    assert results[0].ok, results


def test_compare_applies_rhs1_fsabflow_vs_x_floor(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_rhs1_flow.h5"
    jax_path = tmp_path / "jax_rhs1_flow.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=1,
        constraint_scheme=1,
        fields={"FSABFlow_vs_x": np.asarray([-8.67663610e-08], dtype=np.float64)},
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=1,
        constraint_scheme=1,
        fields={"FSABFlow_vs_x": np.asarray([7.26881129e-08], dtype=np.float64)},
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["FSABFlow_vs_x"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    assert len(results) == 1
    assert results[0].ok, results


def test_compare_applies_rhs1_dkes_fp_fsa_pressure_floor(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_rhs1_dkes_fp.h5"
    jax_path = tmp_path / "jax_rhs1_dkes_fp.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=1,
        constraint_scheme=1,
        geometry_scheme=11,
        collision_operator=0,
        fields={
            "useDKESExBDrift": np.asarray(1, dtype=np.int32),
            "includeXDotTerm": np.asarray(0, dtype=np.int32),
            "includeElectricFieldTermInXiDot": np.asarray(0, dtype=np.int32),
            "FSAPressurePerturbation": np.asarray([-2.7048790208515307e-05], dtype=np.float64),
        },
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=1,
        constraint_scheme=1,
        geometry_scheme=11,
        collision_operator=0,
        fields={
            "useDKESExBDrift": np.asarray(1, dtype=np.int32),
            "includeXDotTerm": np.asarray(0, dtype=np.int32),
            "includeElectricFieldTermInXiDot": np.asarray(0, dtype=np.int32),
            "FSAPressurePerturbation": np.asarray([0.0], dtype=np.float64),
        },
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["FSAPressurePerturbation", "useDKESExBDrift", "includeXDotTerm", "includeElectricFieldTermInXiDot"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    assert all(result.ok for result in results), results


def test_compare_applies_rhs1_fulltraj_fp_fsa_pressure_floor(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_rhs1_fulltraj_fp.h5"
    jax_path = tmp_path / "jax_rhs1_fulltraj_fp.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=1,
        constraint_scheme=1,
        geometry_scheme=11,
        collision_operator=0,
        fields={
            "useDKESExBDrift": np.asarray(0, dtype=np.int32),
            "includeXDotTerm": np.asarray(1, dtype=np.int32),
            "includeElectricFieldTermInXiDot": np.asarray(1, dtype=np.int32),
            "FSAPressurePerturbation": np.asarray([-2.609445006998701e-04], dtype=np.float64),
        },
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=1,
        constraint_scheme=1,
        geometry_scheme=11,
        collision_operator=0,
        fields={
            "useDKESExBDrift": np.asarray(0, dtype=np.int32),
            "includeXDotTerm": np.asarray(1, dtype=np.int32),
            "includeElectricFieldTermInXiDot": np.asarray(1, dtype=np.int32),
            "FSAPressurePerturbation": np.asarray([0.0], dtype=np.float64),
        },
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["FSAPressurePerturbation", "useDKESExBDrift", "includeXDotTerm", "includeElectricFieldTermInXiDot"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    assert all(result.ok for result in results), results


def test_compare_applies_vmec_fulltraj_fp_total_density_floor(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_rhs1_vmec_fulltraj_fp.h5"
    jax_path = tmp_path / "jax_rhs1_vmec_fulltraj_fp.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=1,
        constraint_scheme=1,
        geometry_scheme=5,
        collision_operator=0,
        fields={
            "useDKESExBDrift": np.asarray(0, dtype=np.int32),
            "includeXDotTerm": np.asarray(1, dtype=np.int32),
            "includeElectricFieldTermInXiDot": np.asarray(1, dtype=np.int32),
            "totalDensity": np.asarray([1.0], dtype=np.float64),
            "particleFluxBeforeSurfaceIntegral_vm": np.asarray([1.0e-6], dtype=np.float64),
        },
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=1,
        constraint_scheme=1,
        geometry_scheme=5,
        collision_operator=0,
        fields={
            "useDKESExBDrift": np.asarray(0, dtype=np.int32),
            "includeXDotTerm": np.asarray(1, dtype=np.int32),
            "includeElectricFieldTermInXiDot": np.asarray(1, dtype=np.int32),
            "totalDensity": np.asarray([1.0003], dtype=np.float64),
            "particleFluxBeforeSurfaceIntegral_vm": np.asarray([1.0015e-6], dtype=np.float64),
        },
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=[
            "totalDensity",
            "particleFluxBeforeSurfaceIntegral_vm",
            "useDKESExBDrift",
            "includeXDotTerm",
            "includeElectricFieldTermInXiDot",
        ],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    assert all(result.ok for result in results), results


def test_compare_applies_rhs1_fulltraj_pas_heatflux_rtol(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_rhs1_fulltraj_pas.h5"
    jax_path = tmp_path / "jax_rhs1_fulltraj_pas.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=1,
        constraint_scheme=2,
        geometry_scheme=11,
        collision_operator=1,
        fields={
            "useDKESExBDrift": np.asarray(0, dtype=np.int32),
            "includeXDotTerm": np.asarray(1, dtype=np.int32),
            "includeElectricFieldTermInXiDot": np.asarray(1, dtype=np.int32),
            "heatFlux_vm_psiHat": np.asarray([[8.8089e-04], [1.5274e-04]], dtype=np.float64),
            "heatFlux_vm_psiN": np.asarray([[1.4041302e-01], [2.434723e-02]], dtype=np.float64),
            "heatFlux_vm_rHat": np.asarray([[3.757115e-02], [6.51473e-03]], dtype=np.float64),
            "heatFlux_vm_rN": np.asarray([[3.0922759e-01], [5.361920e-02]], dtype=np.float64),
        },
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=1,
        constraint_scheme=2,
        geometry_scheme=11,
        collision_operator=1,
        fields={
            "useDKESExBDrift": np.asarray(0, dtype=np.int32),
            "includeXDotTerm": np.asarray(1, dtype=np.int32),
            "includeElectricFieldTermInXiDot": np.asarray(1, dtype=np.int32),
            "heatFlux_vm_psiHat": np.asarray([[8.8089150667321e-04], [1.5289160356812458e-04]], dtype=np.float64),
            "heatFlux_vm_psiN": np.asarray([[1.40413260162031e-01], [2.4371395439945133e-02]], dtype=np.float64),
            "heatFlux_vm_rHat": np.asarray([[3.75712142615942e-02], [6.521196091610801e-03]], dtype=np.float64),
            "heatFlux_vm_rN": np.asarray([[3.0922811890201e-01], [5.367241886099425e-02]], dtype=np.float64),
        },
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=[
            "heatFlux_vm_psiHat",
            "heatFlux_vm_psiN",
            "heatFlux_vm_rHat",
            "heatFlux_vm_rN",
            "useDKESExBDrift",
            "includeXDotTerm",
            "includeElectricFieldTermInXiDot",
        ],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    assert all(result.ok for result in results), results


def test_compare_applies_rhs1_vmec_fp_constraint1_fsa_floors(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_rhs1_vmec_fp.h5"
    jax_path = tmp_path / "jax_rhs1_vmec_fp.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=1,
        constraint_scheme=1,
        geometry_scheme=5,
        collision_operator=0,
        fields={
            "useDKESExBDrift": np.asarray(0, dtype=np.int32),
            "FSADensityPerturbation": np.asarray([1.0e-8], dtype=np.float64),
            "FSAPressurePerturbation": np.asarray([0.0], dtype=np.float64),
        },
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=1,
        constraint_scheme=1,
        geometry_scheme=5,
        collision_operator=0,
        fields={
            "useDKESExBDrift": np.asarray(0, dtype=np.int32),
            "FSADensityPerturbation": np.asarray([1.01e-6], dtype=np.float64),
            "FSAPressurePerturbation": np.asarray([1.5e-3], dtype=np.float64),
        },
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["FSADensityPerturbation", "FSAPressurePerturbation"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    assert all(result.ok for result in results), results


def test_compare_applies_rhs1_constraint2_fsabflow_vs_x_floor(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_rhs1_flow_cs2.h5"
    jax_path = tmp_path / "jax_rhs1_flow_cs2.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=1,
        constraint_scheme=2,
        fields={"FSABFlow_vs_x": np.asarray([-8.67663610e-08], dtype=np.float64)},
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=1,
        constraint_scheme=2,
        fields={"FSABFlow_vs_x": np.asarray([7.26881129e-08], dtype=np.float64)},
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["FSABFlow_vs_x"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    assert len(results) == 1
    assert results[0].ok, results


def test_compare_applies_rhs1_constraint2_local_jhat_floor_but_gates_fsab_current(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_rhs1_jhat_cs2.h5"
    jax_path = tmp_path / "jax_rhs1_jhat_cs2.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=1,
        constraint_scheme=2,
        fields={
            "jHat": np.asarray([[-1.0e-9, 2.0e-3]], dtype=np.float64),
            "FSABjHat": np.asarray([4.0e-4], dtype=np.float64),
        },
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=1,
        constraint_scheme=2,
        fields={
            "jHat": np.asarray([[5.0e-8, 2.0e-3 + 5.0e-8]], dtype=np.float64),
            "FSABjHat": np.asarray([4.0e-4 + 1.0e-4], dtype=np.float64),
        },
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["jHat", "FSABjHat"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    by_key = {result.key: result for result in results}
    assert by_key["jHat"].ok, results
    assert not by_key["FSABjHat"].ok, results


def test_compare_skips_missing_and_nonnumeric_datasets(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_mixed.h5"
    jax_path = tmp_path / "jax_mixed.h5"

    with h5py.File(ref_path, "w") as f:
        f["geometryScheme"] = np.asarray(11, dtype=np.int32)
        f["RHSMode"] = np.asarray(3, dtype=np.int32)
        f["constraintScheme"] = np.asarray(1, dtype=np.int32)
        f["shared_numeric"] = np.asarray([1.0, 2.0], dtype=np.float64)
        f["missing_from_jax"] = np.asarray([3.0], dtype=np.float64)
        f["metadata_text"] = np.bytes_("fortran-reference")
    with h5py.File(jax_path, "w") as f:
        f["geometryScheme"] = np.asarray(11, dtype=np.int32)
        f["RHSMode"] = np.asarray(3, dtype=np.int32)
        f["constraintScheme"] = np.asarray(1, dtype=np.int32)
        f["shared_numeric"] = np.asarray([1.0, 2.0 + 5.0e-13], dtype=np.float64)
        f["extra_from_jax"] = np.asarray([4.0], dtype=np.float64)
        f["metadata_text"] = np.bytes_("jax-output")

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["shared_numeric", "missing_from_jax", "extra_from_jax", "metadata_text"],
        rtol=0.0,
        atol=1.0e-12,
    )

    assert [result.key for result in results] == ["shared_numeric"]
    assert results[0].ok


def test_compare_reports_shape_mismatch_as_failed_infinite_error(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_shape.h5"
    jax_path = tmp_path / "jax_shape.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=3,
        constraint_scheme=1,
        fields={"transportCoefficient": np.asarray([1.0, 2.0, 3.0], dtype=np.float64)},
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=3,
        constraint_scheme=1,
        fields={"transportCoefficient": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)},
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["transportCoefficient"],
        rtol=0.0,
        atol=1.0e-12,
    )

    assert len(results) == 1
    assert results[0].key == "transportCoefficient"
    assert not results[0].ok
    assert np.isinf(results[0].max_abs)
    assert np.isinf(results[0].max_rel)


def test_compare_phi1_uses_converged_final_iterate_and_ignores_iteration_metadata(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_phi1.h5"
    jax_path = tmp_path / "jax_phi1.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=1,
        constraint_scheme=1,
        fields={
            "includePhi1": np.asarray(1, dtype=np.int32),
            "NIterations": np.asarray(3, dtype=np.int32),
            "Phi1Hat": np.asarray([100.0, -25.0, 2.5], dtype=np.float64),
        },
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=1,
        constraint_scheme=1,
        fields={
            "includePhi1": np.asarray(1, dtype=np.int32),
            "NIterations": np.asarray(2, dtype=np.int32),
            "Phi1Hat": np.asarray([-1.0, 2.5 + 5.0e-13], dtype=np.float64),
        },
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["NIterations", "Phi1Hat"],
        rtol=0.0,
        atol=1.0e-12,
    )

    assert [result.key for result in results] == ["Phi1Hat"]
    assert results[0].ok, results


def test_compare_vmec_uhat_is_normalized_to_zero_for_v3_parity(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_vmec_uhat.h5"
    jax_path = tmp_path / "jax_vmec_uhat.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=2,
        constraint_scheme=1,
        geometry_scheme=5,
        fields={"uHat": np.asarray([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64)},
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=2,
        constraint_scheme=1,
        geometry_scheme=5,
        fields={"uHat": np.zeros((2, 2), dtype=np.float64)},
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["uHat"],
        rtol=0.0,
        atol=1.0e-12,
    )

    assert len(results) == 1
    assert results[0].ok, results
