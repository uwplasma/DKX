from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pytest

from dkx.ambipolar import radial_current_from_output, solve_ambipolar_from_scan_dir
from dkx.io import read_sfincs_h5, write_sfincs_h5
from dkx.workflows.scans import run_er_scan


def _write_synthetic_scan_point(
    scan_dir: Path,
    *,
    er: float,
    current_vm: float,
    current_vd: float,
    include_phi1: bool,
) -> Path:
    """Write one deterministic two-species Er-scan output.

    The species fluxes are chosen so ``sum_s Z_s Gamma_s`` equals either
    ``current_vm`` or ``current_vd``.  The unused flux channel is intentionally
    allowed to disagree so the ambipolar postprocessor must choose the correct
    SFINCS convention for ``includePhi1``.
    """
    run_dir = scan_dir / f"Er{er:+.3f}".replace("+", "p").replace("-", "m")
    run_dir.mkdir(parents=True, exist_ok=True)

    def _fluxes_for_current(current: float) -> np.ndarray:
        ion_flux = 0.125 + 0.02 * er
        electron_flux = ion_flux - current
        # Include a nonfinal column to assert the SFINCS last-iteration convention.
        return np.asarray(
            [
                [99.0, ion_flux],
                [-99.0, electron_flux],
            ],
            dtype=np.float64,
        )

    write_sfincs_h5(
        path=run_dir / "sfincsOutput.h5",
        data={
            "RHSMode": np.asarray(1, dtype=np.int32),
            "Er": np.asarray(er, dtype=np.float64),
            "Nspecies": np.asarray(2, dtype=np.int32),
            "includePhi1": np.asarray(1 if include_phi1 else 0, dtype=np.int32),
            "Zs": np.asarray([1.0, -1.0], dtype=np.float64),
            "FSABFlow": np.asarray([[0.01, 0.02 + er], [0.03, -0.01 + er]], dtype=np.float64),
            "particleFlux_vm_rHat": _fluxes_for_current(current_vm),
            "particleFlux_vd_rHat": _fluxes_for_current(current_vd),
            "heatFlux_vm_rHat": np.asarray([[1.0, 0.2 + er], [2.0, 0.4 - er]], dtype=np.float64),
            "heatFlux_vd_rHat": np.asarray([[3.0, 0.5 + er], [4.0, 0.6 - er]], dtype=np.float64),
            "FSABjHat": np.asarray([0.0, 0.25 * er], dtype=np.float64),
        },
        fortran_layout=False,
        overwrite=True,
    )
    return run_dir / "sfincsOutput.h5"


def _write_synthetic_scan(
    scan_dir: Path,
    *,
    er_values: list[float],
    current_vm: list[float],
    current_vd: list[float],
    include_phi1: bool,
) -> None:
    scan_dir.mkdir(parents=True, exist_ok=True)
    scan_dir.joinpath("input.namelist").write_text(
        "!ss ErMin = -2\n!ss ErMax = 1\n",
        encoding="utf-8",
    )
    for er, j_vm, j_vd in zip(er_values, current_vm, current_vd, strict=True):
        _write_synthetic_scan_point(
            scan_dir,
            er=er,
            current_vm=j_vm,
            current_vd=j_vd,
            include_phi1=include_phi1,
        )


def _write_named_scan_point(
    scan_dir: Path,
    *,
    var_name: str,
    var_value: float,
    er: float,
    radial_current: float,
    rhs_mode: int = 1,
) -> Path:
    """Write a minimal scan point for parser-only ambipolar tests."""
    run_dir = scan_dir / f"{var_name}{var_value:+.3f}".replace("+", "p").replace("-", "m")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_sfincs_h5(
        path=run_dir / "sfincsOutput.h5",
        data={
            "RHSMode": np.asarray(rhs_mode, dtype=np.int32),
            var_name: np.asarray(var_value, dtype=np.float64),
            "Er": np.asarray(er, dtype=np.float64),
            "rN": np.asarray(0.5, dtype=np.float64),
            "Nspecies": np.asarray(1, dtype=np.int32),
            "includePhi1": np.asarray(0, dtype=np.int32),
            "Zs": np.asarray([1.0], dtype=np.float64),
            "FSABFlow": np.asarray([[100.0, 0.25 + var_value]], dtype=np.float64),
            "particleFlux_vm_rHat": np.asarray([[100.0, radial_current]], dtype=np.float64),
            "heatFlux_vm_rHat": np.asarray([[100.0, 0.5 + var_value]], dtype=np.float64),
            "sources": np.asarray([[100.0, 0.01], [200.0, 0.02]], dtype=np.float64),
            "FSABjHat": np.asarray([100.0, -0.75], dtype=np.float64),
            "nHats": np.asarray([1.0], dtype=np.float64),
            "THats": np.asarray([1.0], dtype=np.float64),
        },
        fortran_layout=False,
        overwrite=True,
    )
    return run_dir / "sfincsOutput.h5"


def test_er_scan_writes_outputs_and_ambipolar_solve_runs(tmp_path: Path) -> None:
    """End-to-end regression test for `scan-er` + `ambipolar-solve` workflow.

    This test is intentionally small but exercises:
      - scan directory layout + `!ss` metadata
      - per-run `sfincsOutput.h5` creation with RHSMode=1 solution-derived fields
      - ambipolar root postprocessing compatible with upstream `sfincsScanPlot_5`
    """
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme11.input.namelist"
    scan_dir = tmp_path / "scan"

    values = [1.0e-3, 0.0, -1.0e-3]
    scan = run_er_scan(
        input_namelist=input_path,
        out_dir=scan_dir,
        values=values,
        compute_solution=True,
        compute_transport_matrix=False,
    )

    assert scan.scan_dir.exists()
    assert (scan.scan_dir / "input.namelist").exists()
    assert scan.variable == "Er"
    assert len(scan.run_dirs) == len(values)
    assert len(scan.outputs) == len(values)

    # Ensure outputs exist and contain the patched Er value.
    for v, out_h5 in zip(scan.values, scan.outputs, strict=True):
        assert out_h5.exists()
        assert out_h5.with_name("sfincsOutput.solver_trace.json").exists()
        d = read_sfincs_h5(out_h5)
        np.testing.assert_allclose(float(np.asarray(d["Er"]).reshape(())), float(v), rtol=0.0, atol=0.0)
        assert "particleFlux_vm_rHat" in d
        assert "Zs" in d
        z_s = np.asarray(d["Zs"], dtype=np.float64).reshape((-1,))
        gamma = np.asarray(d["particleFlux_vm_rHat"], dtype=np.float64)
        gamma_last = gamma[:, -1] if gamma.ndim == 2 else gamma.reshape((-1,))
        np.testing.assert_allclose(radial_current_from_output(d), float(np.sum(z_s * gamma_last)), atol=5e-12)

    res = solve_ambipolar_from_scan_dir(scan_dir=scan_dir, write_pickle=True, write_json=True, n_fine=200)

    pkl_path = scan_dir / "ambipolarSolutions.dat"
    json_path = scan_dir / "ambipolarSolutions.json"
    assert pkl_path.exists()
    assert json_path.exists()

    payload = pickle.loads(pkl_path.read_bytes())
    assert "roots" in payload
    assert "ylabels" in payload
    assert payload["numQuantities"] == len(payload["ylabels"])

    # Roots may or may not exist depending on this tiny fixture; ensure consistency if they do.
    if res.roots_var.size:
        np.testing.assert_allclose(np.asarray(payload["roots"], dtype=np.float64), np.asarray(res.roots_var, dtype=np.float64))


def test_ambipolar_radial_current_closure_uses_symmetric_flux_convention(tmp_path: Path) -> None:
    """Radial-current closure follows the SFINCS scan convention exactly."""
    er_values = [-1.0, 0.0, 1.0]
    vm_currents = [-2.0, -0.5, 1.0]
    vd_currents = [3.0, 4.0, 5.0]

    no_phi1_dir = tmp_path / "no_phi1"
    _write_synthetic_scan(
        no_phi1_dir,
        er_values=er_values,
        current_vm=vm_currents,
        current_vd=vd_currents,
        include_phi1=False,
    )
    no_phi1 = solve_ambipolar_from_scan_dir(scan_dir=no_phi1_dir, write_pickle=False, write_json=False)
    np.testing.assert_allclose(no_phi1.radial_currents, vm_currents, rtol=0.0, atol=5e-12)
    for er, expected in zip(er_values, vm_currents, strict=True):
        data = read_sfincs_h5(no_phi1_dir / f"Er{er:+.3f}".replace("+", "p").replace("-", "m") / "sfincsOutput.h5")
        np.testing.assert_allclose(radial_current_from_output(data), expected, rtol=0.0, atol=5e-12)

    phi1_dir = tmp_path / "with_phi1"
    _write_synthetic_scan(
        phi1_dir,
        er_values=er_values,
        current_vm=vm_currents,
        current_vd=vd_currents,
        include_phi1=True,
    )
    with_phi1 = solve_ambipolar_from_scan_dir(scan_dir=phi1_dir, write_pickle=False, write_json=False)
    np.testing.assert_allclose(with_phi1.radial_currents, vd_currents, rtol=0.0, atol=5e-12)
    for er, expected in zip(er_values, vd_currents, strict=True):
        data = read_sfincs_h5(phi1_dir / f"Er{er:+.3f}".replace("+", "p").replace("-", "m") / "sfincsOutput.h5")
        np.testing.assert_allclose(radial_current_from_output(data), expected, rtol=0.0, atol=5e-12)


def test_synthetic_ambipolar_roots_are_bracketed_zero_current_and_ion_typed(tmp_path: Path) -> None:
    """Synthetic Er scans only report roots supported by sign-changing current brackets."""
    scan_dir = tmp_path / "scan"
    er_values = [-2.0, -1.0, 0.0, 1.0]
    physical_currents = [-1.5, -0.5, 0.5, 1.5]
    misleading_vm_currents = [7.0, 8.0, 9.0, 10.0]
    _write_synthetic_scan(
        scan_dir,
        er_values=er_values,
        current_vm=misleading_vm_currents,
        current_vd=physical_currents,
        include_phi1=True,
    )

    res = solve_ambipolar_from_scan_dir(scan_dir=scan_dir, write_pickle=True, write_json=True, n_fine=200)

    np.testing.assert_allclose(res.var_values, er_values, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(res.radial_currents, physical_currents, rtol=0.0, atol=5e-12)
    assert res.roots_er.size == 1
    np.testing.assert_allclose(res.roots_er[0], -0.5, atol=1e-12)
    assert res.root_types == ["ion"]

    bracketed = False
    for lo_er, hi_er, lo_j, hi_j in zip(
        res.er_values[:-1],
        res.er_values[1:],
        res.radial_currents[:-1],
        res.radial_currents[1:],
        strict=True,
    ):
        if lo_j * hi_j < 0.0 and lo_er <= res.roots_er[0] <= hi_er:
            bracketed = True
            break
    assert bracketed

    radial_current_index = res.outputs_labels.index("radial current")
    np.testing.assert_allclose(res.outputs_at_roots[radial_current_index], np.asarray([0.0]), atol=1e-12)


def test_ambipolar_scan_parser_handles_non_er_scan_variables(tmp_path: Path) -> None:
    """A scan over SFINCS' normalized electric-field input is converted back to Er."""
    scan_dir = tmp_path / "dphi_scan"
    scan_dir.mkdir()
    scan_dir.joinpath("input.namelist").write_text(
        "!ss dPhiHatdrNMin = -1\n!ss dPhiHatdrNMax = 1\n",
        encoding="utf-8",
    )
    _write_named_scan_point(scan_dir, var_name="dPhiHatdrN", var_value=-1.0, er=-2.0, radial_current=-1.0)
    _write_named_scan_point(scan_dir, var_name="dPhiHatdrN", var_value=1.0, er=2.0, radial_current=1.0)

    res = solve_ambipolar_from_scan_dir(scan_dir=scan_dir, write_pickle=True, write_json=True, n_fine=50)

    assert res.var_name == "dPhiHatdrN"
    np.testing.assert_allclose(res.roots_var, np.asarray([0.0]), atol=1e-12)
    np.testing.assert_allclose(res.roots_er, np.asarray([0.0]), atol=1e-12)
    assert res.root_types == ["ion"]
    assert res.radius_actual == pytest.approx(0.5)
    payload = pickle.loads((scan_dir / "ambipolarSolutions.dat").read_bytes())
    assert payload["numQuantities"] == len(res.outputs_labels)


def test_ambipolar_scan_parser_reports_no_root_without_fake_extrapolation(tmp_path: Path) -> None:
    """Same-sign radial-current scans produce an empty, explicit root set."""
    scan_dir = tmp_path / "no_root_scan"
    _write_synthetic_scan(
        scan_dir,
        er_values=[-1.0, 0.0, 1.0],
        current_vm=[1.0, 1.5, 2.0],
        current_vd=[9.0, 9.0, 9.0],
        include_phi1=False,
    )

    res = solve_ambipolar_from_scan_dir(scan_dir=scan_dir, write_pickle=False, write_json=False, n_fine=50)

    assert res.roots_var.size == 0
    assert res.roots_er.size == 0
    assert res.root_types == []
    assert all(values.shape == (0,) for values in res.outputs_at_roots)


def test_ambipolar_scan_parser_failures_are_explicit(tmp_path: Path) -> None:
    """Malformed or incomplete scan directories fail before producing misleading roots."""
    with pytest.raises(FileNotFoundError, match="Missing scan input"):
        solve_ambipolar_from_scan_dir(scan_dir=tmp_path / "missing", write_pickle=False, write_json=False)

    empty_scan = tmp_path / "empty_scan"
    empty_scan.mkdir()
    empty_scan.joinpath("input.namelist").write_text("!ss ErMin = -1\n!ss ErMax = 1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="No run directories"):
        solve_ambipolar_from_scan_dir(scan_dir=empty_scan, write_pickle=False, write_json=False)

    wrong_rhs = tmp_path / "wrong_rhs"
    wrong_rhs.mkdir()
    wrong_rhs.joinpath("input.namelist").write_text("!ss ErMin = -1\n!ss ErMax = 1\n", encoding="utf-8")
    _write_named_scan_point(wrong_rhs, var_name="Er", var_value=-1.0, er=-1.0, radial_current=-1.0, rhs_mode=2)
    _write_named_scan_point(wrong_rhs, var_name="Er", var_value=1.0, er=1.0, radial_current=1.0, rhs_mode=2)
    with pytest.raises(RuntimeError, match="Need at least 2 completed runs"):
        solve_ambipolar_from_scan_dir(scan_dir=wrong_rhs, write_pickle=False, write_json=False)
