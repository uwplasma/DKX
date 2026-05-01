from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_example_module() -> ModuleType:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "examples" / "vmec_jax_finite_beta" / "finite_beta_vmec_to_sfincs.py"
    spec = importlib.util.spec_from_file_location("finite_beta_vmec_to_sfincs", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_finite_beta_example_parses_er_values_and_names_run_dirs() -> None:
    mod = _load_example_module()

    assert mod._parse_er_values("-20,-10,0,10,20") == [-20.0, -10.0, 0.0, 10.0, 20.0]
    assert mod._parse_r_values("0.5,0.15,0.5") == [0.15, 0.5]
    assert mod._format_er_dir(-20.0) == "Erm20"
    assert mod._format_er_dir(2.5) == "Er2.5"
    assert mod._format_r_dir(0.15) == "rN0p15"
    assert mod._psi_n_from_r_n(0.5) == 0.25


def test_finite_beta_example_template_uses_vmec_scheme5_and_er() -> None:
    mod = _load_example_module()

    text = mod._sfincs_template(
        wout_path=Path("/tmp/wout_finite_beta.nc"),
        er=-13.5,
        r_n=0.5,
        ntheta=5,
        nzeta=7,
        nxi=4,
        nl=3,
        nx=3,
        solver_tolerance=1.0e-5,
        nu_n=1.0e-2,
    )

    assert "geometryScheme = 5" in text
    assert 'equilibriumFile = "/tmp/wout_finite_beta.nc"' in text
    assert "inputRadialCoordinateForGradients = 4" in text
    assert "Er = -13.5" in text
    assert "Ntheta = 5" in text
    assert "Nzeta = 7" in text
    assert "solverTolerance = 1e-05" in text


def test_finite_beta_example_finds_synthetic_ambipolar_roots() -> None:
    mod = _load_example_module()
    records = [
        mod.RunRecord(
            r_n=0.5,
            er=er,
            radial_current=current,
            bootstrap_current=-0.02 - 1.0e-4 * er,
            ion_particle_flux_rhat=0.1 + er,
            electron_particle_flux_rhat=-0.1 + er,
            ion_heat_flux_rhat=0.0,
            electron_heat_flux_rhat=0.0,
            output_h5=f"/tmp/Er{er:g}/sfincsOutput.h5",
        )
        for er, current in [(-2.0, 3.0), (-1.0, -1.0), (0.0, -2.0), (1.0, 1.0), (2.0, 3.0)]
    ]

    roots = mod._ambipolar_roots(records)
    assert len(roots) == 2
    assert -2.0 < roots[0] < -1.0
    assert 0.0 < roots[1] < 1.0

    bootstrap_at_roots = mod._root_interpolated_values(records, roots, "bootstrap_current")
    np.testing.assert_allclose(bootstrap_at_roots, [-0.02 - 1.0e-4 * root for root in roots], rtol=0.0, atol=2.0e-12)


def test_finite_beta_example_builds_continuous_radial_branch(tmp_path) -> None:
    mod = _load_example_module()
    scans = {}
    for r_n, roots in [(0.2, [-3.0, 1.0]), (0.5, [-2.0, 1.3]), (0.8, [-1.0, 1.6])]:
        records = []
        for er in [-4.0, roots[0], 0.0, roots[1], 3.0]:
            current = (er - roots[0]) * (er - roots[1])
            records.append(
                mod.RunRecord(
                    r_n=r_n,
                    er=er,
                    radial_current=current,
                    bootstrap_current=-0.02 - 1.0e-3 * r_n + 1.0e-4 * er,
                    ion_particle_flux_rhat=0.0,
                    electron_particle_flux_rhat=0.0,
                    ion_heat_flux_rhat=0.0,
                    electron_heat_flux_rhat=0.0,
                    output_h5=f"/tmp/rN{r_n:g}/Er{er:g}/sfincsOutput.h5",
                )
            )
        scans[r_n] = (records, tmp_path / mod._format_r_dir(r_n))

    profile = mod.build_radial_profile(scans_by_radius=scans, preferred_er=0.0)

    assert [p.r_n for p in profile] == [0.2, 0.5, 0.8]
    np.testing.assert_allclose([p.psi_n for p in profile], [0.04, 0.25, 0.64])
    np.testing.assert_allclose([p.selected_ambipolar_er for p in profile], [1.0, 1.3, 1.6])
    assert all(len(p.roots_er) == 2 for p in profile)


def test_finite_beta_example_convergence_summary_uses_common_surfaces() -> None:
    mod = _load_example_module()
    baseline = [
        mod.SurfaceProfileRecord(0.2, 0.04, [1.0], [-0.02], 1.0, -0.02, "/tmp/base0"),
        mod.SurfaceProfileRecord(0.5, 0.25, [2.0], [-0.03], 2.0, -0.03, "/tmp/base1"),
    ]
    refined = [
        mod.SurfaceProfileRecord(0.2, 0.04, [1.1], [-0.021], 1.1, -0.021, "/tmp/ref0"),
        mod.SurfaceProfileRecord(0.5, 0.25, [1.8], [-0.029], 1.8, -0.029, "/tmp/ref1"),
    ]

    summary = mod.convergence_summary(
        baseline=baseline,
        refined=refined,
        max_abs_er_tolerance=0.25,
        max_abs_bootstrap_tolerance=2.0e-3,
    )

    assert summary["surfaces_checked"] == 2
    np.testing.assert_allclose(summary["max_abs_er"], 0.2)
    np.testing.assert_allclose(summary["max_abs_bootstrap"], 1.0e-3)
    assert summary["passed"] is True


def test_finite_beta_example_convergence_summary_can_fail() -> None:
    mod = _load_example_module()
    baseline = [
        mod.SurfaceProfileRecord(0.5, 0.25, [2.0], [-0.03], 2.0, -0.03, "/tmp/base"),
    ]
    refined = [
        mod.SurfaceProfileRecord(0.5, 0.25, [1.0], [-0.01], 1.0, -0.01, "/tmp/ref"),
    ]

    summary = mod.convergence_summary(
        baseline=baseline,
        refined=refined,
        max_abs_er_tolerance=0.25,
        max_abs_bootstrap_tolerance=2.0e-3,
    )

    assert summary["surfaces_checked"] == 1
    assert summary["passed"] is False


def test_finite_beta_example_refines_bracketed_roots_without_full_solve(tmp_path, monkeypatch) -> None:
    mod = _load_example_module()

    def fake_run_or_load(**kwargs):
        er = float(kwargs["er"])
        return mod.RunRecord(
            r_n=0.5,
            er=er,
            radial_current=er + 5.0,
            bootstrap_current=-0.02,
            ion_particle_flux_rhat=0.0,
            electron_particle_flux_rhat=0.0,
            ion_heat_flux_rhat=0.0,
            electron_heat_flux_rhat=0.0,
            output_h5=f"/tmp/Er{er:g}/sfincsOutput.h5",
        )

    monkeypatch.setattr(mod, "_run_or_load_sfincs_record", fake_run_or_load)
    records = [
        fake_run_or_load(er=-10.0),
        fake_run_or_load(er=0.0),
    ]

    refined = mod._refine_ambipolar_brackets(
        records=records,
        wout_path=tmp_path / "wout.nc",
        scan_dir=tmp_path,
        r_n=0.5,
        ntheta=7,
        nzeta=7,
        nxi=5,
        nl=4,
        nx=4,
        solver_tolerance=1.0e-5,
        nu_n=1.0e-2,
        skip_existing=True,
        verbose=False,
        target_width=2.5,
        max_iterations=3,
    )

    np.testing.assert_allclose([record.er for record in refined], [-10.0, -7.5, -5.0, -2.5, 0.0])


def test_finite_beta_example_plot_summary_with_synthetic_data(tmp_path, monkeypatch) -> None:
    mod = _load_example_module()
    records = [
        mod.RunRecord(
            r_n=0.5,
            er=er,
            radial_current=current,
            bootstrap_current=-0.025 + 1.0e-4 * er,
            ion_particle_flux_rhat=0.01 * (idx + 1),
            electron_particle_flux_rhat=-0.008 * (idx + 1),
            ion_heat_flux_rhat=0.0,
            electron_heat_flux_rhat=0.0,
            output_h5=f"/tmp/Er{er:g}/sfincsOutput.h5",
        )
        for idx, (er, current) in enumerate([(-10.0, 1.0), (0.0, -0.5), (10.0, 1.2)])
    ]
    monkeypatch.setattr(mod, "_load_reference_output", lambda _: {"BHat": np.arange(25.0).reshape(5, 5)})

    png, pdf = mod.plot_summary(
        records=records,
        roots=mod._ambipolar_roots(records),
        profile=[
            mod.SurfaceProfileRecord(
                r_n=0.2,
                psi_n=0.04,
                roots_er=[-1.0],
                bootstrap_current_at_roots=[-0.024],
                selected_ambipolar_er=-1.0,
                selected_bootstrap_current=-0.024,
                scan_dir=str(tmp_path / "rN0p2"),
            ),
            mod.SurfaceProfileRecord(
                r_n=0.5,
                psi_n=0.25,
                roots_er=[0.0],
                bootstrap_current_at_roots=[-0.025],
                selected_ambipolar_er=0.0,
                selected_bootstrap_current=-0.025,
                scan_dir=str(tmp_path / "rN0p5"),
            ),
            mod.SurfaceProfileRecord(
                r_n=0.8,
                psi_n=0.64,
                roots_er=[1.0],
                bootstrap_current_at_roots=[-0.026],
                selected_ambipolar_er=1.0,
                selected_bootstrap_current=-0.026,
                scan_dir=str(tmp_path / "rN0p8"),
            ),
        ],
        convergence_profile=[
            mod.SurfaceProfileRecord(
                r_n=0.2,
                psi_n=0.04,
                roots_er=[-0.9],
                bootstrap_current_at_roots=[-0.0238],
                selected_ambipolar_er=-0.9,
                selected_bootstrap_current=-0.0238,
                scan_dir=str(tmp_path / "conv_rN0p2"),
            ),
            mod.SurfaceProfileRecord(
                r_n=0.5,
                psi_n=0.25,
                roots_er=[0.1],
                bootstrap_current_at_roots=[-0.0249],
                selected_ambipolar_er=0.1,
                selected_bootstrap_current=-0.0249,
                scan_dir=str(tmp_path / "conv_rN0p5"),
            ),
            mod.SurfaceProfileRecord(
                r_n=0.8,
                psi_n=0.64,
                roots_er=[1.1],
                bootstrap_current_at_roots=[-0.0259],
                selected_ambipolar_er=1.1,
                selected_bootstrap_current=-0.0259,
                scan_dir=str(tmp_path / "conv_rN0p8"),
            ),
        ],
        accuracy={
            "surfaces_checked": 3,
            "max_abs_er": 0.1,
            "max_abs_bootstrap": 2.0e-4,
            "passed": True,
        },
        vmec_summary={"fsq_total": 1.0e-6, "normalization_scalars": {"Aminor_p": 2.2}},
        out_dir=tmp_path,
        stem="finite_beta_plot_smoke",
        representative_r_n=0.5,
    )

    assert png.exists()
    assert pdf.exists()
    assert png.stat().st_size > 10_000
    assert pdf.stat().st_size > 1_000
