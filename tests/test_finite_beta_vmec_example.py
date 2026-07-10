from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import pytest


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


def _load_redl_compare_module() -> ModuleType:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "examples" / "vmec_jax_finite_beta" / "compare_landreman_paul_qa_bootstrap_redl.py"
    spec = importlib.util.spec_from_file_location("compare_landreman_paul_qa_bootstrap_redl", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_qs_paper_redl_module() -> ModuleType:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "examples" / "vmec_jax_finite_beta" / "compare_qs_paper_sfincs_jax_redl.py"
    spec = importlib.util.spec_from_file_location("compare_qs_paper_sfincs_jax_redl", script)
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


def test_landreman_paul_redl_comparison_template_uses_scheme5_profile_gradients() -> None:
    mod = _load_redl_compare_module()

    text = mod._sfincs_template(
        wout_path=Path("/tmp/wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc"),
        r_n=0.5,
        er=0.0,
        nu_n=8.31565e-3,
        n_hat=1.2,
        t_i_hat=2.0,
        t_e_hat=2.1,
        ion_mhat=2.0,
        electron_mhat=1.0 / 1836.15267343,
        dn_hat_dpsi_n=-0.3,
        dt_i_hat_dpsi_n=-0.4,
        dt_e_hat_dpsi_n=-0.5,
        collision_operator=1,
        ntheta=5,
        nzeta=7,
        nxi=9,
        nl=4,
        nx=4,
        solver_tolerance=1.0e-6,
    )

    assert "geometryScheme = 5" in text
    assert 'equilibriumFile = "/tmp/wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc"' in text
    assert "inputRadialCoordinateForGradients = 1" in text
    assert "mHats = 2 0.0005446170214876324" in text
    assert "dNHatdpsiNs = -0.3 -0.3" in text
    assert "dTHatdpsiNs = -0.4 -0.5" in text
    assert "collisionOperator = 1" in text
    assert "Ntheta = 5" in text
    assert "Nzeta = 7" in text
    assert "Nxi = 9" in text
    assert "solverTolerance = 1e-06" in text


def test_landreman_paul_redl_comparison_summarizes_synthetic_difference(tmp_path) -> None:
    mod = _load_redl_compare_module()
    args = mod._build_parser().parse_args(["--skip-sfincs", "--out-dir", str(tmp_path)])
    redl = {
        "r_n": np.asarray([0.5]),
        "s": np.asarray([0.25]),
        "jdotb_redl": np.asarray([-4.0]),
        "j_parallel_redl_si": np.asarray([-2.0e6]),
        "fsa_B2": np.asarray([4.0]),
        "epsilon": np.asarray([0.1]),
        "f_t": np.asarray([0.2]),
        "iota": np.asarray([0.42]),
        "nu_e_star": np.asarray([1.0]),
        "nu_i_star": np.asarray([2.0]),
        "L31": np.asarray([0.3]),
        "L32": np.asarray([0.4]),
        "L34": np.asarray([0.5]),
    }
    sfincs_rows = [
        {
            "status": "loaded",
            "r_n": 0.5,
            "s": 0.25,
            "input": "/tmp/input.namelist",
            "output": "/tmp/sfincsOutput.h5",
            "elapsed_s": 1.0,
            "FSABjHat": -1.0,
            "FSABjHatOverRootFSAB2": -0.25,
            "sfincs_j_parallel_si": -1.5e6,
            "sfincs_jdotb_scaled": -6.0e6,
        }
    ]

    payload = mod._write_summary(
        path=tmp_path / "summary.json",
        args=args,
        vmec_input=Path("/tmp/input.LandremanPaul2021_QA_reactorScale_lowres"),
        wout_path=Path("/tmp/wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc"),
        redl=redl,
        sfincs_rows=sfincs_rows,
        scale=6.0e6,
    )

    assert payload["workflow"] == "landreman_paul_qa_sfincs_jax_redl_bootstrap_current_comparison"
    assert payload["comparison"]["n_compared"] == 1
    np.testing.assert_allclose(payload["comparison"]["max_abs_diff_A_per_m2"], 5.0e5)
    np.testing.assert_allclose(payload["comparison"]["max_rel_diff"], 0.25)
    assert "not a Redl-parity claim" in payload["claim_boundary"]
    assert "FSABjHatOverRootFSAB2" in payload["normalization"]["sfincs_si_formula"]
    assert "sqrt(2*(bsq - p))" in payload["normalization"]["redl_geometry_B_convention"]
    assert payload["collisionality_contract"]["sfincs_nu_n"] == args.nu_n
    assert payload["collisionality_contract"]["redl_nu_e_star"] == [1.0]
    assert payload["collisionality_contract"]["redl_nu_i_star"] == [2.0]


def test_landreman_paul_redl_comparison_errorbar_summary_uses_refinement_deltas(tmp_path) -> None:
    mod = _load_redl_compare_module()
    args = mod._build_parser().parse_args(["--skip-sfincs", "--with-errorbars", "--out-dir", str(tmp_path)])
    redl = {
        "r_n": np.asarray([0.5, 0.7]),
        "s": np.asarray([0.25, 0.49]),
        "jdotb_redl": np.asarray([-4.0, -5.0]),
        "j_parallel_redl_si": np.asarray([-2.0e6, -3.0e6]),
        "fsa_B2": np.asarray([4.0, 4.0]),
        "epsilon": np.asarray([0.1, 0.2]),
        "f_t": np.asarray([0.2, 0.3]),
        "iota": np.asarray([0.42, 0.41]),
        "nu_e_star": np.asarray([1.0, 1.1]),
        "nu_i_star": np.asarray([2.0, 2.1]),
        "L31": np.asarray([0.3, 0.31]),
        "L32": np.asarray([0.4, 0.41]),
        "L34": np.asarray([0.5, 0.51]),
    }
    sfincs_rows = [
        {"sfincs_j_parallel_si": -1.5e6},
        {"sfincs_j_parallel_si": -2.5e6},
    ]
    convergence = {
        "enabled": True,
        "definition": "synthetic",
        "baseline_resolution": {"Ntheta": 5, "Nzeta": 5, "Nxi": 5, "NL": 3, "Nx": 4},
        "real_space_resolution": {"Ntheta": 7, "Nzeta": 7, "Nxi": 5, "NL": 3, "Nx": 4},
        "velocity_space_resolution": {"Ntheta": 5, "Nzeta": 5, "Nxi": 7, "NL": 4, "Nx": 5},
        "sfincs_j_parallel_si_errorbar": [2.0e4, 5.0e4],
        "sfincs_j_parallel_si_errorbar_rel_to_baseline": [2.0e4 / 1.5e6, 5.0e4 / 2.5e6],
        "real_space_delta_A_per_m2": [2.0e4, 4.0e4],
        "velocity_space_delta_A_per_m2": [1.0e4, 5.0e4],
        "real_space": [],
        "velocity_space": [],
    }

    payload = mod._write_summary(
        path=tmp_path / "summary_with_errorbars.json",
        args=args,
        vmec_input=Path("/tmp/input.LandremanPaul2021_QA_reactorScale_lowres"),
        wout_path=Path("/tmp/wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc"),
        redl=redl,
        sfincs_rows=sfincs_rows,
        scale=6.0e6,
        convergence=convergence,
    )

    assert payload["comparison"]["n_compared"] == 2
    np.testing.assert_allclose(payload["comparison"]["max_errorbar_A_per_m2"], 5.0e4)
    np.testing.assert_allclose(payload["comparison"]["max_errorbar_rel_to_sfincs"], 0.02)
    assert payload["convergence_errorbars"]["real_space_resolution"]["Ntheta"] == 7


def test_qs_paper_redl_comparison_patches_archived_inputs(tmp_path) -> None:
    mod = _load_qs_paper_redl_module()
    source = tmp_path / "source.input"
    destination = tmp_path / "run" / "input.namelist"
    source.write_text(
        "&resolutionParameters\n"
        "  Ntheta = 25\n"
        "  Nzeta = 39\n"
        "  Nxi = 60\n"
        "  Nx = 7\n"
        "/\n"
        "&otherNumericalParameters\n"
        "  solverTolerance = 1e-9\n"
        "/\n",
        encoding="utf-8",
    )

    mod._write_patched_input(
        source=source,
        destination=destination,
        ntheta=13,
        nzeta=15,
        nxi=21,
        nx=5,
        tolerance=1.0e-6,
    )

    text = destination.read_text(encoding="utf-8")
    assert "Ntheta = 13" in text
    assert "Nzeta = 15" in text
    assert "Nxi = 21" in text
    assert "Nx = 5" in text
    assert "solverTolerance = 1e-06" in text


def test_qs_paper_redl_comparison_parses_surfaces_and_names_dirs() -> None:
    mod = _load_qs_paper_redl_module()

    np.testing.assert_allclose(mod._parse_s_values("0.6,0.5,0.5"), [0.5, 0.6])
    with pytest.raises(ValueError, match="requires a case_root"):
        mod._parse_s_values("all")
    assert mod._surface_dir_name(0.5) == "psiN_0.5"
    assert mod._format_surface_label(0.55) == "s0p550"
    args = mod._build_parser().parse_args(["--solve-method", "sparse_pc_gmres"])
    assert args.solve_method == "sparse_pc_gmres"
    assert args.s_values == "all"
    assert args.verbose_sfincs is False
    verbose_args = mod._build_parser().parse_args(["--verbose-sfincs"])
    assert verbose_args.verbose_sfincs is True
    quick_args = mod._build_parser().parse_args(["--quick"])
    assert quick_args.quick is True
    errorbar_args = mod._build_parser().parse_args(
        [
            "--with-errorbars",
            "--real-ntheta",
            "17",
            "--real-nzeta",
            "19",
            "--velocity-nxi",
            "25",
            "--velocity-nx",
            "6",
        ]
    )
    assert errorbar_args.with_errorbars is True
    assert errorbar_args.real_ntheta == 17
    assert errorbar_args.real_nzeta == 19
    assert errorbar_args.velocity_nxi == 25
    assert errorbar_args.velocity_nx == 6
    claim_args = mod._build_parser().parse_args(["--match-fortran-resolution", "--require-same-resolution"])
    assert claim_args.match_fortran_resolution is True
    assert claim_args.require_same_resolution is True
    custom_fortran_args = mod._build_parser().parse_args(["--fortran-case-root", "/tmp/fortran_lowres"])
    assert custom_fortran_args.fortran_case_root == Path("/tmp/fortran_lowres")
    from_summary_args = mod._build_parser().parse_args(["--from-summary-json", "/tmp/summary.json"])
    assert from_summary_args.from_summary_json == Path("/tmp/summary.json")


def test_qs_paper_redl_comparison_forwards_verbose_to_sfincs_writer(tmp_path, monkeypatch) -> None:
    mod = _load_qs_paper_redl_module()
    source = tmp_path / "source.input"
    source.write_text(
        "&resolutionParameters\n"
        "  Ntheta = 25\n"
        "  Nzeta = 39\n"
        "  Nxi = 60\n"
        "  Nx = 7\n"
        "/\n"
        "&otherNumericalParameters\n"
        "  solverTolerance = 1e-9\n"
        "/\n",
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def fake_writer(**kwargs):
        seen["verbose"] = kwargs["verbose"]
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as h5:
            h5["FSABjHat"] = np.asarray([-1.0])
            h5["rN"] = np.asarray([0.5])
            h5["psiN"] = np.asarray([0.25])
            h5["NIterations"] = np.asarray([1])

    monkeypatch.setattr(mod, "write_sfincs_jax_output_h5", fake_writer)
    args = mod._build_parser().parse_args(["--verbose-sfincs", "--force", "--out-dir", str(tmp_path)])

    row = mod._run_or_read_sfincs_jax(
        source_input=source,
        wout_path=tmp_path / "wout.nc",
        run_dir=tmp_path / "run",
        args=args,
    )

    assert seen["verbose"] is True
    assert row["status"] == "ok"
    assert row["NIterations"] == 1


def test_qs_paper_redl_comparison_env_can_enable_sfincs_verbose(monkeypatch) -> None:
    mod = _load_qs_paper_redl_module()
    args = mod._build_parser().parse_args([])

    assert mod._verbose_sfincs_enabled(args) is False
    monkeypatch.setenv("SFINCS_JAX_EXAMPLE_VERBOSE", "1")
    assert mod._verbose_sfincs_enabled(args) is True


def test_qs_paper_redl_comparison_discovers_all_archived_surfaces(tmp_path) -> None:
    mod = _load_qs_paper_redl_module()
    for value in (0.6, 0.5, 0.5):
        surface = tmp_path / f"psiN_{value:g}"
        surface.mkdir(parents=True, exist_ok=True)
        (surface / "input.namelist").write_text("&general\n/\n", encoding="utf-8")

    np.testing.assert_allclose(mod._parse_s_values("all", case_root=tmp_path), [0.5, 0.6])


def test_qs_paper_redl_comparison_loads_archived_fortran_outputs(tmp_path) -> None:
    mod = _load_qs_paper_redl_module()
    output_path = tmp_path / "psiN_0.55" / "sfincsOutput.h5"
    output_path.parent.mkdir(parents=True)
    with h5py.File(output_path, "w") as h5:
        h5["FSABjHat"] = np.asarray([-1.25])
        h5["Ntheta"] = np.asarray(25)
        h5["Nzeta"] = np.asarray(39)
        h5["Nxi"] = np.asarray(60)
        h5["Nx"] = np.asarray(7)
        h5["NIterations"] = np.asarray(1)
    (output_path.parent / "slurm-123.out").write_text(
        "Done with the main solve. Time to solve: 42.5\n"
        "Memory effectively used, total in Mbytes (INFOG(22)): 1234\n",
        encoding="utf-8",
    )

    rows = mod._load_archived_fortran_outputs(case_root=tmp_path)

    assert len(rows) == 1
    assert rows[0]["s"] == 0.55
    assert rows[0]["Ntheta"] == 25
    assert rows[0]["elapsed_s"] == 42.5
    assert rows[0]["memory_mb"] == 1234.0
    assert rows[0]["memory_metric"] == "mumps_effective_total_mb"
    np.testing.assert_allclose(rows[0]["jdotb_si"], -1.25 * mod.SFINCS_PAPER_CURRENT_FACTOR)


def test_qs_paper_redl_comparison_loads_local_fortran_profile(tmp_path) -> None:
    mod = _load_qs_paper_redl_module()
    output_path = tmp_path / "psiN_0.5" / "sfincsOutput.h5"
    output_path.parent.mkdir(parents=True)
    with h5py.File(output_path, "w") as h5:
        h5["FSABjHat"] = np.asarray([-1.0])
        h5["Ntheta"] = np.asarray(13)
        h5["Nzeta"] = np.asarray(13)
        h5["Nxi"] = np.asarray(21)
        h5["Nx"] = np.asarray(5)
        h5["NIterations"] = np.asarray(1)
        h5["elapsed time (s)"] = np.asarray(0.0)
    (output_path.parent / "sfincs_fortran_stdout.txt").write_text(
        " ** Space in MBYTES used for solve                        :       126\n"
        " Done with the main solve.  Time to solve:    1.1787289999997483       seconds.\n",
        encoding="utf-8",
    )
    (output_path.parent / "sfincs_fortran_stderr.txt").write_text(
        "           319012864  maximum resident set size\n",
        encoding="utf-8",
    )

    rows = mod._load_archived_fortran_outputs(case_root=tmp_path)

    assert len(rows) == 1
    assert rows[0]["elapsed_s"] == pytest.approx(1.1787289999997483)
    assert rows[0]["memory_mb"] == 126.0
    assert rows[0]["memory_metric"] == "mumps_solve_space_mb"
    assert rows[0]["profile"]["max_rss_mb"] == pytest.approx(319.012864)


def test_qs_paper_redl_comparison_same_resolution_gate_is_fail_closed() -> None:
    mod = _load_qs_paper_redl_module()
    selected_fortran_rows = [
        {"status": "ok", "s": 0.5, "Ntheta": 25, "Nzeta": 39, "Nxi": 60, "Nx": 7},
        {"status": "ok", "s": 0.7, "Ntheta": 25, "Nzeta": 39, "Nxi": 60, "Nx": 7},
    ]

    reduced_args = mod._build_parser().parse_args(["--require-same-resolution", "--ntheta", "13"])
    comparison = mod._resolution_comparison(args=reduced_args, selected_fortran_rows=selected_fortran_rows)

    assert comparison["status"] == "mixed_resolution"
    assert comparison["same_resolution_on_compared_surfaces"] is False
    with pytest.raises(SystemExit, match="Refusing mixed-resolution"):
        mod._enforce_same_resolution_requirement(reduced_args, comparison)

    matched_args = mod._build_parser().parse_args(["--match-fortran-resolution", "--require-same-resolution"])
    mod._apply_matching_fortran_resolution(matched_args, selected_fortran_rows)
    matched = mod._resolution_comparison(args=matched_args, selected_fortran_rows=selected_fortran_rows)

    assert (matched_args.ntheta, matched_args.nzeta, matched_args.nxi, matched_args.nx) == (25, 39, 60, 7)
    assert matched["status"] == "same_resolution"
    assert matched["same_resolution_on_compared_surfaces"] is True
    mod._enforce_same_resolution_requirement(matched_args, matched)


def test_qs_paper_redl_comparison_fortran_errorbar_sidecar_is_explicit(tmp_path) -> None:
    mod = _load_qs_paper_redl_module()
    sidecar = tmp_path / "fortran_errorbars.json"
    sidecar.write_text(
        json.dumps(
            {
                "definition": "synthetic refinement bars",
                "surfaces": [
                    {"s": 0.5, "jdotb_si_errorbar": 1.0e5},
                    {"s": 0.7, "errorbar_si": 2.0e5},
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = [{"status": "ok", "s": 0.5}, {"status": "ok", "s": 0.7}, {"status": "ok", "s": 0.9}]

    errorbars = mod._load_fortran_errorbar_map(sidecar)
    mod._apply_fortran_errorbars(rows, errorbars)

    assert errorbars == {0.5: 1.0e5, 0.7: 2.0e5}
    assert rows[0]["jdotb_si_errorbar"] == 1.0e5
    assert rows[1]["jdotb_si_errorbar"] == 2.0e5
    assert "jdotb_si_errorbar" not in rows[2]


def test_qs_paper_redl_comparison_plots_synthetic_payload(tmp_path) -> None:
    mod = _load_qs_paper_redl_module()
    payload = {
        "case_title": "Synthetic QA benchmark",
        "sfincs_resolution_label": "13x13x21x5",
        "resolution_comparison": {
            "same_resolution_on_compared_surfaces": False,
            "sfincs_fortran_v3_resolution_label": "25x39x60x7",
        },
        "redl": {
            "s": [0.45, 0.5, 0.55, 0.6, 0.65],
            "jdotb_si": [-8.2e6, -7.6e6, -7.45e6, -7.2e6, -6.9e6],
        },
        "sfincs_fortran_v3": [
            {"status": "ok", "s": 0.5, "jdotb_si": -7.40e6, "jdotb_si_errorbar": 8.0e4},
            {"status": "ok", "s": 0.55, "jdotb_si": -7.60e6},
            {"status": "ok", "s": 0.6, "jdotb_si": -7.65e6},
        ],
        "sfincs_jax": [
            {"status": "ok", "s": 0.5, "jdotb_si": -7.62e6},
            {"status": "ok", "s": 0.55, "jdotb_si": -7.46e6},
            {"status": "ok", "s": 0.6, "jdotb_si": -7.13e6},
        ],
        "metrics": {
            "requested_points": 3,
            "completed_points": 3,
            "max_relative_difference": 0.05,
            "max_jax_relative_difference_vs_redl": 0.05,
            "max_jax_relative_difference_vs_fortran": 0.07,
            "sfincs_jax_elapsed_s_sum": 12.3,
            "max_errorbar_rel_to_baseline": 0.02,
        },
        "convergence_errorbars": {
            "jdotb_si_errorbar": [1.0e5, 2.0e5, 1.5e5],
            "jdotb_si_errorbar_rel_to_baseline": [0.01, 0.02, 0.015],
        },
    }

    mod._plot(payload, png_path=tmp_path / "qs_redl.png", pdf_path=tmp_path / "qs_redl.pdf")

    assert (tmp_path / "qs_redl.png").exists()
    assert (tmp_path / "qs_redl.pdf").exists()
    assert (tmp_path / "qs_redl.png").stat().st_size > 10_000
    assert (tmp_path / "qs_redl.pdf").stat().st_size > 1_000


def test_qs_paper_redl_comparison_regenerates_plot_from_summary_json(tmp_path) -> None:
    mod = _load_qs_paper_redl_module()
    payload = {
        "case_title": "Synthetic QH benchmark",
        "sfincs_resolution_label": "13x13x21x5",
        "resolution_comparison": {"same_resolution_on_compared_surfaces": True},
        "redl": {"s": [0.45, 0.5, 0.55], "jdotb_si": [-3.0e6, -2.6e6, -2.4e6]},
        "sfincs_fortran_v3": [{"status": "ok", "s": 0.5, "jdotb_si": -2.55e6}],
        "sfincs_jax": [{"status": "ok", "s": 0.5, "jdotb_si": -2.58e6}],
        "metrics": {
            "requested_points": 1,
            "completed_points": 1,
            "max_jax_relative_difference_vs_redl": 0.02,
            "max_jax_relative_difference_vs_fortran": 0.01,
        },
        "performance": {
            "runtime_total_s": {"sfincs_jax": 1.5, "sfincs_fortran_v3": 2.0},
            "memory_peak_mb": {"sfincs_jax": 20.0, "sfincs_fortran_v3": 30.0},
        },
    }
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(payload), encoding="utf-8")

    rc = mod.main(
        [
            "--from-summary-json",
            str(summary),
            "--fig-dir",
            str(tmp_path / "figures"),
            "--stem",
            "rerendered",
        ]
    )

    assert rc == 0
    assert (tmp_path / "figures" / "rerendered.png").stat().st_size > 10_000
    assert (tmp_path / "figures" / "rerendered.pdf").stat().st_size > 1_000
    assert json.loads((tmp_path / "figures" / "rerendered.json").read_text(encoding="utf-8")) == payload


def test_qs_paper_redl_comparison_errorbars_use_refinement_deltas() -> None:
    mod = _load_qs_paper_redl_module()

    baseline = np.asarray([-7.0e6, -6.0e6, np.nan])
    real = np.asarray([-7.1e6, -5.9e6, -4.0e6])
    velocity = np.asarray([-6.8e6, -6.4e6, -3.0e6])

    error = mod._pointwise_max_abs_delta(baseline, [real, velocity])

    np.testing.assert_allclose(error[:2], [2.0e5, 4.0e5])
    assert np.isnan(error[2])


def test_qs_paper_redl_comparison_profiles_ignore_failed_rows() -> None:
    mod = _load_qs_paper_redl_module()
    rows = [
        {"status": "ok", "jdotb_si": -7.0e6},
        {"status": "error", "jdotb_si": -99.0e6},
        {"status": "missing"},
    ]

    profile = mod._sfincs_jdotb_profile_from_rows(rows)

    np.testing.assert_allclose(profile[0], -7.0e6)
    assert np.isnan(profile[1])
    assert np.isnan(profile[2])


def test_qs_paper_redl_comparison_can_hide_fortran_overlay(tmp_path) -> None:
    mod = _load_qs_paper_redl_module()
    payload = {
        "case_title": "Synthetic QA benchmark",
        "hide_fortran": True,
        "sfincs_resolution_label": "13x13x21x5",
        "redl": {"s": [0.5, 0.6], "jdotb_si": [-7.6e6, -7.2e6]},
        "sfincs_fortran_v3": [{"status": "ok", "s": 0.5, "jdotb_si": -99.0e6}],
        "sfincs_jax": [{"status": "ok", "s": 0.5, "jdotb_si": -7.62e6}],
        "metrics": {
            "requested_points": 1,
            "completed_points": 1,
            "max_jax_relative_difference_vs_redl": 0.01,
        },
    }

    mod._plot(payload, png_path=tmp_path / "no_fortran.png", pdf_path=tmp_path / "no_fortran.pdf")

    assert (tmp_path / "no_fortran.png").exists()
    assert (tmp_path / "no_fortran.pdf").exists()


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
