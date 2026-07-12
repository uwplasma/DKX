"""End-to-end tests for the canonical RHSMode=2/3 driver and writer.

- ``sfincs_jax.run.run_transport_matrix`` transport matrices must match the
  recorded Fortran v3 ``sfincsOutput.h5`` goldens on the tiny monoenergetic
  scheme1/scheme11 and RHSMode=2 scheme2 fixtures;
- stdout must contain the exact golden Fortran banner/grid/Goodbye lines;
- importing ``sfincs_jax.run`` must never (re)introduce a legacy
  ``problems``/``operators``/``outputs`` package.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REF = Path(__file__).parent / "ref"

FIXTURES = (
    "monoenergetic_PAS_tiny_scheme1",
    "monoenergetic_PAS_tiny_scheme11",
    "transportMatrix_PAS_tiny_rhsMode2_scheme2",
)

def _read_h5(path: Path) -> dict[str, np.ndarray]:
    import h5py

    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, obj: out.__setitem__(name, obj[...]))
    return out


def _assert_scaled_close(a: np.ndarray, b: np.ndarray, *, tol: float, label: str) -> None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"{label}: shape {a.shape} != {b.shape}"
    if a.size == 0:
        return
    scale = max(1.0, float(np.max(np.abs(a))))
    err = float(np.max(np.abs(a - b)))
    assert err <= tol * scale, f"{label}: max|diff|={err:g} > {tol:g}*scale({scale:g})"


# ---------------------------------------------------------------------------
# Transport-matrix parity vs the recorded Fortran v3 outputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", FIXTURES)
def test_run_transport_matrix_matches_fortran_golden(base: str, tmp_path: Path) -> None:
    from sfincs_jax.run import run_transport_matrix

    run = run_transport_matrix(
        REF / f"{base}.input.namelist",
        out_path=tmp_path / f"{base}.canonical.h5",
        emit=None,
    )

    n = 3 if base.startswith("transportMatrix") else 2
    assert run.transport_matrix.shape == (n, n)
    assert run.state_vectors.shape == (n, run.operator.total_size)
    assert run.solve_result.converged

    # Direct Fortran referee: the canonical transport matrix must match the
    # recorded v3 sfincsOutput.h5.  This re-solves the systems, so the
    # comparison is limited by the finite PETSc KSP tolerance of the recorded
    # solutions; the ill-conditioned tiny scheme1/scheme2 fixtures sit at
    # ~1e-9 relative.  Fortran stores the matrix column-major, so the h5
    # dataset reads back transposed vs mathematical row/column order.
    import h5py

    with h5py.File(REF / f"{base}.sfincsOutput.h5", "r") as f:
        tm_fortran = np.asarray(f["transportMatrix"][...], dtype=np.float64)
    np.testing.assert_allclose(run.transport_matrix.T, tm_fortran, rtol=2e-8, atol=1e-13)

    # The written canonical file must carry the same matrix (Fortran layout).
    canonical = _read_h5(run.output_path)
    _assert_scaled_close(
        canonical["transportMatrix"], tm_fortran, tol=1e-7, label="written transportMatrix"
    )


def test_writer_netcdf_mirrors_h5(tmp_path: Path) -> None:
    from netCDF4 import Dataset

    from sfincs_jax.run import run_transport_matrix

    base = "monoenergetic_PAS_tiny_scheme1"
    run = run_transport_matrix(
        REF / f"{base}.input.namelist", out_path=tmp_path / f"{base}.nc", emit=None
    )
    h5 = _read_h5(
        run_transport_matrix(
            REF / f"{base}.input.namelist", out_path=tmp_path / f"{base}.h5", emit=None
        ).output_path
    )
    with Dataset(run.output_path) as f:
        assert f.getncattr("input_namelist").startswith("!")
        nc_tm = np.asarray(f["transportMatrix"][...], dtype=np.float64)
        assert np.array_equal(nc_tm, np.asarray(h5["transportMatrix"], dtype=np.float64))
        nc_bhat = np.asarray(f["BHat"][...], dtype=np.float64)
        assert np.array_equal(nc_bhat, np.asarray(h5["BHat"], dtype=np.float64))


# ---------------------------------------------------------------------------
# Console parity: exact golden Fortran lines in the run's stdout
# ---------------------------------------------------------------------------


def test_console_lines_for_scheme1_fixture(capsys: pytest.CaptureFixture[str]) -> None:
    from sfincs_jax import console
    from sfincs_jax.run import run_transport_matrix

    run = run_transport_matrix(REF / "monoenergetic_PAS_tiny_scheme1.input.namelist")
    lines = [ln.rstrip() for ln in capsys.readouterr().out.splitlines()]

    for banner_line in console.banner_lines(n_procs=1):
        assert banner_line.rstrip() in lines
    for nml_line in console.namelist_read_lines(
        input_name="monoenergetic_PAS_tiny_scheme1.input.namelist"
    ):
        assert nml_line.rstrip() in lines

    # Grid-summary block, including the exact Fortran matrix-size line
    # (indices.F90:361) for the 487-unknown tiny system.
    assert " ---- Numerical parameters: ----" in lines
    assert " Ntheta             =            9" in lines
    assert " Nzeta              =            9" in lines
    assert " Nxi                =            6" in lines
    assert " Nx                 =            1" in lines
    assert " Nxi_for_x_option:           0" in lines
    assert " x:   1.0000000000000000" in lines
    assert (
        " min_x_for_L:           1           1           1           1           1           1"
        in lines
    )
    assert console.matrix_size_line(matrix_size=487).rstrip() in lines
    assert lines and lines[-1] == " Goodbye!"

    assert console.entering_solver_line().rstrip() in lines
    assert console.main_solve_begin_line().rstrip() in lines
    assert any(ln.startswith(" Done with the main solve.  Time to solve: ") for ln in lines)

    # Transport-matrix block (diagnostics.F90:950-953), with real run values.
    idx = lines.index(" Transport matrix:")
    rendered = console.transport_matrix_lines(run.transport_matrix)
    assert [ln.rstrip() for ln in rendered] == lines[idx : idx + 1 + run.transport_matrix.shape[0]]


def test_transport_matrix_lines_match_fortran_golden_format() -> None:
    """Byte-parity of the transport-matrix block vs the frozen Fortran log."""
    from sfincs_jax import console

    rendered = console.transport_matrix_lines(
        [[-4.2660661992651783e-002, 7.9331964106435927e-004],
         [7.4006536798435429e-004, 1.2804416024249012]]
    )  # fmt: skip
    golden = (
        " Transport matrix:",
        "      -4.2660661992651783E-002   7.9331964106435927E-004",
        "       7.4006536798435429E-004   1.2804416024249012     ",
    )
    assert tuple(ln.rstrip() for ln in rendered) == tuple(ln.rstrip() for ln in golden)


# ---------------------------------------------------------------------------
# Import hygiene: the canonical driver must not touch the legacy stack
# ---------------------------------------------------------------------------


def test_run_module_does_not_import_legacy_stack() -> None:
    code = (
        "import sys\n"
        "import sfincs_jax.run\n"
        "import sfincs_jax.writer\n"
        "bad = sorted(m for m in sys.modules if m.startswith(("
        "'sfincs_jax.problems', 'sfincs_jax.operators', 'sfincs_jax.outputs')))\n"
        "print('LEGACY_IMPORTS=' + '|'.join(bad))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parents[1],
    )
    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith("LEGACY_IMPORTS=")]
    assert marker, proc.stdout
    offenders = [m for m in marker[-1].removeprefix("LEGACY_IMPORTS=").split("|") if m]
    assert offenders == []


def test_run_rejects_rhsmode1() -> None:
    from sfincs_jax.run import run_transport_matrix

    with pytest.raises(NotImplementedError, match="RHSMode"):
        run_transport_matrix(REF / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist", emit=None)
