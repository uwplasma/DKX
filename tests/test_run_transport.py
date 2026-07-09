"""End-to-end tests for the canonical RHSMode=2/3 driver and writer.

Vertical slice #1 referee (plan_final.md "File-Level Execution Queues"):

- ``sfincs_jax.run.run_transport_matrix`` transport matrices must equal the
  legacy ``problems.transport_solve.solve_v3_transport_matrix_linear_gmres``
  result to 1e-10 (scaled) on the tiny monoenergetic scheme1/scheme11 and
  RHSMode=2 scheme2 fixtures;
- the ``sfincs_jax.writer`` h5 file must contain every dataset the legacy
  ``outputs`` writer emits for these modes, equal to 1e-10 (scaled), with the
  known-missing set enumerated explicitly (currently empty);
- stdout must contain the exact golden Fortran banner/grid/Goodbye lines;
- importing ``sfincs_jax.run`` must not import the legacy
  ``problems``/``operators``/``outputs`` packages.
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

# Datasets present in the legacy RHSMode=2/3 output file but not written by the
# canonical writer.  The canonical writer currently covers the full legacy
# field set for these modes; keep this explicit so a future regression is loud.
KNOWN_MISSING: frozenset[str] = frozenset()

# Wall-clock content differs run to run; compare shape/dtype only.
TIMING_KEYS = frozenset({"elapsed time (s)"})

# Scaled comparison tolerance |a-b| <= tol * max(1, max|a|).
DEFAULT_TOL = 1e-10
KEY_TOLERANCES = {
    # flow / totalDensity where totalDensity crosses zero in the unphysical
    # tiny scheme1 monoenergetic fixture: the quotient amplifies the ~1e-13
    # relative state difference between the legacy dense solve and the
    # canonical tier-1 structured solve.
    "velocityUsingTotalDensity": 1e-8,
}


def _read_h5(path: Path) -> dict[str, np.ndarray]:
    import h5py

    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, obj: out.__setitem__(name, obj[...]))
    return out


def _legacy_h5(base: str, tmp_path: Path) -> dict[str, np.ndarray]:
    from sfincs_jax.io import write_sfincs_jax_output_h5

    path = tmp_path / f"{base}.legacy.h5"
    write_sfincs_jax_output_h5(
        input_namelist=REF / f"{base}.input.namelist",
        output_path=path,
        compute_transport_matrix=True,
        overwrite=True,
        verbose=False,
    )
    return _read_h5(path)


def _legacy_transport_matrix(base: str) -> np.ndarray:
    from sfincs_jax.namelist import read_sfincs_input
    from sfincs_jax.problems.transport_solve import solve_v3_transport_matrix_linear_gmres

    result = solve_v3_transport_matrix_linear_gmres(
        nml=read_sfincs_input(REF / f"{base}.input.namelist")
    )
    return np.asarray(result.transport_matrix, dtype=np.float64)


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
# Transport-matrix and h5 equality vs the legacy stack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", FIXTURES)
def test_run_transport_matrix_matches_legacy_stack(base: str, tmp_path: Path) -> None:
    from sfincs_jax.run import run_transport_matrix

    run = run_transport_matrix(
        REF / f"{base}.input.namelist",
        out_path=tmp_path / f"{base}.canonical.h5",
        emit=None,
    )

    # 1) Transport matrix parity vs the legacy whichRHS solve.
    tm_legacy = _legacy_transport_matrix(base)
    _assert_scaled_close(tm_legacy, run.transport_matrix, tol=DEFAULT_TOL, label="transportMatrix")
    n = 3 if base.startswith("transportMatrix") else 2
    assert run.transport_matrix.shape == (n, n)
    assert run.state_vectors.shape == (n, run.operator.total_size)
    assert run.solve_result.converged

    # 2) Every dataset of the legacy writer's file, at the same tolerance.
    legacy = _legacy_h5(base, tmp_path)
    canonical = _read_h5(run.output_path)

    missing = set(legacy) - set(canonical)
    extra = set(canonical) - set(legacy)
    assert missing == set(KNOWN_MISSING)
    assert extra == set()

    for key in sorted(set(legacy) & set(canonical)):
        a, b = legacy[key], canonical[key]
        if a.dtype.kind in "SOU" or b.dtype.kind in "SOU":
            assert np.array_equal(a, b), f"{key}: string dataset mismatch"
            continue
        assert a.dtype == b.dtype, f"{key}: dtype {a.dtype} != {b.dtype}"
        if key in TIMING_KEYS:
            assert a.shape == b.shape
            continue
        _assert_scaled_close(a, b, tol=KEY_TOLERANCES.get(key, DEFAULT_TOL), label=key)


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
