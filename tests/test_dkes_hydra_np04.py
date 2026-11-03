"""DKX against DKES on the HELIAS reactor hydra_Np04_20190108 (IPP).

The DKES values cannot be redistributed, so the repository holds only DKX values and
agreement statistics (``validation/dkes_hydra_np04_v1.json``). The tests that need the DKES
values read them from ``DKX_DKES_HYDRA_DIR`` (the ``DKES_rho_<rho>.h5`` files), the one that
runs DKX from ``DKX_HYDRA_BOOZER`` (J. Geiger, IPP, 19 MB); both skip when unset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.paper_benchmarks.audit_dkes_hydra_np04 import audit
from tools.paper_benchmarks.run_dkes_hydra_np04 import dkes_dir, reference_beidler, reference_rows
from tools.release.artifacts import dkes_transport_to_beidler

ROOT = Path(__file__).resolve().parents[1]
RECORD = json.loads((ROOT / "validation/dkes_hydra_np04_v1.json").read_text())
GEOMETRY = {c["rho"]: c["geometry"] for c in RECORD["columns"]}
BOOZER = os.environ.get("DKX_HYDRA_BOOZER")
needs_dkes = pytest.mark.skipif(dkes_dir() is None, reason="DKX_DKES_HYDRA_DIR (private DKES_rho_<rho>.h5 files) not set")


def test_record_stores_no_dkes_values() -> None:
    text = json.dumps([RECORD["columns"], RECORD["agreement"]])
    assert not (ROOT / "validation/dkes_hydra_np04.csv").exists()
    assert all(token not in text for token in ("D11_S", "D13_S", "D31_S", "D33_S", "Erstar", "nuprime"))
    assert {k for col in RECORD["columns"] for row in col["rows"] for k in row} == {"cmul", "dkx"}


@needs_dkes
def test_collisional_limit_of_the_reference_is_d33_star_one() -> None:
    """D33_S is the SFINCS L33 whose Spitzer scale gives D33* = 1 at cmul = 10, every radius."""
    pytest.importorskip("h5py")
    rows = [r for r in reference_rows() if r["cmul"] == 10.0]
    assert {r["rho"] for r in rows} == set(GEOMETRY)
    for row in rows:
        assert reference_beidler(row, GEOMETRY[row["rho"]])["D33_star"] == pytest.approx(1.0, abs=1.5e-3)


def test_closed_form_conversion_is_the_dkx_function() -> None:
    from dkx.monoenergetic import monoenergetic_dstar_from_transport_matrix as dkx_dstar

    g = GEOMETRY["0.25"]
    tm = [[-2.0e-3, 1.0e-5], [3.0e-5, 2.9742e-3]]
    mine = dkes_transport_to_beidler(tm, cmul=10.0, geometry=g)
    theirs = dkx_dstar(
        tm, nu_prime=mine["nu_prime"], delta=g["delta"], g_hat=g["g"], i_hat=g["I"], iota=g["iota"],
        b0_over_bbar=g["b0"], fsab_hat2=g["fsab2"], r_hat=g["r"], x0=g["x0"], w0=g["w0"], nu_d_hat_x0=g["nud"],
    )
    for key, value in (("D11_star", theirs.d11_star), ("D13_star", theirs.d13_star),
                       ("D31_star", theirs.d31_star), ("D33_star", theirs.d33_star)):
        assert mine[key] == pytest.approx(float(value), rel=1e-12)


def test_sealed_comparison_passes_its_gates() -> None:
    report = audit(dkes_directory=None if dkes_dir() is None else dkes_dir())
    assert report["pass"], report["errors"][:3]


@needs_dkes
def test_stored_statistics_are_the_ones_of_the_private_dkes_data() -> None:
    pytest.importorskip("h5py")
    assert audit()["recomputed_from_dkes"] is True


@pytest.mark.skipif(not (BOOZER and Path(BOOZER).is_file()), reason="DKX_HYDRA_BOOZER (hydra_Np04 Boozer file) not set")
def test_dkx_reproduces_dkes_on_the_boozer_file(tmp_path: Path) -> None:
    from tools.paper_benchmarks.run_dkes_hydra_np04 import run

    pytest.importorskip("h5py")
    if dkes_dir() is None:
        pytest.skip("DKX_DKES_HYDRA_DIR not set")
    cmuls = [1.0, 0.03]
    out = tmp_path / "raw.json"
    run(Path(BOOZER), out, (13, 33, 48), ["0.25"], [0.01], cmuls)
    column = json.loads(out.read_text())["0.25|0.01"]
    for i, cmul in enumerate(cmuls):
        row = next(r for r in reference_rows() if r["rho"] == "0.25" and r["cmul"] == cmul and abs(r["efield"] - 0.01) < 1e-9)
        ref = reference_beidler(row, column["geometry"])
        assert column["d33"][i] == pytest.approx(ref["D33_star"], rel=5e-3)
        assert column["d11"][i] == pytest.approx(ref["D11_star"], rel=0.15)


def test_bc_header_accepts_trailing_text_of_the_ipp_files() -> None:
    from dkx.boozer_bc import _parse_bc_header_line

    ints, reals = _parse_bc_header_line(
        "   36   36   99    4 -6.120000E+01   2.01879  23.61794   2.02211  23.54054 1900.0072   original: a=1.84957 m"
    )
    assert ints == [36, 36, 99, 4]
    assert reals[:2] == [-61.2, 2.01879]
    with pytest.raises(ValueError):
        _parse_bc_header_line("1 2 3 4 5 text")
