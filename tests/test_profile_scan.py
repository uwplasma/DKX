"""``scan-profile``: ``!ss`` parsing, profile files and the per-radius decks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dkx.namelist import read_sfincs_input
from dkx.workflows.profile_scan import parse_ss_lines, read_profiles, run_profile_scan

DECK = """!ss scanType = 5
!ss Nradius = 2
!ss rN_min = 0.2
!ss rN_max = 0.8
&geometryParameters
  geometryScheme = 4
  inputRadialCoordinate = 3  ! rN
  rN_wish = 0.5
/
&speciesParameters
  Zs = 1.0 -1.0
  mHats = 1.0 5.4465e-4
  nHats = 0.5 0.5
  THats = 0.5 0.5
  dNHatdrHats = 0 0
  dTHatdrHats = 0 0
/
&physicsParameters
  Er = 0
/
&resolutionParameters
  Ntheta = 5
  Nzeta = 5
  Nxi = 4
  Nx = 3
/
"""

# rN, nErs, Ermin, Ermax, n_i, T_i, n_e, T_e: n = 1 - rN^2 (ion), T linear in rN (electron).
PROFILE = "# comment\n3\n" + "".join(
    f"{r:g}\t3\t-2\t{2 + 4 * r:g}\t{1 - r * r:g}\t0.5\t{1 - r * r:g}\t{1 - r:g}\n" for r in (0.0, 0.25, 0.5, 0.75, 1.0)
)


def test_parse_ss_lines_reads_fortran_reals_and_strings() -> None:
    ss = parse_ss_lines("!ss ErMax = 20.0d+0 ! top\n!ss NErs = 81\n!ss scanVariable = nu_n\nx = 1\n")
    assert ss == {"ermax": 20.0, "ners": 81.0, "scanvariable": "nu_n"}


def test_parse_ss_lines_names_the_malformed_line() -> None:
    text = "!ss NErs = 81\n!ss ErMin = -20.0d+0\n!ss ErMax = 20:.0d+0\n"
    with pytest.raises(ValueError, match=r"input\.namelist:3: .*'!ss ErMax = 20:\.0d\+0'"):
        parse_ss_lines(text)


def test_read_profiles_checks_columns_and_order(tmp_path: Path) -> None:
    good = tmp_path / "profile_a"
    good.write_text(PROFILE)
    table = read_profiles(good)
    assert table["r_n"].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert table["n_i"][2] == 0.75
    (tmp_path / "profile_b").write_text("3\n0 1 -1 1 1 1 1\n")
    with pytest.raises(ValueError, match="profile_b:2: expected 8 columns"):
        read_profiles(tmp_path / "profile_b")
    (tmp_path / "profile_c").write_text("1\n0.5 1 -1 1 1 1 1 1\n0.4 1 -1 1 1 1 1 1\n")
    with pytest.raises(ValueError, match="strictly increasing"):
        read_profiles(tmp_path / "profile_c")


def test_run_profile_scan_writes_one_deck_per_radius_and_collects_failures(tmp_path: Path) -> None:
    (tmp_path / "input.namelist").write_text(DECK)
    (tmp_path / "profile_x").write_text(PROFILE)
    calls = []

    def _scan_er(*, input_namelist, out_dir, values, **_kwargs):
        calls.append((Path(input_namelist), Path(out_dir), list(values)))
        failed = ((values[0], "stalled"),) if len(calls) == 2 else ()
        return SimpleNamespace(failures=failed)

    result = run_profile_scan(
        input_namelist=tmp_path / "input.namelist", out_dir=tmp_path / "scan", scan_er=_scan_er
    )

    assert result.radii == (0.2, 0.8) and result.n_points == 6
    assert [c[1].name for c in calls] == ["rN0.2", "rN0.8"]
    assert calls[0][2] == pytest.approx(np.linspace(-2, 2.8, 3))  # Ermax = 2 + 4 rN at rN = 0.2
    nml = read_sfincs_input(calls[0][0])
    assert float(nml.group("geometryParameters")["RN_WISH"]) == 0.2
    species = nml.group("speciesParameters")
    assert species["NHATS"][0] == pytest.approx(0.95)  # between 1.0 at rN = 0 and 0.9375 at 0.25
    assert species["THATS"][1] == pytest.approx(0.8)
    # Electron T falls by 1 per unit rN, so dT/drHat = -1 / aHat; the ion T is flat.
    a_hat = -1.0 / species["DTHATDRHATS"][1]
    assert species["DTHATDRHATS"][0] == pytest.approx(0.0)
    assert 0.1 < a_hat < 5.0
    assert result.failures == ((0.8, -2.0, "stalled"),)
    report = json.loads((tmp_path / "scan" / "scan_failures.json").read_text())
    assert report == [{"rN": 0.8, "Er": -2.0, "reason": "stalled"}]


def test_run_profile_scan_rejects_unsupported_decks(tmp_path: Path) -> None:
    (tmp_path / "profile_x").write_text(PROFILE)
    (tmp_path / "input.namelist").write_text(DECK.replace("scanType = 5", "scanType = 2"))
    with pytest.raises(ValueError, match="scanType 5"):
        run_profile_scan(input_namelist=tmp_path / "input.namelist", out_dir=tmp_path / "s")
    (tmp_path / "input.namelist").write_text(DECK.replace("dNHatdrHats", "dNHatdpsiNs"))
    with pytest.raises(ValueError, match="dNHatdrHats"):
        run_profile_scan(input_namelist=tmp_path / "input.namelist", out_dir=tmp_path / "s")


def test_run_profile_scan_refuses_radii_outside_the_profile(tmp_path: Path) -> None:
    (tmp_path / "input.namelist").write_text(DECK)
    (tmp_path / "profile_x").write_text(PROFILE.replace("\n0\t", "\n0.1\t", 1))
    with pytest.raises(ValueError, match=r"radii \[0\.05\] lie outside .*\[0\.1, 1\]"):
        run_profile_scan(input_namelist=tmp_path / "input.namelist", out_dir=tmp_path / "s", radii=[0.05, 0.5], scan_er=None)


def test_scan_profile_command_prints_the_malformed_line_and_exits(tmp_path: Path) -> None:
    from dkx import cli

    (tmp_path / "input.namelist").write_text(DECK.replace("rN_max = 0.8", "rN_max = 0:.8"))
    (tmp_path / "profile_x").write_text(PROFILE)
    with pytest.raises(SystemExit, match=r"scan-profile: .*input\.namelist:4: .*rN_max = 0:\.8"):
        cli.main(["scan-profile", "--input", str(tmp_path / "input.namelist"), "--out-dir", str(tmp_path / "s"), "--quiet"])
