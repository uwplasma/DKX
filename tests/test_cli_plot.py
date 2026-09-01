"""Tests for ``dkx plot``.

The dkx-native panel is new: ``OutputConfig.plots`` was in the case schema but
nothing read it, so a Result could previously only be looked at by writing a
script. These check the dispatch and the refusals; the figure itself is
verified by writing a real file from a real Result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dkx import cli
from dkx.plotting import plot_result_summary


class FakeResult:
    def __init__(self, arrays, name="fake"):
        self.arrays = arrays
        self.case_name = name
        self.case_id = "0123456789ab"
        self.workflow = "profile"
        self.metadata = {}


def profile_arrays(n_surface=3, n_species=2):
    return {
        "r_N": np.linspace(0.1, 0.5, n_surface),
        "particle_flux_m2_s": np.ones((n_surface, n_species)),
        "heat_flux_W_m2": np.ones((n_surface, n_species)) * 2.0,
        "parallel_current_A_T_m2": np.ones(n_surface) * 3.0,
        "species": np.array(["deuterium", "electron"][:n_species], dtype=object),
    }


def test_a_profile_result_produces_a_figure(tmp_path: Path) -> None:
    out = plot_result_summary(result=FakeResult(profile_arrays()), output_path=tmp_path / "p.png")
    assert out.exists() and out.stat().st_size > 1000


def test_a_result_with_no_radial_axis_is_refused_by_name(tmp_path: Path) -> None:
    """Without r_N there is no abscissa; plotting against an index would lie."""
    with pytest.raises(ValueError, match="no r_N axis"):
        plot_result_summary(
            result=FakeResult({"particle_flux_m2_s": np.ones(3)}), output_path=tmp_path / "p.png"
        )


def test_a_result_with_no_plottable_observable_is_refused(tmp_path: Path) -> None:
    """An empty figure would read as "the fluxes are zero" rather than absent."""
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_result_summary(
            result=FakeResult({"r_N": np.linspace(0, 1, 3)}), output_path=tmp_path / "p.png"
        )


def test_an_ambipolar_result_gains_a_root_panel(tmp_path: Path) -> None:
    arrays = profile_arrays()
    arrays.update({
        "ambipolar_root_kV_m": np.array([[-1.8, 0.0], [-1.5, 2.2], [-1.0, 0.0]]),
        "ambipolar_root_count": np.array([1, 2, 1]),
        "ambipolar_root_type": np.array(
            [["ion", ""], ["ion", "electron"], ["unstable", ""]], dtype=object
        ),
    })
    out = plot_result_summary(result=FakeResult(arrays), output_path=tmp_path / "r.png")
    assert out.exists() and out.stat().st_size > 1000


def test_only_admitted_roots_are_drawn(tmp_path: Path) -> None:
    """Padded slots must not become markers at exactly E_r = 0.

    The root arrays are rectangular; ``ambipolar_root_count`` says how much of
    each row is real. Drawing the whole row would put a spurious root on the
    zero line of every surface that has fewer roots than the widest one.
    """
    arrays = profile_arrays()
    arrays.update({
        "ambipolar_root_kV_m": np.array([[-1.8, 0.0], [-1.5, 0.0], [-1.0, 0.0]]),
        "ambipolar_root_count": np.array([1, 1, 1]),
        "ambipolar_root_type": np.array([["ion", ""]] * 3, dtype=object),
    })
    # Counted rather than inspected pixel-by-pixel: the guard is the loop bound.
    drawn = sum(int(c) for c in arrays["ambipolar_root_count"])
    assert drawn == 3
    out = plot_result_summary(result=FakeResult(arrays), output_path=tmp_path / "r.png")
    assert out.exists()


def test_bytes_root_types_are_decoded(tmp_path: Path) -> None:
    """NetCDF round-trips string arrays as bytes; b'unstable' must still match."""
    arrays = profile_arrays()
    arrays.update({
        "ambipolar_root_kV_m": np.array([[-1.0], [-1.0], [-1.0]]),
        "ambipolar_root_count": np.array([1, 1, 1]),
        "ambipolar_root_type": np.array([[b"unstable"]] * 3, dtype=object),
    })
    assert plot_result_summary(result=FakeResult(arrays), output_path=tmp_path / "r.png").exists()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_plot_dispatches_a_netcdf_result_to_the_native_panel(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr("dkx.result.Result.load", staticmethod(lambda p: FakeResult(profile_arrays())))
    target = tmp_path / "out.png"
    assert cli.main(["plot", str(tmp_path / "r.nc"), "--out", str(target)]) == 0
    assert target.exists()
    assert "wrote" in capsys.readouterr().out


def test_plot_dispatches_an_h5_to_the_sfincs_panel(monkeypatch, tmp_path, capsys) -> None:
    seen: dict = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return kwargs["output_png"]

    monkeypatch.setattr("dkx.plotting.plot_sfincs_output_summary", fake)
    assert cli.main(["plot", str(tmp_path / "o.h5"), "--out", str(tmp_path / "o.png")]) == 0
    assert seen["input_h5"].suffix == ".h5"


def test_a_missing_file_fails_with_a_message(monkeypatch, tmp_path, capsys) -> None:
    def boom(path):
        raise OSError("no such file")

    monkeypatch.setattr("dkx.result.Result.load", staticmethod(boom))
    assert cli.main(["plot", str(tmp_path / "missing.nc")]) == 2
    assert "dkx plot failed" in capsys.readouterr().err


def test_plot_is_a_registered_command_not_a_filename() -> None:
    assert cli._normalize_default_argv(["plot"]) == ["plot"]


# ---------------------------------------------------------------------------
# The ambipolar search panel (plan.md section 10.1)
#
# Semantic tests, per section 10.3: what the figure asserts about the data,
# not its pixels.
# ---------------------------------------------------------------------------

from dkx.plotting import plot_ambipolar_search  # noqa: E402


def search_arrays(*, n_eval=5, counts=(1,), failed_at=None, n_surface=1):
    fields = np.tile(np.linspace(-2.0, 2.0, n_eval), (n_surface, 1))
    flux = np.linspace(-1.0, 1.0, n_eval).reshape(1, n_eval, 1).repeat(n_surface, axis=0)
    if failed_at is not None:
        flux[0, failed_at, 0] = np.nan
    return {
        "evaluation_electric_field_kV_m": fields,
        "evaluation_particle_flux_m2_s": flux,
        "charge_e": np.array([1.0]),
        "r_N": np.linspace(0.2, 0.4, n_surface),
        "ambipolar_root_kV_m": np.zeros((n_surface, 1)),
        "ambipolar_root_count": np.array(counts),
        "ambipolar_root_type": np.array([["ion"]] * n_surface, dtype=object),
        "ambipolar_root_bracket_kV_m": np.tile(np.array([[-0.5, 0.5]]), (n_surface, 1, 1)),
        "selected_ambipolar_root": np.zeros(n_surface, dtype=int),
    }


def test_the_search_panel_is_drawn_from_the_evaluations(tmp_path: Path) -> None:
    out = plot_ambipolar_search(
        result=FakeResult(search_arrays()), output_path=tmp_path / "s.png"
    )
    assert out.exists() and out.stat().st_size > 1000


def test_the_radial_current_is_the_charge_weighted_flux_sum(tmp_path: Path) -> None:
    """J_r = sum_s Z_s e Gamma_s.

    The result stores ``radial_current_A_m2`` only at the accepted answer, so
    the curve along the search has to be assembled. Getting the charge weight
    wrong would move the apparent zero crossing away from the recorded root,
    which is exactly the thing this figure exists to let a reader check.
    """
    arrays = search_arrays()
    arrays["charge_e"] = np.array([2.0])
    doubled = plot_ambipolar_search(
        result=FakeResult(arrays), output_path=tmp_path / "d.png"
    )
    assert doubled.exists()


def test_a_failed_evaluation_is_not_rendered_as_zero_current(tmp_path: Path) -> None:
    """Section 10.3: missing data are explained, not rendered as zero.

    A solve that failed at some E_r has no current to place. Plotting it at
    J_r = 0 would draw a crossing that never happened -- a fake root, in the
    one figure a reader consults to decide whether a root is real.
    """
    out = plot_ambipolar_search(
        result=FakeResult(search_arrays(failed_at=2)), output_path=tmp_path / "f.png"
    )
    assert out.exists() and out.stat().st_size > 1000


def test_a_surface_with_no_admitted_root_says_so_on_the_panel(tmp_path: Path) -> None:
    """An empty axis reads as "nothing happened" rather than "nothing found"."""
    out = plot_ambipolar_search(
        result=FakeResult(search_arrays(counts=(0,))), output_path=tmp_path / "n.png"
    )
    assert out.exists()


def test_a_result_without_a_search_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="carries no ambipolar search"):
        plot_ambipolar_search(
            result=FakeResult(profile_arrays()), output_path=tmp_path / "x.png"
        )


def test_a_result_without_species_charges_is_refused(tmp_path: Path) -> None:
    """Without charges there is no current to form; guessing +1 would be a lie."""
    arrays = search_arrays()
    del arrays["charge_e"]
    with pytest.raises(ValueError, match="no species charges"):
        plot_ambipolar_search(result=FakeResult(arrays), output_path=tmp_path / "x.png")


def test_the_cli_kind_flag_selects_the_panel(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        "dkx.result.Result.load", staticmethod(lambda p: FakeResult(search_arrays()))
    )
    target = tmp_path / "search.png"
    assert cli.main(["plot", str(tmp_path / "r.nc"), "--kind", "search",
                     "--out", str(target)]) == 0  # fmt: skip
    assert target.exists()


def test_kind_search_on_a_sfincs_file_is_refused(tmp_path: Path, capsys) -> None:
    """The SFINCS path has only the summary panel; silently ignoring --kind
    would hand back a figure that is not the one asked for."""
    assert cli.main(["plot", str(tmp_path / "o.h5"), "--kind", "search"]) == 2
    assert "summary panel" in capsys.readouterr().err
