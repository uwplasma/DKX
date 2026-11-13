"""Edge cases of the immutable Result container: validation, summary, save/load, plot titles."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from dkx.result import RESULT_SCHEMA_VERSION, Result  # noqa: E402


def _result(arrays=None, dims=None, **kw) -> Result:
    arrays = {"surface": np.array([0.1, 0.4, 0.9])} if arrays is None else arrays
    dims = {k: ("surface",) for k in arrays} if dims is None else dims
    return Result("0123456789abcdef", "unit", "scan", arrays, dims, kw.pop("metadata", {}), **kw)


def test_construction_rejects_bad_schema_rank_and_missing_dimensions() -> None:
    with pytest.raises(ValueError, match="schema 2 is not supported"):
        _result(schema_version=2)
    with pytest.raises(ValueError, match="dimensions missing"):
        _result({"a": np.zeros(3)}, {})
    with pytest.raises(ValueError, match="rank 1 does not match"):
        _result({"a": np.zeros(3)}, {"a": ("x", "y")})


def test_lookup_and_the_operator_bridge_of_a_loaded_result() -> None:
    res = _result({"a": np.arange(3.0)})
    assert res["a"] is res.a and not res.a.flags.writeable
    with pytest.raises(AttributeError):
        res.missing  # noqa: B018
    with pytest.raises(AttributeError):
        res.operator  # noqa: B018
    assert _result(_runtime={"operator": "op"}).operator == "op"


def test_to_dict_and_certificate_are_json_ready() -> None:
    meta = {"converged": True, "timings_s": {"total": np.float64(1.5)}, "grid": (1, 2), "dkx_version": "x"}
    res = _result({"primal_residual": np.array([1e-12])}, {"primal_residual": ("k",)}, metadata=meta, warnings=["w"])
    out = res.to_dict()
    assert out["metadata"]["timings_s"] == {"total": 1.5} and out["metadata"]["grid"] == [1, 2]
    assert out["arrays"]["primal_residual"] == {"shape": [1], "dtype": "float64"} and out["output_path"] is None
    cert = res.certificate()
    assert cert["converged"] and cert["primal_residual"] == [1e-12] and cert["warnings"] == ["w"]
    assert "grid" not in cert


def test_summary_reports_ambipolar_evidence(capsys) -> None:
    arrays = {
        "ambipolar_status": np.array(["bracketed_root", "no_root"]),
        "ambipolar_search_scope": np.array(["explicit_seeded_intervals_only", "scan"]),
        "ambipolar_refinement_status": np.array(["resolved", "refinement_exhausted"]),
        "ambipolar_root_branch_id": np.array([["ion", ""], ["electron", ""]]),
        "ambipolar_branch_event_kind": np.array(["boundary_origin", "fold"]),
    }
    res = _result(arrays, {k: ("surface",) if v.ndim == 1 else ("surface", "root") for k, v in arrays.items()},
                  metadata={"ambipolar_solver_attempts": {"attempt_count": 5, "automatic_true_residual_recovery_count": 1}})
    res.print_summary()
    out = capsys.readouterr().out
    assert "1/2 surfaces bracketed" in out
    assert "1 resolved, 1 refinement exhausted, 0 no bracket observed, 0 not requested" in out
    assert "2 identities; 1 interior events" in out
    assert "5 total; 1 automatic" in out
    unrequested = _result({"ambipolar_refinement_status": np.array(["not_requested"] * 2)}, {"ambipolar_refinement_status": ("surface",)})
    unrequested.print_summary()
    assert "adaptive evidence: not requested" in capsys.readouterr().out
    counted = _result({"ambipolar_status": np.array(["x"]), "ambipolar_root_count": np.array([2])},
                      {"ambipolar_status": ("surface",), "ambipolar_root_count": ("surface",)}, warnings=["slow"])
    counted.print_summary()
    assert "warning: slow" in capsys.readouterr().out


def test_save_refuses_bad_targets_and_inconsistent_dimensions(tmp_path) -> None:
    res = _result({"a": np.zeros(3), "b": np.zeros(2)}, {"a": ("x",), "b": ("x",)})
    with pytest.raises(ValueError, match="needs a path"):
        res.save()
    with pytest.raises(ValueError, match="use a .nc suffix"):
        res.save(tmp_path / "r.txt")
    with pytest.raises(ValueError, match="inconsistent sizes 3 and 2"):
        res.save(tmp_path / "r.nc")
    good = _result({"a": np.zeros(3), "s": np.array(["p", "q", "r"])})
    path = good.save(tmp_path / "r.nc")
    with pytest.raises(FileExistsError):
        good.save(path, overwrite=False)
    assert Result.load(path).arrays["s"].tolist() == ["p", "q", "r"]


def test_load_refuses_another_schema_version(tmp_path) -> None:
    from netCDF4 import Dataset

    path = _result().save(tmp_path / "r.nc")
    with Dataset(str(path), "a") as handle:
        handle.setncattr("dkx_result_schema", RESULT_SCHEMA_VERSION + 1)
    with pytest.raises(ValueError, match="needs an explicit migration"):
        Result.load(path)


def test_plot_titles_name_every_unresolved_ambipolar_surface(tmp_path) -> None:
    n = ("surface",)
    arrays = {
        "surface": np.array([0.1, 0.4, 0.9]),
        "electric_field_kV_m": np.array([-1.0, 0.5, 2.0]),
        "particle_flux_m2_s": np.ones((3, 2)),
        "species": np.array(["e", "i"]),
        "ambipolar_status": np.array(["no_root", "seeded_bracket_partial_failure", "bracketed_root"]),
        "ambipolar_refinement_status": np.array(["refinement_exhausted", "no_bracket_observed", "resolved"]),
        "ambipolar_search_scope": np.array(["explicit_seeded_intervals_only"] * 3),
        "ambipolar_nonsmooth_event": np.array([False, True, False]),
        "ambipolar_root_kV_m": np.array([[np.nan, np.nan], [0.5, 1.0], [2.0, np.nan]]),
        "ambipolar_root_branch_id": np.array([["", ""], ["ion", "electron"], ["ion", ""]]),
    }
    dims = {k: n if v.ndim == 1 else ("surface", "k") for k, v in arrays.items()}
    res = _result(arrays, dims)
    figure = res.plot()
    title = figure.texts[0].get_text()
    for fragment in ("no bracketed root", "seeded brackets failed", "refinement exhausted", "no bracket observed",
                     "seeded intervals only", "branch event warning"):
        assert fragment in title
    assert res.plot(tmp_path / "p.png").exists()
    with pytest.raises(ValueError, match="panels='auto'"):
        res.plot(panels="all")
    with pytest.raises(ValueError, match="no auto-plot observables"):
        _result().plot()
