"""Canonical solver-trace sidecar parity for the run drivers.

``dkx write-output --solver-trace`` now emits the sidecar from the
canonical run drivers (:func:`dkx.run.run_profile` /
:func:`dkx.run.run_transport_matrix`) instead of falling back to the
retired legacy pipeline.  The canonical trace uses
the same :class:`dkx.solver_trace.SolverTrace` schema; the
solver-independent fields (backend, ``rhs_mode``, ``selected_path``,
``geometry_scheme``, ``collision_operator``, sizes, ``device_count``,
convergence, residual target) match the legacy trace, while the retired-GMRES
solver-implementation fields (``solve_method``, per-phase timings) are allowed
to differ.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dkx.api import write_output
from dkx.run import run_profile, run_transport_matrix
from dkx.solver_trace import read_solver_trace_json

REF = Path(__file__).parent / "ref"

_MATCHING_FIELDS = (
    "backend",
    "rhs_mode",
    "selected_path",
    "geometry_scheme",
    "collision_operator",
    "total_size",
    "active_size",
    "device_count",
)


@pytest.mark.parametrize(
    "base, rhs_mode",
    (
        ("pas_1species_PAS_noEr_tiny_scheme1", 1),
        ("transportMatrix_PAS_tiny_rhsMode2_scheme2", 2),
    ),
)
def test_canonical_solver_trace_matches_legacy(base: str, rhs_mode: int, tmp_path: Path) -> None:
    input_path = REF / f"{base}.input.namelist"
    canonical_trace = tmp_path / "canonical_trace.json"
    legacy_trace = tmp_path / "legacy_trace.json"

    if rhs_mode == 1:
        run_profile(input_path, out_path=tmp_path / "c.h5", solver_trace_path=canonical_trace, emit=None)
    else:
        run_transport_matrix(
            input_path, out_path=tmp_path / "c.h5", solver_trace_path=canonical_trace, emit=None
        )
    write_output(input_path, tmp_path / "l.h5", solver_trace_path=legacy_trace)

    canonical = read_solver_trace_json(canonical_trace)
    legacy = read_solver_trace_json(legacy_trace)

    for field in _MATCHING_FIELDS:
        assert getattr(canonical, field) == getattr(legacy, field), field

    # Both solvers converge below the (identical) residual target.
    assert canonical.converged is True
    assert legacy.converged is True
    assert canonical.residual_target == pytest.approx(legacy.residual_target, rel=1e-9)
    assert canonical.residual_norm is not None
    assert canonical.residual_norm <= canonical.residual_target

    # Provenance metadata records the canonical run intent.
    assert canonical.metadata["compute_solution"] is (rhs_mode == 1)
    assert canonical.metadata["compute_transport_matrix"] is (rhs_mode in (2, 3))
    assert canonical.metadata["differentiable"] is False
    assert canonical.metadata["input_namelist"] == str(input_path.resolve())


@pytest.mark.parametrize("suffix", [".h5", ".nc", ".npz"])
def test_geometry_only_trace_preserves_intent_without_claiming_a_solve(tmp_path: Path, suffix) -> None:
    import numpy as np

    from dkx.io import read_sfincs_output_file

    input_path = REF / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    output, trace_path = tmp_path / f"geometry{suffix}", tmp_path / "geometry.json"
    write_output(input_path, output, geometry_only=True, solver_trace_path=trace_path)
    trace = read_solver_trace_json(trace_path)
    data = read_sfincs_output_file(output)
    assert trace.selected_path == "geometry_only"
    assert trace.geometry_scheme == 1 and trace.rhs_mode == 1
    assert trace.total_size > 0 and 0 < trace.active_size <= trace.total_size
    assert trace.solve_method is None and trace.residual_norm is None and trace.converged is None
    assert not trace.metadata["compute_solution"] and not trace.metadata["compute_transport_matrix"]
    assert trace.metadata["output_path"] == str(output.resolve())
    assert trace.elapsed_s >= 0.0
    assert np.all(np.isfinite(data["BHat"])) and np.min(data["BHat"]) > 0.0
