from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_fortran_jax_operator.py"
_SPEC = importlib.util.spec_from_file_location("audit_fortran_jax_operator", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = audit
_SPEC.loader.exec_module(audit)


def test_parse_petsc_matlab_sparse_and_vector(tmp_path: Path) -> None:
    mat_path = tmp_path / "matrix.m"
    mat_path.write_text(
        "\n".join(
            [
                "%Mat Object: matrix 1 MPI process",
                "% Size = 2 2",
                "% Nonzeros = 3",
                "zzz = zeros(3,3);",
                "zzz = [",
                "1 1  2.0",
                "1 2  -1.0",
                "2 2  3.0",
                "];",
                " matrix = spconvert(zzz);",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    vec_path = tmp_path / "stateVector.m"
    vec_path.write_text(
        "\n".join(
            [
                "%Vec Object: stateVector 1 MPI process",
                "stateVector = [",
                "1.0",
                "-2.0D+00",
                "];",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    matrix = audit.parse_petsc_matlab_sparse(mat_path)
    vector = audit.parse_petsc_matlab_vector(vec_path)

    assert matrix.shape == (2, 2)
    assert matrix.nnz == 3
    assert np.asarray(matrix @ vector).tolist() == pytest.approx([4.0, -6.0])
    assert audit.summarize_vector(np.asarray([0.0, -3.0])).argmax_1based == 2


def test_solve_sparse_system_report_returns_true_residual() -> None:
    matrix = audit.sp.csr_matrix(np.asarray([[2.0, -1.0], [0.0, 3.0]], dtype=np.float64))
    rhs = np.asarray([4.0, -6.0], dtype=np.float64)
    reference = np.asarray([1.0, -2.0], dtype=np.float64)

    report = audit.solve_sparse_system_report(matrix, rhs, reference_state=reference)

    assert report["true_residual"]["norm2"] == pytest.approx(0.0, abs=1.0e-14)
    assert report["state_difference"]["norm2"] == pytest.approx(0.0, abs=1.0e-14)
    assert report["elapsed_s"] >= 0.0
