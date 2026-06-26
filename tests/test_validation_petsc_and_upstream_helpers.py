from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest

from sfincs_jax.validation.petsc_binary import read_petsc_mat_aij, read_petsc_vec
from sfincs_jax.workflows.postprocess_upstream import find_upstream_utils_dir, run_upstream_util


def test_petsc_vec_reader_roundtrips_big_endian_fixture(tmp_path: Path) -> None:
    path = tmp_path / "vec.dat"
    np.asarray([1211214, 3], dtype=">i4").tofile(path)
    with path.open("ab") as stream:
        np.asarray([1.5, -2.0, 4.25], dtype=">f8").tofile(stream)

    vec = read_petsc_vec(path)

    assert vec.size == 3
    np.testing.assert_allclose(vec.values, [1.5, -2.0, 4.25])


def test_petsc_vec_reader_rejects_short_and_wrong_class_files(tmp_path: Path) -> None:
    short = tmp_path / "short.vec"
    short.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="too small"):
        read_petsc_vec(short)

    wrong = tmp_path / "wrong.vec"
    np.asarray([999, 0], dtype=">i4").tofile(wrong)
    with pytest.raises(ValueError, match="Unexpected PETSc Vec classid"):
        read_petsc_vec(wrong)


def test_petsc_aij_reader_roundtrips_sorted_csr_fixture(tmp_path: Path) -> None:
    path = tmp_path / "mat.dat"
    np.asarray([1211216, 2, 3, 3], dtype=">i4").tofile(path)
    with path.open("ab") as stream:
        np.asarray([2, 1], dtype=">i4").tofile(stream)
        np.asarray([0, 2, 1], dtype=">i4").tofile(stream)
        np.asarray([1.0, 3.0, -2.0], dtype=">f8").tofile(stream)

    mat = read_petsc_mat_aij(path)

    assert mat.shape == (2, 3)
    np.testing.assert_array_equal(mat.row_ptr, [0, 2, 3])
    np.testing.assert_array_equal(mat.col_ind, [0, 2, 1])
    np.testing.assert_allclose(mat.data, [1.0, 3.0, -2.0])
    assert mat.get(0, 0) == pytest.approx(1.0)
    assert mat.get(0, 1) == pytest.approx(0.0)
    assert mat.get(1, 1) == pytest.approx(-2.0)


def test_petsc_aij_reader_rejects_bad_headers_and_row_counts(tmp_path: Path) -> None:
    short = tmp_path / "short.mat"
    short.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="too small"):
        read_petsc_mat_aij(short)

    wrong = tmp_path / "wrong.mat"
    np.asarray([999, 0, 0, 0], dtype=">i4").tofile(wrong)
    with pytest.raises(ValueError, match="Unexpected PETSc Mat classid"):
        read_petsc_mat_aij(wrong)

    bad_rows = tmp_path / "bad_rows.mat"
    np.asarray([1211216, 2, 2, 3], dtype=">i4").tofile(bad_rows)
    with bad_rows.open("ab") as stream:
        np.asarray([1, 1], dtype=">i4").tofile(stream)
        np.asarray([0, 1, 1], dtype=">i4").tofile(stream)
        np.asarray([1.0, 2.0, 3.0], dtype=">f8").tofile(stream)
    with pytest.raises(ValueError, match="row pointers do not sum"):
        read_petsc_mat_aij(bad_rows)


def test_find_upstream_utils_dir_resolves_override_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    utils = tmp_path / "utils"
    utils.mkdir()
    assert find_upstream_utils_dir(override=utils) == utils

    with pytest.raises(FileNotFoundError, match="utils dir does not exist"):
        find_upstream_utils_dir(override=tmp_path / "missing")

    monkeypatch.setenv("SFINCS_JAX_UPSTREAM_UTILS_DIR", str(utils))
    assert find_upstream_utils_dir() == utils

    monkeypatch.setenv("SFINCS_JAX_UPSTREAM_UTILS_DIR", str(tmp_path / "missing_env"))
    with pytest.raises(FileNotFoundError, match="SFINCS_JAX_UPSTREAM_UTILS_DIR"):
        find_upstream_utils_dir()


def test_run_upstream_util_executes_noninteractive_script(tmp_path: Path) -> None:
    utils = tmp_path / "utils"
    case_dir = tmp_path / "case"
    utils.mkdir()
    case_dir.mkdir()
    script = utils / "make_marker.py"
    script.write_text(
        "from pathlib import Path\n"
        "import builtins\n"
        "import os\n"
        "import sys\n"
        "Path('marker.txt').write_text('|'.join([os.environ['MPLBACKEND'], builtins.input('x'), *sys.argv[1:]]))\n",
        encoding="utf-8",
    )
    messages: list[str] = []

    run_upstream_util(
        util="make_marker.py",
        case_dir=case_dir,
        args=("a", "b"),
        utils_dir=utils,
        emit=lambda _level, message: messages.append(message),
    )

    assert (case_dir / "marker.txt").read_text(encoding="utf-8") == "Agg||a|b"
    assert messages and "make_marker.py" in messages[0]


def test_run_upstream_util_rejects_missing_script_or_case(tmp_path: Path) -> None:
    utils = tmp_path / "utils"
    case_dir = tmp_path / "case"
    utils.mkdir()
    case_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Upstream util not found"):
        run_upstream_util(util="missing.py", case_dir=case_dir, utils_dir=utils)

    script = utils / "ok.py"
    script.write_text("pass\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="case_dir does not exist"):
        run_upstream_util(util="ok.py", case_dir=tmp_path / "missing_case", utils_dir=utils)


def test_run_upstream_util_propagates_script_failure(tmp_path: Path) -> None:
    utils = tmp_path / "utils"
    case_dir = tmp_path / "case"
    utils.mkdir()
    case_dir.mkdir()
    script = utils / "fail.py"
    script.write_text("raise SystemExit(3)\n", encoding="utf-8")

    with pytest.raises(subprocess.CalledProcessError):
        run_upstream_util(util="fail.py", case_dir=case_dir, utils_dir=utils)
