from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_ENABLE_X64", "True")
from jax import config as jax_config

import pytest

jax_config.update("jax_enable_x64", True)


def _ensure_reference_goldens() -> None:
    """Materialise the lzma-compressed Fortran goldens before any test reads them.

    The heavy ``tests/ref`` binaries are committed only as ``*.xz`` (see
    ``tests/_golden_cache.py``); decompress them into place on session start so
    parity tests read byte-identical fixtures from a lightweight checkout.
    """

    import importlib.util

    helper = Path(__file__).resolve().parent / "_golden_cache.py"
    spec = importlib.util.spec_from_file_location("_sfincs_golden_cache", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ensure_decompressed()


def pytest_configure() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    os.environ.setdefault("SFINCS_JAX_FORTRAN_STDOUT", "0")
    _ensure_reference_goldens()


@pytest.fixture(autouse=True)
def _restore_sfincs_jax_environment():
    """Prevent solver-policy environment overrides from leaking across tests."""

    prefix = "SFINCS_JAX_"
    before = {key: value for key, value in os.environ.items() if key.startswith(prefix)}
    yield
    for key in [key for key in os.environ if key.startswith(prefix) and key not in before]:
        os.environ.pop(key, None)
    for key, value in before.items():
        os.environ[key] = value


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep CI runtime under control by skipping the slowest integration tests."""
    if os.environ.get("SFINCS_JAX_CI", "0") != "1":
        return

    # Skip only the heaviest end-to-end tests in CI; keep unit + fast parity checks.
    slow_mark = pytest.mark.skip(reason="Skipped slow integration test in CI mode.")
    slow_patterns = (
        "tests/test_transport_matrix_rhsmode2_parity.py::",
        "tests/test_transport_matrix_rhsmode3_parity.py::",
        "tests/test_transport_matrix_write_output_end_to_end.py::",
        "tests/test_transport_parallel.py::",
        "tests/test_output_h5_scheme11_parity.py::",
        "test_full_system_newton_krylov",
        "test_full_system_gmres_solution_parity",
    )
    for item in items:
        nodeid = item.nodeid
        if any(pat in nodeid for pat in slow_patterns):
            item.add_marker(slow_mark)
