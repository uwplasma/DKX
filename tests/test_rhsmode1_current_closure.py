from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.io import read_sfincs_h5


RHS_MODE1_CURRENT_FIXTURES = (
    "output_scheme4_2species_quick",
    "pas_1species_PAS_noEr_tiny_scheme1",
    "pas_1species_PAS_noEr_tiny_withPhi1_linear",
    "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision",
)


def _as_species_iteration_array(value: object, *, n_species: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1 and arr.size == n_species:
        return arr.reshape((n_species, 1))
    if arr.ndim == 1 and n_species == 1:
        return arr.reshape((1, arr.size))
    if arr.ndim == 2 and arr.shape[0] == n_species:
        return arr
    raise AssertionError(f"unexpected species/iteration shape {arr.shape} for {n_species} species")


def _as_iteration_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape((-1,))


@pytest.mark.parametrize("base", RHS_MODE1_CURRENT_FIXTURES)
def test_rhsmode1_bootstrap_current_is_charge_weighted_parallel_flow(base: str) -> None:
    """SFINCS RHSMode=1 current diagnostics must be moment closures, not fitted values."""

    data = read_sfincs_h5(Path(__file__).parent / "ref" / f"{base}.sfincsOutput.h5")
    n_species = int(np.asarray(data["Nspecies"]).reshape(-1)[0])
    charges = np.asarray(data["Zs"], dtype=np.float64).reshape((n_species,))
    flow = _as_species_iteration_array(data["FSABFlow"], n_species=n_species)
    current = _as_iteration_array(data["FSABjHat"])

    expected_current = np.sum(charges[:, None] * flow, axis=0)
    np.testing.assert_allclose(current, expected_current, rtol=0.0, atol=5.0e-12)

    if "FSABjHatOverB0" in data and "B0OverBBar" in data:
        b0 = float(np.asarray(data["B0OverBBar"], dtype=np.float64).reshape(-1)[0])
        assert np.isfinite(b0)
        assert abs(b0) > 0.0
        np.testing.assert_allclose(_as_iteration_array(data["FSABjHatOverB0"]), current / b0, rtol=0.0, atol=5.0e-12)

    if "FSABjHatOverRootFSAB2" in data and "FSABHat2" in data:
        fsab2 = float(np.asarray(data["FSABHat2"], dtype=np.float64).reshape(-1)[0])
        assert np.isfinite(fsab2)
        assert fsab2 > 0.0
        np.testing.assert_allclose(
            _as_iteration_array(data["FSABjHatOverRootFSAB2"]),
            current / np.sqrt(fsab2),
            rtol=0.0,
            atol=5.0e-12,
        )
