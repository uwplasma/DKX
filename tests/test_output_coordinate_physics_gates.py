from __future__ import annotations

from pathlib import Path

import numpy as np

from sfincs_jax.io import read_sfincs_h5


def _quick_2species_output() -> dict[str, object]:
    repo = Path(__file__).resolve().parents[1]
    return read_sfincs_h5(repo / "tests" / "ref" / "output_scheme4_2species_quick.sfincsOutput.h5")


def test_output_flux_coordinate_conversions_match_radial_normalization() -> None:
    """Regression gate for radial-flux normalization used in SFINCS-style plots.

    The neoclassical literature commonly plots radial particle, heat, and momentum
    fluxes against several radial coordinates. These arrays must differ only by
    the derivative chain-rule factors between ``psiHat``, ``psiN``, ``rHat``, and
    ``rN``; otherwise cross-code comparisons and ambipolar-root plots become
    silently inconsistent.
    """
    data = _quick_2species_output()
    psi_a_hat = float(np.asarray(data["psiAHat"]))
    a_hat = float(np.asarray(data["aHat"]))
    r_n = float(np.asarray(data["rN"]))
    root = np.sqrt(r_n * r_n)
    ddpsi_n_to_ddpsi_hat = 1.0 / psi_a_hat
    ddr_hat_to_ddpsi_hat = a_hat / (2.0 * psi_a_hat * root)
    ddr_n_to_ddpsi_hat = 1.0 / (2.0 * psi_a_hat * root)

    for base in ("particleFlux_vm", "heatFlux_vm", "momentumFlux_vm"):
        flux = np.asarray(data[f"{base}_psiHat"], dtype=np.float64)
        np.testing.assert_allclose(
            np.asarray(data[f"{base}_psiN"], dtype=np.float64),
            flux * ddpsi_n_to_ddpsi_hat,
            rtol=0.0,
            atol=5e-13,
        )
        np.testing.assert_allclose(
            np.asarray(data[f"{base}_rHat"], dtype=np.float64),
            flux * ddr_hat_to_ddpsi_hat,
            rtol=0.0,
            atol=5e-13,
        )
        np.testing.assert_allclose(
            np.asarray(data[f"{base}_rN"], dtype=np.float64),
            flux * ddr_n_to_ddpsi_hat,
            rtol=0.0,
            atol=5e-13,
        )


def test_output_profile_gradient_conversions_match_radial_normalization() -> None:
    data = _quick_2species_output()
    psi_a_hat = float(np.asarray(data["psiAHat"]))
    a_hat = float(np.asarray(data["aHat"]))
    r_n = float(np.asarray(data["rN"]))
    root = np.sqrt(r_n * r_n)
    ddpsi_hat_to_ddpsi_n = psi_a_hat
    ddpsi_hat_to_ddr_hat = 2.0 * psi_a_hat * root / a_hat
    ddpsi_hat_to_ddr_n = 2.0 * psi_a_hat * root

    for base in ("dnHatd", "dTHatd"):
        grad = np.asarray(data[f"{base}psiHat"], dtype=np.float64)
        np.testing.assert_allclose(
            np.asarray(data[f"{base}psiN"], dtype=np.float64),
            grad * ddpsi_hat_to_ddpsi_n,
            rtol=0.0,
            atol=5e-8,
        )
        np.testing.assert_allclose(
            np.asarray(data[f"{base}rHat"], dtype=np.float64),
            grad * ddpsi_hat_to_ddr_hat,
            rtol=0.0,
            atol=5e-8,
        )
        np.testing.assert_allclose(
            np.asarray(data[f"{base}rN"], dtype=np.float64),
            grad * ddpsi_hat_to_ddr_n,
            rtol=0.0,
            atol=5e-8,
        )
