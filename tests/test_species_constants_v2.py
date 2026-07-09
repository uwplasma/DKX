"""Referee tests for Phase 3.1c: constants_v2.py / species_v2.py == old code paths.

New modules must reproduce the existing scattered implementations to 1e-15:
- ``physics/collisions.py`` (nuDHat deflection frequency),
- ``outputs/transport.py`` (radial-coordinate conversion factors),
- ``operators/profile_system.py`` (namelist -> species arrays + psiHat gradients),
- ``operators/profile_fblock.py`` / ``outputs/writer.py`` (Fortran defaults),
- ``input_compat.py`` (gradient-coordinate inference).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax import constants_v2, species_v2
from sfincs_jax.input_compat import (
    infer_species_input_radial_coordinate_for_gradients,
    scheme4_radial_constants,
)
from sfincs_jax.namelist import parse_sfincs_input_text, read_sfincs_input
from sfincs_jax.operators.profile_fblock import _V3_DEFAULT_DELTA, _V3_DEFAULT_NU_N
from sfincs_jax.outputs.transport import conversion_factors_to_from_dpsi_hat
from sfincs_jax.physics.collisions import _V3_PI, _V3_SQRTPI, nu_d_hat_pitch_angle_scattering_v3

FORTRAN_EXAMPLE = (
    "/Users/rogerio/local/sfincs/fortran/version3/examples/"
    "quick_2species_FPCollisions_noEr/input.namelist"
)

# Species parameter sets with 1, 2, and 3 species.
SPECIES_CASES = {
    "1sp_hydrogen": dict(z=[1.0], m_hat=[1.0], n_hat=[1.0], t_hat=[1.0]),
    "2sp_example": dict(z=[1.0, 6.0], m_hat=[1.0, 6.0], n_hat=[0.6, 0.009], t_hat=[0.5, 0.8]),
    "3sp_with_electrons": dict(
        z=[1.0, 6.0, -1.0],
        m_hat=[1.0, 6.0, 5.446170214e-4],
        n_hat=[0.6, 0.009, 0.654],
        t_hat=[0.5, 0.8, 0.7],
    ),
}


def _species_set(case: dict, dn=None, dt=None) -> species_v2.SpeciesSet:
    s = len(case["z"])
    return species_v2.SpeciesSet(
        z=jnp.asarray(case["z"], dtype=jnp.float64),
        m_hat=jnp.asarray(case["m_hat"], dtype=jnp.float64),
        n_hat=jnp.asarray(case["n_hat"], dtype=jnp.float64),
        t_hat=jnp.asarray(case["t_hat"], dtype=jnp.float64),
        dn_hat_dpsi_hat=jnp.asarray(dn if dn is not None else [0.0] * s, dtype=jnp.float64),
        dt_hat_dpsi_hat=jnp.asarray(dt if dt is not None else [0.0] * s, dtype=jnp.float64),
    )


def test_defaults_match_fortran_globalvariables() -> None:
    # globalVariables.F90 lines 133-135 (via operators/profile_fblock.py literals).
    assert constants_v2.DEFAULT_DELTA == _V3_DEFAULT_DELTA == 4.5694e-3
    assert constants_v2.DEFAULT_NU_N == _V3_DEFAULT_NU_N == 8.330e-3
    assert constants_v2.DEFAULT_ALPHA == 1.0
    # globalVariables.F90 lines 16-17 (via physics/collisions.py literals).
    assert constants_v2.PI_V3 == _V3_PI
    assert constants_v2.SQRT_PI_V3 == _V3_SQRTPI
    # globalVariables.F90 line 155.
    assert constants_v2.DEFAULT_NU_PRIME == 1.0
    assert constants_v2.DEFAULT_E_STAR == 0.0
    # globalVariables.F90 line 97 (adiabatic defaults; also outputs/writer.py).
    ad = species_v2.AdiabaticSpecies()
    assert (ad.z, ad.m_hat, ad.n_hat, ad.t_hat) == (-1.0, 5.446170214e-4, 1.0, 1.0)


@pytest.mark.parametrize(
    ("psi_a_hat", "a_hat", "r_n"),
    [
        (-0.384935, 0.5109, 0.5),  # geometryScheme=4 constants
        (0.15596, 0.5585, 0.25),  # geometryScheme=1 defaults, off-mid radius
        (2.3, 1.7, 0.9),
    ],
)
def test_radial_conversions_match_old_transport_helper(psi_a_hat: float, a_hat: float, r_n: float) -> None:
    old = conversion_factors_to_from_dpsi_hat(psi_a_hat=psi_a_hat, a_hat=a_hat, r_n=r_n)
    new = constants_v2.RadialCoordinates(psi_a_hat=psi_a_hat, a_hat=a_hat, r_n=r_n)
    assert new.d_dpsi_n_to_d_dpsi_hat == old["ddpsiN2ddpsiHat"]
    assert new.d_dr_hat_to_d_dpsi_hat == old["ddrHat2ddpsiHat"]
    assert new.d_dr_n_to_d_dpsi_hat == old["ddrN2ddpsiHat"]
    assert new.d_dpsi_hat_to_d_dpsi_n == old["ddpsiHat2ddpsiN"]
    assert new.d_dpsi_hat_to_d_dr_hat == old["ddpsiHat2ddrHat"]
    assert new.d_dpsi_hat_to_d_dr_n == old["ddpsiHat2ddrN"]
    # Flux-surface labels (radialCoordinates.F90 lines 143-145).
    assert new.psi_n == r_n * r_n
    assert new.psi_hat == psi_a_hat * r_n * r_n
    assert new.r_hat == a_hat * r_n


@pytest.mark.parametrize("case_name", sorted(SPECIES_CASES))
def test_nu_d_hat_matches_physics_collisions(case_name: str) -> None:
    case = SPECIES_CASES[case_name]
    x = jnp.asarray([0.05, 0.35, 0.9, 1.7, 2.4, 4.1], dtype=jnp.float64)
    old = np.asarray(
        nu_d_hat_pitch_angle_scattering_v3(
            x=x,
            z_s=jnp.asarray(case["z"], dtype=jnp.float64),
            m_hats=jnp.asarray(case["m_hat"], dtype=jnp.float64),
            n_hats=jnp.asarray(case["n_hat"], dtype=jnp.float64),
            t_hats=jnp.asarray(case["t_hat"], dtype=jnp.float64),
        )
    )
    new = np.asarray(_species_set(case).nu_d_hat(x))
    np.testing.assert_allclose(new, old, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("case_name", sorted(SPECIES_CASES))
def test_gradient_variants_match_writer_arithmetic(case_name: str) -> None:
    case = SPECIES_CASES[case_name]
    s = len(case["z"])
    rng = np.random.default_rng(20260709)
    dn = rng.normal(size=s)
    dt = rng.normal(size=s)
    sset = _species_set(case, dn=dn, dt=dt)
    radial = constants_v2.RadialCoordinates(psi_a_hat=-0.384935, a_hat=0.5109, r_n=0.5)
    conv = conversion_factors_to_from_dpsi_hat(psi_a_hat=-0.384935, a_hat=0.5109, r_n=0.5)
    # outputs/writer.py: dnHatd{psiN,rHat,rN} = ddpsiHat2dd* * dnHatdpsiHat.
    for grads, base in ((sset.density_gradients(radial), dn), (sset.temperature_gradients(radial), dt)):
        np.testing.assert_allclose(np.asarray(grads.d_dpsi_hat), base, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(np.asarray(grads.d_dpsi_n), conv["ddpsiHat2ddpsiN"] * base, rtol=1e-15)
        np.testing.assert_allclose(np.asarray(grads.d_dr_hat), conv["ddpsiHat2ddrHat"] * base, rtol=1e-15)
        np.testing.assert_allclose(np.asarray(grads.d_dr_n), conv["ddpsiHat2ddrN"] * base, rtol=1e-15)


def _nml_text(species_block: str, geometry_block: str = "geometryScheme = 4") -> str:
    return f"""
&geometryParameters
  {geometry_block}
/
&speciesParameters
{species_block}
/
&physicsParameters
/
"""


@pytest.mark.parametrize(
    "species_block",
    [
        "  Zs = 1\n  mHats = 1\n  nHats = 1\n  THats = 1\n  dNHatdrHats = -0.5\n  dTHatdrHats = -0.3",
        "  Zs = 1 6\n  mHats = 1 6\n  nHats = 0.6 0.009\n  THats = 0.5 0.8\n"
        "  dNHatdpsiNs = -0.3 -0.001\n  dTHatdpsiNs = -0.3 -0.2",
        "  Zs = 1 6 -1\n  mHats = 1 6 5.446170214d-4\n  nHats = 0.6 0.009 0.654\n"
        "  THats = 0.5 0.8 0.7\n  dNHatdrNs = -0.5 -0.01 -0.56\n  dTHatdrNs = -0.4 -0.3 -0.2",
        "  Zs = 1 -1\n  mHats = 1 5.446170214d-4\n  nHats = 1 1\n  THats = 1 1\n"
        "  dNHatdpsiHats = -0.7 -0.7\n  dTHatdpsiHats = -0.2 -0.1",
        "  Zs = 1",  # readInput.F90 defaults: unit species, zero gradients
    ],
)
def test_namelist_species_match_old_writer_path(species_block: str) -> None:
    nml = parse_sfincs_input_text(_nml_text(species_block))
    species = nml.group("speciesParameters")
    geom_params = nml.group("geometryParameters")

    psi_a_hat, a_hat = scheme4_radial_constants()
    r_n = 0.5  # v3 forces rN=0.5 for geometryScheme=4
    radial = constants_v2.RadialCoordinates(psi_a_hat=psi_a_hat, a_hat=a_hat, r_n=r_n)
    new = species_v2.species_set_from_namelist(nml, radial=radial)

    # Old path: input_compat inference + outputs/transport conversion factors,
    # exactly as outputs/writer.py and operators/profile_system.py compute them.
    coord_old = infer_species_input_radial_coordinate_for_gradients(
        geom_params=geom_params, species_params=species, default=4
    )
    assert (
        species_v2.infer_gradient_coordinate(geom_params=geom_params, species_params=species) == coord_old
    )
    conv = conversion_factors_to_from_dpsi_hat(psi_a_hat=psi_a_hat, a_hat=a_hat, r_n=r_n)
    factor_old = {
        0: 1.0,
        1: conv["ddpsiN2ddpsiHat"],
        2: conv["ddrHat2ddpsiHat"],
        3: conv["ddrN2ddpsiHat"],
        4: conv["ddrHat2ddpsiHat"],
    }[coord_old]
    suffix = {0: "PSIHATS", 1: "PSINS", 2: "RHATS", 3: "RNS", 4: "RHATS"}[coord_old]

    def _old_arr(key: str, default: float) -> np.ndarray:
        v = species.get(key, default)
        return np.atleast_1d(np.asarray(v, dtype=np.float64))

    s = _old_arr("ZS", 1.0).size
    np.testing.assert_allclose(np.asarray(new.z), _old_arr("ZS", 1.0), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(new.m_hat), _old_arr("MHATS", 1.0), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(new.n_hat), _old_arr("NHATS", 1.0), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(new.t_hat), _old_arr("THATS", 1.0), rtol=0.0, atol=0.0)
    dn_old = factor_old * np.broadcast_to(_old_arr(f"DNHATD{suffix}", 0.0), (s,))
    dt_old = factor_old * np.broadcast_to(_old_arr(f"DTHATD{suffix}", 0.0), (s,))
    np.testing.assert_allclose(np.asarray(new.dn_hat_dpsi_hat), dn_old, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(np.asarray(new.dt_hat_dpsi_hat), dt_old, rtol=1e-15, atol=0.0)


def test_fortran_example_namelist_matches_full_system_operator() -> None:
    """Parity vs the heaviest old path: operators/profile_system full-system build."""
    from sfincs_jax.operators.profile_system import full_system_operator_from_namelist

    nml = read_sfincs_input(FORTRAN_EXAMPLE)
    op = full_system_operator_from_namelist(nml=nml)

    psi_a_hat, a_hat = scheme4_radial_constants()
    radial = constants_v2.RadialCoordinates(psi_a_hat=psi_a_hat, a_hat=a_hat, r_n=0.5)
    new = species_v2.species_set_from_namelist(nml, radial=radial)

    assert new.n_species == 2
    np.testing.assert_allclose(np.asarray(new.z), np.asarray(op.z_s), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(new.m_hat), np.asarray(op.m_hat), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(new.n_hat), np.asarray(op.n_hat), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(new.t_hat), np.asarray(op.t_hat), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(new.dn_hat_dpsi_hat), np.asarray(op.dn_hat_dpsi_hat), rtol=1e-15, atol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(new.dt_hat_dpsi_hat), np.asarray(op.dt_hat_dpsi_hat), rtol=1e-15, atol=0.0
    )
    # Physics-parameter values from the same namelist (Delta/alpha explicit in the file).
    assert float(op.delta) == 4.5694e-3
    assert float(op.alpha) == constants_v2.DEFAULT_ALPHA
    # No adiabatic species in this example.
    assert species_v2.adiabatic_species_from_namelist(nml) is None

    # Derived collisionality on the operator's own x grid, new vs physics/ code.
    x = jnp.asarray(op.x, dtype=jnp.float64)
    old_nu = np.asarray(
        nu_d_hat_pitch_angle_scattering_v3(
            x=x, z_s=op.z_s, m_hats=op.m_hat, n_hats=op.n_hat, t_hats=op.t_hat
        )
    )
    np.testing.assert_allclose(np.asarray(new.nu_d_hat(x)), old_nu, rtol=1e-15, atol=0.0)


def test_rhs_mode3_nu_n_and_phi_gradient_match_writer_formulas() -> None:
    # sfincs_main.F90 lines 153-154, replicated in outputs/writer.py's RHSMode=3 branch.
    nu_prime, e_star = 0.7, 0.3
    b0, g_hat, i_hat, iota = 1.1, 3.7481, 0.05, 0.4542
    alpha, delta = 1.0, 4.5694e-3
    nu_n_old = float(nu_prime) * float(b0) / (float(g_hat) + float(iota) * float(i_hat))
    dphi_old = 2.0 / (alpha * delta) * e_star * iota * b0 / g_hat
    assert (
        constants_v2.nu_n_from_nu_prime(nu_prime=nu_prime, b0_over_bbar=b0, g_hat=g_hat, i_hat=i_hat, iota=iota)
        == nu_n_old
    )
    assert (
        constants_v2.d_phi_hat_d_psi_hat_from_e_star(
            e_star=e_star, alpha=alpha, delta=delta, iota=iota, b0_over_bbar=b0, g_hat=g_hat
        )
        == dphi_old
    )
    # Round trip.
    assert constants_v2.nu_prime_from_nu_n(
        nu_n=nu_n_old, b0_over_bbar=b0, g_hat=g_hat, i_hat=i_hat, iota=iota
    ) == pytest.approx(nu_prime, rel=1e-15)


def test_adiabatic_species_from_namelist_defaults_and_overrides() -> None:
    nml = parse_sfincs_input_text(_nml_text("  Zs = 1\n  withAdiabatic = .true."))
    ad = species_v2.adiabatic_species_from_namelist(nml)
    assert ad is not None
    # Defaults from globalVariables.F90 line 97 (same literals as outputs/writer.py).
    assert (ad.z, ad.m_hat, ad.n_hat, ad.t_hat) == (-1.0, 5.446170214e-4, 1.0, 1.0)

    nml2 = parse_sfincs_input_text(
        _nml_text("  Zs = 1\n  withAdiabatic = .true.\n  adiabaticZ = -2.0\n  adiabaticNHat = 0.5")
    )
    ad2 = species_v2.adiabatic_species_from_namelist(nml2)
    assert ad2 is not None and ad2.z == -2.0 and ad2.n_hat == 0.5 and ad2.t_hat == 1.0


def test_species_set_is_a_pytree_and_v_hat() -> None:
    import jax

    sset = _species_set(SPECIES_CASES["2sp_example"])
    leaves = jax.tree_util.tree_leaves(sset)
    assert len(leaves) == 6
    rebuilt = jax.tree_util.tree_map(lambda a: a, sset)
    np.testing.assert_allclose(np.asarray(rebuilt.z), np.asarray(sset.z))
    # v_hat = sqrt(THat/mHat), the `sqrt_t_over_m` factor used across operators/.
    np.testing.assert_allclose(
        np.asarray(sset.v_hat),
        np.sqrt(np.asarray(sset.t_hat) / np.asarray(sset.m_hat)),
        rtol=0.0,
        atol=0.0,
    )
