"""Build a :class:`~dkx.drift_kinetic.KineticOperator` from a SFINCS v3 namelist.

Reads the geometry (analytic, Boozer or VMEC), species, resolution and drive
settings, and returns the operator with the context a run needs alongside it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
from typing import Any, NamedTuple

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from dkx.collisions import (  # noqa: E402
    make_fokker_planck_v3_operator,
    make_fokker_planck_v3_phi1_operator,
    make_improved_sugama_v3_operator,
    make_pitch_angle_scattering_v3_operator,
    resolve_rosenbluth_method,
)
from dkx.constants import (  # noqa: E402
    DEFAULT_DELTA,
    DEFAULT_NU_N,
    RadialCoordinates,
    d_phi_hat_d_psi_hat_from_e_star,
    nu_n_from_nu_prime,
)
from dkx.input_compat import (  # noqa: E402
    effective_equilibrium_file,
    effective_psi_a_hat,
    effective_psi_n_wish,
    infer_phi_input_radial_coordinate_for_gradients,
    infer_species_input_radial_coordinate_for_gradients,
)
from dkx.magnetic_geometry import (  # noqa: E402
    FluxSurfaceGeometry,
    psi_a_hat_from_wout,
    read_boozer_bc,
    read_vmec_wout,
    selected_r_n_from_bc,
    vmec_radial_interpolation,
)
from dkx.drift_kinetic import KineticOperator  # noqa: E402
from dkx.paths import repository_root, resolve_existing_path  # noqa: E402
from dkx.phase_space import (  # noqa: E402
    Grids,
    legendre_coupling_lower,
    legendre_coupling_upper,
    make_grids,
)
from dkx.species import (  # noqa: E402
    SpeciesSet,
    species_set_from_namelist,
)

# =============================================================================
# Small namelist-access helpers (same conventions as readInput.F90 defaults).
# =============================================================================

def _get_int(group: dict, key: str, default: int) -> int:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return int(v)

def _get_float(group: dict, key: str, default: float) -> float:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return float(v)

def _get_bool(group: dict, key: str, default: bool = False) -> bool:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return bool(v)

def _get_str_or_none(group: dict, key: str) -> str | None:
    """A namelist string value, or ``None`` when the key is absent or blank."""
    v = group.get(key.upper(), None)
    if isinstance(v, list):
        v = v[0] if v else None
    if v is None:
        return None
    text = str(v).strip()
    return text or None

# =============================================================================
# Namelist construction (readInput.F90 defaults; createGrids.F90 overrides)
# =============================================================================

def _flux_surface_averages_effective(
    *, grids: Grids, geom: FluxSurfaceGeometry
) -> tuple[float, float, float]:
    """Effective ``(B0OverBBar, GHat, IHat)`` from surface averages.

    For VMEC geometry (scheme 5) the flux functions are stored as 0.0
    placeholders and v3 recomputes them in ``computeBIntegrals``:
    ``B0OverBBar = <B^3>/<B^2>``, ``GHat = <B_sub_zeta>``, ``IHat = <B_sub_theta>``
    (the latter two averaged over 4*pi^2 with the quadrature weights).
    """
    w = np.asarray(grids.theta_weights)[:, None] * np.asarray(grids.zeta_weights)[None, :]
    d_hat = np.asarray(geom.d_hat)
    b_hat = np.asarray(geom.b_hat)
    vprime = float(np.sum(w / d_hat))
    fsab2 = float(np.sum(w * b_hat**2 / d_hat) / vprime)
    b0_eff = float(np.sum(w * b_hat**3 / d_hat) / (vprime * fsab2))
    denom = 4.0 * math.pi * math.pi
    g_eff = float(np.sum(w * np.asarray(geom.b_hat_sub_zeta)) / denom)
    i_eff = float(np.sum(w * np.asarray(geom.b_hat_sub_theta)) / denom)
    return b0_eff, g_eff, i_eff

def _geometry_and_radial(
    *, nml: Any, grids: Grids, compute_grad_psi_dot_grad_b: bool = False
) -> tuple[FluxSurfaceGeometry, RadialCoordinates]:
    """Geometry + radial-coordinate conversions for the supported geometrySchemes.

    ``compute_grad_psi_dot_grad_b`` requests the Sugama normal-curvature factor
    ``gradpsidotgradB_overgpsipsi`` (magneticDriftScheme 5/6), available for the
    file-based geometries (5/11/12) only.
    """
    geom_params = nml.group("geometryParameters")
    phys = nml.group("physicsParameters")
    scheme = _get_int(geom_params, "geometryScheme", -1)

    if scheme == 1:
        psi_a_hat = effective_psi_a_hat(geom_params=geom_params, phys_params=phys, default=0.15596)
        a_hat = _get_float(geom_params, "aHat", 0.5585)
        psi_n_wish = effective_psi_n_wish(
            geom_params=geom_params, default_r_n=0.5, psi_a_hat=psi_a_hat, a_hat=a_hat
        )
        r_n = math.sqrt(float(psi_n_wish))
        geom = FluxSurfaceGeometry.from_scheme(
            1,
            theta=grids.theta,
            zeta=grids.zeta,
            epsilon_t=_get_float(geom_params, "epsilon_t", -0.07053),
            epsilon_h=_get_float(geom_params, "epsilon_h", 0.05067),
            epsilon_antisymm=_get_float(geom_params, "epsilon_antisymm", 0.0),
            iota=_get_float(geom_params, "iota", 0.4542),
            g_hat=_get_float(geom_params, "GHat", 3.7481),
            i_hat=_get_float(geom_params, "IHat", 0.0),
            b0_over_bbar=_get_float(geom_params, "B0OverBBar", 1.0),
            helicity_l=_get_int(geom_params, "helicity_l", 2),
            helicity_n=_get_int(geom_params, "helicity_n", 10),
            helicity_antisymm_l=_get_int(geom_params, "helicity_antisymm_l", 1),
            helicity_antisymm_n=_get_int(geom_params, "helicity_antisymm_n", 0),
        )
        return geom, RadialCoordinates(psi_a_hat=float(psi_a_hat), a_hat=float(a_hat), r_n=r_n)

    if scheme == 2:
        # v3 fixed simplified LHD model; rN forced to 0.5.
        a_hat = 0.5585
        return (
            FluxSurfaceGeometry.from_scheme(2, theta=grids.theta, zeta=grids.zeta),
            RadialCoordinates(psi_a_hat=(a_hat * a_hat) / 2.0, a_hat=a_hat, r_n=0.5),
        )

    if scheme == 3:
        # v3 fixed LHD inward-shifted analytic Boozer model (geometry.F90 case 3);
        # aHat=0.5400, psiAHat=aHat^2/2, rN forced to 0.5 (rN_wish ignored).  The
        # B-field harmonics/flux functions come from from_scheme(3).
        a_hat = 0.5400
        return (
            FluxSurfaceGeometry.from_scheme(3, theta=grids.theta, zeta=grids.zeta),
            RadialCoordinates(psi_a_hat=(a_hat * a_hat) / 2.0, a_hat=a_hat, r_n=0.5),
        )

    if scheme == 4:
        # v3 built-in W7-X standard model; rN forced to 0.5.
        return (
            FluxSurfaceGeometry.from_scheme(4, theta=grids.theta, zeta=grids.zeta),
            RadialCoordinates(psi_a_hat=-0.384935, a_hat=0.5109, r_n=0.5),
        )

    if scheme in {11, 12}:
        path = _resolve_equilibrium_path(nml=nml, geom_params=geom_params)
        header, _surfaces = read_boozer_bc(path, geometry_scheme=scheme)
        psi_n_wish = effective_psi_n_wish(geom_params=geom_params, default_r_n=0.5)
        r_n_wish = math.sqrt(float(psi_n_wish))
        vmec_radial_option = _get_int(geom_params, "VMECRadialOption", 1)
        r_n = float(
            selected_r_n_from_bc(
                path=str(path),
                geometry_scheme=scheme,
                r_n_wish=r_n_wish,
                vmec_radial_option=vmec_radial_option,
            )
        )
        geom = FluxSurfaceGeometry.from_boozer(
            path,
            theta=grids.theta,
            zeta=grids.zeta,
            r_n_wish=r_n_wish,
            vmec_radial_option=vmec_radial_option,
            geometry_scheme=scheme,
            compute_grad_psi_dot_grad_b=compute_grad_psi_dot_grad_b,
        )
        return geom, RadialCoordinates(psi_a_hat=float(header.psi_a_hat), a_hat=float(header.a_hat), r_n=r_n)

    if scheme == 5:
        path = _resolve_equilibrium_path(nml=nml, geom_params=geom_params, vmec=True)
        w = read_vmec_wout(path)
        psi_a_hat = float(psi_a_hat_from_wout(w))
        a_hat = float(w.aminor_p)
        psi_n_wish = effective_psi_n_wish(
            geom_params=geom_params, default_r_n=0.5, psi_a_hat=psi_a_hat, a_hat=a_hat
        )
        vmec_radial_option = _get_int(geom_params, "VMECRadialOption", 1)
        interp = vmec_radial_interpolation(
            w=w, psi_n_wish=float(psi_n_wish), vmec_radial_option=vmec_radial_option
        )
        r_n = float(interp.psi_n) ** 0.5
        geom = FluxSurfaceGeometry.from_vmec(
            w,
            theta=grids.theta,
            zeta=grids.zeta,
            psi_n_wish=float(psi_n_wish),
            vmec_radial_option=vmec_radial_option,
            vmec_nyquist_option=_get_int(geom_params, "VMEC_NYQUIST_OPTION", 1),
            min_bmn_to_load=_get_float(geom_params, "MIN_BMN_TO_LOAD", 0.0),
            ripple_scale=_get_float(geom_params, "RIPPLESCALE", 1.0),
            helicity_n=_get_int(geom_params, "HELICITY_N", 0),
            helicity_l=_get_int(geom_params, "HELICITY_L", 0),
            compute_grad_psi_dot_grad_b=compute_grad_psi_dot_grad_b,
        )
        return geom, RadialCoordinates(psi_a_hat=psi_a_hat, a_hat=a_hat, r_n=r_n)

    if scheme == 13:
        # Namelist Boozer |B| spectrum (geometry.F90 case 13, the STELLOPT/BMNC
        # optimization path).  ``boozer_bmnc(m,n)`` / ``boozer_bmns(m,n)`` are 2-D
        # indexed arrays in geometryParameters; NPeriods, psiAHat, aHat, iota,
        # GHat, IHat are read from the namelist.  The field is analytic
        # (nearbyRadiiGiven=.false., radial derivatives 0, rN = rN_wish), so the
        # radial coordinate mirrors the scheme-1 analytic pattern.
        psi_a_hat = effective_psi_a_hat(geom_params=geom_params, phys_params=phys, default=0.15596)
        a_hat = _get_float(geom_params, "aHat", 0.5585)
        psi_n_wish = effective_psi_n_wish(
            geom_params=geom_params, default_r_n=0.5, psi_a_hat=psi_a_hat, a_hat=a_hat
        )
        r_n = math.sqrt(float(psi_n_wish))
        idx = nml.indexed.get("geometryparameters", {})
        bmnc_map = idx.get("BOOZER_BMNC", {})
        bmns_map = idx.get("BOOZER_BMNS", {})
        if not bmnc_map and not bmns_map:
            raise ValueError(
                "geometryScheme=13 requires at least one boozer_bmnc/boozer_bmns amplitude "
                "in geometryParameters (validateInput.F90)."
            )
        # from_fourier takes 1-D (amp, m, n) arrays with the (0,0) mode included
        # (it extracts B0OverBBar = bmnc(0,0) internally).  When a sine spectrum is
        # present, align it to the same (m,n) order as the cosine spectrum by
        # unioning the mode keys and zero-filling both sides.
        if bmns_map:
            keys = sorted(set(bmnc_map) | set(bmns_map))
            bmnc = jnp.asarray([float(bmnc_map.get(k, 0.0)) for k in keys], dtype=jnp.float64)
            bmns = jnp.asarray([float(bmns_map.get(k, 0.0)) for k in keys], dtype=jnp.float64)
        else:
            keys = sorted(bmnc_map)
            bmnc = jnp.asarray([float(bmnc_map[k]) for k in keys], dtype=jnp.float64)
            bmns = None
        m_arr = jnp.asarray([int(k[0]) for k in keys])
        n_arr = jnp.asarray([int(k[1]) for k in keys])
        geom = FluxSurfaceGeometry.from_fourier(
            theta=grids.theta,
            zeta=grids.zeta,
            bmnc=bmnc,
            m=m_arr,
            n=n_arr,
            bmns=bmns,
            n_periods=max(1, _get_int(geom_params, "Nperiods", 1)),
            iota=_get_float(geom_params, "iota", 0.4542),
            g_hat=_get_float(geom_params, "GHat", 3.7481),
            i_hat=_get_float(geom_params, "IHat", 0.0),
        )
        return geom, RadialCoordinates(psi_a_hat=float(psi_a_hat), a_hat=float(a_hat), r_n=r_n)

    raise NotImplementedError(
        f"KineticOperator.from_namelist supports geometryScheme in {{1,2,3,4,5,11,12,13}}; got {scheme}."
    )

def _resolve_equilibrium_path(*, nml: Any, geom_params: dict, vmec: bool = False) -> Path:
    eq = effective_equilibrium_file(geom_params=geom_params)
    if eq is None:
        raise ValueError("This geometryScheme requires equilibriumFile in geometryParameters.")
    base_dir = nml.source_path.parent if nml.source_path is not None else None
    repo_root = repository_root()
    extra = (
        (repo_root / "tests" / "ref", repo_root / "src" / "dkx" / "data" / "equilibria")
        if repo_root is not None
        else ()
    )
    try:
        return resolve_existing_path(str(eq), base_dir=base_dir, extra_search_dirs=extra).path
    except FileNotFoundError:
        if vmec:
            # Allow `.txt -> .nc` fallback for VMEC wout files (matches the old reader).
            p2 = Path(str(eq).strip().strip('"').strip("'")).with_suffix(".nc")
            return resolve_existing_path(str(p2), base_dir=base_dir, extra_search_dirs=extra).path
        raise

def _n_periods_from_namelist(*, nml: Any) -> int:
    """NPeriods for grid construction (createGrids.F90 / geometry.F90)."""
    geom_params = nml.group("geometryParameters")
    scheme = _get_int(geom_params, "geometryScheme", -1)
    if scheme == 1:
        return max(1, _get_int(geom_params, "helicity_n", 10))
    if scheme in {2, 3}:
        return 10
    if scheme == 4:
        return 5
    if scheme in {11, 12}:
        path = _resolve_equilibrium_path(nml=nml, geom_params=geom_params)
        header, _ = read_boozer_bc(path, geometry_scheme=scheme)
        return int(header.n_periods)
    if scheme == 5:
        path = _resolve_equilibrium_path(nml=nml, geom_params=geom_params, vmec=True)
        return int(read_vmec_wout(path).nfp)
    if scheme == 13:
        # geometry.F90 case 13: NPeriods is read from the namelist.
        return max(1, _get_int(geom_params, "Nperiods", 1))
    raise NotImplementedError(
        f"KineticOperator.from_namelist supports geometryScheme in {{1,2,3,4,5,11,12,13}}; got {scheme}."
    )

def _load_external_phi1(*, nml: Any, phys: dict, grids: Grids) -> jnp.ndarray:
    """Read the FIXED external Phi1(theta,zeta) field for ``readExternalPhi1``.

    Reuses :func:`dkx.io.read_sfincs_h5` (the ``sfincsOutput.h5`` reader) on
    ``externalPhi1Filename`` (default ``externalPhi1.h5``, resolved relative to the
    input deck's directory), takes the LAST ``NIterations`` slice, and returns it
    as a ``(Ntheta, Nzeta)`` array (readHDF5Input.F90:264-320; SFINCS stores
    ``Phi1Hat`` as ``(Nzeta, Ntheta, NIterations)``).  The external (theta, zeta)
    grid must equal the run grid; external-to-run grid interpolation is a
    documented follow-up.
    """
    from dkx.io import read_sfincs_h5  # noqa: PLC0415

    raw = phys.get("EXTERNALPHI1FILENAME", "externalPhi1.h5")
    if isinstance(raw, list):
        raw = raw[0] if raw else "externalPhi1.h5"
    filename = str(raw).strip().strip('"').strip("'")
    base_dir = nml.source_path.parent if nml.source_path is not None else None
    repo_root = repository_root()
    extra = (repo_root / "tests" / "ref",) if repo_root is not None else ()
    path = resolve_existing_path(filename, base_dir=base_dir, extra_search_dirs=extra).path

    data = read_sfincs_h5(path)
    if "Phi1Hat" not in data:
        raise ValueError(f"external Phi1 file {path} has no Phi1Hat dataset.")
    phi1 = np.asarray(data["Phi1Hat"], dtype=np.float64)
    if phi1.ndim == 3:  # (Nzeta, Ntheta, NIterations) -> last iteration
        phi1 = phi1[..., -1]
    phi1 = phi1.T  # (Ntheta, Nzeta)
    if phi1.shape != (grids.n_theta, grids.n_zeta):
        raise ValueError(
            f"external Phi1Hat has shape {phi1.shape}; run grid is "
            f"{(grids.n_theta, grids.n_zeta)}.  External-to-run grid interpolation is a "
            "documented follow-up; make the external grid equal the run grid."
        )
    for name, ext_key, run in (("theta", "theta", grids.theta), ("zeta", "zeta", grids.zeta)):
        if ext_key in data and not np.allclose(
            np.asarray(data[ext_key], dtype=np.float64), np.asarray(run, dtype=np.float64), rtol=0.0, atol=1e-10
        ):
            raise NotImplementedError(
                f"external Phi1 {name} grid differs from the run grid; external-to-run grid "
                "interpolation is a documented follow-up (make the external grid equal the run grid)."
            )
    return jnp.asarray(phi1, dtype=jnp.float64)

class KineticOperatorBuild(NamedTuple):
    """One operator build together with the grids/geometry it was derived from.

    The single-build context of :func:`kinetic_operator_build_from_namelist`:
    downstream consumers (the run drivers, the writer, diagnostics) share these
    objects instead of re-deriving them from the namelist.

    Attributes:
        operator: the assembled :class:`KineticOperator`.
        grids: the :class:`dkx.phase_space.Grids` the operator uses
            (theta/zeta nodes and quadrature weights are not stored on the
            operator itself).
        geometry: the full :class:`dkx.magnetic_geometry.FluxSurfaceGeometry`
            (radial-derivative fields included).
        radial: the :class:`dkx.constants.RadialCoordinates` conversions
            for the selected flux surface.
    """

    operator: KineticOperator
    grids: Grids
    geometry: FluxSurfaceGeometry
    radial: RadialCoordinates

def kinetic_operator_from_namelist(nml: Any) -> KineticOperator:
    """Build a :class:`KineticOperator` from a parsed SFINCS input namelist.

    Thin facade over :func:`kinetic_operator_build_from_namelist` returning the
    operator only; callers that also need the grids, geometry, or radial
    conversions use the build function to avoid re-deriving them.
    """
    return kinetic_operator_build_from_namelist(nml).operator

def kinetic_operator_build_from_namelist(nml: Any) -> KineticOperatorBuild:
    """Build a :class:`KineticOperator` plus its grids/geometry/radial context.

    ``nml`` is a :class:`dkx.namelist.Namelist`.  Grids come from
    :func:`dkx.phase_space.make_grids`, geometry from
    :class:`dkx.magnetic_geometry.FluxSurfaceGeometry`, species from
    :func:`dkx.species.species_set_from_namelist`, radial-coordinate
    and monoenergetic conversions from :mod:`dkx.constants`, and
    collision matrices from :mod:`dkx.collisions`.  The returned
    :class:`KineticOperatorBuild` carries the grids, flux-surface geometry, and
    radial conversions alongside the operator so one build serves the solve,
    diagnostics, and output-writing consumers.

    Supports ``geometryScheme`` in {1, 2, 3, 4, 5, 11, 12, 13} (analytic schemes
    1/2/3/4, file-based 5/11/12, and the namelist Boozer |B| spectrum 13) and
    every ``magneticDriftScheme`` 0-9 (the tangential magnetic drifts need a
    geometryScheme in {5, 11, 12} that carries the radial B-field derivatives;
    scheme 4 is geometryScheme 11/12 only, as in validateInput.F90).  Raises
    ``NotImplementedError`` for the deferred features listed in the module
    docstring (Fokker-Planck collisions on the uniform/Chebyshev x grids).
    """
    general = nml.group("general")
    phys = nml.group("physicsParameters")
    other = nml.group("otherNumericalParameters")
    res = nml.group("resolutionParameters")
    geom_params = nml.group("geometryParameters")

    # ---- Phi1 / quasineutrality configuration (includePhi1 vertical slice) ----
    species_params = nml.group("speciesParameters")
    include_phi1_input = _get_bool(phys, "includePhi1", False)
    read_external_phi1 = include_phi1_input and _get_bool(phys, "readExternalPhi1", False)
    # ``readExternalPhi1`` holds Phi1 fixed (read from an external file), so the
    # DKE stays LINEAR and the state is f-only: the operator's ``include_phi1``
    # (the QN block + Phi1 unknown + lambda row) is off, but the Phi1-in-kinetic /
    # Phi1-in-collision term coefficients still evaluate at the external field.
    include_phi1 = include_phi1_input and not read_external_phi1
    # Default TRUE, matching version3/globalVariables.F90:152 (and dkx's own
    # SfincsInput field, which already carried the right default).  Defaulting
    # it off silently dropped the Phi1-in-kinetic coupling from every deck that
    # enabled Phi1 without naming this switch, which costs O(Phi1^2): the term
    # is Phi1 times f, and f itself carries an O(Phi1) response.
    include_phi1_in_kinetic = include_phi1_input and _get_bool(
        phys, "includePhi1InKineticEquation", True
    )
    include_phi1_in_collision = include_phi1_input and _get_bool(phys, "includePhi1InCollisionOperator", False)
    if include_phi1_in_collision and not include_phi1_in_kinetic:
        raise NotImplementedError(
            "includePhi1InCollisionOperator=.true. requires includePhi1InKineticEquation=.true. "
            "(populateMatrix.F90 assembles the Phi1-in-collision densities alongside the "
            "Phi1-in-kinetic couplings)."
        )
    quasineutrality_option = _get_int(phys, "quasineutralityOption", 1)
    if include_phi1 and quasineutrality_option not in {1, 2}:
        raise NotImplementedError(
            f"quasineutralityOption={quasineutrality_option} is not supported "
            "(only 1 (full) and 2 (EUTERPE) are consolidated)."
        )
    # withAdiabatic only enters quasineutrality; for readExternalPhi1 (no QN) it is
    # physically inert but still echoed to output, so track the namelist flag.
    with_adiabatic = include_phi1_input and _get_bool(species_params, "withAdiabatic", False)
    adiabatic_z = _get_float(species_params, "adiabaticZ", 1.0)
    adiabatic_n_hat = _get_float(species_params, "adiabaticNHat", 0.0)
    adiabatic_t_hat = _get_float(species_params, "adiabaticTHat", 1.0)

    magnetic_drift_scheme = _get_int(phys, "magneticDriftScheme", 0)
    if not 0 <= magnetic_drift_scheme <= 9:
        # validateInput.F90: "magneticDriftScheme must be >= 0" / "<= 9".
        raise ValueError(f"magneticDriftScheme must be between 0 and 9 (got {magnetic_drift_scheme}).")
    with_magnetic_drifts = magnetic_drift_scheme > 0
    if with_magnetic_drifts:
        md_geometry_scheme = _get_int(geom_params, "geometryScheme", -1)
        if md_geometry_scheme not in {5, 11, 12}:
            raise NotImplementedError(
                f"magneticDriftScheme={magnetic_drift_scheme} requires a geometryScheme carrying "
                "the radial (psi) B-field derivatives; only 5 (VMEC), 11, and 12 (Boozer .bc) are "
                f"consolidated (Fortran validateInput allows 5/6/7/11/12), got {md_geometry_scheme}."
            )
        if magnetic_drift_scheme == 4 and md_geometry_scheme not in {11, 12}:
            # validateInput.F90: "magneticDriftScheme 4 has only been implemented
            # for geometryScheme 11 and 12."
            raise ValueError(
                "magneticDriftScheme 4 has only been implemented for geometryScheme 11 and 12 "
                f"(got geometryScheme={md_geometry_scheme})."
            )
        # validateInput.F90 allows geometryScheme 5/6/11/12 for schemes 5 and 6;
        # the {5, 11, 12} gate above already enforces the consolidated subset.
    collision_operator = _get_int(phys, "collisionOperator", 0)
    if collision_operator not in {0, 1, 3}:
        raise NotImplementedError(f"collisionOperator={collision_operator} is not supported.")

    rhs_mode = _get_int(general, "RHSMode", 1)
    constraint_scheme = _get_int(phys, "constraintScheme", -1)
    if constraint_scheme < 0:
        # collisionOperator=3 (improved Sugama) acts on the L=0 block through its
        # energy-diffusion + energy/particle field terms, so its speed null space
        # is {density, temperature} per species (2 sources) like the Fokker-Planck
        # operator -- not the per-speed PAS null space (constraintScheme=2).
        constraint_scheme = 1 if collision_operator in {0, 3} else 2
    if constraint_scheme not in {0, 1, 2, 3, 4}:
        raise NotImplementedError(f"constraintScheme={constraint_scheme} is not supported.")

    x_grid_scheme = _get_int(other, "xGridScheme", 5)
    if not 1 <= x_grid_scheme <= 8:
        raise ValueError(f"xGridScheme must be between 1 and 8 (got {x_grid_scheme}).")
    if collision_operator == 0 and x_grid_scheme in {3, 4, 7, 8}:
        raise NotImplementedError(
            f"collisionOperator=0 with xGridScheme={x_grid_scheme} is not supported: the "
            "Fokker-Planck Rosenbluth-potential interpolation matrices for the uniform/"
            "Chebyshev speed grids (interpolationMatrix.F90 / "
            "ChebyshevInterpolationMatrix.F90) are not ported."
        )
    x_dot_derivative_scheme = _get_int(other, "xDotDerivativeScheme", 0)
    point_at_x0 = x_grid_scheme in {2, 3, 4, 6, 7, 8}

    # ---- grids (phase_space) ----
    grids = make_grids(
        n_theta=_get_int(res, "Ntheta", 15),
        n_zeta=_get_int(res, "Nzeta", 15),
        n_xi=_get_int(res, "Nxi", 16),
        n_x=_get_int(res, "Nx", 5),
        n_l=_get_int(res, "NL", 4),
        n_periods=_n_periods_from_namelist(nml=nml),
        theta_derivative_scheme=_get_int(other, "thetaDerivativeScheme", 2),
        zeta_derivative_scheme=_get_int(other, "zetaDerivativeScheme", 2),
        magnetic_drift_derivative_scheme=_get_int(other, "magneticDriftDerivativeScheme", 3),
        x_grid_scheme=x_grid_scheme,
        x_grid_k=_get_float(other, "xGrid_k", 0.0),
        x_max=_get_float(res, "xMax", 5.0),
        x_dot_derivative_scheme=x_dot_derivative_scheme,
        n_xi_for_x_option=_get_int(other, "Nxi_for_x_option", 1),
        monoenergetic=(rhs_mode == 3),
    )

    # ---- readExternalPhi1: read the FIXED external Phi1(theta,zeta) field ----
    external_phi1_hat = _load_external_phi1(nml=nml, phys=phys, grids=grids) if read_external_phi1 else None

    # ---- geometry + radial conversions (magnetic_geometry / constants) ----
    geom, radial = _geometry_and_radial(
        nml=nml,
        grids=grids,
        compute_grad_psi_dot_grad_b=magnetic_drift_scheme in {5, 6},
    )
    fsab_hat2 = float(geom.fsab_hat2(theta_weights=grids.theta_weights, zeta_weights=grids.zeta_weights))

    # ---- species (species) ----
    species_grad_coord = infer_species_input_radial_coordinate_for_gradients(
        geom_params=geom_params, species_params=nml.group("speciesParameters"), default=4
    )
    species: SpeciesSet = species_set_from_namelist(
        nml, radial=radial, input_radial_coordinate_for_gradients=species_grad_coord
    )
    n_species = species.n_species

    alpha = _get_float(phys, "alpha", 1.0)
    delta = _get_float(phys, "Delta", DEFAULT_DELTA)
    nu_n = _get_float(phys, "nu_n", DEFAULT_NU_N)
    krook = _get_float(phys, "Krook", 0.0)
    er = _get_float(phys, "Er", 0.0)

    # ---- flux functions (effective values for VMEC placeholders) ----
    b0_eff = float(geom.b0_over_bbar)
    g_eff = float(geom.g_hat)
    i_eff = float(geom.i_hat)
    if abs(g_eff) < 1e-30 or abs(b0_eff) < 1e-30:
        b0_eff, g_eff, i_eff = _flux_surface_averages_effective(grids=grids, geom=geom)

    # ---- dPhiHat/dpsiHat: drive value (phi gradient coordinate) and kinetic value ----
    phi_grad_coord = infer_phi_input_radial_coordinate_for_gradients(
        geom_params=geom_params, phys_params=phys, default=4
    )
    if phi_grad_coord == 0:
        dphi_rhs = _get_float(phys, "dPhiHatdpsiHat", 0.0)
    elif phi_grad_coord == 1:
        dphi_rhs = radial.d_dpsi_n_to_d_dpsi_hat * _get_float(phys, "dPhiHatdpsiN", 0.0)
    elif phi_grad_coord == 2:
        dphi_rhs = radial.d_dr_hat_to_d_dpsi_hat * _get_float(phys, "dPhiHatdrHat", 0.0)
    elif phi_grad_coord == 3:
        dphi_rhs = radial.d_dr_n_to_d_dpsi_hat * _get_float(phys, "dPhiHatdrN", 0.0)
    elif phi_grad_coord == 4:
        dphi_rhs = radial.d_dr_hat_to_d_dpsi_hat * (-er)
    else:
        raise NotImplementedError(f"Unsupported inputRadialCoordinateForGradients={phi_grad_coord} for Phi.")

    if rhs_mode == 3:
        # sfincs_main.F90: EStar overwrites dPhiHatdpsiHat for monoenergetic runs.
        e_star = _get_float(phys, "EStar", 0.0)
        dphi_kinetic = d_phi_hat_d_psi_hat_from_e_star(
            e_star=e_star, alpha=alpha, delta=delta, iota=float(geom.iota), b0_over_bbar=b0_eff, g_hat=g_eff
        )
        dphi_rhs = dphi_kinetic
    else:
        # The ExB/Er kinetic terms follow the Er input (v3 examples convention).
        if er != 0.0 and phi_grad_coord != 4:
            raise NotImplementedError(
                "Er != 0 with a non-Er inputRadialCoordinateForGradients is not supported."
            )
        dphi_kinetic = radial.d_dr_hat_to_d_dpsi_hat * (-er)

    # These three defaults must match globalVariables.F90:144-146, because a
    # deck is free to omit the key and inherit it. includeXDotTerm and
    # includeElectricFieldTermInXiDot default to .true. upstream; defaulting
    # them to False here silently dropped both E_r trajectory terms from any
    # compact deck with a finite E_r, which is a wrong answer rather than a
    # missing feature. useDKESExBDrift really is .false. upstream.
    include_xdot = _get_bool(phys, "includeXDotTerm", True)
    include_er_xidot = _get_bool(phys, "includeElectricFieldTermInXiDot", True)
    use_dkes_exb = _get_bool(phys, "useDKESExBDrift", False)
    has_er = float(dphi_kinetic) != 0.0

    # ---- collisions (dkx.collisions builders) ----
    pas = None
    fp = None
    fp_phi1 = None
    sugama = None
    # &otherNumericalParameters RosenbluthMethod selects how the Fokker-Planck
    # Rosenbluth response matrices are integrated ('quadpack' for Fortran
    # parity, 'hybrid', or 'analytic').  Resolved eagerly so a mistyped deck
    # fails on any collisionOperator, not only the branches that consume it.
    rosenbluth_method = resolve_rosenbluth_method(_get_str_or_none(other, "RosenbluthMethod"))
    if include_phi1_in_collision and collision_operator != 0:
        raise NotImplementedError(
            "includePhi1InCollisionOperator=.true. requires collisionOperator=0 "
            "(the linearized Fokker-Planck operator carries the Phi1-shifted densities)."
        )
    if collision_operator == 3:
        # Improved Sugama momentum/energy-conserving model operator (research
        # extension beyond v3; Sugama et al. Phys. Plasmas 26, 102108 (2019);
        # Frei et al. arXiv:2202.06293).  Same block layout as the FP operator.
        sugama = make_improved_sugama_v3_operator(
            x=np.asarray(grids.x, dtype=np.float64),
            x_weights=np.asarray(grids.x_weights, dtype=np.float64),
            ddx=np.asarray(grids.ddx, dtype=np.float64),
            d2dx2=np.asarray(grids.d2dx2, dtype=np.float64),
            z_s=np.asarray(species.z, dtype=np.float64),
            m_hats=np.asarray(species.m_hat, dtype=np.float64),
            n_hats=np.asarray(species.n_hat, dtype=np.float64),
            t_hats=np.asarray(species.t_hat, dtype=np.float64),
            nu_n=nu_n,
            krook=krook,
            n_xi=grids.n_xi,
            n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=np.int32),
        )
    elif collision_operator == 1:
        nu_n_use = nu_n
        if rhs_mode == 3:
            nu_n_use = nu_n_from_nu_prime(
                nu_prime=_get_float(phys, "nuPrime", 1.0),
                b0_over_bbar=b0_eff,
                g_hat=g_eff,
                i_hat=i_eff,
                iota=float(geom.iota),
            )
        pas = make_pitch_angle_scattering_v3_operator(
            x=grids.x,
            z_s=species.z,
            m_hats=species.m_hat,
            n_hats=species.n_hat,
            t_hats=species.t_hat,
            nu_n=float(nu_n_use),
            krook=krook,
            n_xi_for_x=grids.n_xi_for_x,
            n_xi=grids.n_xi,
        )
    elif include_phi1_in_collision:
        # includePhi1InCollisionOperator=.true.: the poloidally varying FP
        # operator (collisions.FokkerPlanckV3Phi1Operator); the collision
        # densities are shifted by exp(-Z*alpha*Phi1Hat/THat) at apply time.
        fp_phi1 = make_fokker_planck_v3_phi1_operator(
            rosenbluth_method=rosenbluth_method,
            x=np.asarray(grids.x, dtype=np.float64),
            x_weights=np.asarray(grids.x_weights, dtype=np.float64),
            ddx=np.asarray(grids.ddx, dtype=np.float64),
            d2dx2=np.asarray(grids.d2dx2, dtype=np.float64),
            x_grid_k=_get_float(other, "xGrid_k", 0.0),
            z_s=np.asarray(species.z, dtype=np.float64),
            m_hats=np.asarray(species.m_hat, dtype=np.float64),
            n_hats=np.asarray(species.n_hat, dtype=np.float64),
            t_hats=np.asarray(species.t_hat, dtype=np.float64),
            nu_n=nu_n,
            krook=krook,
            n_xi=grids.n_xi,
            nl=grids.n_l,
            alpha=alpha,
            n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=np.int32),
        )
    else:
        import os  # noqa: PLC0415

        strict_env = os.environ.get("DKX_FP_STRICT_PARITY", "").strip().lower()
        if strict_env in {"0", "false", "no", "off"}:
            strict_parity = False
        elif strict_env in {"1", "true", "yes", "on"}:
            strict_parity = True
        else:
            strict_parity = bool(rhs_mode == 1 and n_species > 1)
        fp = make_fokker_planck_v3_operator(
            rosenbluth_method=rosenbluth_method,
            x=np.asarray(grids.x, dtype=np.float64),
            x_weights=np.asarray(grids.x_weights, dtype=np.float64),
            ddx=np.asarray(grids.ddx, dtype=np.float64),
            d2dx2=np.asarray(grids.d2dx2, dtype=np.float64),
            x_grid_k=_get_float(other, "xGrid_k", 0.0),
            z_s=np.asarray(species.z, dtype=np.float64),
            m_hats=np.asarray(species.m_hat, dtype=np.float64),
            n_hats=np.asarray(species.n_hat, dtype=np.float64),
            t_hats=np.asarray(species.t_hat, dtype=np.float64),
            nu_n=nu_n,
            krook=krook,
            n_xi=grids.n_xi,
            nl=grids.n_l,
            n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=np.int32),
            strict_parity=strict_parity,
        )

    # &otherNumericalParameters SfincsMatrixThreshold: a dkx parity switch, off by
    # default.  SFINCS v3 inserts every matrix entry through sparsify.F90, which
    # skips |value| <= 1d-12; on hot-electron decks that deletes part of the
    # ion->electron field-particle collision block, and the electron flow and
    # bootstrap current follow it (2026-09-13 sfincs-sparsify-threshold study).  Applied to the dense Fokker-Planck matrix, where the
    # difference has been measured; physics results keep every entry.
    sfincs_threshold = _get_float(other, "SfincsMatrixThreshold", 0.0)
    if not math.isfinite(sfincs_threshold) or sfincs_threshold < 0.0:
        raise ValueError(
            f"SfincsMatrixThreshold must be finite and non-negative, got {sfincs_threshold!r}."
        )
    if sfincs_threshold > 0.0:
        if fp is None:
            raise ValueError(
                "SfincsMatrixThreshold applies to collisionOperator=0 without "
                "includePhi1InCollisionOperator; this deck has no Fokker-Planck "
                "matrix to threshold."
            )
        fp = replace(fp, mat=jnp.where(jnp.abs(fp.mat) > sfincs_threshold, fp.mat, 0.0))

    # ---- inductive E_parallel ----
    e_parallel_hat = _get_float(phys, "EParallelHat", 0.0)
    epar_spec_raw = phys.get("EPARALLELHATSPEC", None)
    if epar_spec_raw is None:
        e_parallel_hat_spec = jnp.zeros((n_species,), dtype=jnp.float64)
    else:
        e_parallel_hat_spec = jnp.atleast_1d(jnp.asarray(epar_spec_raw, dtype=jnp.float64))
        if e_parallel_hat_spec.shape == (1,) and n_species > 1:
            e_parallel_hat_spec = jnp.broadcast_to(e_parallel_hat_spec, (n_species,))
        if e_parallel_hat_spec.shape != (n_species,):
            raise ValueError(
                f"EParallelHatSpec must have {n_species} entries, got {e_parallel_hat_spec.shape}"
            )

    # Tangential magnetic-drift geometry + upwinded stencils (magneticDriftScheme 1-9).
    # The radial (psi) B-field derivatives come straight from the Boozer/VMEC
    # geometry; the plus/minus stencils implement magneticDriftDerivativeScheme.
    # The flux-function scalars feed the scheme 3/4/8 shear terms and the scheme 6
    # pressure term; the Sugama normal-curvature factor feeds schemes 5/6.
    if with_magnetic_drifts:
        magnetic_drift_arrays: dict[str, Any] = {
            "b_hat_sub_psi": geom.b_hat_sub_psi,
            "db_hat_dpsi_hat": geom.db_hat_dpsi_hat,
            "db_hat_sub_psi_dtheta": geom.db_hat_sub_psi_dtheta,
            "db_hat_sub_psi_dzeta": geom.db_hat_sub_psi_dzeta,
            "db_hat_sub_theta_dpsi_hat": geom.db_hat_sub_theta_dpsi_hat,
            "db_hat_sub_zeta_dpsi_hat": geom.db_hat_sub_zeta_dpsi_hat,
            "ddtheta_magdrift_plus": grids.ddtheta_magdrift_plus,
            "ddtheta_magdrift_minus": grids.ddtheta_magdrift_minus,
            "ddzeta_magdrift_plus": grids.ddzeta_magdrift_plus,
            "ddzeta_magdrift_minus": grids.ddzeta_magdrift_minus,
            "iota": jnp.asarray(geom.iota, dtype=jnp.float64),
            "g_hat": jnp.asarray(g_eff, dtype=jnp.float64),
            "diota_dpsi_hat": jnp.asarray(geom.diota_dpsi_hat, dtype=jnp.float64),
            "p_prime_hat": jnp.asarray(geom.p_prime_hat, dtype=jnp.float64),
            "grad_psi_dot_grad_b_over_gpsipsi": geom.grad_psi_dot_grad_b_over_gpsipsi,
        }
    else:
        magnetic_drift_arrays = {}

    op = KineticOperator(
        n_species=n_species,
        n_x=grids.n_x,
        n_xi=grids.n_xi,
        n_theta=grids.n_theta,
        n_zeta=grids.n_zeta,
        rhs_mode=rhs_mode,
        constraint_scheme=constraint_scheme,
        point_at_x0=point_at_x0,
        use_dkes_exb=use_dkes_exb,
        with_exb=has_er,
        with_er_xidot=bool(include_er_xidot and has_er),
        with_er_xdot=bool(include_xdot and has_er),
        x=grids.x,
        x_weights=grids.x_weights,
        ddx=grids.ddx,
        ddtheta=grids.ddtheta,
        ddzeta=grids.ddzeta,
        theta_weights=grids.theta_weights,
        zeta_weights=grids.zeta_weights,
        n_xi_for_x=grids.n_xi_for_x,
        xi_coupling_lower=jnp.asarray(legendre_coupling_lower(grids.n_xi)),
        xi_coupling_upper=jnp.asarray(legendre_coupling_upper(grids.n_xi)),
        b_hat=geom.b_hat,
        db_hat_dtheta=geom.db_hat_dtheta,
        db_hat_dzeta=geom.db_hat_dzeta,
        d_hat=geom.d_hat,
        b_hat_sup_theta=geom.b_hat_sup_theta,
        b_hat_sup_zeta=geom.b_hat_sup_zeta,
        b_hat_sub_theta=geom.b_hat_sub_theta,
        b_hat_sub_zeta=geom.b_hat_sub_zeta,
        fsab_hat2=jnp.asarray(fsab_hat2, dtype=jnp.float64),
        z_s=species.z,
        m_hat=species.m_hat,
        t_hat=species.t_hat,
        n_hat=species.n_hat,
        dn_hat_dpsi_hat=species.dn_hat_dpsi_hat,
        dt_hat_dpsi_hat=species.dt_hat_dpsi_hat,
        alpha=jnp.asarray(alpha, dtype=jnp.float64),
        delta=jnp.asarray(delta, dtype=jnp.float64),
        dphi_hat_dpsi_hat=jnp.asarray(dphi_rhs, dtype=jnp.float64),
        dphi_hat_dpsi_hat_kinetic=jnp.asarray(dphi_kinetic, dtype=jnp.float64),
        e_parallel_hat=jnp.asarray(e_parallel_hat, dtype=jnp.float64),
        e_parallel_hat_spec=e_parallel_hat_spec,
        pas=pas,
        fp=fp,
        sugama=sugama,
        fp_phi1=fp_phi1,
        include_phi1=bool(include_phi1),
        quasineutrality_option=int(quasineutrality_option),
        include_phi1_in_kinetic=bool(include_phi1_in_kinetic),
        with_adiabatic=bool(with_adiabatic),
        adiabatic_z=jnp.asarray(adiabatic_z, dtype=jnp.float64),
        adiabatic_n_hat=jnp.asarray(adiabatic_n_hat, dtype=jnp.float64),
        adiabatic_t_hat=jnp.asarray(adiabatic_t_hat, dtype=jnp.float64),
        phi1_hat_base=(
            jnp.zeros((grids.n_theta, grids.n_zeta), dtype=jnp.float64) if include_phi1 else None
        ),
        phi1_lin_state=None,
        external_phi1_hat=external_phi1_hat,
        ddx_xdot_plus=grids.ddx_xdot_plus,
        ddx_xdot_minus=grids.ddx_xdot_minus,
        with_magnetic_drifts=with_magnetic_drifts,
        magnetic_drift_scheme=magnetic_drift_scheme if with_magnetic_drifts else 0,
        **magnetic_drift_arrays,
    )
    return KineticOperatorBuild(operator=op, grids=grids, geometry=geom, radial=radial)
