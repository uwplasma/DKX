"""DKX reverse-mode sensitivities against DKX central finite differences, with a step sweep.

SFINCS's own RHSMode 4/5 adjoint differs from its finite-difference check by 0.05% to 66% on
W7-X, so it cannot be the reference. DKX replaces the mode by ``jax.grad`` through the implicit
solve; this script checks that derivative against central differences of the same DKX solve.

Two cases, both two-species (H+, e-) full linearized Fokker-Planck with a radial electric field:

- ``tokamak``: a circular large-aspect-ratio field, ``|B| = B00 + B10 cos(theta)``, parameters
  (B00, B10, iota, GHat, IHat);
- ``w7x``: the W7-X standard configuration at rN = 0.5, parameters the three largest Boozer |B|
  harmonics (0,1), (1,1), (1,0), iota, GHat and IHat.

Observables are the RHSMode 4/5 families: ion particle flux, ion heat flux, bootstrap current
``<j.B>`` and ion parallel flow. The step sweep is absolute, 1e-2 to 1e-5.

Run from the repository root:
  python tools/paper_benchmarks/adjoint_sensitivity_verification.py --out validation/adjoint_sensitivity_v1.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import dkx  # noqa: E402
from dkx.boozer_bc import read_boozer_bc  # noqa: E402
from dkx.run import profile_moments_from_operator  # noqa: E402
from dkx.solve import solve  # noqa: E402
from dkx.workflows.geometry_adapters import kinetic_operator_on_boozer_surface  # noqa: E402

STEPS = (1e-2, 1e-3, 1e-4, 1e-5)
OBSERVABLES = ("particleFlux_ion", "heatFlux_ion", "FSABjHat", "FSABFlow_ion")
SPECIES = dict(
    Zs=[1.0, -1.0], mHats=[1.0, 5.446170214e-4], nHats=[0.5, 0.5], THats=[1.0, 1.0],
    dNHatdrHats=[-0.3, -0.3], dTHatdrHats=[-1.0, -1.0], collisionOperator=0, Delta=4.5694e-3,
    alpha=1.0, nu_n=8.33e-3, inputRadialCoordinateForGradients=4,
)  # fmt: skip
SOLVER = dict(
    method="auto", tol=1e-10, max_restarts=60, restart=80, adjoint_residual_factor=1e3
)


def _template(geometry: dict, grid: dict):
    return dkx.run(**geometry, **SPECIES, **grid, emit=None).operator


def _observables(operator) -> jnp.ndarray:
    solved = solve(operator, operator.rhs(), differentiable=True, tier1_keep_lowest=operator.n_xi,
                   emit=None, **SOLVER)  # fmt: skip
    m = profile_moments_from_operator(operator, solved.x.reshape(-1))
    return jnp.stack([m["particleFlux_vm_psiHat"][0], m["heatFlux_vm_psiHat"][0],
                      m["FSABjHat"].reshape(()), m["FSABFlow"][0]])  # fmt: skip


def _tokamak():
    geometry = dict(geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3, epsilon_t=-0.1, epsilon_h=0.0,
                    iota=0.4, GHat=1.0, IHat=0.2, B0OverBBar=1.0, helicity_l=1, helicity_n=0,
                    psiAHat=0.045, aHat=0.1, Er=-0.5)  # fmt: skip
    grid = dict(Ntheta=11, Nzeta=1, Nxi=14, NL=4, Nx=4)
    template = _template(geometry, grid)
    m, n, b, nfp = np.array([0, 1]), np.array([0, 0]), np.array([1.0, 0.1]), 1
    base = dict(iota=0.4, g_hat=1.0, i_hat=0.2)
    names = ["B00", "B10", "iota", "GHat", "IHat"]

    def field(p):
        return (jnp.asarray(b).at[0].add(p[0]).at[1].add(p[1]), base["iota"] + p[2],
                base["g_hat"] + p[3], base["i_hat"] + p[4])  # fmt: skip

    return template, (m, n, nfp), names, field, dict(geometry=geometry, grid=grid)


def _w7x():
    from dkx.validation.data_fetch import resolve_external_equilibrium  # noqa: PLC0415

    path = resolve_external_equilibrium(Path("w7x_standardConfig.bc"))
    geometry = dict(
        geometryScheme=11,
        equilibriumFile=str(path),
        inputRadialCoordinate=3,
        rN_wish=0.5,
        Er=-2.0,
    )
    grid = dict(Ntheta=7, Nzeta=15, Nxi=12, NL=3, Nx=3)
    template = _template(geometry, grid)
    header, surfaces = read_boozer_bc(path, geometry_scheme=11)
    s = min(surfaces, key=lambda q: abs(q.r_n - 0.5))
    nfp = int(header.n_periods)
    m = np.concatenate([[0], s.m])
    n = np.concatenate([[0], s.n]) * nfp
    b = np.concatenate([[s.b0_over_bbar], s.b_amp])
    keep = np.abs(b) >= 1e-3 * b[0]
    m, n, b = m[keep], n[keep], b[keep]
    harmonics = [(0, 1), (1, 1), (1, 0)]
    slots = [
        int(np.flatnonzero((m == hm) & (n == hn * nfp))[0]) for hm, hn in harmonics
    ]
    names = [f"B{hm}{hn}" for hm, hn in harmonics] + ["iota", "GHat", "IHat"]

    def field(p):
        bm = jnp.asarray(b)
        for k, slot in enumerate(slots):
            bm = bm.at[slot].add(p[k])
        return bm, s.iota + p[3], s.g_hat + p[4], s.i_hat + p[5]

    geometry = {**geometry, "equilibriumFile": "w7x_standardConfig.bc"}
    return template, (m, n, nfp), names, field, dict(geometry=geometry, grid=grid)


def run_case(name: str) -> dict:
    template, (m, n, nfp), names, field, deck = {"tokamak": _tokamak, "w7x": _w7x}[
        name
    ]()

    def observe(p):
        bmnc, iota, g_hat, i_hat = field(p)
        operator = kinetic_operator_on_boozer_surface(template, bmnc_b=bmnc, ixm_b=m, ixn_b=n, nfp=nfp,
                                                      iota=iota, g_hat=g_hat, i_hat=i_hat)  # fmt: skip
        return _observables(operator)

    value = jax.jit(observe)
    p0 = jnp.zeros(len(names))
    adjoint = np.asarray(jax.jit(jax.jacrev(observe))(p0))  # (observable, parameter)
    sweep = {}
    for h in STEPS:
        columns = []
        for k in range(len(names)):
            e = jnp.zeros(len(names)).at[k].set(h)
            columns.append(
                (np.asarray(value(p0 + e)) - np.asarray(value(p0 - e))) / (2 * h)
            )
        sweep[f"{h:g}"] = np.stack(columns, axis=1).tolist()
    return dict(id=name, deck=deck, parameters=names, observables=list(OBSERVABLES), value=np.asarray(value(p0)).tolist(),
                adjoint=adjoint.tolist(), central_difference=sweep, solver=SOLVER)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["tokamak", "w7x"])
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    git = lambda *a: subprocess.run(
        ["git", *a], cwd=root, capture_output=True, text=True
    ).stdout.strip()  # noqa: E731
    payload = dict(
        schema="dkx.adjoint_sensitivity.v1",
        claim_scope="dkx_reverse_mode_sensitivities_against_dkx_central_differences_small_grids",
        source=dict(dkx_commit=git("merge-base", "HEAD", "origin/main"), jax=jax.__version__, python=platform.python_version()),
        steps=list(STEPS),
        cases=[run_case(name) for name in args.cases],
        exclusions=EXCLUSIONS,
    )  # fmt: skip
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return 0


EXCLUSIONS = [
    "grids of 7x15x12x3 (W7-X) and 11x1x14x4 (tokamak-like); the agreement is a property of the implicit-differentiation algebra, not a converged physics result",
    "the reference is DKX itself; SFINCS RHSMode 4/5 output is not used because it disagrees with its own finite differences by 0.05% to 66%",
    "one ion observable set (particle flux, heat flux, bootstrap current, ion parallel flow) and one surface per case; geometry parameters only, no profile or collisionality derivatives",
    "the solves use adjoint_residual_factor = 1e3: the certified-adjoint guard at its default refuses the ion parallel-flow cotangent on W7-X",
    "experimental validation",
]

if __name__ == "__main__":
    sys.exit(main())
