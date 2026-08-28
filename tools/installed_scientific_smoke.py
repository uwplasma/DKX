"""Exercise an installed DKX distribution with a small finite kinetic solve."""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()

    import numpy as np
    import dkx

    package = Path(dkx.__file__).resolve().parent
    if "site-packages" not in package.parts:
        raise SystemExit(f"dkx resolved outside site-packages: {package}")
    if args.workspace is not None and args.workspace.resolve() in package.parents:
        raise SystemExit(f"dkx resolved inside the source checkout: {package}")
    installed = metadata.version("dkx")
    if dkx.__version__ != installed:
        raise SystemExit(
            f"version mismatch: metadata={installed}, import={dkx.__version__}"
        )

    result = dkx.run(
        geometryScheme=1,
        inputRadialCoordinate=3,
        rN_wish=0.3,
        B0OverBBar=1.0,
        GHat=1.0,
        IHat=0.0,
        iota=1.31,
        epsilon_t=0.1,
        epsilon_h=0.0,
        psiAHat=0.045,
        aHat=0.1,
        Zs=[1.0],
        mHats=[1.0],
        nHats=[1.0],
        THats=[0.5],
        dNHatdrHats=[-6.0],
        dTHatdrHats=[-3.0],
        Ntheta=9,
        Nzeta=1,
        Nxi=6,
        NL=3,
        Nx=4,
        Delta=4.5694e-3,
        alpha=1.0,
        nu_n=8.4774e-3,
        Er=0.0,
        collisionOperator=1,
    )
    particle_flux = float(result.moments["particleFlux_vm_psiHat"][0])
    bootstrap = float(result.moments["FSABjHat"])
    if not np.isfinite(particle_flux) or not np.isfinite(bootstrap):
        raise SystemExit(
            f"kinetic smoke is non-finite: flux={particle_flux}, FSABjHat={bootstrap}"
        )
    print(
        f"dkx {installed} from {package}; particle flux={particle_flux:+.6e}; "
        f"FSABjHat={bootstrap:+.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
