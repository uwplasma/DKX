#!/usr/bin/env python
"""Gradients: dkx implicit differentiation vs SFINCS Fortran v3 finite differences.

Computes the same derivative two ways and reports accuracy *and* cost.

The objective is ``FSABjHat`` -- the flux-surface-averaged bootstrap current,
which is what stellarator optimization actually targets -- and the parameters
are the kinetic gradient drives ``dNHatdrHats`` and ``dTHatdrHats``, one per
species.  Those are the inputs a profile-optimization or transport-inversion
loop varies, so the comparison is the one that decides whether gradient-based
work is affordable at all.

The cost asymmetry is the point.  A finite-difference gradient of ``N``
parameters costs ``2N`` converged nonlinear solves (central differences); it is
the only option for a code that cannot differentiate itself, and it is why
gradient-based optimization against Fortran neoclassical solvers is usually
replaced by gradient-free search.  Implicit differentiation costs *one*
transposed solve regardless of ``N``: the adjoint is defined by the linear
equation at the converged solution, not by differentiating the iteration.

Accuracy is compared on the same footing by chain-ruling dkx's operator-level
derivative back to the namelist parameter.  The map from ``dNHatdrHats`` to the
operator's ``dn_hat_dpsi_hat`` is exactly linear (a geometry Jacobian factor),
so its slope is recovered exactly from two operator builds and carries no
finite-difference error of its own.

Finite differences also have no exact answer to converge to: the step size
trades truncation error against solver-noise amplification, and the Fortran
solve's own Krylov tolerance sets a floor below which the difference is noise.
``--step`` sweeps that, so the comparison shows the FD error floor rather than
quoting one lucky step.

Usage::

    python tools/benchmarks/ad_vs_fortran_fd.py \
        --deck /path/to/examples/quick_2species_FPCollisions_noEr \
        --fortran-binary /path/to/sfincs \
        --fortran-launcher 'micromamba run -n sfincs-fortran' \
        --steps 1e-2 1e-3 1e-4 --out ad_vs_fd.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from dkx.drift_kinetic import kinetic_operator_from_namelist  # noqa: E402
from dkx.moments import rhsmode1_moments  # noqa: E402
from dkx.namelist import read_sfincs_input  # noqa: E402
from dkx.solve import solve  # noqa: E402
from dkx.writer import operator_containers  # noqa: E402


def _fsabjhat(operator) -> jnp.ndarray:
    """Species-summed bootstrap current of one differentiable RHSMode=1 solve."""
    state = solve(operator, operator.rhs(), tol=1e-12, differentiable=True).x
    layout, vgrid, surface, species = operator_containers(operator)
    table = rhsmode1_moments(
        layout, vgrid, surface, species, jnp.reshape(state, (-1,)),
        delta=operator.delta, alpha=operator.alpha,
    )  # fmt: skip
    return jnp.reshape(table["FSABjHat"], ())


def dkx_gradient(deck: Path) -> dict:
    """AD gradient of FSABjHat w.r.t. the per-species density/temperature drives."""
    base = kinetic_operator_from_namelist(read_sfincs_input(str(deck)))

    def objective(drives: jnp.ndarray) -> jnp.ndarray:
        n_species = base.dn_hat_dpsi_hat.shape[0]
        return _fsabjhat(
            replace(
                base,
                dn_hat_dpsi_hat=drives[:n_species],
                dt_hat_dpsi_hat=drives[n_species:],
            )
        )

    drives = jnp.concatenate(
        [jnp.asarray(base.dn_hat_dpsi_hat), jnp.asarray(base.dt_hat_dpsi_hat)]
    )

    start = time.perf_counter()
    value, gradient = jax.value_and_grad(objective)(drives)
    jax.block_until_ready(gradient)
    cold_s = time.perf_counter() - start

    start = time.perf_counter()
    gradient = jax.grad(objective)(drives)
    jax.block_until_ready(gradient)
    warm_s = time.perf_counter() - start

    return {
        "value": float(value),
        "grad_wrt_operator_drives": [float(g) for g in gradient],
        "cold_s": round(cold_s, 3),
        "warm_s": round(warm_s, 3),
        "n_parameters": int(drives.size),
        "solves": "1 forward + 1 transposed (independent of N)",
    }


def namelist_to_operator_slopes(deck: Path, names: tuple[str, ...]) -> dict:
    """Exact d(operator drive)/d(namelist parameter), one entry per species.

    The map is linear, so two builds recover the slope with no truncation
    error; this is what puts the AD gradient and the Fortran finite difference
    in the same units.
    """
    text = deck.read_text()
    base = kinetic_operator_from_namelist(read_sfincs_input(str(deck)))
    slopes: dict[str, list[float]] = {}
    with tempfile.TemporaryDirectory() as scratch:
        for name, field in zip(names, ("dn_hat_dpsi_hat", "dt_hat_dpsi_hat")):
            values = _read_vector(text, name)
            bumped = [v + 1.0 for v in values]  # unit step; the map is linear
            path = Path(scratch) / "bumped.namelist"
            path.write_text(_write_vector(text, name, bumped))
            other = kinetic_operator_from_namelist(read_sfincs_input(str(path)))
            slopes[name] = [
                float(b - a)
                for a, b in zip(getattr(base, field), getattr(other, field))
            ]
    return slopes


def _read_vector(text: str, name: str) -> list[float]:
    match = re.search(rf"^\s*{name}\s*=\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    if match is None:
        raise SystemExit(f"{name} not found in the deck")
    return [float(tok.replace("d", "e").replace("D", "e"))
            for tok in match.group(1).split("!")[0].split()]


def _write_vector(text: str, name: str, values: list[float]) -> str:
    joined = " ".join(f"{v:.16e}" for v in values)
    return re.sub(
        rf"^(\s*){name}\s*=.*$", rf"\g<1>{name} = {joined}",
        text, count=1, flags=re.MULTILINE | re.IGNORECASE,
    )


_FSABJHAT = re.compile(r"FSABjHat \(bootstrap current\):\s*([-\d.eEdD+]+)")


def _run_fortran(example_dir: Path, text: str, binary: Path, launcher: list[str]) -> float:
    """One Fortran solve on a perturbed deck; returns FSABjHat."""
    with tempfile.TemporaryDirectory(prefix="dkx_fd_") as scratch:
        work = Path(scratch) / "run"
        shutil.copytree(example_dir, work)
        (work / "input.namelist").write_text(text)
        proc = subprocess.run(
            launcher + [str(binary)], cwd=work, capture_output=True, text=True,
        )
        match = _FSABJHAT.search(proc.stdout)
        if match is None:
            raise SystemExit(f"no FSABjHat in Fortran output:\n{proc.stdout[-1500:]}")
        return float(match.group(1).replace("D", "e").replace("d", "e"))


def fortran_fd_gradient(
    example_dir: Path, binary: Path, launcher: list[str], step: float,
    names: tuple[str, ...],
) -> dict:
    """Central-difference gradient: 2N converged Fortran solves."""
    deck = example_dir / "input.namelist"
    text = deck.read_text()
    gradient: list[float] = []
    n_solves = 0
    start = time.perf_counter()
    for name in names:
        values = _read_vector(text, name)
        for index in range(len(values)):
            scale = abs(values[index]) or 1.0
            h = step * scale
            plus, minus = list(values), list(values)
            plus[index] += h
            minus[index] -= h
            high = _run_fortran(example_dir, _write_vector(text, name, plus), binary, launcher)
            low = _run_fortran(example_dir, _write_vector(text, name, minus), binary, launcher)
            n_solves += 2
            gradient.append((high - low) / (2.0 * h))
    return {
        "step": step,
        "grad_wrt_namelist": gradient,
        "wall_s": round(time.perf_counter() - start, 2),
        "n_solves": n_solves,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deck", type=Path, required=True, help="example directory")
    parser.add_argument("--fortran-binary", type=Path, required=True)
    parser.add_argument("--fortran-launcher", default="")
    parser.add_argument("--steps", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    names = ("dNHatdrHats", "dTHatdrHats")
    example = args.deck.resolve()
    deck = example / "input.namelist"
    launcher = args.fortran_launcher.split() if args.fortran_launcher else []

    print("dkx: implicit-differentiation gradient ...", file=sys.stderr)
    ad = dkx_gradient(deck)
    slopes = namelist_to_operator_slopes(deck, names)
    chain = [s for name in names for s in slopes[name]]
    ad_namelist = [g * s for g, s in zip(ad["grad_wrt_operator_drives"], chain)]

    report = {
        "case": example.name,
        "objective": "FSABjHat",
        "parameters": [f"{n}[{i}]" for n in names for i in range(len(slopes[n]))],
        "dkx": {**ad, "grad_wrt_namelist": ad_namelist},
        "fortran_fd": [],
    }

    for step in args.steps:
        print(f"fortran: central differences at step {step:g} ...", file=sys.stderr)
        fd = fortran_fd_gradient(example, args.fortran_binary, launcher, step, names)
        reference = np.asarray(ad_namelist)
        candidate = np.asarray(fd["grad_wrt_namelist"])
        scale = max(float(np.max(np.abs(reference))), 1e-300)
        fd["max_rel_diff_vs_ad"] = float(np.max(np.abs(candidate - reference))) / scale
        report["fortran_fd"].append(fd)
        print(f"    rel diff {fd['max_rel_diff_vs_ad']:.3e} "
              f"in {fd['wall_s']}s / {fd['n_solves']} solves", file=sys.stderr)

    best = min(report["fortran_fd"], key=lambda r: r["max_rel_diff_vs_ad"])
    report["summary"] = {
        "n_parameters": len(report["parameters"]),
        "dkx_warm_s": ad["warm_s"],
        "fortran_best_step": best["step"],
        "fortran_wall_s": best["wall_s"],
        "fortran_solves": best["n_solves"],
        "best_agreement_rel": best["max_rel_diff_vs_ad"],
        "speedup_vs_fd": round(best["wall_s"] / max(ad["warm_s"], 1e-9), 1),
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
