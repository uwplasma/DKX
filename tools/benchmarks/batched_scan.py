"""Batched-vs-serial-loop throughput for an ``E_r`` scan and a surface scan.

Measures the win from :mod:`sfincs_jax.batch` — a single ``jax.vmap``ped solve
over many kinetic problems that share a discretization — against the naive
serial Python loop of one :func:`sfincs_jax.solve.solve` per element, on the
same physics.  Two batch axes:

* **``E_r`` scan** — a vector of radial-electric-field values on one geometry
  (:func:`sfincs_jax.batch.batched_er_scan`); and
* **surface scan** — a batch of flux-surface operators sharing grids/layout
  (:func:`sfincs_jax.batch.batched_surface_scan`).

For each axis the tool warms up (compiles), times the batched call and the
serial loop, checks they agree, and prints per-backend throughput (solves/s)
and the batched speedup.  Even on CPU ``vmap`` amortizes per-solve dispatch, so
the speedup is > 1; the large win is on GPU, where the batch fills the device.

Usage::

    python tools/benchmarks/batched_scan.py
    python tools/benchmarks/batched_scan.py --batch 32 --surfaces 8
    python tools/benchmarks/batched_scan.py --ntheta 9 --nzeta 9 --nxi 12 --nx 4
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Callable

import numpy as np


def _deck(
    *,
    er: float = 0.0,
    epsilon_h: float = 0.05,
    dndr: float = -0.5,
    n_theta: int = 7,
    n_zeta: int = 7,
    n_xi: int = 8,
    n_x: int = 3,
) -> str:
    """A tiny non-axisymmetric two-species PAS deck (RHSMode=1, Er knob)."""
    return f"""&general
/
&geometryParameters
  geometryScheme = 1
  inputRadialCoordinate = 3
  rN_wish = 0.3
  B0OverBBar = 1.0
  GHat = 1.0
  IHat = 0.0
  iota = 1.31
  epsilon_t = 0.1
  epsilon_h = {epsilon_h}
  helicity_l = 2
  helicity_n = 5
  psiAHat = 0.045
  aHat = 0.1
/
&speciesParameters
  Zs = 1 -1
  mHats = 1.0 0.000545509
  nHats = 1.0 1.0
  THats = 1.0 1.0
  dNHatdrHats = {dndr} {dndr}
  dTHatdrHats = -1.0 -1.0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0
  nu_n = 8.4774d-3
  Er = {er}
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = {n_theta}
  Nzeta = {n_zeta}
  Nxi = {n_xi}
  NL = 4
  Nx = {n_x}
  solverTolerance = 1d-10
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
"""


def _timed(fn: Callable[[], Any], *, repeat: int = 3) -> tuple[float, Any]:
    """Best-of-``repeat`` wall seconds of ``fn`` after one warm-up (compile)."""
    import jax  # noqa: PLC0415

    out = jax.block_until_ready(fn())
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn())
        best = min(best, time.perf_counter() - t0)
    return best, out


def _write_deck(tmp: "Path", text: str, name: str) -> "Path":  # noqa: F821
    path = tmp / name
    path.write_text(text)
    return path


def bench_er_scan(tmp: "Path", *, batch: int, res: dict[str, int]) -> dict[str, Any]:  # noqa: F821
    """Batched vs serial-loop throughput of an ``E_r`` scan on one geometry."""
    import jax.numpy as jnp  # noqa: PLC0415

    from sfincs_jax import batch as batch_mod  # noqa: PLC0415
    from sfincs_jax import er as er_mod  # noqa: PLC0415

    deck = _deck(**res)
    problem = er_mod.prepare(_write_deck(tmp, deck, "er_scan.namelist"), er_bracket=(-5.0, 5.0))
    er_values = jnp.asarray(np.linspace(-3.0, 1.0, batch), dtype=jnp.float64)
    er_list = [float(v) for v in np.asarray(er_values)]

    def batched() -> Any:
        return batch_mod.batched_er_scan(problem, er_values).radial_current

    def serial() -> Any:
        return jnp.stack([er_mod.radial_current(problem, e)[0] for e in er_list])

    t_batched, jr_batched = _timed(batched)
    t_serial, jr_serial = _timed(serial)
    max_diff = float(np.max(np.abs(np.asarray(jr_batched) - np.asarray(jr_serial))))
    return {
        "batch": batch,
        "total_size": int(problem.operator.total_size),
        "t_batched_s": t_batched,
        "t_serial_s": t_serial,
        "speedup": t_serial / t_batched,
        "throughput_batched": batch / t_batched,
        "throughput_serial": batch / t_serial,
        "max_abs_diff": max_diff,
        "chunk_size": int(batch_mod.batched_er_scan(problem, er_values).chunk_size),
    }


def bench_surface_scan(tmp: "Path", *, surfaces: int, res: dict[str, int]) -> dict[str, Any]:  # noqa: F821
    """Batched vs serial-loop throughput of a multi-surface scan."""
    import jax.numpy as jnp  # noqa: PLC0415

    from sfincs_jax import batch as batch_mod  # noqa: PLC0415
    from sfincs_jax.drift_kinetic import kinetic_operator_from_namelist  # noqa: PLC0415
    from sfincs_jax.inputs import load_sfincs_input  # noqa: PLC0415
    from sfincs_jax.run import profile_moments_from_operator  # noqa: PLC0415
    from sfincs_jax.solve import solve  # noqa: PLC0415

    ripples = np.linspace(0.03, 0.08, surfaces)
    gradients = np.linspace(-0.4, -0.7, surfaces)
    operators = []
    for idx, (eh, dn) in enumerate(zip(ripples, gradients)):
        deck = _deck(epsilon_h=float(eh), dndr=float(dn), **res)
        path = _write_deck(tmp, deck, f"surface_{idx}.namelist")
        operators.append(kinetic_operator_from_namelist(load_sfincs_input(path).raw))

    def batched() -> Any:
        return batch_mod.batched_surface_scan(operators).moments["FSABjHat"]

    def serial() -> Any:
        out = []
        for op in operators:
            state = jnp.reshape(solve(op, op.rhs(), method="auto", tol=1e-10).x, (-1,))
            out.append(profile_moments_from_operator(op, state)["FSABjHat"])
        return jnp.stack(out)

    t_batched, boot_batched = _timed(batched)
    t_serial, boot_serial = _timed(serial)
    max_diff = float(np.max(np.abs(np.asarray(boot_batched) - np.asarray(boot_serial))))
    return {
        "surfaces": surfaces,
        "total_size": int(operators[0].total_size),
        "t_batched_s": t_batched,
        "t_serial_s": t_serial,
        "speedup": t_serial / t_batched,
        "throughput_batched": surfaces / t_batched,
        "throughput_serial": surfaces / t_serial,
        "max_abs_diff": max_diff,
        "chunk_size": int(batch_mod.batched_surface_scan(operators).chunk_size),
    }


def _print_report(backend: str, er: dict[str, Any], surf: dict[str, Any]) -> None:
    print(f"\n=== sfincs_jax batched scan throughput  [backend: {backend}] ===")
    for title, r, count_key in (
        ("E_r scan (one geometry)", er, "batch"),
        ("surface scan (multi-geometry)", surf, "surfaces"),
    ):
        print(f"\n{title}")
        print(f"  elements                {r[count_key]}")
        print(f"  unknowns per solve      {r['total_size']}")
        print(f"  auto chunk size         {r['chunk_size']}")
        print(f"  batched wall            {r['t_batched_s'] * 1e3:9.2f} ms "
              f"({r['throughput_batched']:8.2f} solves/s)")
        print(f"  serial-loop wall        {r['t_serial_s'] * 1e3:9.2f} ms "
              f"({r['throughput_serial']:8.2f} solves/s)")
        print(f"  batched speedup         {r['speedup']:7.2f}x")
        print(f"  max |batched - serial|  {r['max_abs_diff']:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16, help="E_r scan length.")
    parser.add_argument("--surfaces", type=int, default=6, help="number of flux surfaces.")
    parser.add_argument("--ntheta", type=int, default=7)
    parser.add_argument("--nzeta", type=int, default=7)
    parser.add_argument("--nxi", type=int, default=8)
    parser.add_argument("--nx", type=int, default=3)
    args = parser.parse_args()

    import tempfile  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    res = {"n_theta": args.ntheta, "n_zeta": args.nzeta, "n_xi": args.nxi, "n_x": args.nx}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        er = bench_er_scan(tmp, batch=args.batch, res=res)
        surf = bench_surface_scan(tmp, surfaces=args.surfaces, res=res)
    _print_report(jax.default_backend(), er, surf)


if __name__ == "__main__":
    main()
