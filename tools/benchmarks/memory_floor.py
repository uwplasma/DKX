"""Decompose the resident memory a `dkx` run pays before any physics.

DKX is lighter than SFINCS Fortran on only 3 of 32 upstream decks, and half of
that gap is at the *small* end, where a fixed floor exceeds the whole Fortran
process (0.1-0.2 GB).  Blaming "JAX overhead" is not a finding; this measures
what the floor is actually made of, so that the part which can be reduced is
separated from the part which cannot.

Each stage runs in its own subprocess and reports its own peak RSS, so the
stages are independent measurements rather than one cumulative trace in which
a later import silently inherits an earlier one's cost.

Usage::

    python tools/benchmarks/memory_floor.py --deck tests/ref/<name>.input.namelist

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

#: Stages, coarsest first.  Each is standalone Python run in a fresh interpreter.
STAGES: dict[str, str] = {
    "bare python": "pass",
    "+ numpy": "import numpy",
    "+ jax imported": "import jax, jax.numpy as jnp",
    "+ jax backend live": "import jax, jax.numpy as jnp\njax.block_until_ready(jnp.zeros(1) + 1)",
    "+ dkx imported": "import dkx.run",
}

#: Stages that need the deck path substituted in.
DECK_STAGES: dict[str, str] = {
    "+ operator built": (
        "import sys\n"
        "from dkx.drift_kinetic import KineticOperator\n"
        "from dkx.inputs import read_sfincs_input\n"
        "op = KineticOperator.from_namelist(read_sfincs_input(sys.argv[1]))\n"
        "print('UNKNOWNS', op.total_size)"
    ),
    "+ solved": (
        "import sys\n"
        "import jax.numpy as jnp\n"
        "from dkx.drift_kinetic import KineticOperator\n"
        "from dkx.inputs import read_sfincs_input\n"
        "from dkx.solve import solve\n"
        "op = KineticOperator.from_namelist(read_sfincs_input(sys.argv[1]))\n"
        "result = solve(op, jnp.ones((op.total_size,)))\n"
        "print('UNKNOWNS', op.total_size)\n"
        "print('ROUTE', result.method)"
    ),
}

#: XLA allocator knobs, against the fullest stage.  None of them moves the floor
#: -- run them repeatedly rather than once, because a single sample of each is
#: how a 50 MB difference gets mistaken for a lever it is not (docs/performance.rst).
ALLOCATOR_VARIANTS: dict[str, dict[str, str]] = {
    "default": {},
    "XLA_PYTHON_CLIENT_PREALLOCATE=false": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    "XLA_PYTHON_CLIENT_ALLOCATOR=platform": {"XLA_PYTHON_CLIENT_ALLOCATOR": "platform"},
    "JAX_DISABLE_JIT=1": {"JAX_DISABLE_JIT": "1"},
}


def _run(code: str, args: list[str], env_extra: dict[str, str]) -> tuple[float | None, dict]:
    """Peak RSS in GB and any UNKNOWNS/ROUTE the stage printed."""
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", os.getcwd())
    env.setdefault("JAX_ENABLE_X64", "1")
    env.update(env_extra)
    proc = subprocess.run(
        ["/usr/bin/time", "-l", sys.executable, "-c", code, *args],
        capture_output=True, text=True, env=env, check=False,
    )
    peak = None
    for line in (proc.stderr or "").splitlines():
        if "maximum resident set size" in line:  # macOS reports bytes
            peak = int(line.split()[0]) / 2**30
        elif "Maximum resident set size" in line:  # GNU time reports kB
            peak = int(line.split()[-1]) * 1024 / 2**30
    facts = {}
    for line in (proc.stdout or "").splitlines():
        if line.startswith(("UNKNOWNS", "ROUTE")):
            key, _, value = line.partition(" ")
            facts[key.lower()] = value
    return peak, facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--deck", type=Path,
        default=Path("tests/ref/pas_1species_PAS_noEr_tiny_scheme1.input.namelist"),
        help="a deck small enough that its physics is not the memory",
    )
    parser.add_argument("--json", type=Path, help="also write the raw numbers here")
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="samples per allocator knob; one sample is not enough to call a null result",
    )
    args = parser.parse_args()
    if not args.deck.exists():
        parser.error(f"deck not found: {args.deck}")

    measured: dict[str, float] = {}
    print(f"{'stage':26s} {'peak RSS':>10s} {'step':>10s}")
    previous = 0.0
    for name, code in STAGES.items():
        peak, _ = _run(code, [], {})
        if peak is None:
            print(f"{name:26s} {'FAILED':>10s}")
            continue
        measured[name] = peak
        print(f"{name:26s} {peak:9.3f}G {peak - previous:+9.3f}G")
        previous = peak

    unknowns = route = "?"
    for name, code in DECK_STAGES.items():
        peak, facts = _run(code, [str(args.deck)], {})
        unknowns = facts.get("unknowns", unknowns)
        route = facts.get("route", route)
        if peak is None:
            print(f"{name:26s} {'FAILED':>10s}")
            continue
        measured[name] = peak
        print(f"{name:26s} {peak:9.3f}G {peak - previous:+9.3f}G")
        previous = peak

    print(f"\ndeck {args.deck}  ({unknowns} unknowns, route {route})")
    print(f"\n{'knob':38s} {'peak RSS':>10s}   (best of {args.repeats}, fullest stage)")
    for name, env_extra in ALLOCATOR_VARIANTS.items():
        samples = [
            _run(DECK_STAGES["+ solved"], [str(args.deck)], env_extra)[0]
            for _ in range(args.repeats)
        ]
        samples = [s for s in samples if s is not None]
        if not samples:
            print(f"{name:38s} {'FAILED':>10s}")
            continue
        measured[f"knob:{name}"] = min(samples)
        spread = max(samples) - min(samples)
        print(f"{name:38s} {min(samples):9.3f}G  (spread {spread:.3f}G)")

    if args.json:
        args.json.write_text(
            json.dumps({"deck": str(args.deck), "unknowns": unknowns,
                        "route": route, "peak_rss_gb": measured}, indent=2) + "\n"
        )  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
