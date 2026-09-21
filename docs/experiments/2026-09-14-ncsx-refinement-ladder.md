# NCSX resolution ladder and baseline algebraic error estimates on the new route

## Hypothesis

Phase 1 step 2 and item Q9 of the #230 handoff. Two NCSX resolution checks were blocked on memory: theta25 was gated at "56.1 GiB available", and the baseline adjoint campaign at 42 GiB. With #231, #233 and single-threaded BLAS, those grids should fit in 20 GiB. Three things are tested:

- The refinement ladder and the four baseline adjoints run on one shared CPU host, with unchanged physics and tolerances.
- The code changes reproduce the archived moments to rounding.
- The ladder yields per-axis error bars, or states which axes are not yet in the asymptotic range.

Owner: independent review. Budget: one afternoon of CPU time.

## Admission test

**Setup.**
- Deck: NCSX single-species, full Fokker–Planck with full-trajectory `E_r` (the historical baseline reconstructed below). Only the resolution is changed.
- Code: DKX `main` `2b7641d`, SOLVAX 0.21.0 (PyPI), JAX 0.10.2, x64.
- Host: Xeon W-2295, cores 8–11. Environment `DKX_CORES=4` with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` all set to 1. One fresh process per point.
- Watchdog: stops a point above 30 GiB RSS, below 8 GiB host-available, or at the time cap.

**Acceptance per point.** Original relative residual ≤ 1e-10, with the complete state returned by the Krylov route.

**Reproduction.** Points repeated from the earlier campaign must agree to ≤ 1e-12 relative. The public input reconstruction below identifies the case; the historical raw campaign is not bundled.

**Error bars.**
- Discretization: `dkx.workflows.converge.richardson_uncertainty` (fine-grid GCI; it refuses non-monotone ladders).
- Algebraic: `dkx.sensitivity.linear_observable_algebraic_error`, using a transposed GCROT with the solve's own transposed preconditioner at 1e-10 and 1e-12, to check that the estimate is stable.

## Result

**Ladder, first pass** (relative differences against the baseline):

| Point | Grid (θ, ζ, ξ, x) | Iterations | Build + solve | Peak RSS | Δ flow | Δ particle flux | Δ heat flux |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | (21, 37, 61, 8) | 53 | 43 s | 7.7 GiB | — | — | — |
| theta25 | (25, 37, 61, 8) | 114 | 98 s | 10.4 GiB | +3.06e-4 | −1.44e-4 | −3.33e-4 |
| theta29 | (29, 37, 61, 8) | 111 | 146 s | 13.3 GiB | +4.16e-4 | −3.35e-4 | −6.26e-4 |
| zeta43 | (21, 43, 61, 8) | 53 | 76 s | 10.0 GiB | −2.26e-4 | +2.73e-3 | +2.82e-3 |
| pitch60 | (21, 37, 60, 8) | 54 | 53 s | 7.8 GiB | +2.03e-4 | −1.49e-2 | −1.55e-2 |
| pitch81 | (21, 37, 81, 8) | 53 | 86 s | 10.1 GiB | +8.43e-5 | −7.11e-3 | −7.33e-3 |
| speed9 | (21, 37, 61, 9) | 60 | 70 s | 8.8 GiB | −1.69e-3 | −1.43e-4 | +2.27e-5 |
| joint | (25, 43, 81, 9) | 137 | 258 s | 19.7 GiB | −1.46e-3 | −6.24e-3 | −6.29e-3 |

- **Acceptance.** Every point converged with original residual between 4.9e-11 and 9.8e-11.
- **Reproduction.** The baseline, speed9, zeta43 and pitch81 reproduce the archived moments to 1.6e-14, 2.0e-14, 2.2e-14 and 3.1e-14 relative. So #231, #233, SOLVAX 0.21.0 and the BLAS setting did not change an answer.
- **Cost.** The archived runs of the same three grids took 128–278 s at 15.7–16.2 GiB. Here they take 70–86 s at 8.8–10.1 GiB. theta25, which the memory gate had refused, completes at 10.4 GiB.

**Poloidal ladder** `Ntheta = 21 → 25 → 29` (GCI):

| Moment | Result |
| --- | --- |
| Flow and current | asymptotic, observed order 5.3, GCI 1.2e-4 |
| Heat flux | asymptotic but at order 0.23, GCI 1.1e-2; the low order makes the band wide |
| Particle flux | refused (differences change behaviour; not in the asymptotic range) |

**Pitch neighbours.** Relative to `Nxi = 61`, `Nxi = 60` moves the particle and heat fluxes by −1.5% and `Nxi = 81` by −0.7%. The pitch resolution is therefore the dominant discretization uncertainty in the fluxes, at the percent level, and an even/odd neighbour differs by twice the 61 → 81 change. The flow is insensitive to pitch at the 2e-4 level.

**Iteration counts** roughly double when `Ntheta` rises from 21 to 25 or 29 (53 → 111–114) and reach 137 on the joint grid. The coarse preconditioner loses quality with poloidal resolution on this deck; this is recorded, not yet explained.

**Ladder, second pass** (same host and settings, relative differences against the baseline):

| Point | Grid (θ, ζ, ξ, x) | Iterations | Build + solve | Peak RSS | Δ flow | Δ particle flux | Δ heat flux |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| theta33 | (33, 37, 61, 8) | 102 | 212 s | 16.8 GiB | +4.96e-4 | −3.82e-4 | −7.23e-4 |
| zeta49 | (21, 49, 61, 8) | 53 | 85 s | 12.4 GiB | −2.73e-4 | +3.67e-3 | +3.80e-3 |
| pitch101 | (21, 37, 101, 8) | 53 | 115 s | 12.2 GiB | +8.53e-5 | −7.45e-3 | −7.75e-3 |
| pitch121 | (21, 37, 121, 8) | 53 | 135 s | 14.3 GiB | +8.53e-5 | −7.47e-3 | −7.77e-3 |
| speed10 | (21, 37, 61, 10) | 77 | 93 s | 9.7 GiB | −8.60e-4 | −4.24e-4 | −1.96e-4 |

**Ladder, third pass: speed** (same host and settings, same code `2b7641d` with SOLVAX 0.21.0, relative differences against the baseline):

| Point | Grid (θ, ζ, ξ, x) | Iterations | Solve call | Peak RSS | Δ flow | Δ particle flux | Δ heat flux |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| speed11 | (21, 37, 61, 11) | 82 | 100 s | 10.7 GiB | −1.342e-3 | −1.18e-4 | −5.62e-5 |
| speed12 | (21, 37, 61, 12) | 90 | 138 s | 11.6 GiB | −1.348e-3 | −2.40e-4 | −1.67e-4 |

Both points converged, with original residuals of 9.6e-11 and 5.9e-11.

**Per axis**, from three-rung ladders. `R` is the ratio of successive differences; `richardson_uncertainty` supports an estimate only for monotone convergence (`0 < R < 1`) at an observed order ≤ 12.

| Axis (rungs) | Flow | Particle flux | Heat flux |
| --- | --- | --- | --- |
| θ (25, 29, 33) | asymptotic, GCI 5.6e-4 | asymptotic, GCI 2.7e-5 | asymptotic, GCI 8.5e-5 |
| ζ (37, 43, 49) | monotone (R = 0.21) but order ≈ 12.1 > 12, refused; last change −4.7e-5 | asymptotic, GCI 8.8e-4 | asymptotic, GCI 9.1e-4 |
| ξ (81, 101, 121) | monotone (R = 0.011), order ≈ 25, refused; last change 1.2e-8 | monotone (R = 0.042), order ≈ 18, refused; last change −1.4e-5 | monotone (R = 0.050), order ≈ 17, refused; last change −2.1e-5 |
| x (8, 9, 10) | oscillatory (R = −0.49), spread 1.7e-3 | divergent (R = 2.0), spread 4.2e-4 | oscillatory (R = −9.6), spread 2.2e-4 |
| x (10, 11, 12) | monotone (R = 0.013), faster than order 12; last change −6.4e-6 | oscillatory (R = −0.40) | oscillatory (R = −0.79) |
| x (8, 10, 12) | asymptotic, GCI 1.6e-3 | oscillatory (R = −0.43) | oscillatory (R = −0.15) |

- **Pitch (ξ) is converged by `Nxi = 101`.** The Legendre expansion converges faster than any power, so the helper's power-law order check refuses it. The last difference (≤ 2e-5 in the fluxes) is the honest estimate.
- **Baseline pitch error.** Refining pitch lowers both fluxes: at `Nxi = 121` the particle flux is 0.75% and the heat flux 0.78% below the baseline. So the baseline grid `Nxi = 61` overstates both fluxes by about 0.75%. This is the largest single discretization error at the baseline.
- **Speed (x), flow.** The flow settles by `Nx = 11`: `Nx = 11` and `12` differ by 6e-6 relative, and both sit 1.35e-3 below the baseline, so the baseline understates |flow| by 0.135%. The even ladder `(8, 10, 12)` is asymptotic with GCI 1.6e-3. `(10, 11, 12)` also counts as faster than order 12, but `(9, 10, 11)` oscillates, so the close agreement of the last two rungs may be partly accidental. The even-ladder bar is the one to quote.
- **Speed (x), fluxes.** They alternate with the parity of `Nx` at a few 1e-4 through `Nx = 12`, and no speed ladder supports an estimate. Their spread over `Nx = 8–12` is 4.2e-4 for the particle flux and 2.2e-4 for the heat flux, well below the pitch error.

**Baseline algebraic error estimates.** One primal solve at `(21,37,61,8)`: 53 iterations, residual 8.54e-11. For each moment, a transposed GCROT used the transposed preconditioner from that solve. The audit took 4 min in total, peaking at 9.7 GiB; the campaign had been gated at 42 GiB available.

| Moment | Adjoint iterations (1e-10 / 1e-12) | Adjoint residual (1e-10 / 1e-12) | Signed correction `λᵀr` | Relative to the moment |
| --- | ---: | ---: | ---: | ---: |
| `FSABFlow` = `FSABjHat` | 33 / 81 | 9.0e-11 / 6.4e-12 | −1.206e-11 | +1.8e-10 |
| particle flux | 30 / 37 | 7.6e-11 / 4.7e-13 | −8.407e-17 | −1.6e-10 |
| heat flux | 28 / 34 | 9.7e-11 / 8.9e-13 | −8.362e-17 | −4.5e-11 |

- **Requested-tolerance acceptance.** The flow transpose at a requested `1e-12` reached `6.4e-12`, so it did not meet that request. Its stable correction is diagnostic evidence, not an admitted `1e-12` solve. The listed `1e-10` transposes and the tighter particle/heat-flux transposes meet their requests.
- **Stability.** Tightening the adjoint from 1e-10 to 1e-12 changes each correction by less than 3e-11 of itself. The flow adjoint at 1e-12 stopped at its restart cap with residual 6.4e-12; the estimate remained stable despite missing the requested tolerance.
- **Interpretation.** These are estimates, not bounds. The algebraic error of every reported moment is about 1e-10 relative: eight orders of magnitude below the discretization differences above.

Raw states and historical logs remain outside Git. The public reconstruction below permits independent reruns; it does not recreate unavailable timing logs.

## Decision

Continue Phase 1 at a revised reporting grid, and record three follow-ups.

**Error budget.** The algebraic error (≈ 1e-10) is negligible. The signed changes below are refined value minus baseline, relative to the baseline `(21,37,61,8)`.

| Source | Fluxes | Flow |
| --- | --- | --- |
| Pitch | −0.75% (the baseline overstates them) | — |
| ζ | +0.37% | — |
| θ | ≲ 7e-4 | — |
| x | oscillates ≲ 4e-4 | −0.135% (the baseline understates \|flow\|), even-ladder GCI 1.6e-3 |

**Reporting grid.** For the positioning figure use at least `Nxi = 101` and `Nzeta = 49` for the fluxes, and `Nx ≥ 11` for the flow. Those bounds describe separate historical rungs, not a grid combining all refinements. A combined reporting grid and its joint refinement need fresh memory admission.

**Follow-ups:**
1. Done in #239. `richardson_uncertainty` reports monotone ladders faster than `max_order` with `max(safety, 3)` times the last difference, instead of refusing them.
2. The coupling attribution proposed in #240 is historical. Later exact-retention and balancing experiments did not support it as a successful remedy; follow the bounded diagnostic in `plan.md` instead.
3. Repeat the ladder on a two-species deck and on the collaborator's HSX-like deck at its resonant `E_*`, where the SFINCS manual expects the speed resolution to matter most.


## Installed-candidate replay (2026-09-21)

The installed DKX 2.5.0 candidate from public source
[`a48c94e5`](https://github.com/uwplasma/dkx/commit/a48c94e5cec0b5425e0c3a503defc80935dd5fe7)
replayed only the baseline and pitch81 pair. The installed wheel's SHA256 was
`96c6ad67731c11e6f1fea2a69c81f6aed2c1e0fd5fdf078629f6046407d75908`.
Runtime: Xeon W-2295 CPU, four CPUs, single-threaded BLAS, Python 3.11.15,
JAX/jaxlib 0.10.2 with x64 enabled, and SOLVAX 0.24.0.
The reconstructed baseline input SHA256 was
`78dc5d545409cec765bebb0a7a02f43e897070a7a415fdc372d0a92ac6a340a9`.
Physics and solver tolerance `1e-10` were unchanged.

| Quantity | Baseline (21, 37, 61, 8) | Pitch81 (21, 37, 81, 8) | Signed change / absolute baseline |
| --- | ---: | ---: | ---: |
| Complete-state original relative L2 residual | 8.539365599620979e-11 | 9.504394422102867e-11 | — |
| `FSABFlow` = `FSABjHat` | -0.06860781541031573 | -0.06860203463881590 | +0.008426% |
| `particleFlux_vm_psiHat` | 5.198212888227198e-7 | 5.161234890990345e-7 | -0.711360% |
| `heatFlux_vm_psiHat` | 1.857282155175711e-6 | 1.843674992420304e-6 | -0.732638% |

Both points passed the original-equation gate. Total wall time was 91.60 s;
the pitch refinement recorded 49.40 s. Peak RSS was 11,024,564 KiB (10.51 GiB)
from GNU time; the sampled sum over the run's processes peaked at 10.52 GiB.
The runner uses one process for both points, so baseline-only wall time and
separate per-point RSS were not recorded.

After reconstructing the inputs below, the replay command is:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 DKX_CORES=4 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
DKX_TIER2_MEMORY_GUARD=on DKX_COARSE_FACTOR_DTYPE=float64 \
/usr/bin/time -v timeout --signal=TERM --kill-after=30s 20m \
python -m dkx converge baseline.input.namelist --cores 4 \
  --axes pitch --factor 1.3278688524590163 --no-joint \
  --tolerance 0.01 --format json > report.json 2> run.log
```

The CLI tolerance is the 1% observable-change criterion, not the solver
tolerance. External monitoring required at least 32 GiB host-available memory
at admission, and imposed 20 GiB summed run RSS, 12 GiB minimum host-available
memory, and a 20-minute wall limit, targeting only the run's process groups.
Start availability was 47.65 GiB; the monitored minimum was 34.15 GiB.
The initial group monitor omitted the child group created by `timeout`;
session-wide monitoring corrected that omission during execution. No limit
violation was observed, and the command exited successfully. The command
above includes the wall limit and DKX's route admission guard; it does not
implement the external RSS/host-memory watchdog.

This is an accepted partial replay with pairwise changes below 1%. It does
not rerun the historical ladder or adjoints, establish joint convergence,
or supply a three-grid uncertainty estimate. The combined reporting grid,
joint refinement, and revised-grid uncertainty/adjoint evidence remain pending.

### Reporting-grid attempt: time limit

One subsequent `(33,49,101,11)` attempt used the same installed candidate,
wheel, runtime and unchanged physics/tolerance above. The reporting input
SHA256 was `edde8b27a69c81d0d02af9e19e716bb333f77caccec254890ecad601e3b0c8c2`.
The guard selected reusable float64 Schur factors: 21.650 GiB factor storage,
27.063 GiB with the route's resident multiplier, versus 94.149 GiB for dense
bands with their multiplier. These estimates do not bound whole-process RSS.

| Outcome | Wall at watchdog stop | Peak sampled run RSS | Minimum sampled host available |
| --- | ---: | ---: | ---: |
| Time limit; no accepted result | 1200.23 s | 26.94 GiB | 17.25 GiB |

Admission required 44 GiB available and observed 44.49 GiB. An external
watchdog sampled all processes in the created session before releasing the
solver, including the child process group created by `timeout`, at roughly
0.5-second intervals (maximum observed interval 0.573 s). Limits were 40 GiB
summed RSS, 12 GiB host reserve and 20 minutes wall time. Only owned session
groups were targeted at the deadline; no memory-limit violation was observed, and no
session members remained afterward. No restart or larger joint run followed.

The exact public command, under that external watchdog, was:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 DKX_CORES=4 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
DKX_TIER2_MEMORY_GUARD=on DKX_COARSE_FACTOR_DTYPE=float64 \
/usr/bin/time -v timeout --signal=TERM --kill-after=2s 20m \
python -m dkx write-output --input input.namelist \
  --out reporting.h5 --solver-trace reporting.json --cores 4 \
  --no-overwrite > reporting.log 2>&1
```

The command alone does not implement the external RSS/reserve watchdog.
Neither the requested HDF5 result nor the solver-trace JSON was produced;
no normalized original residual, complete returned state or moments were
available. The log confirms route selection but does not separate assembly,
preconditioner build and solve time; GNU time's final summary was also absent
after session termination. This is a time-limit outcome, not evidence of
numerical nonconvergence. Phase timing is required before another large
attempt; combined-grid acceptance and uncertainty remain unestablished.

## Public reconstruction and next refinement

The exact equilibrium and source input are in
[yancc commit `33e1ce9`](https://github.com/f0uriest/yancc/tree/33e1ce9b208f6d3209fdb55aeba8712e6d6a4223),
under its [MIT license](https://github.com/f0uriest/yancc/blob/33e1ce9b208f6d3209fdb55aeba8712e6d6a4223/LICENSE).
This preparation downloads them into the current directory, verifies both
upstream files, and reconstructs the historical baseline before changing only
its grid. Run it in an empty directory outside the repository. It runs no solver.

```python
from hashlib import sha256
from pathlib import Path
import re
from urllib.request import urlopen

base = "https://raw.githubusercontent.com/f0uriest/yancc/33e1ce9b208f6d3209fdb55aeba8712e6d6a4223/"
source = ("publications/conlin2026/20251212-01-sfincs_for_yancc_benchmarks/"
          "20251212-01-030_collisionality_scan/10/input.namelist")

def download(path, expected):
    with urlopen(base + path, timeout=60) as response:
        data = response.read()
    assert sha256(data).hexdigest() == expected, path
    return data

equilibrium = download("tests/data/wout_NCSX.nc",
    "78e60753b960e1e50c5e320e06e7485bd573d37c72dec97dbc4de39c1f182f02")
text = download(source,
    "4c50a95d0cb079ec5679a9c1351a47af20c48c304db3b603c4e0b64a708dd2fe").decode()

def set_value(text, key, value):
    text, count = re.subn(rf"(?im)^(\s*{key}\s*=\s*).+$",
                          lambda match: match[1] + str(value), text)
    assert count == 1, key
    return text

for key, value in dict(equilibriumFile='"equilibrium.nc"', Ntheta=21,
                       Nzeta=37, Nxi=61, Nx=8, solverTolerance="1d-10").items():
    text = set_value(text, key, value)
assert sha256(text.encode()).hexdigest() == "78dc5d545409cec765bebb0a7a02f43e897070a7a415fdc372d0a92ac6a340a9"
Path("baseline.input.namelist").write_text(text)
for key, value in dict(Ntheta=33, Nzeta=49, Nxi=101, Nx=11).items():
    text = set_value(text, key, value)
assert sha256(text.encode()).hexdigest() == "edde8b27a69c81d0d02af9e19e716bb333f77caccec254890ecad601e3b0c8c2"
Path("equilibrium.nc").write_bytes(equilibrium)
Path("input.namelist").write_text(text)
```

After checking available memory and reserving the machine, the existing runner
can attempt this next study with an installed, pinned DKX version:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 DKX_CORES=4 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m dkx converge input.namelist --cores 4 \
  --axes theta zeta pitch speed --factor 1.1 --tolerance 0.01 \
  --format json > report.json 2> run.log
```

The reporting grid is `(33,49,101,11)`; separate refinements reach theta 37,
zeta 55, pitch 111 and speed 12; the joint grid is `(37,55,111,12)`.
The reporting grid was attempted once and stopped at the time limit above;
the separate refinements and larger joint have **not** been run in this update.
Current route estimates
require about 27 GiB available for reusable factors at the reporting grid and
52 GiB at the joint grid, excluding additional reservation headroom. Retaining
the historical dense route instead requires roughly 179 GiB available at the
joint grid. These are admission estimates, not measured peak RSS or guarantees.
A route change must be recorded and timed afresh.

Require complete states and original relative residuals at most `1e-10` for
every RHS, and finite signed flow, bootstrap current, particle flux and heat
flux. The separate and joint observable changes must meet the application
budget. One refinement per axis measures change; it does not supply a
three-grid uncertainty estimate or the revised-grid observable adjoints.
Those remain required before claiming the plan's 1% uncertainty milestone.
