# Reuse economics and one redundant inverse

## Hypothesis

Phase 2, step 3: saved preconditioner setup can outweigh extra Krylov work
beyond the tiny field sweeps. Measure existing interfaces before adding a
refresh policy. Owner: rogeriojorge. This record also follows the application
cost the audit exposed into its generic owner, SOLVAX.

## Admission test

DKX `9e9536d9`, SOLVAX 0.20.0; start from
`tests/reduced_inputs/geometryScheme4_2species_noEr.input.namelist` and set
`(Ntheta, Nzeta, Nxi, Nx) = (15, 17, 24, 6)`, tolerance `1e-10`.
The full-Fokker–Planck, full-trajectory, no-Phi1 W7-X case has two species and
73,444 rectangular unknowns. Fields are `0, 0.2, 0.4, 0.6` in deck units.
The input SHA256 is
`f9b12309ca821471d9676cb2393a7dee4345f43435311020117147c0eb7261f4`.
This is a scaling diagnostic, **not a converged research grid**.

Both arms pass the previous distribution as `x0`. One rebuilds the
preconditioner per point; one holds the first preconditioner. Both pass
`recycle=None`, `restart=30`, `recycle_dim=8`, `max_restarts=12` and explicit
`method="gmres"`, avoiding automatic escalation. This disables transferring
recycle vectors **between solves**, not recycling within GCROT; DKX rejects
`recycle_dim=0`. Recover the complete state, check the original equation at
every point and stop on failure. Bound each process at ten minutes.

Discard the first sequence for timing; alternate arm order over five repeats.
`wall_s` covers field/operator/RHS preparation and synchronized solver work,
including preconditioner construction. It **excludes initial problem preparation
and the subsequent moment/residual audit**, so the table is not end-to-end
certified-sweep performance. Both arms are warm in `x0`, not independent cold
solutions. These omissions must be closed for the step's full admission.

## Result

Local M3 Max, Python 3.11.14, JAX/jaxlib 0.9.2; office A4000 GPU 1,
JAX/jaxlib 0.10.2. Both use float64. Median and min–max below sum the four
solve-path times in each repeated sequence; compare arms within each backend,
not hardware with differing software environments.

| Backend | Rebuild, seconds | Fixed, seconds | Iterations, rebuild → fixed |
| --- | ---: | ---: | ---: |
| CPU | 18.31 (17.39–20.97) | 14.64 (14.12–16.18) | 177 → 191 |
| GPU 1 | 25.53 (24.74–26.14) | 22.71 (21.04–23.35) | 181 → 195 |

All 48 solves in each campaign satisfy the original `1e-10` tolerance.
The host was shared; GPU process sampling at 0.5-second intervals found no
competing compute PID during the GPU 1 campaign. This does not establish the
plan's fully idle publication protocol. GPU live peak was 2,164,308,992 bytes;
allocator-pool peak was 4,292,870,144 bytes, a different metric.

An exploratory GPU 0 run had inconsistent times and competing processes;
its timings are excluded. The guarded GPU 0 attempt stopped before solving.
A separate profiled run recorded eight `jit(while)` compilations/cache loads
of 0.32–0.43 seconds, with persistent-cache hits. First invocation is therefore
not synonymous with fresh compilation. The Chrome export covers only part of
the run (1,000,049 events); it cannot support full kernel attribution.
`profile-kernels.py` disables Python-call tracing for a smaller follow-up,
but has **not run**. The complete XPlane stays on office outside Git.

### Implementation found by the audit

SOLVAX forms `W = a_inv(B)` during setup, discards it, and applies `a_inv`
twice per residual. For its documented linear inverse,
`a_inv(r_x - B y) = a_inv(r_x) - W y`. Retaining W in place of B removes
one inverse call. The retained array's shape is unchanged; mixed dtypes can
change bytes. This is the same [bordered factorization](https://doi.org/10.1017/S0962492904000212),
with different floating-point evaluation order, not a new physics model.

[Draft SOLVAX #101](https://github.com/uwplasma/SOLVAX/pull/101), source
`bd52aea19b1e6545a9da01fcd683ddbc543a0a46`, implements the rearrangement.
Its suite had 749 passes and six optional-backend skips; 31 operator tests
include inverse-call count, nonsymmetric real/complex references, nonzero
border blocks and derivatives through setup. Ruff and Sphinx `-W` passed.
Sixteen DKX GPU solves completed with maximum original residual `9.334e-11`;
each reported species flux/flow agrees with the baseline within `4e-9` relative.
The timing campaign stopped on a competing GPU PID and is **not admitted**.
The broader local DKX integration run was interrupted without a final pytest
summary; it is **not a passing gate**. Post-change repeated CPU timing remains
unexecuted. Before the change, 110 ambipolar tests and 44 documentation-contract
tests passed locally, and DKX's Sphinx `-W` build passed.

## Decision and handoff

Keep explicit preconditioner reuse, but do not promote a refresh default from
these measurements. First finish #101's downstream checks and recalibrate
application cost; then complete step 3's root, single-species, cold-equivalence,
profile/history and bounded stale-recovery admission. Charge audits, pilots and
retries in end-to-end timing. Preserve Phase 1's separate grid/observable-error
requirements. No release or phase completion is claimed.

Raw drivers, inputs, source archives, logs and checksums are in
`/Users/rogeriojorge/local/dkx-reuse-evidence-20260912`; office copies are in
`/home/rjorge/local/dkx-reuse-admission-20260912`. `README.md` there maps every
run to its status, and `manifest.json` checks integrity, not scientific validity.
The PR body contains restart commands and dependency provenance. No owned jobs
remain running. At handoff, local simulations were busy and office had only
about 3 GiB available host RAM despite idle GPUs: recheck **both** resources
before restarting. Keep large traces outside Git and do not overwrite evidence.
