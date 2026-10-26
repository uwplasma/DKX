# Changelog

## Unreleased

- Validation: the W7-X ambipolar profile is not promoted. A spectral-angle probe at (21, 71, 192, 8), the rung `docs/benchmarks/validation_matrix.md` says is needed, is refused by the structured direct route on the 62 GiB office host (dense-band estimate 152.6 GiB at float64, 20.3 GiB at float32) and the Krylov fallback had not finished one solve in 24 minutes, so the promotion gate could not be run; the matrix page records the numbers and what the gate needs.

## v2.12.0 — 2026-10-25

Self-consistent bootstrap current at the ambipolar E_r for optimization
(`KineticBootstrapMismatch(ambipolar=True)`, differentiated through the root),
spectral angles by default, the improved Sugama l = 1 correction (Spitzer-Harm
within 0.65%), Beidler et al. (2011)'s sign convention for D31*/D13*, and a
W7-X comparison against the digitized ICNTS curves.

- Feature: `KineticBootstrapMismatch(ambipolar=True)` evaluates the traced bootstrap row at each surface's ambipolar `E_r`, with `er_kV_per_m` also accepting a callable of `s`, and `spectral_angles=True`. The root comes from a damped Newton iteration warm-started at the previous root (same branch), converged on the Newton gap `|J_r / J_r'|` and run as its own compiled program from the host, so the program VMEX differentiates holds one solve per surface and no loop: cold Jacobian 228 s to 57 s, warm residual 2.8 s (one surface, 9x9x32x4, loaded 8-core office host). The derivative through the root is `j_p - j_E J_p / J_E` (implicit function theorem). Against central differences of VMEX re-solves (QI example seed, s = 0.3 and 0.6, three boundary coefficients): at fixed `E_r = -2` kV/m 3e-3 or better; at the ambipolar root 2e-2 or better. Default ambipolar grid `9 x 9 x 32 x 5` (`DEFAULT_AMBIPOLAR_RESOLUTION`): within 2.1% and 1.0% of `13 x 13 x 64 x 5` for `<j.B>` and 2% for the roots; `Nx = 4` is 30% high at s = 0.3 whatever `Nxi`. `unconverged` counts rows whose root search failed; `examples/advanced/ambipolar_bootstrap_dkx.py` and `tests/test_bootstrap_ambipolar.py`.
- Changed: `D31*` and `D13*` of the monoenergetic coefficients follow Beidler et al. (2011)'s sign convention: `D31*` is positive in the banana limit whatever the sign of iota (`D13* = -D31*` at `EStar = 0`). Up to 2.11.0 they carried `sign(iota)`, so every stored `D31*`/`D13*` of an `iota < 0` deck (W7-X, HSX, TJ-II, NCSX, DSHAPE) changes sign; `D11*` and `D33*` are unchanged. The sign is applied once, in `monoenergetic_dstar_from_transport_matrix`, and its docstring no longer claims the orientation was standardized. Sealed records re-sealed with the sign flipped: `validation/independent_cross_code_v1.json` (DKX and reference `D31*`/`D13*` of DSHAPE 0.015800 -> -0.015800, NCSX -0.137747 -> 0.137747, W7-X EIM 0.013317 -> -0.013317; reference `cross_orientation` -1 -> +1; relative errors unchanged), `validation/cross_code_gaps_v1.json` (all 13 `D31*` ladder values, e.g. DKX 0.013317 -> -0.013317), the five `docs/_static/figures/paper_benchmarks` JSONs (HSX, TJ-II, W7-X, low-collisionality, Shaing-Callen; Shaing-Callen limit at the tiny grid -0.059023 -> 0.059023), the compact-output and registry checksums (DKX output sha256: DSHAPE b195f55b -> 974801b8, NCSX 4be87097 -> e684c24f, W7-X afb539e0 -> dfcc52b8). `dkes_to_beidler` defaults to `cross_orientation = +1`. The #382 W7-X comparison no longer flips the sign by hand.
- Default: `[resolution] angles` is now `"spectral"` (Fourier collocation); `finite_difference` stays an option and is written to the canonical case when chosen, while `spectral` is omitted. A case that omitted `angles` keeps its case ID but now names different numerics (a cached result of the old default has the same ID). Reason: at equal grid the spectral error is lower (see the table below) and the warm time is the same. A converted SFINCS namelist keeps `angles = "finite_difference"` explicitly, so deck parity is unchanged; the kernel-parity tests in `tests/test_native_execution_result.py` pin the stencil for the same reason.
- `[resolution] auto = true` (opt-in) fills missing `theta`, `zeta`, `pitch`, `speed` from the measured ladder of `docs/user_guide/convergence.md` at the case's lowest ion collisionality (`dkx.config.auto_resolution`; tokamak rungs for `file = "tokamak"`). It stays opt-in because it is not calibrated: on an HSX deck (n = 2e19 m^-3, T_i = 0.1 keV) it chose 13 x 17 x 24 x 5, 33% from a 17 x 33 x 48 x 6 reference. Without `auto`, every size is required as before.
- `[run] check_grid = true` is the input-file form of `dkx run --check-grid` (omitted from the canonical case when false). The refined solves of the certificate and of `dkx converge` do not re-certify.
- Measurements (office, 8 pinned cores of a loaded host; error is the largest relative difference of the particle flux, heat flux and parallel current against a reference of the same case, warm time after one cold run, process peak RSS). Tolerance sweep, spectral angles, reference at 1e-12 (tokamak: 1e-10, since 1e-12 does not converge there): HSX deck 11 x 25 x 32 x 5 (Krylov) 1e-4 / 1e-6 / 1e-8 / 1e-10: error 6e-5 / 2e-7 / 2e-9 / 1e-10, time 21 / 25 / 30 / 56 s, 1.2 GB. Tokamak 21 x 1 x 20 x 6: error 6e-10 / 6e-10 / 2e-11 / 0 at 0.3 / 0.3 / 1.1 / 0.8 s, 0.3-0.5 GB; W7-X 11 x 13 x 32 x 5 uses the direct route, whose answer does not depend on the tolerance (error 0, 15 s, 1.5 GB). The solver error at 1e-8 is below the discretization error by 6 orders, so the default tolerance is unchanged. Defaults, tolerance 1e-10, against a finer spectral reference: tokamak finite differences / spectral / automatic grid 2.5e-3 / 1.5e-3 / 1.4e-3 (0.8 / 0.8 / 0.9 s); HSX 2.0 / 4.1e-2 / 3.3e-1 (32 / 28 / 13 s, 1.2 / 1.3 / 1.0 GB; reference 185 s, 4.4 GB). W7-X finite differences and spectral at 11 x 13 x 32 x 5 both take 11 s; its 15 x 33 x 64 x 8 reference and 21 x 49 x 96 x 8 automatic grid did not finish in 40 minutes.
- The pitch-160 fallback to Krylov is a cost rule, not a tolerance miss: `method = "auto"` leaves the speed-coupled direct route above 150 GFlop per right-hand side (W7-X, 9 x 13, two species, full FP, pitch 80 and 160). Forced with `structured_direct`, the three fixed defect-correction sweeps reach a residual of 1e-17 at pitch 80 (108 s, 2.9 GB) and 160 (246 s, 5.5 GB), so the proposed longer refinement is not needed and is not included. The tokamak PAS truncated direct route does stall near 5e-10 and falls back to Krylov for tolerances below 1e-6; at 1e-12 the Krylov solve ends at 4e-12 and the run raises (open).
- Test: W7-X Phi1 in the kinetic equation with spectral angles is accepted and within 1.5% of the stencil flux at 7 x 7 x 8 x 3 (`tests/test_native_phi1.py`).
- Plan: section 19, literature round 2 (improved Sugama correction, unequal-temperature gaps, ICNTS and MONKES reference data). No code change.
- Validation: `validation/beidler2011_w7x_monoenergetic.csv` holds D11*, D31* and D33* of the W7-X standard configuration at rho = 0.5, digitized from Figs 17, 21 and 26 of Beidler et al. (2011) with the method and errors in its header (3% on D11*, 0.003 on D31*, 0.002 on D33*). `tests/test_beidler_w7x_reference.py` runs DKX's monoenergetic route on `w7x_standardConfig.bc` against it: D11* within +1.6 to +3.0% (+5.4 and +7.7% at a finer grid, 1.1e-3 and 3.6e-4), D31* within 0.007, D33* within 1.5%, and D11* at v_E = 3e-3 inside the scatter of the DKES and Monte Carlo symbols. DKX's D31* takes the sign of iota, opposite to Beidler's convention for this file (iota < 0); the test applies the flip and the cause is not changed here.
- Feature: improved Sugama (`collisionOperator = 3`) adds the l = 1 flow-moment correction of Sugama et al. (2019), Eqs. 40-43. The Landau l = 1 field block minus the model's rank-1 momentum term is projected onto the first K = 5 Laguerre flow moments (`flow_order`), keeping momentum exact and the shifted Maxwellian a null vector. The Spitzer-Harm conductivity moves from 11.1 / 9.8 / 7.8 / 3.1 % low to within 0.6 % at Z = 1, 2, 4, 16 (K = 2: within 2.1 %; Frei et al. 2022 compare K = 2 and 5). The same projection is valid at T_a != T_b: flow and heat-flow friction match the Landau operator to 2e-3 (T_e/T_i = 10, m_e/m_i = 1/1836), and the l = 0 energy exchange converges to 2019 Eq. A1. `tests/test_physics_limits.py` and `tests/test_collisions_improved_sugama.py` pin both.

## v2.11.0 — 2026-10-21

Validation and robustness. `dkx scan` covers SFINCS scanTypes 1-5. Phi1 is
benchmarked on W7-X with an impurity, and its memory falls from 11 to 2.4 GB.
The W7-X monoenergetic gap to MONKES/yancc was a radial-normalization error in
the benchmark deck; D11* now agrees to 2e-4. Improved Sugama builds its
momentum restoring per species pair. The bootstrap current agrees with Redl to
1.5-1.8% in a tokamak and within the angular-resolution spread on precise QA
and QH. Large-E_r Fokker-Planck stalls are fixed at their cause, SuperLU memory
is calibrated, MUMPS can run out of core, and `--check-grid` reports whether a
result is resolved.

- Fix: W7-X EIM `D11*` looked 10 to 12% apart between DKX, YANCC and MONKES because the deck's `psiAHat` was B0 aHat^2/2 (0.3195) instead of the equilibrium's toroidal flux (0.3032), so DKX divided by a different dpsi/dr. Corrected, and with each code's own dpsi/dr mapped by `dkes_to_beidler(psi_p=...)`, the converged ladders agree to 7e-4 (`validation/cross_code_gaps_v1.json`); the sealed W7-X case is re-sealed at a 3% gate. `tests/test_physics_limits.py` pins D11 per psi^2 against NTX and the 1/psiAHat scaling of D11*.
- Memory: the Phi1 bordered preconditioner no longer materializes the border (matrix-free `A^-1 B` only), is passed to the Krylov solve without baked-in constants, and is reused across Newton steps. W7-X 9x35x32x4, three species: 11.1 GB to 2.4 GB peak, 275 s to 115 s; 13x45x48x5: over 14 GB to 9.3 GB, 621 s (4 cores, shared host). SFINCS: about 1.3 and 5.9 GB.
- Fix: the improved Sugama momentum-restoring response was built from the test operator summed over all background species, so electron-ion Lorentz scattering entered the electron-electron restoring shape and the like-species operator lost its shifted-Maxwellian null vector. It is now one term per ordered species pair. Spitzer-Harm conductivity deficit: 22.6 to 5.4 % before, 11.1 to 3.1 % after (Z = 1 to 16); the rest is open, since the published operator is within 1 % (Frei et al. 2022), so the field-particle part is still incomplete (`tests/test_physics_limits.py`).
- `dkx scan` also runs SFINCS scanTypes 1 (convergence), 3 (one variable) and 4 (radius), so no repository checkout is needed, and names radius folders `rN_<r>` as upstream (an earlier `rN<r>` folder is reused).
- Fix: the `coupled` preconditioner no longer stalls at large |E_r|. Its float32 apply is a fixed map only to single precision, and the wide restart window reconstructs `M^-1 (V y)` from the Arnoldi basis alone (no stored `Z`), which amplified that error: the first wide cycle ended at 80 x `||b||` on the HSX QHS full-FP deck (66,004 unknowns) and the solve stalled after 892 to 1,253 iterations. The wide window now stores `Z` for `coupled` and the wide-restart budget counts both bases. Forced `coupled`, E_r = 15 and 20, 4 threads of a loaded laptop (load 7 to 10): stall then escalation to `coarse_triangle` 60 s and 67 s (stall alone 42 s and 50 s); now 19 s and 23 s, 487 and 572 iterations, one rung. E_r = 0, 5, 10 converge before the wide window and are unchanged (5, 11, 10 s). A residual-rate test on the short windows was measured and rejected: restart-stagnant decks that converge in the wide window (coarse, Nx = 12 and 16, 79,204 and 105,604 unknowns) sit at 0.73 to 0.83 after the probe window and 0.66 after the middle one, no different from the stalled `coupled` solve.
- The SuperLU peak-memory estimate is calibrated on six sizes instead of one. HSX QHS full-FP deck, E_r terms on, 4 laptop threads (load 7 to 10), whole-process peak: 19,804 / 33,004 / 39,604 (Nxi = 40) / 46,204 / 85,804 unknowns took 3.6 / 5.7 / 6.7 / 7.8 / 14.4 GB and 31 / 111 / 73 / 375 / 2,341 s, residuals 6e-15 to 4e-14; with the earlier 66,004-unknown point (9.9 GB) the fit is 1.25 x (2.04 + 8.59 (n/66,004)^1.28) GB, within 8% of every point. The old law (1.25 x 9.9 GB x (n/66,004)^1.6) was 1.3 x too high at 85,804 and 2 x too low at 19,804; it put the 633,604-unknown gap deck at 460 GB, this one at 198 GB (an extrapolation past the largest measured size). Time grows as about n^3 above 46,000.
- `SolverOptions(direct_out_of_core=True)` / `solve(direct_out_of_core=True)`: the MUMPS direct route keeps its factors on disk (needs the SOLVAX change that adds `out_of_core` and the `python-mumps` wheels; `ImportError` otherwise). End to end through `solve(method="direct", direct_backend="mumps")` on the HSX QHS full-FP deck, 66,004 unknowns, 8 pinned cores of the shared office host (load 18 to 26): 84 s in core and 75 s out of core, including the 3,008-product assembly, relative residual 6e-14 and 8e-14 (SuperLU: 47 min). MUMPS analysis of the 633,604-unknown gap deck, 51 s: 94 GB in core (PORD ordering 86 GB), or 13 GB of memory and about 77 GB of disk out of core, 1.5e14 flops. Block low-rank compression gave no memory gain at 66,004 unknowns and is not exposed.
- `[resolution] angles = "spectral"` selects Fourier collocation for the `theta` and `zeta` derivatives of a native case (default `finite_difference`, the fourth-order stencils, unchanged; the default is omitted from the canonical case so existing case IDs hold). On analytic W7-X at 1e20 m^-3, 1 keV, pitch 64, speed 8, the particle and heat flux error against the 21 x 49 spectral value falls from 3.7% to 0.3% at 13 x 17 points with spectral angles, where fourth-order differences still leave 2.1%; on the LHD 1/nu case at the same density, finite differences are still 1.5% off at 21 x 49 and spectral angles are below 1% from 13 x 17 (laptop, 4 threads). Warm solve time is the same at equal grid (analytic W7-X 15 x 25 x 48 x 6: 1.0 s finite differences, 0.9 s spectral).
- `dkx run --check-grid [AXIS ...]`, `dkx scan --check-grid` and `run_case(check_grid=...)` attach a per-radius resolution certificate with one extra solve: all of theta, zeta, pitch and speed (or the named ones) are refined together by 1.25 and each surface is `resolved` when its particle and heat fluxes move by less than 2% and, for an ambipolar profile, its root by less than 0.05 kV/m. The result carries `resolution_change`, `resolution_resolved` (and `resolution_root_change_kV_m`), `metadata["resolution_certificate"]` and a warning naming the unresolved radii; a scan gains a `resolution_change` column. `dkx.workflows.converge.certify_resolution` is the Python entry.
- Tutorial 5 and `13_convergence.py` use a deck that converges: the old one (n = 8e19 m^-3, T = 1 keV, tokamak with `zeta = 1`) moved the particle flux by 8-17% per theta rung up to 41 points. The new one (n = 2e20 m^-3, T = 0.2 keV) is within 2% of its finest rung from 13/12/5 on, and `dkx converge` at theta 21, pitch 20, speed 6 reports 0.8%, converged. The page quotes the real output.
- W7-X ambipolar profile re-measured with spectral angles (validation matrix): roots moved by up to 3.3 kV/m from the scaffold, and the electron-root flux still moves 2.7% for $N_\zeta$ 55 to 67 at $N_\xi$ 192, so the profile is not promoted.
- Docs: `docs/user_guide/convergence.md` gains the per-radius certificate and a measured table of the grid to choose by collisionality (W7-X, LHD, tokamak) and by angle scheme.
- Validation: `cross_code_gaps` gains W7-X Phi1 in the kinetic equation at E_r = -2 with a C6+ impurity against SFINCS (four resolutions, fluxes within 1.2e-7, flows within 1.8e-5 scaled; DKX peaks at 5 to 11 GB where SFINCS needs about 1 GB, and 13x45x48x5 exceeds 14 GB), and a converged monoenergetic ladder on W7-X EIM. DKX, YANCC and MONKES each converge, but D11* is 0.2656, 0.2947 and 0.3024 (DSHAPE agrees to 0.09%), so the 6% gate is not tightened. The sealed 5.5% came from finite-difference angles. The improved Sugama model is pinned against Spitzer-Harm (22.6% to 5.4% low for Z = 1 to 16). RHSMode 4/5: SFINCS's adjoint output differs from its own finite-difference check by up to 66% on W7-X, so there is no Fortran oracle.
- The bootstrap current against Redl is a resolution and field effect, not a convention error. Tutorial 12's 0.64 (FP over Redl) is the default grid; on that non-quasi-symmetric seed it falls to 0.57 at 29x29 angles (Nxi=48, Nx=6, 0.57 to 0.58 across 21 to 37 angles). DKX and SFINCS (v3, 1 rank, MUMPS) agree on the same decks to 1e-4 to 4e-4 (QA beta 2.5%, s=0.5, four grids) and 2e-4 (seed, 17x17x64x7). Where Redl is exact, a large-aspect-ratio tokamak, FP is 1.015 and 1.018 of it at n = 1e19 and 1e20 m^-3 (41 angles, Nx=7, Nxi=48), pitch-angle scattering 1.15; new `tests/test_bootstrap_redl_tokamak.py` pins this at 3% (43 s cold). Precise-QA beta 2.5%, s=0.5, FP over Redl: 0.98 at 25x37x64x7, 0.89 at 25x37x32x5, but 0.82 and 1.32 at the angular grids 19x29 and 21x31 (both codes), so check angles as well as Nxi and Nx. Office host, 8 pinned cores, load 18 to 30, JAX 0.12.0. Docs: tutorial page 8; tutorial 12 comment.
- The bootstrap current against Redl is a resolution and field effect, not a convention error. Tutorial 12's 0.64 (FP over Redl) is the default grid; on that non-quasi-symmetric seed it falls to 0.57 at 29x29 angles (Nxi=48, Nx=6, 0.57 to 0.58 across 21 to 37 angles). DKX and SFINCS (v3, 1 rank, MUMPS) agree on the same decks to 1e-4 to 5e-4 (QA beta 2.5%, s=0.5, four grids) and 2e-4 (seed, 17x17x64x7). Where Redl is exact, a large-aspect-ratio tokamak, FP is 1.015 and 1.018 of it at n = 1e19 and 1e20 m^-3 (41 angles, Nx=7, Nxi=48), pitch-angle scattering 1.15; new `tests/test_bootstrap_redl_tokamak.py` pins this at 3% (43 s cold). Precise-QA beta 2.5%, s=0.5, FP over Redl: 0.98 at 25x37x64x7, 0.89 at 25x37x32x5, but 0.82 and 1.32 at the angular grids 19x29 and 21x31 (both codes), so check angles as well as Nxi and Nx. Office host, 8 pinned cores, load 18 to 30, JAX 0.12.0. Docs: tutorial page 8; tutorial 12 comment.
- Bootstrap current against Redl in the precise quasi-helical field of Landreman and Paul (nfp 4, reactor scale, Zeff = 1, Te = Ti, helicity N = -4), s = 0.25, 0.5, 0.75, full Fokker-Planck at 31x31 angles, Nxi=64, Nx=6: 1.050, 1.069, 1.012 of Redl, within 1e-3 of SFINCS v3 (1 rank, MUMPS) on the same decks. At s = 0.5 the ratio is 1.131, 1.069, 1.037 for 25, 31, 37 angles (s = 0.75: 1.012, 1.003), so angles still move it by 3%. Nxi=32 gives 0.94 and is under-resolved (SFINCS agrees). New `tests/test_bootstrap_redl_qh.py` (86 s, 4 cores, marked slow) pins DKX to SFINCS at 1% and to 1.12 of Redl at 25x25, Nxi=48, Nx=5; the wout is the simsopt test file (MIT, 0.8 MB as xz). Office host, JAX 0.12.0, 4 to 8 pinned cores, load 18 to 25.


## v2.10.1 — 2026-10-20

One CLI: `dkx run` and `dkx scan` replace the `dkx sfincs …` spellings (kept as
deprecated aliases), `dkx.scan(...)` drives scans from Python, and a page on
bringing an existing SFINCS scan workflow to DKX.

- One CLI: `dkx run` and `dkx scan` accept a SFINCS `input.namelist` as well as a case file. `dkx run` with no argument in a folder holding `input.namelist` writes `sfincsOutput.h5`, as the SFINCS executable does. `dkx scan` reads the deck's `!ss` lines (scanType 2: E_r, 5: radius x E_r) and writes the `sfincsScan` folder tree; `--profiles`, `--er MIN MAX N`, `--radii`, `--jobs`, `--skip-existing` override or tune it. `dkx.scan(namelist, profiles=..., er=...)` is the same from Python.
- `dkx sfincs <command>` is hidden and prints a deprecation note; use `dkx <command>`. `scan-er` and `scan-profile` stay as hidden aliases.
- New docs page, "Bring your SFINCS workflow".

## v2.10.0 — 2026-10-20

Production scans that finish: a robust ambipolar root search, radial profiles
that bracket only certified points and continue between radii, scans that keep
going past failed points, and `dkx sfincs scan-profile` for radius x Er scans
from a SFINCS namelist and profile file. Large-|E_r| Fokker-Planck points that
stalled for 20 minutes now solve in about a minute. Memory: monoenergetic runs
use 1.3 GB instead of 24 GB, and multi-surface runs stay flat without clearing
the compile cache. New cross-code validation against SFINCS, an 8-page guided
tutorial, and a faster CI.

- Fix: host memory grew with every surface of an ambipolar profile because each batched solve (`batched_solve`, one per E_r evaluation) ran an eager `lax.map` over a fresh closure: one executable compiled and kept per call, 11 per surface. The compiled map is now cached on the operator it closes over and freed with it, and `dkx run` no longer calls `jax.clear_caches()` between surfaces. W7-X wout, 5 surfaces, 15x15x12x4 FP ambipolar, 8 pinned cores of the shared office host (load 10-13), cold/warm compile cache: RSS after the last surface 1.59/1.40 GB with the clear on main, 4.5/4.4 GB without it, 0.80/0.61 GB now; wall time 198/201 s on main, 170/113 s now; backend compiles 481 (main), 82 (now). Plain profile, same case: 33.6/24.3 s to 21.0/12.4 s, 461 to 97 compiles. Single-device and sharded maps now agree to the last ulp (3.5e-18 on states of 1e-3) rather than bitwise.
- Fix: `scan_failures.json` (and the failed-point list of `scan-er` and `scan-profile`) recorded the failed E_r rounded to 4 significant figures, read back from the point's directory name. It now records the float that was solved.
- The ramped truncated structured solve is one compiled executable per layout. It ran its `lax.scan` sweeps eagerly over fresh closures, so every solve (and every E_r field of a scan) retraced and compiled one executable per `Nxi_for_x` group (8 on the tokamak decks). Now `_truncated_sweeps` is a module-level `jit` taking the operator as an argument. `tokamak_1species_PASCollisions_noEr` through `dkx sfincs write-output`, 2 pinned cores of the shared office host, alternating A/B/A/B: cold 14.5-15.0 s to 9.5-9.6 s (235 to 131 compiles), second solve in the same process 0.87-0.94 s to 0.13 s (8 to 2 compiles). Peak RSS +50-70 MB (465 to 515-537 MB).
- Tests: the dense-route memory budget is pinned for the suite (`tests/conftest.py`). `_coarse_memory_budget` reads available memory, so on a loaded host (office, 0 GB free) six `test_speed_triangle_precond` and `test_tier2_memory_guard` tests failed on a correct tree; with the budget pinned all pass at 0 GB free.
- `dkx sfincs scan-er` no longer loses a scan to one bad point. A point that raises is recorded in `scan_failures.json`, the other points are written, and the command exits 1. When the shared-operator batch cannot converge a point, that point is retried alone. When the batch would be a traced GMRES (full Fokker–Planck with the E_r terms, which has no preconditioner escalation and no early exit) the scan goes point by point from the start; a batch of 9 E_r values for an HSX QHS full-FP deck at 66,004 unknowns had written no point after 50 minutes (8 pinned cores of the shared office host, load 25).
- New `dkx sfincs scan-profile`: the namelist-route equivalent of upstream `sfincsScan` with `scanType = 5` (radius × E_r from a profile file and the deck's `!ss` lines). A malformed `!ss` or profile line raises an error naming the file and line, and a radius outside the profile file's range is refused (it used to take the edge value and a zero gradient).
- Krylov escalation at large |E_r| with the xDot/xiDot terms. On an HSX QHS full-FP deck (66,004 unknowns, 8 pinned cores of the shared office host) the points at |E_r| = 15 and 20 took 19 to 24 minutes each: the `coupled` preconditioner, which factors the operator without the E_r terms, stalled after 768 to 1,262 iterations, and each escalation rung then retried at the probe restart of 30, where only the last rung (wide restart) converged. `auto` now starts with `coarse_triangle` when the E_r terms are active and nonzero and its bands fit, and every preconditioner rung follows the restart policy of the first attempt: the same four points take 44 to 114 s and all nine E_r points from -20 to 20 converge (main: 16 to 1,415 s).
- Fix: the `sparse_triangle` rung of the escalation always failed with `UnexpectedTracerError`. SOLVAX's compile reuse turns arrays closed over by a jitted callback into jit arguments, including those of the nested host callback, so `upper` and the operator reached it as tracers. The host state is now on one opaque object.
- Fix: the stall message named the requested preconditioner (`coarse`) instead of the one `auto` had used.
- The last escalation rung runs SuperLU above `max_dense_size` when its estimated peak fits the memory available now (1.25 x 9.9 GB x (n/66,004)^1.6; the HSX QHS full-FP deck at E_r = -20, 66,004 unknowns, converged on the sparse direct route in 47 minutes at 9.9 GB on 8 pinned cores of a loaded host). Before, any stall above 8,192 unknowns raised after the iterative rungs, and the message asked for MUMPS. When the estimate does not fit, the error states it.
- Docs: the tutorial track is rewritten as eight guided pages (first run, reading results, geometry and species, collisions and solvers, scans and resolution, ambipolar `E_r`, transport matrices, `Phi1`/bootstrap/gradients) with the real output and figure of every step, replacing the four-part track. `13_convergence.py` gains a resolution ladder. Figures are palette-compressed (`docs/figures/publish_tutorial_figures.py`, 244 KB for 13) and registered in `docs/figure_provenance.json`.
- Fix: `find_ambipolar_er` kept every field's full solve state, preconditioner included, in its `J_r` memo, so RSS grew with each field (a W7-X 15x45x80x6 FP run passed 14 GB after 5 fields). The memo now keeps the warm start only, and the single reuse preconditioner is held only when `precond_window > 0`.
- CI: the coverage shards restore a per-shard JAX compilation cache (written by pushes to main, read by every run, keyed on jax version and `pyproject.toml`) and the matrix drops from 13 to 9 shards. Required check names are unchanged.
- Fix: host RSS grew over an ambipolar profile because executables were compiled and kept per field and per surface. The coupled preconditioner now applies through one module-level jitted sweep (it ran an eager `lax.scan` over fresh closures on every application), and `dkx run` clears JAX compile caches between surfaces. W7-X, 5 surfaces, 13x37x40x5 FP, 2 species, 6 cores of the shared office host: peak RSS 13.5 to 6.9 GB, 5/5 surfaces converged both times, total 1450 to 1611 s at load 9 to 22.
- `dkx/cli.py` (2,481 lines) is split by owner: `cli.py` keeps the SFINCS-compat commands and `main` (1,096 lines), `cli_user.py` the user commands, `cli_parsers.py` the argparse builders, `cli_front.py` the argv front door. `dkx.cli` re-exports every name tests and entry points use; no behaviour change.
- `dkx/representative.py` (2,164 lines) loses its figure panels and plotting entry points to `representative_plots.py` (649 lines; `representative.py` is 1,559). `dkx.representative` re-exports them; no behaviour change.
- `dkx/workflows/optimization.py` (2,124 lines) is split by owner: `optimization.py` keeps objectives and scan promotion (1,078 lines), `optimization_evidence.py`, `optimization_comparison.py` and `optimization_ladder.py` hold the evidence plans, payload comparison and resolution ladder. `dkx.workflows.optimization` re-exports every public name; no behaviour change.
- Tests: four small modules folded into their natural owners (four fewer files, same 89 tests collected): `test_plot_pdf_paging_*` into `test_plotting`, `test_audit_suite_output_keys` into `test_audit_suite` (was `test_audit_suite_runtime_drift`), `test_repo_size_policy` into `test_release_size_contracts`, `test_cached_grids_readonly` into `test_phase_space`.
- `dkx/multigrid.py` (1,901 lines) is split by owner: `multigrid_levels.py` (settings, simplified operator, transfers, coarsening), `multigrid_smoothers.py` (level matvec, coarse solve, line and plane smoothers), `multigrid_diagnostics.py` (pitch-basis measurements); `multigrid.py` keeps the docstring and the preconditioner builders (586 lines). `dkx.multigrid` re-exports every name; no behaviour change.
- `dkx/phase_space.py` (1,877 lines) is split by owner: `phase_space_angles.py` (uniform and widened-upwind angle grids), `phase_space_speed.py` (speed grid and its derivative matrices); `phase_space.py` keeps the Legendre pitch machinery, `Grids` and `make_grids` (668 lines). `dkx.phase_space` re-exports every name; no behaviour change.
- Monoenergetic solves stay lean. When the truncated block-Thomas solve misses `tol` by rounding (low collisionality), `auto` now applies one defect correction before falling back to dense-preconditioned Krylov. W7-X at ν*=3e-4, grid 23×55×140, office Xeon on 8 pinned cores, NTX's cross-code harness: peak RSS 23.9 GB (recorded run) to 1.26 GB, warm 101 to 85 s. Alternating A/B/A/B at 19×41×100 with a 1 GB structured budget: 6.1–7.4 to 1.09 GB, warm 47–49 to 19–21 s. D11, D31, D33 unchanged to 1e-9.
- Validation: production-resolution DKX/SFINCS cells for W7-X ambipolar roots at three radii, precise QA and QH, a Boozer `.bc` surface, the trajectory-model Er sweep and RHSMode 2/3 (`validation/cross_code_gaps_v1.json`); fluxes agree within 2.7e-4.

- Fix: the memoized speed grids (`make_x_grid`, `make_speed_grid`) and parsed `.bc` surfaces returned shared, writable arrays, so one caller's in-place edit changed every later result for the same arguments. Their arrays are now read-only.
- Fix: `KineticBootstrapCurrent` read `aHat` with netCDF4 only and fell back to `aHat = 1` on any error. A LIBSTELL text `wout` (which `geometryScheme = 5` accepts) or a missing `Aminor_p` therefore scaled every `d/drHat` gradient in the deck by `aHat` without warning. It now reads through `read_vmec_wout` and raises when `Aminor_p` is missing or not positive.
- The persistent compilation cache keys on the XLA:CPU thread count: entries live
  under `threads<NPROC>/`, so an executable compiled under one `--cores`
  setting is not reloaded under another.
- Smaller clone: committed PNGs are palette-quantized and recompressed (1.91 MB to 1.15 MB, same pixels to the eye, same generators), 15 evidence JSONs that nothing reads are deleted, and `tools/parity/output_parity_vs_fortran_fixture.py` is gone (the CLI driver CI runs does the same comparison).
- The namelist-to-operator builder moved from `dkx/drift_kinetic.py` (2,697 to 1,877 lines) to `dkx/operator_build.py`. `dkx.drift_kinetic` still exports `kinetic_operator_from_namelist`, `kinetic_operator_build_from_namelist` and `KineticOperatorBuild`; no behaviour change.
- The namelist-to-operator builder moved from `dkx/drift_kinetic.py` (2,697 to 1,890 lines) to `dkx/operator_build.py`. `dkx.drift_kinetic` still exports `kinetic_operator_from_namelist`, `kinetic_operator_build_from_namelist` and `KineticOperatorBuild`; no behaviour change.
- `dkx/magnetic_geometry.py` (2,373 lines) keeps geometry construction (1,471 lines); the Boozer `.bc` reader moved to `boozer_bc.py` and the VMEC `wout` reader to `vmec_wout.py`. `dkx.magnetic_geometry` re-exports both; no behaviour change.
- Ten `tests/test_output_h5_*_parity.py` modules (1,232 lines) are one parametrized `tests/test_output_h5_parity.py` (254 lines): 34 deck cases over the writer and canonical routes, every compared key and tolerance kept (checked key-for-key against the old lists).
- `dkx/solve.py` (3,641 lines) is split by route: `solve.py` keeps routing and `solve()` (986 lines), `solve_structured.py` the structured direct route, `solve_krylov.py` recycled Krylov, `solve_direct.py` sparse direct, `solve_common.py` the result types and residual guards. `dkx.solve` re-exports its public names; no behaviour change.
- `dkx/solve.py` (3,641 lines) is split by route: `solve.py` keeps routing and `solve()` (965 lines), `solve_structured.py` the structured direct route, `solve_krylov.py` recycled Krylov, `solve_direct.py` sparse direct, `solve_common.py` the result types and residual guards. `dkx.solve` re-exports its public names; no behaviour change.
- `dkx/solve.py` (3,641 lines) is split by route: `solve.py` keeps routing and `solve()` (966 lines), `solve_structured.py` the structured direct route, `solve_krylov.py` recycled Krylov, `solve_direct.py` sparse direct, `solve_common.py` the result types and residual guards. `dkx.solve` re-exports its public names; no behaviour change.
- `examples/` is a numbered tutorial ladder, `examples/tutorials/01..15`, from a first native run to a shape-optimization loop. Each script reads top to bottom: imports, input parameters, run with progress printed, then plot, save and print a summary. `DKX_EXAMPLES_CI=1` runs any of them in seconds. The heavy VMEX scripts moved to `examples/advanced/`; the old numbered folders, `getting_started/`, `autodiff/`, `transport/`, `vmex_finite_beta/` and the notebooks are gone.
- The docs have a four-part tutorial track (`docs/tutorials/`) that follows the scripts: first steps, building a case, transport physics, and bootstrap current, gradients and optimization. Each part gives the equations, the command and the output it printed. The README links the new example paths and the quick start runs as written.
- `docs/design_decisions.md` gives the reasons for the velocity discretization (Legendre modes, polynomial speed grid) and for the native defaults (Fokker–Planck collisions, DKES trajectories).
- `docs/design_decisions.md` compares DKX's solver with DKES, MONKES, SFINCS, yancc, NEO-2 and KNOSOS and gives the route per run type; plan section 17 records the decisions and the GPU kernel measurement.
- `radial_profiles` brackets only fields whose solve passes the original-equation check (`algebraic_converged`, with the full state retained) and lists the rest in `uncertified_er`. Before, a failed field entered the bracketing. Each surface also samples its neighbour's root (`er_seed`), extension probes merge into the scan instead of a second full scan, and a failure anywhere on a surface, including root refinement, is recorded for that surface rather than ending the profile. The two silent `except Exception: break` fallbacks are gone. Records carry their `resolution`.
- `find_ambipolar_er` no longer stalls on a stale preconditioner. Brent's first two fields are the bracket ends, and reusing the preconditioner built at -12 kV/m at +12 kV/m held a W7-X FP solve (r/a = 0.5, 15x31x32x6) past 2,400 s. Preconditioners are now rebuilt at every field by default and each field starts from the nearest solved state; `precond_window` opts into reuse near the build field, as bounded GMRES (about 200 iterations, under `auto` too) with a fresh-build retry. See `docs/numerics/factor_reuse.md`.
- `find_ambipolar_er` solves each field once: `J_r` is memoized, the cold duplicate solve at the root is gone (every accepted state already passes the original-equation check), and `all_roots` reuses Brent's fields and reports the Brent root itself for its own sign change, so the enumerated and selected roots agree exactly. `all_roots` is now opt-in. `wall_time_s` and `n_solves` report the whole call.
- `current_tol` is relative to the largest `|J_r|` seen while bracketing (`current_scale`), default `1e-6`; it was an absolute `1e-10` in normalized units. `dkx sfincs ambipolar --current-tolerance` and `run_ambipolar_brent` follow.
- A field whose solves all fail is retried cold, then under `auto` with a 100x tighter kinetic tolerance; if that fails too the result is `status="solve_failure"` with the attempts in `failures`, instead of an exception.

## v2.9.0 — 2026-10-07

Exact direct solves for Fokker–Planck decks that fit, a coupled preconditioner
that `auto` chooses when memory allows, native and validated Phi1 with an
adjoint, spectral angles for monoenergetic databases, a better default CPU
thread split, and case inputs without a `schema` key.

- Default CPU threads: with neither `DKX_CORES` nor `NPROC` set, XLA's pool gets `min(8, cores)` and BLAS `min(4, cores)` threads, counting the cores the process may run on (taskset aware, not `os.cpu_count()`). Before, BLAS was single-threaded, which left XLA's sequential LAPACK custom calls (`getrf`, `trsm`) on one core. Explicit `DKX_CORES`, `NPROC` and BLAS variables are unchanged, and forced host devices keep one BLAS thread. Office Xeon, warm, main to this change: W7-X monoenergetic 2.55 to 1.67 s pinned to 4 cores and 2.60 to 1.75 s unpinned; NCSX full FP 5.2 to 5.0 s and 5.3 to 4.8 s; sparse direct unchanged at 23.4 s; the bounce-averaged surrogate unchanged. A 14-core laptop is within noise.

- `tools/benchmarks/cross_code_speed.py --dkx-xla-threads N` sets DKX's XLA pool apart from YANCC's. On four pinned x86 cores, XLA's pool and OpenBLAS compete inside every LAPACK call, and one XLA thread makes DKX's warm monoenergetic solve 2–3× faster (W7-X 3.4 to 1.6 s, HSX 19.5 to 6.8 s; MONKES 0.32 s and 1.5 s).

- `monoenergetic_database` now uses spectral (Fourier) angular derivatives when the deck does not set `thetaDerivativeScheme`/`zetaDerivativeScheme`. A deck that sets them keeps its choice. On the matched W7-X/HSX benchmark this is the discretization MONKES uses, and it converges like MONKES: HSX reaches 3% at 19x41x64 instead of 25x51x96, and W7-X is at 4e-5 one rung above 15x31x48 (finite differences: 0.6% at 25x51x96). Values at a fixed grid change within discretization error. To pair a database with a full-kinetic solve of the same deck, set the schemes in the deck.
- The full-band structured direct route (`build_tier1_solver`) builds its Legendre rows one at a time inside the elimination scan, stores only the Schur LUs, and regenerates the off-diagonal blocks in the substitution sweeps. Answers are unchanged (1e-12). On W7-X 15x31x48 the peak RSS falls from 1.95 GB to 0.94 GB and the first solve without a compilation cache from 8.2 s to 3.7 s; the warm solve is 8% slower. With the default persistent cache the first solve takes 1.4 s.
- `tools/benchmarks/cross_code_speed.py` has a `cached` stage that times DKX's first solve with the persistent compilation cache on.

- Phi1 is promoted from `compatibility_only` to `validated_limited`. On the geometryScheme 4 Phi1 example deck and a trace-carbon variant, DKX matches Fortran SFINCS v3 to 2.0e-6 over an eight-rung resolution ladder (`validation/phi1_sfincs_benchmark_v1.json`). `stable_candidate` is withheld: there is one geometry, Fokker-Planck collisions only, zero Er, and the trace-impurity flux is resolved only to about 3% in Nx.
- `dkx.phi1.phi1_solution` gives the converged Phi1 state with a matrix-free implicit adjoint: GMRES on the transposed Jacobian (a VJP of the coupled residual) on the active Legendre-truncated subspace, preconditioned by the last Newton step's transpose preconditioner. On a native `phi1 = "kinetic"` case the gradient matches central finite differences to 1.2e-9 and the Taylor remainder falls as h^2 (ratios 4.02, 4.01).
- Case inputs no longer carry a `schema` key. Older files with `schema = 1`
  still load (the key is ignored); other values are refused. Case IDs are
  unchanged.
- `dkx schema` is renamed `dkx template` (`schema` stays a hidden alias for one
  release). `Case` no longer has a `schema` attribute and `SCHEMA_VERSION` is
  no longer exported; case IDs are unchanged.

### Solvers

- Full Fokker–Planck and improved Sugama decks with DKES trajectories, no
  tangential drifts and no Phi1 now have an exact structured direct route: a
  block elimination along the Legendre index whose blocks couple all species
  and speeds (`dkx.structured_direct.build_coupled_solver`). It eliminates from
  the highest Legendre mode down, lifts the momentum null space of the `L = 1`
  block with a rank-`Nspecies` term removed again by Woodbury, and handles every
  `constraintScheme` border. `method="auto"` takes it within the structured
  memory budget and 150 GFlop per right-hand side; on the upstream suite it
  solves `transportMatrix_geometryScheme2` in 2.8 s against 33 s (1,181 GCROT
  iterations) and `transportMatrix_geometryScheme11` in 9.9 s against 48 s, and
  leaves `geometryScheme4_2species_noEr` and the W7-X/HSX decks, where Krylov is
  faster or the blocks do not fit, on recycled Krylov. A speed-triangular sweep
  was measured first and is not exact in this speed basis (lower triangle 1-3%
  of the diagonal, electron-ion block as large as its diagonal).
- `preconditioner="coupled"`: the speed-coupled elimination as a Krylov
  preconditioner, exact for Legendre blocks `L < 2` and `(species, x)`-diagonal
  (float32) in the tail; `E_r` and drift `L ± 2` terms are dropped from the
  factored operator. Iterations fall from 46 to 17 on a 396k-unknown HSX FP
  deck at equal wall time.
  It keeps the `L`-diagonal magnetic drifts at `L <= 2` (23 to 17 iterations on
  the reduced W7-X drift deck) and stores tail bands as per-species blocks times
  the speed scale, `Nx` times smaller.
  The tail is now eliminated and swept with `lax.scan` (padded with identity
  blocks where `Nxi_for_x` truncates), its bands are kept as two scalars per `L`
  over shared float32 streaming and mirror blocks, and the dense `L < 2` Schur
  step is formed one column pair at a time: the compiled HSX FP production
  factorization drops from 11.5 GB output + 26.6 GB temporaries to
  11.7 GB + 4.4 GB.
  Its tail and dense `L < 2` blocks are now factored in float32: on the
  production HSX FP deck (1.88M unknowns, 8 cores) 17 iterations, 508 s and
  16.6 GB, against 1,100 s and 15.0 GB for the coarse preconditioner.
- `method="auto"` now preconditions Fokker–Planck and Sugama Krylov solves with
  `coupled` when its estimated peak fits the Krylov memory budget (default: half
  the available memory), else `coarse`.
- `dkx.solve` keeps its API; the structured route's applicability tests and
  memory model moved to `dkx.structured_direct`. Structured solves refine
  against the pinned operator (identity rows on truncated `Nxi_for_x` DOFs).
- Native `[physics] phi1 = "kinetic"` runs the coupled kinetic + quasineutrality + gauge Newton–Krylov solve on `profile` cases and is accepted by the nonlinear residual; `dkx convert` maps kinetic-only Phi1 decks to it. `"full"`, adiabatic species, quasineutrality option 2 and the ambipolar workflow stay refused. Status remains `compatibility_only`.

## v2.8.0 — 2026-10-05

A converged kinetic bootstrap row compiled as one program, a momentum-conserving
correction for pitch-angle-scattering solves, refined ambipolar roots, and the
options `vmex --neoclassical` uses.

### Documentation and repository

- Documentation rebuilt (#292): getting started, user guide (every case field,
  outputs, CLI, Python API), tutorials, how DKX works, physics, numerics,
  benchmarks, examples, features and design decisions. The README defines the
  case schema before using it.
- Historical development files removed (#293, #298); dead doc citations
  repointed; `dkx run --out` help, the `dkx.run` method docstring and the
  analytic tokamak wording corrected; `[output].plots` wired.
- Ambipolar root stability is classified from the outward current (#301); the
  optional Boozer backend is declared as the `booz` extra (#302).

### Bootstrap objective

- `KineticBootstrapMismatch` defaults to `Nxi = 48` (was 16). Pitch-angle
  convergence study (plan.md 16.7): on the VMEX QA beta = 2.5% deck with its
  published profiles, `<j.B>` at `Nxi = 48` is within 1.1% of `Nxi = 96-128`
  for |E_r| >= 5 kV/m and within 3% at |E_r| <= 1 kV/m; `Nxi = 16` was 16-44%
  low. A realistic E_r does not converge at far lower `Nxi` (one step of the
  ladder at most), so the row keeps `E_r = 0` by default. Rows built with an
  explicit `resolution` are unchanged.

- `KineticBootstrapCurrent(ambipolar=True)` refines its bracketed ion root
  with the same Brent refinement as the representative radial scan (at most
  4 more single-field solves) and reads `<j.B>` from the solve at the refined
  root instead of interpolating across the scan bracket.

### Representative run

- The ambipolar root of each surface is refined after bracketing: Brent's
  method on the bracket, at most 4 more single-field solves, stopping at
  |J_r| <= 1e-6 max|J_r| or a 0.01 kV/m bracket. Moments are read at the
  refined root instead of interpolated across a 10-20 kV/m bracket. On the
  VMEX QA beta = 2.5% bootstrap deck with its published profiles (default
  grid) the roots move from -11.8, -25.8, -49.0, -39.8, -43.3 to -11.9, -26.1,
  -46.5, -43.4, -39.0 kV/m, and <j.B> at r/a = 0.55 from -4755 to -3192
  kA/m^2, for 18 extra solves (radial-scan wall time 59 s to 117 s). The
  outer roots stay non-monotone: there J_r is within 1-5% of zero over tens
  of kV/m and its crossings move by tens of kV/m on the `full` grid, so that
  spread is a resolution effect, not a bracket artifact.
- The ambipolarity panel spans its row; the two empty slots beside it are gone.
- `run_representative` accepts `surfaces`, `er` (ambipolar root or a
  prescribed kV/m), `collision_operator` (`fp`, `pas`,
  `pas+momentum_correction`), explicit `profiles` (`ne_coeffs`, `Te_coeffs`,
  `Ti_coeffs` polynomials in s, as VMEX's `KineticProfiles`; T_i may differ
  from T_e) and `redl_jdotb`, an external Redl curve drawn on the bootstrap
  panel. Defaults are unchanged.
- Fix: the radial scan solved every surface with the n, T and gradients of
  r/a = 0.5. Each surface now uses its own plasma. On the VMEX QA beta = 2.5%
  bootstrap deck this removes the r/a = 0.55 outlier (-46 kV/m between -31
  and -28). The Er profile is now monotone: -13, -27, -50, -60, -61 kV/m at
  r/a = 0.25 to 0.85. A dense scan confirmed that the root itself was a real
  crossing for the plasma it was given, not a root-pick or bracket error.
- The evaluated root is the most negative *stable* root under the
  outward-current rule of #301. The ambipolarity panel is now drawn for every
  surface, with each root labelled ion, unstable or electron.
- Fix: the caption no longer clips at the canvas edges. Long lines wrap, and
  the layout rect's height no longer runs past the top of the figure.

### Physics

- Rewrite `dkx.momentum_correction` as the Sugama–Nishimura moment method on the
  structured pitch-angle solve (Phys. Plasmas 9, 4637 (2002); Maassberg, Beidler &
  Turkin, Phys. Plasmas 16, 072504 (2009)). The parallel particle flow, heat flow and
  next Sonine moment of every species are coupled through the full Fokker–Planck
  operator (friction `l^ab_ij` and field-particle restoration from DKX's own
  Rosenbluth-potential operator), the projection is weighted by `nu_D` so the model
  conserves total parallel momentum exactly, and `<j.B>`, flows and back-substituted
  radial fluxes are read off one corrected state. `momentum_corrected_solve(pas, fp)`
  is traceable and differentiable; `friction_drives` caches the velocity-space part.
  `KineticBootstrapMismatch(collision_model="pas+momentum_correction")` uses it as a
  VMEX objective row. Accuracy against full Fokker–Planck on the same grid is in
  `docs/physics/reduced_models.md`.

### Removed

- The single-moment, database-based correction (`ParallelViscosity`,
  `parallel_viscosity`, `parallel_friction_matrix`, `solve_corrected_flows` and the
  database signature of `momentum_corrected_bootstrap`). It carried the parallel
  particle flow only, so the temperature-gradient bootstrap drive was lost, and closed
  the system with an ad hoc `M0 V_unc` drive. Accessing the old names raises an error
  that names the replacement; `dkx.api.momentum_corrected_bootstrap` now takes the
  pitch-angle operator and its Fokker–Planck twin.

### Performance

- Retain the Fokker-Planck speed triangle in the sparse Krylov preconditioner
  (#303). `solve(preconditioner="sparse_triangle")` keeps the self-species
  upper speed triangle that `"sparse"` drops, reusing the same sparse LU
  factors by back-substitution over speed. The stalled-solve escalation now
  tries it in place of `"sparse"` on Fokker-Planck and Sugama decks; an
  explicit `preconditioner="sparse"` is unchanged. On the HSX Fokker-Planck
  DKES deck it cut GCROT iterations from 46-55 to 16-18 across a four-rung
  ladder (54,564 to 786,244 unknowns) and on the W7-X paper deck from 22-26
  to 10-11, with currents and fluxes equal to the solver tolerance.
- Compile the kinetic bootstrap-current row of a VMEX optimization as one
  program. `KineticBootstrapMismatch` traced one Boozer transform and one
  kinetic solve per surface as an unrolled loop, was dispatched op by op when
  VMEX evaluated it eagerly at set-up, and was differentiated forward through
  the kinetic solve once per boundary dof by VMEX's block Jacobian. The
  surfaces are now one batch axis of one compiled program, an eager call
  compiles once per runtime, and the forward derivative applies the
  reverse-mode Jacobian (one adjoint solve per surface) to every tangent
  (`batched_with_reverse_jvp`). The Boozer plan is also built under an outer
  `jit`, which used to raise. Values and Jacobian rows are unchanged to
  3e-14 and 1.4e-12 relative (8.6e-6 for the Fokker-Planck Krylov route,
  whose tangent solves become adjoint solves at the same tolerance). On a
  shared 36-core host (8 pinned cores, load 8-33), the row's added cost over
  the same VMEX problem without it fell from 92-122 s to 56-74 s for set-up
  plus one residual and one Jacobian (three surfaces, default grid); an eager evaluation from 68-80 s cold and
  6-7 s warm to 20-26 s and 0.6 s; the example's CI smoke pass from 358 s
  to 298 s.

## v2.7.0 — 2026-09-27

Faster by default and ready for optimization: a memory-aware Krylov restart
that removes the high-`Nx` stall, a self-compiling structured route for
callers that do not `jit`, and a drift-kinetic bootstrap-current objective for
VMEX optimization.

### Correctness

- Refuse failed bootstrap-current evidence (#274). `KineticBootstrapCurrent`
  replaced a nonfinite surface current with zero, so an unsuccessful solve
  looked like an optimal one; nonfinite currents, residuals and squared terms
  and a non-positive reference current are now refused, and the ambipolar
  interpolation requires every scan point to be accepted and finite. The
  scan-promotion audit likewise substituted zero when an output had neither
  `FSABjHatOverRootFSAB2` nor `FSABjHat`, passing the gate with a zero
  objective; a missing or nonfinite bootstrap current is now refused.

- Convert the handedness of the Boozer route in one place
  (`boozer_route_psi_a_hat`, and its traced form
  `convert_boozer_route_handedness`): `psiAHat = |phi_edge|/(2 pi) signgs
  sign(G + iota I)`. A VMEX equilibrium has `signgs = -1` and `booz_xform`
  returns `G > 0`, so taking `psiAHat = +|phi_edge|/(2 pi)` flipped every flux
  and `<j.B>` on that route against the VMEC-file route and Redl.
  `optimize_QA_bootstrap.py`, `optimize_QH_bootstrap.py` and the gradient hook
  of `bootstrap_consistency_kinetic_loop.py` printed the flipped sign; their
  objectives were squares or magnitudes, so the optimizations were unaffected.

### Performance

- Retain only the active pitch rows in reusable coarse factors (#273). The
  generated reusable preconditioner factored every subsystem through all `Nxi`
  rows, including the masked identity rows past `Nxi_for_x`; it now factors
  `max(1, Nxi_for_x)` rows and applies the uncoupled tails as
  `r / (1 + floor)` at factor precision. On the recorded NCSX layout that is
  744 rows instead of 1,111 (predicted float64 LU and pivots 14.5 GiB instead of
  21.7 GiB); a traced `Nxi_for_x` keeps the rectangular layout. The operator,
  masks, pinning and residual acceptance are unchanged.

- Memory-aware Krylov restart by default. `solve(restart=None)` and
  `SolverOptions(restart=None)`, the new defaults, run five FGMRES cycles of
  30 and two of 100, then continue from the iterate at the longest restart, at
  most 1,000, whose basis (`2 * restart * unknowns * 8` bytes, since flexible
  GMRES stores `V` and `Z`) fits `krylov_memory_budget_gb`
  (`DKX_KRYLOV_MEMORY_BUDGET_GB`, else a quarter of available memory). The
  previous policy widened only to 100 and only while two such bases fit in
  256 MiB, so decks above 167,772 unknowns never widened. On the HSX-like
  `Nx` ladder the iteration growth from `Nx = 10` to 16 was restart
  stagnation: 2,788 iterations at restart 200 against 357 at 1,000, and 7,167
  against 390 at `(Nxi, Nx) = (40, 16)`
  (`docs/experiments/2026-09-23-restart-and-direct-reach.md`). On the public
  tokamak full-Fokker-Planck deck at `(Nx, Nxi) = (24, 40)`, 87,362 unknowns,
  the default converges in 1,736 iterations where the previous one stalled
  and failed every rung of the stall ladder (`docs/performance.rst`, "Krylov
  restart length"). Decks that converged within 350 iterations follow the
  same path as before. The stall
  ladder's larger-budget rung spends its budget at the wide restart. An integer
  `restart` fixes the cycle size as before; explicit `method="gmres"`,
  differentiable and traced solves, and caller-supplied preconditioners keep a
  fixed size of 30 unless given one.
- `dkx solve-v3 --restart` was printed but never applied; its default reads
  `auto` and its help says so.

### Optimization

- `dkx.bootstrap.KineticBootstrapMismatch`: the drift-kinetic bootstrap
  current as a traced VMEX objective term, with the interface and normalized
  residual of VMEX's `RedlBootstrapMismatch`. On each kinetic surface the chain
  VMEX state -> `boozer_input_tables` -> `booz_xform_jax` -> DKX structured
  solve -> `<j.B>` is traced, so VMEX's implicit Jacobian differentiates it;
  `mismatch=False` gives a pure `<j.B>` row. Pitch-angle scattering is the
  default collision operator (cheap, no momentum restoration).
- `examples/optimization/QA_optimization_bootstrap_dkx.py` is VMEX's
  `QA_optimization_bootstrap.py` with the DKX row in place of Redl's
  (`BOOTSTRAP_MODEL = "dkx" | "redl" | "both"`), replacing the
  finite-difference version; the seed deck ships in `examples/data`.
- The structured direct route compiles itself for callers that do not `jit`
  (#279). The factorization, the refined substitution, the whole
  differentiable solve (one `custom_linear_solve` with the residual guard
  inside it), `KineticOperator.rhs` and `profile_moments_from_operator` each
  run as one cached executable per operator structure, so an eager
  `jax.value_and_grad` of an objective no longer dispatches, linearizes and
  transposes thousands of small operations one at a time. On the
  16,230-unknown structured deck, on heavily loaded hosts, the eager primal
  went from 5.0–5.9× to 1.18–1.30× of its compiled time and the eager
  gradient from 7.3–9.6× to 1.24–1.38×; the fixed eager overhead fell from
  1.33 s to 0.017 s (primal) and from 2.5 s to 0.10 s (gradient). Answers are
  unchanged to round-off, and the plain and differentiable solves return the
  same bits. The first call of each new operator structure pays a one-time
  compile.

## v2.6.0 — 2026-09-21

Acceptance tightened where it could pass a wrong answer, MUMPS as an explicit
direct backend, and convergence and optimization reports that no longer claim
more than their evidence supports.

### Correctness

- Admit each right-hand side against its own norm (#265). A multi-column solve
  admitted every column against the *largest* right-hand side's norm, so with
  norms `[1, 1e6]`, residuals `[1e-5, 0]` and a tolerance of `1e-10` the small
  column passed while missing its own target by a factor of 100,000. Each
  column now meets `max(atol, tol * ||b_j||)`, reused factors included; zero
  columns use the requested absolute tolerance, and nonfinite inputs, negative
  residual norms and invalid tolerances are refused.
- Include the magnetic drift's upwind support in the sparse assembly (#268).
  Streaming uses centred derivatives and the tangential magnetic drift separate
  upwind ones, whose toroidal radius is three points rather than two on the
  public reduced W7-X case; the pattern and the column groups now use the union
  of the active supports. The assembly's own `1e-10` check against the operator
  refused the old pattern, so no wrong matrix was ever factored; decks without
  magnetic drifts assemble exactly as before.
- Reject invalid optimization evidence and require a genuine refinement before
  promotion (#264): nonfinite or negative residuals, a single baseline grid
  labelled as a ladder, and a refinement that changes radius or species count
  are refused, and an absent backend comparison is reported as `untested`.

### Execution

- Select MUMPS explicitly as the sparse direct backend,
  `SolverOptions(method="direct", direct_backend="mumps", memory_budget_gb=...)`
  (#267). SuperLU stays the default and the default path is unchanged. The
  budget is checked after assembly and before factorization, against measured
  process memory, the COO storage, the right-hand-side buffers and the host's
  currently available memory, and again before a retained factorization is
  applied to a wider right-hand side. An explicit MUMPS request that cannot be
  met is refused and never silently falls back to SuperLU. It needs the MUMPS
  adapter of SOLVAX 0.25.0 and PyMUMPS; with SOLVAX 0.24 it raises an
  `ImportError` naming what is missing.

### Reporting

- Report grid convergence and original-equation acceptance as separate
  verdicts (#269). `dkx converge` keeps `converged` for the observable changes
  and adds `original_equations_accepted`, which requires complete, finite,
  typed residual evidence within the requested tolerance on every rung; the
  exit status is zero only when both pass. Native runs reapply the operator to
  the returned state and reject a false solver success.
- Compare bootstrap current in `dkx converge`, keep earlier rungs when a later
  one fails, and record failed and refused refinements without claiming
  convergence (#266). The NCSX record gains an executable reconstruction from
  pinned public sources with four SHA-256 checks.
- Report memory in one unit (#271). Traces mixed decimal MB with MiB and could
  present device capacity or a historical peak as current usage; profiling now
  uses MiB throughout and keeps current RSS, peak RSS and current device usage
  distinct.

### Development

- Let a cancelled CI run stop before the final gate (#270).

### Known limitation

- The sparse direct route's transposed solve can stall above its tolerance for
  a general cotangent. On the operator of
  `tests/test_operator_assembly.py::DECK` (1,204 unknowns) with the cotangent
  `numpy.random.default_rng(0).standard_normal(1204)`, it reaches a relative
  residual of `3.18e-8` against the forward solve's `1e-10` on the same
  factors, in 2.5.0 and here alike; further refinement sweeps do not move it.
  The route reports `converged=False` rather than a value, so this is an
  accuracy limit that is reported, not a wrong answer. The 1,962-unknown
  full-FP grid of #265 reaches `2e-13`, so the limit depends on the operator.

## v2.5.0 — 2026-09-20

The production solver program of #253: the operator assembled from products
with it, the sparse direct route rebuilt on that assembly and made reusable,
and the gap deck's iteration growth traced to the speed resolution with six
candidate causes measured and excluded.

### Execution

- Return the factorization the two direct routes build, in
  `SolveResult.factors`, and take it back through `solve(..., factors=...)`;
  `solve(..., transpose=True)` solves `A^T x = b` from those same factors. A
  transport matrix delivered as three separate calls cost three factorizations
  and now costs none: 0.96 s against 0.28 s on a 16,230-unknown structured
  deck, 1.19 s against 0.28 s on a 1,962-unknown sparse one. The sparse direct
  route had no adjoint at all; a transposed solve is now 0.15 of a primal,
  reached by substituting through the stored `L` and `U` in the other order.
  Reuse across a *neighbouring* operator is permitted and checked rather than
  assumed: the defect is measured against the operator actually passed in, and
  a solve that misses its tolerance refactorizes once, so recovery is bounded
  by a factorization instead of an unbounded Krylov wait.
- Take the operator's transposition once per solve rather than once per
  application. `_transposed_apply` called `jax.linear_transpose` inside the
  callable it returned, so every adjoint application re-traced the whole
  operator. The focused solve suites run in 247 s where they took 406 s.
- Assemble the sparse direct route's matrix in fewer products by dropping the
  requirement that the angular separation divide the grid. The grouping was
  already optimal for its own conflict structure -- every group held exactly 72
  columns and `633,600 / 72 = 8,800` -- but at `Ntheta = 11`, a prime, the only
  divisor above twice the stencil radius is 11 itself, so every theta became
  its own class. Checking the wrap-around gap directly instead takes the
  633,604-unknown deck from 8,800 products to 4,800, each product being one
  operator application; the recovered matrix is unchanged entrywise.

- Build the sparse direct route's matrix from products with the operator rather
  than one column at a time, scale it before factoring, and correct the defect
  once. Sampling costs one operator application per column, which is what
  `max_dense_size` bounds and why the route refused every production deck; the
  assembly costs one per group of columns that share no row, 5,508 for the
  66,004-unknown collaborator grid. That deck now solves in 846 s to a relative
  residual of `1.3e-14`, where the route previously refused it.
- Apply the exact speed triangle (`preconditioner="coarse_triangle"`) by
  back-substitution over `x`, rather than reaching the same inverse as a
  nilpotent series of `n_x` sweeps of the whole band. The map, the factors and
  the memory are unchanged, and one application at NCSX `(25, 37, 61, 8)` costs
  3.7x less: 2.3x a `coarse` application against 8.5x.
- Continue every escalation rung from the stalled iterate, rather than from the
  original guess, when that iterate is finite and improved on the right-hand
  side it started from. On a forced NCSX stall the rung that converges takes 18
  iterations instead of 27, for the same answer. A diverged iterate is refused,
  so a rung never starts further away than the original guess.
- Escalate a stalled recycled-Krylov solve to that triangle before `sparse` and
  `multigrid`. Those three are inverses of one simplified operator, so none of
  them answers a stall caused by the Fokker-Planck speed coupling that operator
  drops; retaining its upper triangle changes the operator being inverted, cuts
  NCSX iterations 2.3-4.1x, and reuses the factors the coarse route already
  built. A deck with no dense collision operator skips the rung, because there
  the triangle is the stalled preconditioner under another name.

### Research records

- The speed-triangle back-substitution measurement, and the batched LU dispatch
  measurement on CPU and GPU, in `docs/experiments/`.
- Why the gap deck is slow, in four records. The dropped collision coupling is
  not the cause although it is 0.9999997 of `A - M`; retaining it exactly costs
  17% *more* iterations. The cost is carried by `Nx` and not `Nxi`, and is
  independent of the tolerance. The preconditioned operator is strongly
  non-normal and far more so with `Nx`, and a diagonal similarity that removes
  1.5e10 of that eigenvector conditioning moves the iteration count by at most
  1.05x, so the balanced solve is killed on its own criterion. Six candidate
  causes are now measured and excluded.
- What one factorization serving many solves does and does not buy: the
  transport matrix admitted on both direct routes, the gradient admitted on the
  sparse one, and `jax.grad` measured to miss its gate for a reason factor
  reuse cannot address, since reverse mode re-executes the forward pass and the
  adjoint already runs on the primal's factors.

## v2.4.0 — 2026-09-15

The integrated #169–#191 stack, the reconciled #190 plan and the work from the
2026-09-13 independent review (#229–#246).

### Correctness and differentiation

- Report stalled solves without attributing an unmeasured physical cause or
  recommending changes to the requested electric field and accuracy.
- Add the opt-in `SfincsMatrixThreshold` parity switch, which reproduces SFINCS
  v3's `1d-12` matrix sparsification on the Fokker-Planck operator. It accounts
  for a 12-19% bootstrap-current gap on a hot-electron HSX-like deck; the
  default keeps every entry.
- Add `dkx.validity.normalized_radial_electric_field` and its operator-build
  form: the per-species `E_*` of Landreman et al. (2014), whose magnitude above
  about 1/3 marks the E x B resonance regime where trajectory models separate and
  resolution must follow `E_r`.

- Add opt-in bounded GMRES reuse to the host ambipolar root, with one cold retry
  after failed admission and independent final acceptance. Keep retry policy out
  of root differentiation.
- Correct the ambipolar teaching case to quasineutral analytic W7-X with full-FP
  collisions. JIT the geometry-optimization value/gradient, check its original
  kinetic residual and retain a three-step finite-difference window.
- Report monotone refinement ladders that converge faster than `max_order`, as
  spectral pitch and angle directions do, with three times the last difference
  as the grid uncertainty instead of refusing them. Diverging and oscillating
  ladders are still refused.

- Keep discrete pitch layouts static under JIT; refresh full-FP density kernels
  and opt-in temperature-dependent coefficients for prepared profile derivatives
  (#173, #174, #178, #186).
- Prepare immutable native profile/field scans with per-input original-equation
  status and complete kinetic-state recovery. Differentiate supported regular
  ambipolar roots with original primal/transpose admission (#182, #184, #187, #188).
- Reject invalid kinetic states, roots and adjoint references; qualify physical
  profile sensitivities with independent cold solves and Taylor checks
  (#180, #184, #188).

### Execution

- Bound the first automatic Krylov attempt and reuse its factors with a wider
  restart window when the basis-size guard permits. Preserve explicit solver
  settings and the differentiated path.

- Propagate nested legacy scan failures, distinguish failed progress from success,
  and retry incomplete Er points while preserving previous attempt files.
- Default the host BLAS pools to one thread. XLA's threadpool already runs
  batched CPU LAPACK in parallel; four BLAS threads per call made the NCSX
  `(21,37,61,8)` coarse factorization about 3x slower. Explicit settings win.

- Build the transposed coarse preconditioner on its first application, and
  synchronize coarse builds on their factors instead of a discarded zero-vector
  application. Non-differentiable solves no longer pay two transposed coarse
  applications per build; the preconditioner maps are unchanged.
- Factor the dense coarse preconditioner straight from its pinned row generator
  inside one compiled computation, so no band is materialized before
  elimination, and store only the rows each subsystem's `Nxi_for_x` keeps. The
  elimination and both substitution sweeps run one Legendre row at a time,
  batched over the subsystems active at that row, so the kernels launched per
  application scale with the longest chain. The truncated rows are an uncoupled
  `(1 + floor) I` and are applied as such. The map is unchanged: the f-block
  inverse agrees with the reusable Schur-LU route to rounding, GCROT iteration
  counts are identical, and the band guard is left as it was, conservative for
  ramped decks (`tests/test_coarse_ragged_chains.py`).

- Preserve independent-device batch sharding through JIT and gradients, including
  uneven batches (#179). Each complete system still resides on one device.
- Reuse generated SOLVAX Schur factors within supported PAS solve executions for
  multiple right-hand sides, transpose solves and refinement, subject to memory
  policy (#188). Persistent reuse across changed operators is still future work.
- Require SOLVAX 0.21.0, whose bordered Krylov preconditioner applies the coarse
  inverse once per iteration instead of twice, matching the release used for
  recent DKX timings.

### Benchmarking and development

- Supervise benchmark process groups and cancellation; reject invalid reference
  outputs and stale resumed results; retain original-equation checks and evidence
  (#170, #183, #185).
- Preserve requested PETSc options and observed backends; make source inventories
  reproducible; correct sparse reference matrix interpretation (#169, #176, #177).
- Preserve the tested package tree when GitHub merge refs move (#181), and balance
  thirteen coverage shards using measured test durations (#191).
- Adopt one authoritative figure-first plan, a concise workflow-oriented README,
  grouped documentation and citation metadata (#189, #190). Establish decision
  records and an experiment template in Phase 0.
- Revise the plan from the 2026-09-13 independent review (#230). Take
  finite-difference gradient references at `1e-12`, bound the sparse-map `pas`
  check at `1e-7`, and regenerate `.test_durations` from one complete run on one
  host so the thirteen coverage shards fit their time cap (#238, #241).

### Research records

- NCSX `(21,37,61,8)` refinement ladder with baseline adjoint estimates, its
  `Nx = 11, 12` speed rungs, and the cause of GCROT iteration growth with
  `Ntheta`: the dropped Fokker-Planck speed coupling (#237, #240, #242).
- SFINCS v3's `1e-12` sparsification threshold explains the 12-19% current gap
  on the HSX-like deck, confirmed from the SFINCS side and reported upstream as
  landreman/sfincs#27 (#243).
- SOLVAX operator couplings in the coarse preconditioner halve its memory but
  run 3x slower, so the dense route stays; explicit-inverse storage is rejected
  as not backward stable (#244).
- The HSX-like deck at resonant `E_*` is resolved in pitch by `Nxi = 120` but
  not in speed for the ion channel; no bootstrap current or ion flux at its two
  points is admitted (#245, #246).

These changes do not certify converged production performance, general persistent
restart reuse, full native Phi1 support or complete equilibrium-boundary optimization.
Historical measurements and their limits remain in the performance documentation.
