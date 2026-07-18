# Paper Benchmarks

Community-standard benchmark cases for the methods paper, in the validation
category: each script produces one publication-style figure plus a JSON record
of the computed coefficients, resolutions, convergence checks, and (when the
Fortran v3 executable is available) direct cross-check numbers.

Scripts are flat and self-documenting: parameters at the top, printed progress,
and a single figure + JSON pair written to
`docs/_static/figures/paper_benchmarks/`.

## Cases

- `monoenergetic_icnts_w7x.py`: ICNTS-style monoenergetic transport
  coefficients (`D11*`, `D31*` versus `nuPrime` at several `EStar`) on the
  W7-X standard configuration at r/a = 0.5, with matched-deck SFINCS Fortran
  v3 cross-check points [C.D. Beidler et al., Nucl. Fusion 51, 076001 (2011)].
- `monoenergetic_icnts_tjii.py`: the same scan on the TJ-II standard
  configuration at s = 0.493 (strong-ripple limit of the benchmark set), with
  the Boozer |B| spectrum supplied through `geometryScheme = 13` and the same
  Fortran v3 cross-check recipe (plus a MUMPS pivot/refinement note recorded
  in the JSON).
- `monoenergetic_icnts_hsx.py`: the same scan on the HSX quasi-helically
  symmetric configuration at r/a = 0.5 (`hsx3free.bc`), showing the
  QH-suppressed 1/nu branch in contrast to W7-X and TJ-II.
- `shaing_callen_convergence.py`: the low-collisionality "hard mode" test --
  the bootstrap coefficient `D31*` on the W7-X standard configuration scanned
  to `nuPrime = 3e-4` at `EStar = 0` and a small finite `EStar`, compared
  against the collisionless Shaing-Callen asymptote evaluated for the same
  surface [K.C. Shaing and J.D. Callen, Phys. Fluids 26, 3315 (1983)], with a
  per-point `Nxi` resolution schedule, split 1.3x convergence gates, and
  Fortran v3 cross-check points.
- `gradient_verification.py`: the AD-vs-FD gradient-verification table
  (three derivatives through the monoenergetic-database, RHSMode=1 solve,
  and ambipolar-root paths; JSON + rst snippet).
- `electron_root_optimization.py`: the differentiable ambipolar-Er /
  electron-root optimization workflow (roadmap item 3).  Resolves the full
  `J_r(E_r)` S-curve with all ambipolar roots classified ion / unstable /
  electron by the `dJr/dEr` sign, scans `T_e/T_i` to bracket the
  ion-root -> electron-root transition (the bifurcation diagram, including the
  saddle-node fold where the electron/unstable pair is born), and
  differentiates the selected ambipolar `E_r` through the implicit root
  (`dkx.er.ambipolar_er`, verified AD-vs-FD) with a small gradient-descent
  demo that drives the ambipolar `E_r` toward a target [H. Maassberg,
  C.D. Beidler & E.E. Simmet, Plasma Phys. Control. Fusion 41, 1135 (1999);
  Yu. Turkin et al., Phys. Plasmas 18, 022505 (2011); D.A. Spong, Phys.
  Plasmas 12, 056114 (2005)].  The electron channel is brought into the
  laptop-resolvable `E_r` window with a reduced ion/electron mass ratio; the
  JSON `modeling_note` documents why (physical `m_e/m_i` suppresses it by
  `(rho_e/rho_i)^2`).  Checkpointed under `output/electron_root/`
  (`DKX_EROOT_FORCE=1` to recompute).
- `bootstrap_consistency_kinetic_loop.py`: the workflow case -- the
  self-consistent-bootstrap equilibrium iteration with the actual
  drift-kinetic solve inside the loop (in place of the Redl analytic proxy
  [A. Redl et al., Phys. Plasmas 28, 022502 (2021)]) on a finite-beta
  precise-QA reactor-scale configuration: a damped Picard iteration
  equilibrium -> kinetic `<J.B>`(s) -> prescribed toroidal-current profile
  -> equilibrium, the kinetic-vs-Redl discrepancy profile at the converged
  state (the proxy error the loop removes), split resolution-refinement
  error bars, and one end-to-end `jax.value_and_grad` of the total
  bootstrap current through the differentiable equilibrium/Boozer/kinetic
  chain [M. Landreman, S. Buller & M. Drevlak, Phys. Plasmas 29, 082501
  (2022)].  Requires the optional vmex + booz_xform_jax companions;
  checkpointed and resumable (`DKX_BOOT_LOOP_MAX_NEW_STAGES`).
- `impurity_transport.py`: the classical/neoclassical impurity-transport case
  for a bulk hydrogenic plasma plus a high-Z trace impurity.  A Fortran v3
  parity anchor on the committed two-species carbon example
  (`quick_2species_FPCollisions_noEr`, neoclassical impurity
  `particleFlux_vm_psiHat` vs the Fortran golden), a mixed-collisionality scan
  on the W7-X standard configuration (full multi-species RHSMode=1 kinetic
  solve for the neoclassical flux and the algebraic `dkx.impurity` classical
  flux, checked against dkx's own `classicalTransport.F90` counterpart to
  machine precision), a `vmap`-over-charge-state Z-scan, and the
  temperature-screening diagnostic (exact `-Z` density peaking, the `1/2`
  collisional screening coefficient, and an AD-vs-FD ion-temperature-gradient
  derivative) [S.I. Braginskii, Rev. Plasma Phys. 1, 205 (1965); P.H.
  Rutherford, Phys. Fluids 17, 1782 (1974); F.L. Hinton & R.D. Hazeltine, Rev.
  Mod. Phys. 48, 239 (1976); P. Helander & D.J. Sigmar, *Collisional Transport
  in Magnetized Plasmas*, CUP (2002)].  Checkpointed and resumable.

## Running

From the repo root:

```bash
python examples/paper_benchmarks/monoenergetic_icnts_w7x.py
```

Set `SFINCS_FORTRAN_EXE=/path/to/sfincs` to enable the Fortran cross-check
points (skipped otherwise).  Equilibrium files are fetched into the local
`dkx` data cache on first use.  Expect several minutes per script at
production resolution; the CI-sized regression version of each case lives in
the test suite (tests/test_paper_benchmark_monoenergetic in the repo root).
