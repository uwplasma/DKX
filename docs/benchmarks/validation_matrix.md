# Validation matrix

A validation entry is one self-contained piece of evidence: a physics claim or
benchmark figure together with its literature anchor, the script that generates
it, the artifact it produces, the tests that protect it, and the criteria it
must meet. This page describes how entries are recorded and gated, and lists
every entry with its claim status.

## The machine-readable manifest

The manifest is `tools/publication_figures/validation_manifest.json`, one record
per entry. Each record carries:

| Field | Meaning |
| --- | --- |
| `id`, `kind`, `status` | identity and category of the entry |
| `literature` | the published result the entry reproduces or compares against |
| `scripts` | the commands that regenerate its artifacts |
| `artifacts` | the checked-in JSON and figures |
| `source_code` | the implementation files that define it |
| `tests` | the tests that protect it |
| `acceptance_gates` | the concrete criteria required before it can support a claim |
| `release_gate` | claim status, evidence, release decision and promotion gate |

`tests/test_validation_manifest_schema.py` enforces the schema. Every listed
source file, test, script and artifact must exist, including for deferred
entries, so that a closed entry keeps its anchors and criteria and can be
reopened. Sealed evidence under `validation/` is re-audited by the `audit_*.py`
scripts in `tools/paper_benchmarks/`; `python -m tools.release.registry` runs all
of them.

## Release claim gate metadata

Every record has a `release_gate` block, checked by
`python -m tools.release.release check-gates` and
`tests/test_release_gate_metadata.py`. The allowed `claim_status` values are:

`release_ready`
: checked-in artifacts support the documented release-scope claim, and the
  listed tests are the fast check for that claim.

`regression_scaffold`
: checked-in bounded artifacts are useful for CI, branch validation or
  manuscript layout, but a broader or full-resolution claim is deliberately not
  made.

`bounded_proxy`
: checked-in artifacts support a narrower proxy or normalization claim, while
  the full literature reproduction stays closed until its `promotion_gate` is
  met.

`closed_deferred`
: the entry is explicitly closed for the tagged release as post-release or
  nightly research work.

No record may set `blocks_current_release=true` unless the release process is
meant to stop on it. An entry that is not ready is either absent or recorded as
`closed_deferred` with a concrete reason and `promotion_gate`, so that scaffold
scripts, run plans and proxy figures cannot be mistaken for closed evidence.

## Entries

| Entry | Claim status | What it shows | Where |
| --- | --- | --- | --- |
| `sfincs2014_fig1_lhd_collisionality` | `release_ready` | LHD transport matrix against collisionality, FP and PAS | {doc}`analytic_limits` |
| `sfincs2014_fig2_w7x_collisionality` | `release_ready` | W7-X transport matrix against collisionality, FP and PAS | {doc}`analytic_limits` |
| `sfincs2014_fig1_lhd_collisionality_reaudit_fast`, `sfincs2014_fig2_w7x_collisionality_reaudit_fast` | `regression_scaffold` | fast four-point reruns for CI | — |
| `sfincs2014_fig3_high_collisionality_limit` | `closed_deferred` | Simakov–Helander high-collisionality reproduction | {doc}`analytic_limits` |
| `sfincs2014_high_collisionality_trend_proxy` | `bounded_proxy` | inverse-$\nu$ tail fits of $L_{11}$, $L_{12}$ | {doc}`analytic_limits` |
| `sfincs2014_simakov_helander_limit_audit` | `bounded_proxy` | Appendix-B normalization audit and high-$\nu'$ run plan | {doc}`analytic_limits` |
| `sfincs2014_er_sweep_tokamak_reference` | `release_ready` | DKES, partial and full trajectory models against $E_r$, tokamak | {doc}`cross_code` |
| `sfincs2014_er_sweep_stellarator_fast_scaffold` | `regression_scaffold` | the same sweep on a stellarator, fast grid | {doc}`cross_code` |
| `publication_validation_dashboard` | `release_ready` | dashboard of the collisionality scans and trajectory sweeps | {doc}`cross_code` |
| `fortran_v3_cpu_gpu_suite_benchmark_summary` | `release_ready` | 39-case CPU/GPU parity audit and runtime/memory benchmark | {doc}`sfincs`, {doc}`performance` |
| `sfincs_full_kinetic_tokamak_profile` | `release_ready` | full-FP tokamak profile, DKX against SFINCS, 2.69e-10 | {doc}`sfincs` |
| `sfincs_full_kinetic_finite_er_tokamak` | `release_ready` | the same at normalized $E_r = -30$, 1.88e-9 | {doc}`sfincs` |
| `sfincs_full_kinetic_stellarator_profile` | `release_ready` | full-FP W7-X surface, 1.37e-8 | {doc}`sfincs` |
| `monkes_monoenergetic_overlap` | `regression_scaffold` | DSHAPE, NCSX, W7-X EIM against YANCC and MONKES, within 6% | {doc}`cross_code` |
| `native_w7x_physical_flux_sfincs` | `release_ready` | native-path W7-X fluxes in physical units against SFINCS | {doc}`cross_code` |
| `w7x_fixed_field_resolution_referee` | `release_ready` | fixed-field W7-X resolution referee | this page |
| `native_w7x_ambipolar_profile_certificate` | `regression_scaffold` | five-surface W7-X ambipolar profile from a case file | this page |
| `native_w7x_ambipolar_phase_space_ladder`, `_phase_space_axes`, `_pitch_budget`, `_pitch_speed_groups`, `_pitch_explicit_groups`, `_joint_speed_zeta_tail` | `regression_scaffold` | resolution diagnosis of that profile | this page |
| `w7x_seeded_bracket_discovery`, `w7x_admitted_grid_seeded_envelope` | `bounded_proxy` | seeded ambipolar root bracketing on admitted grids | this page |
| `w7x_ambipolar_er_validation` | `closed_deferred` | ambipolar $E_r$ against a reconstructed W7-X discharge | {doc}`cross_code` |
| `adjoint_sensitivity_gradient_checks` | `release_ready` | autodiff and implicit gradients against central differences | {doc}`../numerics/differentiation` |

The counts per status are printed by `python -m tools.release.release check-gates`.

## Figures of the scaffold, proxy and deferred entries

These artifacts are what the non-release-ready entries point at. They are kept so
the release gate can check that each recorded artifact exists; none of them is a
release claim.

```{figure} ../_static/figures/paper/dkx_fig1_lhd_collisionality_reaudit_fast.png
:alt: Fast four-point LHD collisionality rerun, FP and PAS.
:width: 80%

`sfincs2014_fig1_lhd_collisionality_reaudit_fast` (`regression_scaffold`): the
bounded LHD collisionality lane, four points, FP and PAS separated
(`tools/publication_figures/generate_sfincs_paper_figs.py`).
```

```{figure} ../_static/figures/paper/dkx_fig2_w7x_collisionality_reaudit_fast.png
:alt: Fast four-point W7-X collisionality rerun, FP and PAS.
:width: 80%

`sfincs2014_fig2_w7x_collisionality_reaudit_fast` (`regression_scaffold`): the
same bounded lane on W7-X (`tools/publication_figures/generate_sfincs_paper_figs.py`).
```

```{figure} ../_static/figures/paper/dkx_fig3_simakov_helander.png
:alt: Transport matrix at high collisionality against the Simakov-Helander limit.
:width: 80%

`sfincs2014_fig3_high_collisionality_limit` (`closed_deferred`): the scan does
not yet reach the collisionality where the analytic asymptote applies, so the
reproduction stays open ({doc}`analytic_limits`).
```

```{figure} ../_static/figures/paper/dkx_high_collisionality_trend_proxy.png
:alt: Power-law fits of the high-collisionality L11 and L12 tails.
:width: 80%

`sfincs2014_high_collisionality_trend_proxy` (`bounded_proxy`): positive
power-law slopes of the PAS $L_{11}$ and $L_{12}$ tails on both collisionality
artifacts (`tools/release/artifacts.py`).
```

```{figure} ../_static/figures/paper/dkx_autodiff_sensitivity_map.png
:alt: Sensitivity map of transport outputs to input parameters from implicit differentiation.
:width: 80%

`adjoint_sensitivity_gradient_checks` (`release_ready`): implicit-differentiation
sensitivities through a pinned full-system solve, which agree with centered
finite differences below the recorded `1e-4` relative-error gate
(`tools/publication_figures/generate_autodiff_sensitivity_validation.py`).
```

## The suite benchmark entry

```{figure} ../_static/figures/paper/dkx_fortran_suite_benchmark_summary.png
:alt: Runtime and active-memory bars for SFINCS Fortran v3 and DKX CPU and GPU across the example suite.
:width: 92%

The `fortran_v3_cpu_gpu_suite_benchmark_summary` entry: runtime and active
memory for SFINCS v3 and DKX CPU and GPU, cold and warm, over the rows inside
the reference-runtime-window (Fortran runtime at least 10 s).
```

Its acceptance tests require all 39 audited cases to stay `parity_ok` on both
backends, with zero strict mismatches and no `jax_error` or `max_attempts`
failures. The summary JSON recomputes runtime, memory, ratios, top offenders and
the excluded short-reference rows from the checked-in reports; excluded rows
remain parity checks until they are rerun at a production-comparison
resolution. Numbers are on {doc}`sfincs` and {doc}`performance`.

## The ambipolar profile entries

A five-surface W7-X standard-configuration profile (hydrogen and electrons,
PAS collisions, DKES drifts, all-root search, radial branch continuation) is
driven from a checked TOML case and sealed in
`validation/native_ambipolar_profile_v1.json`
(`python tools/paper_benchmarks/audit_native_ambipolar_profile.py`). It keeps
root counts `[1, 1, 3, 1, 1]`, seven branch events and all 222 solver
attempts; every final bracket is 0.0048828125 kV/m wide.

The companion resolution ladder
(`validation/ambipolar_phase_space_ladder_v1.json`) refines
$(N_\theta, N_\zeta, N_\xi, N_x)$ through $(13, 31, 32, 5)$, $(15, 37, 36, 6)$
and $(17, 37, 40, 6)$. Root counts and classifications stay the same, but the
reference-to-fine movement fails its criteria:

| Quantity | Movement | Threshold | Result |
| --- | ---: | ---: | --- |
| ambipolar root | 1.6259765625 kV/m | 0.005 kV/m | fail |
| selected particle flux | 4.08% | 2% | fail |
| selected heat flux | 7.81% | 2% | fail |
| maximum accepted true residual | 3.92e-13 | 1e-12 | pass |

The recorded outcome is `refinement_exhausted`. The follow-on entries separate
the axes and find pitch-angle resolution to be the dominant unconverged
direction. This negative result is what keeps the profile a
`regression_scaffold` rather than a phase-space-converged validation, and it is
the reason every example and tutorial asks for a convergence check before a
number is quoted ({doc}`../user_guide/convergence`).

## Open and deferred work

These are not release blockers, and none is claimed:

- the Simakov–Helander high-collisionality reproduction (needs scans to
  $\nu' \approx 100$);
- W7-X ambipolar $E_r$ against a reconstructed discharge (needs pinned
  equilibrium and profile provenance; without it artifacts stay
  `w7x_like_scaffold`);
- KNOSOS overlap and low-collisionality trend comparison with MONKES;
- production-resolution QI ladders and a true device-resident QI route;
- single-case multi-device strong scaling;
- full VMEC-boundary-to-kinetic-transport gradients ({doc}`../tutorials/vmex_optimization`).

A new entry is added to both this page and the manifest in the same change,
with its `release_gate` block.
