# Agreement with SFINCS v3

DKX solves the same drift-kinetic system as SFINCS Fortran v3 (Landreman et al.
2014) and reads the same `input.namelist` decks
({doc}`../user_guide/sfincs_namelist`). On a matched deck the only differences
left should be solver tolerance and floating-point summation order. This page
records how close the two codes are, how the comparison is run, and the places
where the Fortran reference itself is the limiting factor.

## Discrete parity on the upstream suite

Every deck in the upstream `fortran/version3/examples` suite (vendored under
`examples/sfincs_examples/`) is run through both codes and compared key by key
on the HDF5 output.

```{figure} ../_static/figures/readme/canonical_parity.png
:alt: Parity envelopes of DKX against SFINCS v3 for profile outputs, state vectors and transport matrices.
:width: 85%

Parity envelopes of the referee tests. RHSMode=1 output tables agree to a
maximum scaled difference of 8e-14 (`tests/test_run_rhsmode1.py`); structured
direct state vectors agree with recorded reference state vectors to 1e-11, and
RHSMode=2/3 transport matrices with Fortran golden data to 6e-13 to 9e-9
depending on conditioning (`tests/test_run_transport.py`). Regenerate with
`python tools/benchmarks/readme_figures.py`.
```

| Measurement | Value | Source |
| --- | --- | --- |
| Upstream decks run end to end | 38 | `tools/benchmarks/parity_performance_matrix.py` |
| Median relative difference on shared output moments | 4.1e-6 | same |
| Full Fokker–Planck decks | agree to 1e-8 | same |
| CPU/GPU suite audit, cases `parity_ok` on both backends | 39 of 39, zero strict mismatches | `tools/publication_figures/artifacts/dkx_fortran_suite_benchmark_summary.json` |

The 39-case CPU/GPU audit requires every case to stay `parity_ok` on both
backends, with zero strict mismatches and no `jax_error` or `max_attempts`
failures. Each row in the summary JSON records the number of common HDF5 keys
compared (for example 265 for `geometryScheme4_2species_noEr_withPhi1InDKE`)
and the number that differ (0).

## Matched full-kinetic profiles

The upstream decks are small. Three further comparisons run a physical profile
problem with full linearized Fokker–Planck collisions at two resolutions in
both codes, against a pinned SFINCS build (PETSc 3.23.6, MUMPS 5.8.1, no source
edits). Each is sealed in a JSON artifact under `validation/` and re-audited by
`python tools/paper_benchmarks/audit_full_kinetic_sfincs_validation.py`.

| Case | Artifact | SFINCS unknowns | Largest scaled DKX/SFINCS difference | True residuals |
| --- | --- | --- | --- | --- |
| Tokamak, $E_r = 0$ | `full_kinetic_sfincs_v1.json` | 6,887 → 12,509 | 2.69e-10 | below 1.82e-11 |
| Tokamak, normalized $E_r = -30$ | `full_kinetic_sfincs_finite_er_v1.json` | 6,887 → 12,509 | 1.88e-9 | below 5.25e-11 |
| W7-X SC1 Boozer surface, $r_N = 0.5$ | `full_kinetic_sfincs_stellarator_v1.json` | 54,407 → 98,126 | 1.37e-8 | below 1.82e-12 |

Particle flux and NTV vanish by axisymmetric cancellation in the tokamak cases,
so their criterion is an absolute scale (1e-12 and 2e-11) rather than a
relative error. The stellarator case's largest scaled difference is in NTV and
corresponds to an absolute difference of about 8.31e-13. These are surface
comparisons at prescribed field; they are not $E_r$ scans, ambipolar-root,
$\Phi_1$ or experimental validations.

## Runtime and memory on the suite

```{figure} ../_static/figures/paper/dkx_fortran_suite_benchmark_summary.png
:alt: Runtime and active-memory comparison of SFINCS Fortran v3 and DKX CPU and GPU, cold and warm, across the example suite.
:width: 92%

Runtime (left) and active memory (right) for SFINCS Fortran v3 and DKX CPU and
GPU, cold and warm, on every suite row inside the reference-runtime-window,
ordered by best warm DKX speed-up. Reproduce with
`python tools/publication_figures/generate_fortran_suite_benchmark_summary.py`.
```

The plot keeps only rows whose Fortran runtime is at least 10 s: the
reference-runtime-window. Below that, process launch, file I/O and JIT
compilation dominate both codes, so those rows stay parity checks and are not
used for performance claims. Median DKX/Fortran ratios over the plotted rows,
from `tools/publication_figures/artifacts/dkx_fortran_suite_benchmark_summary.json`:

| Ratio (DKX / Fortran) | CPU | GPU |
| --- | --- | --- |
| cold wall clock | 0.021x | 0.037x |
| active memory | 2.89x | 3.71x |

DKX is faster and uses more memory. Fortran memory is the process maximum RSS;
DKX active memory subtracts the fixed Python/JAX/XLA baseline from the profiled
RSS. The full-process ratios and the whole-suite split by solver route are on
{doc}`performance`.

## Reference limits of SFINCS

Agreement with SFINCS is the right test only where SFINCS is right. Two
measurements found places where it is not, and the numbers below are the
record of both.

```{figure} ../_static/figures/readme/sfincs_reference_limits.png
:alt: Left, the HSX bootstrap-current gap between released SFINCS and DKX and its removal with the sparsification cutoff at zero. Right, residuals reached on the HSX-like gap deck by each SFINCS route and the DKX Krylov route.
:width: 92%

Left: the bootstrap-current gap on an HSX deck, before and after setting
SFINCS's matrix-entry cutoff to zero. Right: the HSX-like gap deck, where no
route of either code reaches the requested tolerance on a 36 GiB host.
Regenerate with `python tools/publication_figures/generate_readme_showcase.py`.
```

### The sparsification threshold

On a quasi-helically symmetric HSX VMEC equilibrium (nfp = 4, $T_i = 59$ eV,
$T_e = 1.37$ keV, two species, full Fokker–Planck, full-trajectory $E_r$ terms,
$(N_\theta, N_\zeta, N_\xi, N_x) = (11, 15, 20, 10)$, 66,004 unknowns), released
SFINCS v3 and DKX differ by 12–19% in the bootstrap current: 19% at
$r_N = 0.187$, $E_r = 15$ and 12% at $r_N = 0.367$, $E_r = 14.39$, while the
right-hand sides agree to 6e-15.

The cause is in SFINCS's matrix assembly. `MatSetValueSparse`
(`fortran/version3/sparsify.F90`) skips every entry with
`abs(value) <= threshholdForInclusion = 1d-12`. On a cold-ion, hot-electron deck
($\sqrt{T_e m_i / (T_i m_e)} = 207$ here) the ion→electron field-particle
collision block has entries from 1e-26 to 8e-7, and the cutoff removes 442 of
its 8,000 entries at the first point. Electron viscosity is weak on a
quasisymmetric surface, so the electron parallel flow, and with it the current,
is sensitive to exactly those entries. Applying the same cutoff to DKX's
collision tensor reproduces the SFINCS current to 2e-10.

With the cutoff set to zero and SFINCS v3 (`8df5453`) rebuilt (PETSc 3.20.2,
MUMPS 5.6.2 with AMD ordering, one rank, relative true residuals at most
2.5e-12), the two codes agree:

| Point | SFINCS, cutoff `1d-12` | SFINCS, cutoff `0` | SFINCS (cutoff 0) relative to DKX |
| --- | ---: | ---: | ---: |
| $r_N = 0.187$ | 1.36166376e-2 | 1.10337063e-2 | 6.6e-11 |
| $r_N = 0.367$ | −3.56737204e-2 | −3.14056428e-2 | 1.4e-11 |

That is agreement to 7e-11 or better. Keeping the small entries adds about 2%
nonzeros and leaves the SFINCS solve time unchanged (165 s to 167 s). DKX keeps
every entry, so it is the more faithful discretization of the stated model; the
fix is proposed upstream as
[landreman/sfincs#27](https://github.com/landreman/sfincs/pull/27).

For parity on decks with $\sqrt{T_e m_i / (T_i m_e)} \gtrsim 30$, a SFINCS
reference is admitted only when built with the cutoff disabled.

### A deck neither code converges

The same HSX-like configuration at $r_N = 0.187$, $E_r = 15$ with $N_\xi = 120$
and $N_x = 16$ (633,604 unknowns) is not solved by any route of either code on a
36 GiB laptop host. SFINCS v3 with the cutoff disabled, one rank, one-hour cap
per route:

| SFINCS route | Outcome |
| --- | --- |
| Full direct factorization (MUMPS) | killed by the operating system for memory after 22.5 min |
| Default GMRES, preconditioned by the simplified matrix (`preconditioner_x = 1`) | stagnant at $\lVert r\rVert/\lVert b\rVert = 0.9955$ after 66 iterations and one hour |
| `preconditioner_x = 2`, keeping the upper speed triangle | killed for memory after 30.6 min, factoring its preconditioner |

DKX's recycled Krylov route with the coarse preconditioner, which inverts the
same simplified operator, ran 20,000 iterations to a relative residual of
2.5e-5. The stall is a property of the preconditioning method both codes share,
not of either implementation; SFINCS's remedy is a factorization that keeps the
speed coupling, and both of its routes that attempt one run out of memory.

DKX's operator is matrix-free, so its sparse direct route assembles the matrix
from operator products, one product per group of columns that share no row
(`dkx.assembly`). On this deck the assembly needs 4,800 products instead of one
per column, and the assembled matrix is identical entry for entry to
column-by-column sampling. That assembly makes the 66,004-unknown reduction of
the same configuration solvable by `solve(method="direct")` to a relative
residual of 1.3e-14. The full deck remains open: extrapolating the factorization
cost of the direct route from 66,004 and 158,404 unknowns gives of order
fourteen hours and 190 GiB at 633,604 unknowns, beyond a 62 GiB host.

The measurement protocol and scripts for both findings (column comparison,
threshold rebuild, three-route SFINCS campaign) are summarized in
`tools/publication_figures/generate_readme_showcase.py`, which draws the figure
from the recorded values.

### Reuse of one factorization

SFINCS factors its matrix once per run. DKX's direct routes return their
factorization in `SolveResult.factors` and accept it back through
`solve(..., factors=...)`; `solve(..., transpose=True)` solves the adjoint from
the same factors ({doc}`../numerics/factor_reuse`).

```{figure} ../_static/figures/readme/factor_reuse.png
:alt: Wall time and factorization count for repeated and adjoint solves on the structured and sparse direct routes.
:width: 92%

Median wall time of nine interleaved repeats and the number of factorizations
per arm, on the structured direct route (16,230 unknowns) and the sparse direct
route (1,962 unknowns, full Fokker–Planck).
```

Three right-hand sides delivered as three separate calls cost three
factorizations and 0.9593 s on the structured route and 1.1931 s on the sparse
route; with stored factors they cost none, 0.2822 s and 0.2779 s. On the sparse
route the transposed solve costs 0.15 of a primal. These are medians on an
Apple M3 Max CPU (SOLVAX 0.24.0, JAX 0.10.2, float64) on a host shared with other
work, so the wall times carry that uncertainty; the factorization counts do not.
