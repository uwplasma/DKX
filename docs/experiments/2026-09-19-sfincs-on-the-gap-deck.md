# SFINCS on the deck DKX cannot converge, and what transposes

## Hypothesis

`2026-09-18-speed-triangle-back-substitution.md` left one point unresolved: the
collaborator's HSX-like deck at point A (`rN = 0.187`, `Er = 15`) with
`Nxi = 120`, `Nx = 16`, 633,604 unknowns, which no DKX route converges in three
hours. The working assumption was that SFINCS solves this class of deck and DKX
lacks something it has. This record tests that assumption, and asks which of
SFINCS's methods transposes.

Owner: independent review. Budget: one day.

## Admission test

- **SFINCS.** v3 with the sparsification threshold disabled, so its operator is
  DKX's (`2026-09-13-sfincs-sparsify-threshold.md`). PETSc 3.20.2, MUMPS 5.6.2
  with AMD ordering, one rank, on a 36 GiB laptop with an hour's cap per arm.
  Three arms on the same deck: a full direct factorization
  (`useIterativeLinearSolver = .false.`); the default iteration, GMRES
  preconditioned by a MUMPS factorization of the *simplified* matrix
  (`preconditioner_x = 1`); and `preconditioner_x = 2`, which keeps the
  collision operator's upper speed triangle, as DKX's `coarse_triangle` does.
- **DKX.** The same deck through the assembly of `dkx.assembly`, and the
  matrices that follow from it.
- A route counts only if it returns a converged answer. An arm stopped by the
  operating system or the clock is recorded as that.

## Result

**SFINCS does not solve this deck on this host by any route.**

| Arm | Outcome |
| --- | --- |
| Full direct (MUMPS) | Killed by the operating system for memory after 22.5 min |
| Default iteration, `preconditioner_x = 1` | Stagnant: `‖r‖/‖b‖ = 0.9955` after 66 iterations and one hour |
| `preconditioner_x = 2` | Killed for memory after 30.6 min, factoring its preconditioner |

- **The simplified preconditioner fails in both codes.** SFINCS's residual falls
  by 0.45% in 66 iterations; DKX's `coarse` route, which inverts the same
  simplified operator, ran 20,000 iterations to 2.5e-5. The wall at `Nx = 16` is
  a property of the method both codes share, not of either implementation.
- **SFINCS's remedy is memory.** Its two routes that would work factor a matrix
  that keeps the speed coupling, and both exhausted 36 GiB.
- SFINCS assembles the full 29,781,431-nonzero matrix analytically in 1.55 s.

**What transposes is the assembly, which DKX gains here.** The operator is
matrix-free, so its sparse direct route sampled one column per product, 633,604
products here, and was guarded off. The couplings are known, so a product with a
whole group of columns that share no row carries all of them
(`solvax.compression`, SOLVAX#111):

| Deck | Unknowns | Entries per row | Products | Assembly | Against the operator |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test grid `(5, 5, 6, 4)` | 1,204 | 80 | 1,008 | 1.9 s | entrywise identical |
| Test grid `(8, 9, 10, 4)` | 6,484 | 80 | 3,248 | 6.4 s | entrywise identical |
| Collaborator grid `(10, 15, 20, 10)` | 66,004 | 198 | 5,508 | 22 s | 1.1e-16 |
| Gap deck `(10, 15, 120, 16)` | 633,604 | 198 | 8,800 | about 1.2 min | not run to completion |

- Against sampling every column the recovered matrix is identical entry for
  entry, not merely close. On the gap deck one product costs 8.4 ms, so sampling
  would take 89 minutes and the recovery takes 1.4% of that.
- **The raw operator cannot be factored at all.** The rectangular layout keeps
  the `l >= Nxi_for_x(x)` degrees of freedom as exact zero rows. The solver
  poses `A M + (I - M)`, identity on those rows, and the assembly does the same.

**The direct route works, and the factorization is what it lacks.** On the
collaborator grid, 66,004 unknowns and 2,210,951 nonzeros, DKX assembled the
pinned operator in 5.4 s and SuperLU factored and solved it:

| Code | Factorization | Fill | Peak RSS | Relative residual |
| --- | ---: | ---: | ---: | ---: |
| SFINCS, MUMPS with AMD | about 40 s, whole run | not reported | 3.8 GiB | 8.3e-14 |
| DKX, SuperLU with COLAMD | 801 s | 389× | 8.4 GiB | 2.8e-11, then 1.7e-13 after one refinement step |

- The answer is a converged one: one step of iterative refinement takes the
  residual from 2.8e-11 to 1.7e-13, and the solve itself costs 1.8 s, so every
  further right-hand side or adjoint of that operator is nearly free.
- The factorization is 20× slower than MUMPS and takes twice the memory. COLAMD
  orders for `A^T A` and fills 389× on a matrix whose speed coupling is dense;
  the ordering that was best for the simplified subsystems is the wrong one
  here. The SFINCS figures are from `2026-09-12-reuse-admission.md`, on the same
  deck with the sparsification threshold in place.

**Two methods that did not transpose.**

- **Fill-reducing ordering.** MUMPS needed AMD to factor this class of matrix at
  all, so SuperLU's default was the suspect. On the matrices DKX's sparse
  preconditioner factors (19,800 rows, 99.8% structurally symmetric) it is the
  other way round: COLAMD fills 2.12×, the natural order 2.29×, and the AMD-like
  `MMD_AT_PLUS_A` 4.12× at three times the factorization time. The subsystems
  are block tridiagonal with banded blocks, which the default already suits.
- **An incomplete LU of the full operator.** It keeps the coupling the
  simplified preconditioner drops, at a fraction of a complete factorization's
  memory. On the collaborator grid it built in 133 s at 9.5× fill, and GMRES
  then diverged: 19,967 iterations to a relative residual of 6.1. The matrix is
  too ill conditioned for a drop-tolerance factorization without the scaling and
  pivoting a complete one applies.

Scripts and records are kept outside Git in
`dkx-review-evidence-20260913/sfincs_gap/` and `q11_assembly/`:
`run_gap.sh`, `gap.log`, the three SFINCS run directories,
`ordering_bench.py`, `assembly_check.py`, `assembly_stats.py`, `ilu_probe.py`,
`direct_probe.py` and their JSON outputs.

## Decision

- **Land the assembly.** It is exact, it is checked against the operator before
  it is returned, and it removes the reason the direct route was confined to
  small decks.
- **The simplified preconditioner is the shared limit**, so matching SFINCS is
  not the goal on this deck class; a preconditioner or a factorization that
  keeps the speed coupling within memory is.
- **Keep SuperLU's default ordering**, and do not pursue a drop-tolerance
  incomplete factorization.
- **The direct route needs a multifrontal factorization**, not a different
  assembly: 801 s against 40 s is the gap, and it is the first step of the
  production solver program in `plan.md`.
