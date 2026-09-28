# Differentiation

The drift-kinetic operator, its right-hand side and the moment diagnostics are
pure JAX functions. A scalar built from a solved distribution (a flux, the
bootstrap current, an ambipolar $E_r$, a transport coefficient) can be passed
to `jax.grad` and returns the derivative with respect to geometry harmonics,
profiles or collisionality. Derivatives through the linear solve come from the
implicit function theorem, not from differentiating solver iterations.

```{figure} ../_static/figures/paper/dkx_autodiff_gradient_check.png
:alt: Reverse-mode gradients of DKX observables overlaid on centered finite differences.
:width: 88%

Reverse-mode `jax.grad` derivatives of kinetic observables (points) against
centered finite differences (line). Regenerate with
`tools/publication_figures/generate_autodiff_sensitivity_validation.py`;
`examples/autodiff/gradients_tour.py` runs the same check interactively.
```

## Implicit differentiation through the solve

Every route solves $A(p)\,u = b(p)$, with $p$ the differentiable parameters. For
an objective $J = g(u, p)$ the tangent $du/dp$ satisfies

$$
A\,\frac{du}{dp} = \frac{\partial b}{\partial p} - \frac{\partial A}{\partial p}\,u .
$$

The reverse-mode adjoint is one transposed solve at the converged state,

$$
A^\mathsf{T}\lambda = \left(\frac{\partial g}{\partial u}\right)^{\!\mathsf{T}},
\qquad
\frac{dJ}{dp} = \frac{\partial g}{\partial p}
 + \lambda^\mathsf{T}\left(\frac{\partial b}{\partial p} - \frac{\partial A}{\partial p}\,u\right).
$$

`solve(op, rhs, differentiable=True)` wraps the structured direct and recycled
Krylov routes in `solvax.implicit.linear_solve`
(`jax.lax.custom_linear_solve`). The transposed solve reuses the forward work:
on the structured route it is the same block-Thomas factors swept with
`transpose=True`; on the Krylov route it is a transposed-preconditioner GCROT
solve built from the same coarse operator. The outer root problems, the
ambipolar $E_r$ (`dkx.er.ambipolar_er`) and the nonlinear $\Phi_1$ Newton solve
(`dkx.phi1.phi1_state`), are wrapped in `solvax.implicit.root_solve`
(`jax.lax.custom_root`), so their derivatives also follow from the implicit
function theorem.

The factorization and substitutions take a gradient-free copy of the operator;
the gradient flows through the operator's action and its transpose at the
converged state. Forward mode (`jax.jvp`, `jax.jacfwd`) works through the same
wrapper.

The sparse direct route runs on the host and refuses `jax.grad`. A host workflow
that needs its adjoint passes the stored factors back with
`solve(..., factors=result.factors, transpose=True)`, at 0.15 of a primal
({doc}`factor_reuse`).

## What is differentiable

```{list-table}
:header-rows: 1

* - Target
  - What flows
  - Entry point
* - Geometry
  - Boozer harmonics $\hat B_{mn}$ and derived metric coefficients, from analytic schemes and JAX-native producers
  - `KineticOperator.apply`, `dkx.workflows.geometry_adapters`
* - Profiles
  - densities, temperatures, their radial gradients, `nu_n`, the $E_r$ drive
  - `KineticOperator.rhs` and the operator coefficients; `dkx.prepare_er_scan(..., differentiable_profiles=True)`
* - Ambipolar $E_r$
  - the root of $J_r(E_r) = 0$ and any function of it
  - `dkx.er.ambipolar_er`
* - $\Phi_1$ state
  - $\Phi_1(\theta,\zeta)$ from the quasineutrality Newton solve
  - `dkx.phi1.phi1_state` (requires `Nxi_for_x_option = 0`)
* - Monoenergetic transport matrix
  - RHSMode 3 coefficients and the energy-convolved thermal $L_{ij}$
  - `dkx.monoenergetic.monoenergetic_database_from_operator(..., differentiable=True)`
* - Kinetic bootstrap current in a VMEX optimization
  - $\langle \mathbf{j}\cdot\mathbf{B}\rangle$ through `vmex -> booz_xform_jax -> dkx`
  - `dkx.bootstrap.KineticBootstrapMismatch`
```

Not differentiable: the file readers (`input.namelist`, `.bc` Boozer files,
`wout_*.nc`), which are provenance and parity tools, and the host
NumPy/QUADPACK Fokker–Planck coefficient builder. Full Fokker–Planck profile
derivatives use the opt-in prepared builder
`dkx.collisions.prepare_fokker_planck_v3_profiles`, a Gauss–Legendre quadrature
(128 points per panel by default) that refreshes all four collision kernels on
device. Masses, charges, the normalization and the Coulomb logarithm stay fixed.
The differentiable $\Phi_1$ helper raises `NotImplementedError` with an active
`Nxi_for_x` ramp.

## Admission of the adjoint

A gradient is only as good as its transposed solve, and a stalled adjoint is
invisible from outside: the forward solution and every field of `SolveResult`
look healthy while the vector-Jacobian product is wrong. This happens on
operators whose null space the constraint scheme does not span; the singular
full Fokker–Planck system with `constraintScheme = 1` is the recorded case. DKX
therefore recomputes $\|A^\mathsf{T}\lambda - g\|$ from the operator after the
transposed solve, never the Krylov estimate, records it in
`SolveResult.adjoint`, and raises at execution time if either the forward or
the transposed residual misses `max(atol, tol * ||rhs||)`. `check_adjoint=False`
disables the raise but still records the residuals;
`adjoint_residual_factor` (default 1) relaxes the gate explicitly. A
homogeneous equation requires entrywise zero defect.

The gate covers the recycled Krylov route and full-state structured solves.
Partial recovery on the truncated structured kernel (next section) has no
full-equation adjoint admission, because its zero-padded state is not a
full-equation solution.

## The truncated structured kernel

The truncated kernel ({doc}`solver_routes`) inverts a reduced,
Schur-complemented operator on the lowest `tier1_keep_lowest` Legendre blocks.
Using its zero-padded state with a full-operator $A^\mathsf{T}$ adjoint would be
inconsistent, so it stays outside the implicit wrapper and its blocks are
generated on the fly. Plain `jax.grad` tapes the generated sweeps, and the
reverse pass costs $\mathcal{O}(N_\xi m^2)$ per $(s, x)$ chain.

`solve(..., tier1_adjoint_window=w)` routes through SOLVAX's
structure-preserving custom VJP instead. The right-hand-side gradient is an
exactly generated truncated solve of the transposed operator, and coefficient
gradients are pulled back through the block assembly on the leading `keep + w`
blocks, so reverse mode costs $\mathcal{O}((K + w) m^2)$. The right-hand-side
gradient has no window error; the coefficient-gradient error decays as
$\mathcal{O}(\rho^{2w})$ for the block-dominant collisional operators this
kernel serves, and $w \ge N_\xi$ reproduces the taped gradient exactly. The
default `None` keeps the taped gradient. The option is reverse mode only: JAX
cannot push `jvp` through a `custom_vjp`.

Compiled temporary working set of `jax.grad` through the truncated solve on the
tiny scheme-1 pitch-angle-scattering fixture, window `w = 4`
(XLA `memory_analysis().temp_size_in_bytes`):

```{list-table}
:header-rows: 1

* - $N_\xi$
  - taped (MiB)
  - `tier1_adjoint_window=4` (MiB)
* - 8
  - 0.161
  - 0.068
* - 32
  - 0.528
  - 0.068
* - 128
  - 1.994
  - 0.068
* - 256
  - 3.949
  - 0.068
```

With `tier1_keep_lowest = Nxi` and `differentiable=True`, full generated
recovery uses the implicit linear solve and, when the storage estimate fits
`tier1_memory_budget_gb`, retains the generated Schur LU factors for the forward
and transposed solves.

## Measured accuracy

Each differentiable path is checked against centered finite differences:

```{list-table}
:header-rows: 1

* - Target
  - Reverse-mode route
  - `grad` against finite difference
* - Pitch-angle scattering with $E_r$, kinetic outputs
  - recycled Krylov transposed solve
  - `2.9e-6`
* - Ramped pitch-angle scattering, RHSMode 1, partial recovery
  - truncated block-Thomas, taped reverse mode
  - agree at `rtol = 1e-6`
* - Monoenergetic $L_{11}$ with respect to $\hat B_{mn}$
  - structured direct plus energy convolution
  - `5.5e-10`
* - Energy convolution to thermal $L_{ij}$
  - closed form against a full RHSMode 2 solve
  - `5.8e-14`
* - `FSABjHat` with respect to density and temperature gradient drives, four upstream decks
  - structured direct and recycled Krylov
  - `4.7e-10` to `4.8e-7`
```

Finite differences have no exact answer to converge to; the step trades
truncation against solver noise. `tools/benchmarks/ad_vs_fortran_fd.py` sweeps
the step and reports the error floor rather than one value. The four decks of
the last row are restricted to those whose reference solve is well converged in
the true residual, since a finite difference of an under-converged reference
measures its noise.

## Measured cost

### Against the primal

On an analytic `geometryScheme=1` pitch-angle-scattering deck at
`(Ntheta, Nzeta, Nxi, Nx) = (13, 13, 16, 6)`, 16,230 unknowns, objective
`FSABjHat`, parameters scaling `THat`, `nHat` and `dTHat/dpsiHat`
(`tools/benchmarks/derivative_cost.py`; Xeon W-2295 pinned to four cores, one
XLA/BLAS thread, float64, nine repeats after warm-up):

```{list-table}
:header-rows: 1

* - Arm, under `jax.jit`
  - median s
  - min s
* - primal, `differentiable=True`
  - 0.373
  - 0.304
* - `jax.grad`
  - 0.373
  - 0.338
* - `jax.value_and_grad`
  - 0.366
  - 0.316
```

A compiled gradient costs 1.00 (median) to 1.11 (minimum) of the differentiable
primal. Its components, compiled: the band assembly and factorization
(`build_tier1_solver`) take 0.31 s, about 85 % of a primal; one block-Thomas
substitution 0.030 s; an operator application, a transposed application, and
the vector-Jacobian product of $A(p)u - b(p)$ at fixed $u$ about 0.0002 s each.
A gradient adds one transposed substitution with refinement and the residual
VJP, which is the adjoint pattern's cost (Paul, Abel, Landreman & Dorland,
*J. Plasma Phys.* 2019); XLA removes the unused tangent of the factorization.

Executed operation by operation, the same gradient costs about twice its primal,
because each of its 13,698 primitives (against 5,783 for a plain primal and
8,460 for a differentiable one) is dispatched separately. The structured route
therefore compiles itself: band assembly and elimination run as one compiled
program per operator structure, and `Tier1Solver` is a pytree so a neighbouring
operator reuses the executable. On loaded hosts the eager primal and gradient
run within 1.2 to 1.4 of their compiled times, and eager and compiled gradients
agree to `4e-16`. Wrapping the objective in `jax.jit` remains the fastest form;
see {doc}`compilation_and_parallelism`.

### Against finite differences

A central-difference gradient of $k$ parameters costs $2k$ converged solves.
The implicit gradient costs one transposed solve whatever $k$ is.

```{figure} ../_static/figures/paper_benchmarks/gradient_cost_scaling.png
:alt: Gradient wall time against parameter count, and agreement across four configurations.
:width: 95%

Four upstream decks spanning one and two species, pitch-angle and
Fokker–Planck collisions, and zero and finite `Er`. The objective is
`FSABjHat`; the parameters are the per-species density and temperature
gradient drives. Regenerate with `tools/paper_benchmarks/gradient_cost_scaling.py`.
```

The finite-difference cost is measured at every $k$, not extrapolated: on the
two-species decks the four points are `2.86, 5.71, 8.56, 11.37` s, linear at
3.1 s per parameter, against a flat one-adjoint cost. At $k = 4$ the ratio is
only 1.4× to 7.1×, because eight solves are the same order as one forward plus
one adjoint. The claim is the slope: profile and geometry optimization runs at
$k$ in the tens, where the slope dominates.

## The optimization chain

Stellarator optimization with a kinetic objective closes the loop from the
plasma boundary to a neoclassical figure of merit under automatic
differentiation:

$$
\text{boundary}
\xrightarrow{\ \texttt{vmex}\ } \{\hat B_{mn}\}
\xrightarrow{\ \texttt{booz\_xform\_jax}\ } \text{Boozer geometry}
\xrightarrow{\ \texttt{dkx}\ } \langle \mathbf{j}\cdot\mathbf{B}\rangle,\ D_{ij},\ \Gamma_s .
$$

`dkx.bootstrap.KineticBootstrapMismatch` exposes the kinetic
$\langle \mathbf{j}\cdot\mathbf{B}\rangle$ as a VMEX objective term with the
interface of VMEX's Redl term, so VMEX's implicit Jacobian differentiates
through the Boozer transform and an exact structured drift-kinetic solve on
each surface. `examples/optimization/QA_optimization_bootstrap_dkx.py` runs it;
the geometry link alone is `examples/autodiff/vmex_to_boozer_sfincs_pipeline.py`.
The workflow is in {doc}`../tutorials/vmex_optimization`.

These scripts implement the full differentiable chain; running one is not by itself a
qualified optimization. A qualified result also needs the converged final equilibrium,
the original-equation residual of every kinetic solve, derivative checks against finite
differences, a finer-grid repeat of the objective and an independent reference, as set out
in `plan.md`.

## Algebraic error of a linear moment

`dkx.sensitivity.linear_observable_algebraic_error` uses the same adjoint to
estimate the algebraic error of a linear moment $c^\mathsf{T}u$: with
$r = b - Au$ and $A^\mathsf{T}\lambda = c$, the signed correction is
$\lambda^\mathsf{T} r$. An approximate adjoint leaves the remainder
$(c - A^\mathsf{T}\lambda)^\mathsf{T}(u_\mathrm{exact} - u)$, which a small
transposed residual does not bound on an ill-conditioned operator (Pierce &
Giles, *J. Comput. Phys.* **200**, 2004). The estimate is an error correction,
not an error bound; its use in convergence reports is in
{doc}`../user_guide/convergence`.

## Worked examples

- `examples/07_gradients/run.py`: gradients on the operator lane.
- `examples/autodiff/gradients_tour.py`: `jax.grad` of kinetic outputs checked
  against finite differences.
- `examples/autodiff/matrix_free_residual_and_jvp.py`: matrix-free residual and
  Jacobian-vector products.
- `examples/autodiff/implicit_diff_through_gmres_solve_scheme5.py`: implicit
  differentiation through a Krylov solve on a VMEC geometry.
- `examples/autodiff/differentiable_geometry_gradients.py`: a geometry scalar
  differentiated with respect to harmonic amplitudes.
- {doc}`../tutorials/gradients`.
