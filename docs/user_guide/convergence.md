# Convergence

A DKX number carries three separate kinds of error, and each is checked
separately:

| error | check | where |
|---|---|---|
| algebraic: the linear system was not solved exactly | the original residual $\lVert Ax-b\rVert/\lVert b\rVert$ against `solver.relative_tolerance` | every run; `primal_residual`, `certificate()` |
| $E_r$ search: a root was missed or poorly located | adaptive midpoint refinement of the ambipolar search | `[convergence]` in the case |
| discretization: the grid is too coarse | refine $N_\theta, N_\zeta, N_\xi, N_x$ and watch the observables | `dkx converge` |

A small residual says nothing about the grid, and a settled grid says nothing
about whether the solves on it were accepted. `dkx converge` reports both.

## Algebraic acceptance

Every native solve is accepted only if the returned state satisfies the
original, unpreconditioned equations: DKX reapplies the operator to the
complete state and requires
$\lVert Ax-b\rVert_2 \le \epsilon\,\lVert b\rVert_2$ with
$\epsilon$ = `solver.relative_tolerance`. A solve that fails raises; nothing
is returned with an unaccepted state. The residual and drive norms of each
surface are kept (`primal_residual`, `primal_rhs_norm`) with
`metadata["original_residual_evidence"]` naming the norm, the tolerance and
whether the state was complete ({doc}`outputs`).

Choose `relative_tolerance` well below the relative accuracy you need from the
observables. The default is $10^{-10}$; the examples use $10^{-8}$.

## Refining the $E_r$ search

For `workflow = "ambipolar_profile"`, the `[convergence]` table refines the
electric-field search, not the grid:

```toml
[convergence]
enabled = true
observables = ["particle_flux", "heat_flux", "electric_field"]
relative_tolerance = 0.02
max_refinements = 2
```

Each level inserts the midpoint of every sampled interval and solves there. The
hierarchy always runs to `max_refinements`, so an early stable root cannot hide
a pair of crossings a finer level would expose. `ambipolar_refinement_status`
is `resolved` only when the last two levels keep the same nonzero root count
and the roots, the requested observables and the bracket widths meet the
tolerances; `refinement_exhausted` means roots were seen but did not settle;
`no_bracket_observed` means no sign change was seen, which does not prove there
is no root. The retained-evaluation bound is computed before anything runs,
and `dkx validate` prints it.

## Refining the grid: `dkx converge`

```console
dkx converge case.toml
dkx converge case.toml --axes pitch speed --factor 1.5 --tolerance 0.01
dkx converge input.namelist --format json
```

For a case at resolution $(N_\theta, N_\zeta, N_\xi, N_x)$ the study runs:

1. the baseline;
2. one refinement per requested axis, that axis multiplied by `--factor`
   (default 1.5) and raised by at least one point;
3. unless `--no-joint`, one run with every requested axis refined together.

That is `len(axes) + 2` solves. An axisymmetric case (`zeta = 1`) skips the
`zeta` axis, because refining it would change the physics rather than the
accuracy. For a namelist deck, angular sizes are kept odd, as the SFINCS grid
builder requires.

Each refinement is compared with the baseline entrywise, on matching surfaces
and species, with signs kept: a sign reversal or a change in a small species'
flux is not hidden behind a larger one. For each observable the reported change
is

$$
\max_i \frac{|f_i^{\text{refined}} - f_i^{\text{baseline}}|}{\max(|f_i^{\text{baseline}}|,\ a/\tau)}\ ,
$$

reduced to the worst entry, where $\tau$ is the tolerance and $a$ an optional
absolute tolerance (zero by default). The default observables for a case are
`particle_flux_m2_s`, `heat_flux_W_m2` and `parallel_current_A_T_m2`; for a
deck they are `FSABFlow`, `FSABjHat`, `particleFlux_vm_psiHat` and
`heatFlux_vm_psiHat` (`RHSMode = 1`) or every transport-matrix entry
(`RHSMode = 2/3`). Missing, empty, nonfinite or reshaped arrays make the rung fail rather than
report a change.

The report has two verdicts:

| verdict | meaning |
|---|---|
| grid changes within tolerance (`converged`) | every per-axis change and the joint change are below the tolerance, and no rung failed |
| `original_equations_accepted` | the baseline and every rung carry complete, finite original-residual evidence within their tolerance |

The exit status is 0 only when both hold. With `--format json` the report also
carries every rung's resolution, per-observable changes, observables, residuals
and status.

### Why the joint run matters

Axes couple. A resolution that looks adequate in $\theta$ may only look so
because $N_\xi$ is too coarse to expose the error. The checked-in
`examples/01_tokamak_profile/case.toml` records the case: at `pitch = 8` the
`theta` refinement moves the outputs by 0.2%, and at `pitch = 40` the same
refinement moves them by 74%. On that teaching grid `theta` 9 → 14 alone moves
the particle flux by 1187%. When the joint change exceeds twice the largest
single-axis change, `dkx converge` says the per-axis rows are not a safe summary
of the case.

### Python

```python
from dkx.workflows.converge import converge_case, converge_sfincs_input

report = converge_case(
    case,
    axes=("theta", "pitch"),
    factor=1.5,
    tolerance=0.02,
    absolute_tolerances={"particle_flux_m2_s": 1e15},   # m^-2 s^-1, for a flux near zero
    richardson=True,
)
report.converged, report.original_equations_accepted, report.worst_grid_uncertainty
```

`absolute_tolerances` gives a physical error budget, in the array's units, for
an observable that is near zero; each entry must then change by less than the
larger of the relative and absolute budgets. `converge_sfincs_input` does the
same for a linear deck (`RHSMode` 1–3, `includePhi1 = .false.`,
`forceOddNthetaAndNzeta = .true.`) without converting it; every original
right-hand-side residual must pass the deck's `solverTolerance`, and no files
are written.

### A discretization error bar

A two-grid difference is a resolution check, not an error estimate.
`richardson=True` adds a third rung per axis at `factor**2` (one more solve per
axis) and reports, per observable, an observed order, an extrapolated value and
the fine-grid convergence index (GCI) of Roache and ASME V&V 20. The observed
order $p$ solves

$$
p = \frac{\left|\ln|d_{32}/d_{21}| + q(p)\right|}{\ln r_{21}},\qquad
q(p) = \ln\frac{r_{21}^{\,p} - s}{r_{32}^{\,p} - s},
$$

with $d_{21}$ the fine-minus-medium and $d_{32}$ the medium-minus-coarse
differences and the refinement ratios taken from the actual resolutions.
`tests/test_converge_workflow.py` reproduces the worked example of Celik et al.
(2008), ASME J. Fluids Eng. 130, 078001: order 1.53, extrapolated value 6.1685,
fine-grid GCI 2.2%.

The estimate is **refused** unless $R = d_{21}/d_{32}$ lies in $(0, 1)$: a
diverging ($R > 1$) or oscillating ($R < 0$) ladder has not reached the
asymptotic range, and extrapolating through it would invent accuracy. A
monotone ladder whose order exceeds `max_order` (12) is reported with the status
`monotone, faster than max_order`, no extrapolation, and a bar of
$\max(F_s, 3)\,|d_{21}|/|f_1|$; spectral directions (Legendre pitch, Fourier
angles) behave this way. `report.worst_grid_uncertainty` is infinite when any
requested estimate was refused, because an estimate that could not be made is
not a small one. The algebraic error bar is separate
({doc}`../numerics/differentiation`); neither bounds model error.

## Choosing a resolution

- **Start coarse, then refine jointly.** The checked-in examples use teaching
  grids (for example `theta = 9, zeta = 1, pitch = 8, speed = 4`) that run in
  seconds and are not converged. Raise the axes until `dkx converge` passes
  both verdicts at the tolerance you intend to quote.
- **Low collisionality needs pitch and angle.** The trapped–passing boundary
  layer makes $N_\xi$ and $N_\zeta$ the sensitive axes at low
  collisionality, while $N_x$ changes more slowly.
- **Strong $E_r$ needs speed and pitch.** The normalized field
  $E_* = \dfrac{c\,G}{\iota\,v_s B_0}\dfrac{d\Phi_0}{d\psi}$ of Landreman et
  al. (Phys. Plasmas 21, 042503, 2014) compares $E_r$ with the value at which
  the $E\times B$ precession cancels parallel streaming for thermal particles.
  Below $|E_*| \approx 1/3$ (`dkx.validity.E_STAR_TRAJECTORY_AGREEMENT`) the
  trajectory models agree closely; above it they separate, the bootstrap
  current can change sign, and the speed and pitch resolutions must grow with
  $E_r$. `dkx.validity.normalized_radial_electric_field_of` returns $E_*$ per
  species from an operator build. Cold ions against hot electrons reach this
  regime first.
- **Record the pitch allocation.** Under `pitch_speed_ramp = 1` or `2`,
  raising `pitch` changes the truncation at several speed nodes at once; the
  result records the active modes per speed node
  (`metadata["phase_space"]`), and a study should quote them.
- **Check the ambipolar search separately.** Grid convergence at one field does
  not certify root discovery; use `[convergence]` for the search and
  `dkx plot --kind search` to look at it.

{doc}`../numerics/discretization` describes the grids, and
{doc}`../benchmarks/validation_matrix` lists which configurations have been
checked against reference codes.
