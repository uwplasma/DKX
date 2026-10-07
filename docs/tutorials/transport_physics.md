# 3. Transport physics: $E_r$, coefficients, $\Phi_1$

With a geometry, an operator and species in hand, this part asks the questions
neoclassical transport is used for: what radial electric field the plasma
sets, what its transport coefficients are, and what the in-surface potential
$\Phi_1$ changes.

Scripts: `examples/tutorials/08_ambipolar_er.py` and `.toml`
(`08_ambipolar_er_w7x.toml` is the production-scale case),
`09_monoenergetic.py` and `.namelist`, `10_transport_matrix.py`, `11_phi1.py`.

## The ambipolar radial electric field

In a stellarator the electron and ion particle fluxes are not automatically
equal: each depends on $E_r$, and $E_r$ adjusts until no net charge leaves the
surface. The ambipolarity condition is

$$
J_r(E_r) \;=\; \sum_s Z_s e\,\Gamma_s(E_r) \;=\; 0 .
$$

$J_r(E_r)$ can have one root or three: an **ion root** at small or negative
$E_r$, an **electron root** at large positive $E_r$, and an unstable root
between them. A root is stable when a small increase of $E_r$ drives an outward
current that pushes it back, which DKX reads from the sign of $dJ_r/dE_r$
({doc}`../physics/electric_field`). In a tokamak the fluxes are intrinsically
ambipolar (they balance at any $E_r$), so this workflow is for stellarators.

Two fields turn a profile case into an ambipolar one:

```toml
[run]
workflow = "ambipolar_profile"

[electric_field]
mode = "ambipolar"
search_kV_m = [-5.0, 5.0]
find_all_roots = true
continue_branches = true
search_points = 5
root_tolerance_kV_m = 0.05
```

`08_ambipolar_er.toml` runs deuterium and electrons with FP collisions on the
analytic W7-X field. DKX samples $J_r$ at `search_points` fields across
`search_kV_m`, brackets every sign change, refines each bracket and classifies
the root:

```console
python examples/tutorials/08_ambipolar_er.py
dkx run examples/tutorials/08_ambipolar_er.toml --out er.nc
dkx roots er.nc
```

```text
  psi_N=0.09: 1 root(s), selected #0
    Er = -3.6719 kV/m  J_r = +2.716e-03 A m^-2  dJ_r/dEr = +1.658e+00 A m^-2 (kV/m)^-1  [ion]
    at the selected root: Er = -3.6719 kV/m  Gamma = +1.7654e+18 m^-2 s^-1  Q = +3.1118e+03 W m^-2
  psi_N=0.16: 1 root(s), selected #0
    Er = -3.5156 kV/m  J_r = +4.691e-03 A m^-2  dJ_r/dEr = +8.711e-01 A m^-2 (kV/m)^-1  [ion]
    at the selected root: Er = -3.5156 kV/m  Gamma = +6.3518e+17 m^-2 s^-1  Q = +1.2298e+03 W m^-2
  selection rule: continue the selected branch identity; nearest zero on the first surface and nearest prior field after branch loss
  all surfaces bracketed: True
```

Equal ion and electron temperatures at these collisionalities give an ion root
on both surfaces, at a few kV/m negative. The residual current at the root is
within `root_tolerance_kV_m` times the slope; tighten the tolerance to shrink
it. The fluxes reported for each surface are those at its selected root. The
other roots, their slopes and the selection reason stay in the result
(`ambipolar_root_kV_m`, `ambipolar_root_type`, `selected_ambipolar_root`),
and `dkx roots` prints them. A root is only found if its sign change falls
between two samples, so a coarse `search_points` can step over a close pair.

```{figure} ../_static/figures/docs/ambipolar_er_roots.png
:alt: Radial current against radial electric field with ion, unstable and electron roots.
:width: 70%

$J_r(E_r)$ with ion, unstable and electron roots.
```

```{figure} ../_static/figures/paper_benchmarks/w7x_ambipolar_er.png
:alt: Radial current against electric field on W7-X with every sampled evaluation and the classified roots.
:width: 85%

At benchmark scale on W7-X: every sampled evaluation and the classified roots
(`tools/paper_benchmarks/w7x_ambipolar_er.py`).
```

$E_r$ is a resolution-sensitive output: the recorded five-surface W7-X
ambipolar profile still moves under refinement at the resolutions tried
({doc}`../benchmarks/validation_matrix`). Check it with `dkx converge`.

**Scans in one process.** For many $E_r$ values on one surface,
`dkx.prepare_er_scan(case, surface_index=...)` builds geometry, grids and
collisions once and `dkx.batched_er_scan(problem, er_values)` evaluates them in
one compiled call, reusing factors and the Krylov recycle subspace. For one
surface from a SFINCS deck, `dkx.find_ambipolar_er` runs a Brent search, and
`dkx sfincs scan-er` plus `dkx sfincs ambipolar-solve` reproduce the
`sfincsScan` directory workflow ({doc}`../user_guide/scans_and_parallelism`).

## Monoenergetic coefficients $D_{11}$, $D_{31}$, $D_{33}$

The fluxes depend on speed through the collision frequency and the
$E\times B$ drift. Freezing the speed gives the monoenergetic problem: one
pitch-angle equation per speed, with PAS collisions,

$$
\xi\,\mathbf{b}\cdot\nabla f
- \frac{1-\xi^2}{2}\,(\mathbf{b}\cdot\nabla\ln B)\,\frac{\partial f}{\partial\xi}
+ \frac{\mathbf{E}\times\mathbf{B}}{v\,B^2}\cdot\nabla f
- \frac{\nu}{2v}\,\frac{\partial}{\partial\xi}\Bigl[(1-\xi^2)\frac{\partial f}{\partial\xi}\Bigr]
= s(\theta,\zeta,\xi) ,
$$

where $s$ is the radial-drift or the inductive source, parameterized by two numbers only: the collisionality $\nu' = \nu R/(\iota v)$
and the normalized field $E^* = E_r/(v B)$. Its solution gives a $2\times2$
matrix of monoenergetic coefficients,

$$
\begin{pmatrix} \text{radial flux} \\ \text{parallel flow} \end{pmatrix}
= -\begin{pmatrix} D_{11} & D_{13} \\ D_{31} & D_{33} \end{pmatrix}
\begin{pmatrix} \text{radial gradient force} \\ \text{parallel (inductive) force} \end{pmatrix},
$$

with Onsager symmetry $D_{13} = -D_{31}$. $D_{11}$ is the radial diffusion
(the $1/\nu$, $\sqrt\nu$ and plateau regimes of a stellarator), $D_{31}$ the
bootstrap coefficient and $D_{33}$ the parallel conductivity. DKX reports them
in the Beidler et al. (2011) normalization: $D_{11}^*$ against the plateau value
of the equivalent tokamak, $D_{31}^*$ against the banana-regime bootstrap
coefficient, and $D_{33}^*$ against the collisional (Spitzer) conductivity, so
$D_{33}^* \to 1$ at high $\nu'$. The thermal coefficients follow by a
convolution over a Maxwellian in speed ({doc}`../physics/reduced_models`).

```python
import numpy as np
import dkx

database = dkx.run_monoenergetic_database(
    "examples/tutorials/09_monoenergetic.namelist",
    (1.0e-3, 1.0e-2, 1.0e-1, 1.0e0, 1.0e1),   # nu'
    (0.0, 1.0e-1),                             # E*
    output_path="monoenergetic.npz", solve_method="auto", tol=1.0e-10,
)
d11 = np.asarray(database.d11_star)   # shape (n_nu, n_estar)
```

`python examples/tutorials/09_monoenergetic.py` on the circular-tokamak deck:

```text
  nu_prime    E_star          D11*          D31*          D33*
     0.001         0   2.84592e-01   5.53420e-01   8.29166e-01
      0.01         0   1.91628e+00   3.62242e-01   8.88181e-01
       0.1         0   3.18302e+00   3.36226e-02   9.89621e-01
         1         0   1.57073e+01   4.66465e-04   9.99856e-01
        10         0   1.54828e+02   4.68502e-06   9.99999e-01
  D33* at the highest nu', E*=0: 0.999999 (collisional limit is 1)
  D11* monotone in nu' at E*=0: True
```

Read it as a tokamak should behave: $D_{11}^*$ rises with collisionality from
the banana regime through the plateau into Pfirsch–Schlüter, $D_{31}^*$ (the
bootstrap coefficient) is largest at low collisionality and falls away, and
$D_{33}^*$ approaches 1 from below. A non-monotone $D_{11}^*$ at $E^* = 0$
means the pitch grid is too coarse, whatever the residual says: the boundary
layer at the trapped–passing boundary narrows as $\nu'$ falls, and `Nxi` must
grow like $1/\sqrt{\nu'}$ or faster. The CLI form is
`dkx sfincs monoenergetic-database --input DECK --nu-prime ... --e-star ... --out db.npz`.

**Angular discretization.** Since 2.9.0, `monoenergetic_database` uses
spectral (Fourier) derivatives in $\theta$ and $\zeta$ unless the deck sets
`thetaDerivativeScheme`/`zetaDerivativeScheme`. That is MONKES's
discretization, and it converges like MONKES: on HSX, 3% at
$19\times41\times64$ instead of $25\times51\times96$ with finite differences. To
pair a database with a full-kinetic solve of the same deck, set the schemes in
the deck so both use the same angles.

```{figure} ../_static/figures/paper_benchmarks/monoenergetic_icnts_w7x.png
:alt: D11 and D31 against collisionality on W7-X at several E*, with SFINCS points.
:width: 80%

The same scan at benchmark resolution on W7-X, with SFINCS v3 points
({doc}`../benchmarks/cross_code`).
```

## The thermal transport matrix

Across all speeds, the fluxes of one species are linear in three forces: a
density-gradient force (which also carries $E_r$), a temperature-gradient
force and the inductive parallel field $\langle E_\parallel B\rangle$. SFINCS's
`RHSMode = 2` solves one unit of each, and the $3\times3$ matrix $L_{ij}$
relates them to the particle flux, the heat flux and the parallel flow
in SFINCS's normalization (exact definitions in
{doc}`../physics/drives_and_rhs_modes`). Any gradients then follow without another solve.

```python
from dkx.run import run_transport_matrix

run = run_transport_matrix(
    "examples/sfincs_examples/transportMatrix_geometryScheme2/input.namelist", tol=1e-10)
print(run.transport_matrix)
print("route:", run.solve_result.method)
```

```text
[[-1.1376e-02 -4.1675e-02  2.7708e-02]
 [-4.1669e-02 -3.1588e-01 -2.2964e-02]
 [ 2.7685e-02 -2.3100e-02  2.5827e+01]]
route: block_tridiagonal
```

The three right-hand sides share one factorization of the speed-coupled
structured direct route ({doc}`../numerics/factor_reuse`); on the upstream
suite this deck solves in 2.8 s where the Krylov route took 33 s. The matrix is
nearly symmetric: Onsager symmetry $L_{ij} = L_{ji}$ holds in the continuum,
and the remaining asymmetry (1e-4 to 1e-3 relative here) is discretization error that falls under refinement.
`10_transport_matrix.py` computes the `RHSMode = 2` and `RHSMode = 3` forms and
draws each as a heat map. CLI:
`dkx sfincs transport-matrix-v3 --input DECK --out-matrix transportMatrix.npy`.

## The in-surface potential $\Phi_1$

The electrostatic potential is not exactly a flux function. Its variation
$\Phi_1(\theta,\zeta)$ is small, $e\Phi_1/T \sim \Delta$, but it enters the
kinetic equation through a parallel electric field and a tangential
$E\times B$ drift, and the density of every species through a Boltzmann factor.
It is fixed by quasineutrality on the surface,

$$
\sum_s Z_s\,n_s(\theta,\zeta) = \sum_s Z_s\left[\,n_{s0}\,e^{-Z_s e\Phi_1/T_s}
+ \int d^3v\, f_{s1}\right] = 0 ,
$$

which makes the problem nonlinear in $\Phi_1$; DKX solves the coupled kinetic,
quasineutrality and gauge system by Newton–Krylov ({doc}`../physics/phi1_and_impurities`).
$\Phi_1$ matters most for high-$Z$ impurities, whose Boltzmann factor carries
$Z$ and whose radial flux picks up $\mathbf{v}_{E1}\cdot\nabla\psi$.

Natively it is one field: `[physics] phi1 = "kinetic"` on a `profile` case.
Hydrogen and electrons on the analytic W7-X field with FP collisions, the
particle flux on two surfaces:

```text
off      route=block_tridiagonal   Gamma_H = ['+1.088e+18', '+7.231e+17']
kinetic  route=phi1_newton_krylov  Gamma_H = ['+1.391e+18', '+9.117e+17']
```

$\Phi_1$ changes the main-ion flux by about 25% on this coarse grid. The route
is no longer a single linear solve but the Newton–Krylov iteration.
`11_phi1.py` runs the same comparison with a carbon impurity.

Status, stated precisely: $\Phi_1$ is `validated_limited`. On the
`geometryScheme = 4` $\Phi_1$ example deck and a trace-carbon variant, DKX
matches Fortran SFINCS v3 to 2.0e-6 over an eight-rung resolution ladder
(`validation/phi1_sfincs_benchmark_v1.json`). That is one geometry, FP
collisions only and zero $E_r$, and the trace-impurity flux is resolved only to
about 3% in `Nx`. `phi1 = "full"`, quasineutrality option 2, adiabatic species
and the ambipolar workflow with $\Phi_1$ are refused natively and go through a
SFINCS namelist. `dkx.phi1.phi1_solution` returns the converged state with a
matrix-free implicit adjoint, so $\Phi_1$ results can be differentiated (next
part).

## Next

{doc}`bootstrap_gradients_optimization`: the bootstrap current, gradients and
optimization.
