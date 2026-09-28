# Transport matrices and monoenergetic coefficients

A profile solve answers one question: the fluxes for these gradients. A
transport matrix answers all of them at once. On one flux surface the
neoclassical fluxes are linear in the thermodynamic forces, so the solve for a
unit of each force gives a matrix from which any combination of gradients
follows. DKX computes both forms SFINCS defines:

| Form | SFINCS setting | Size | Relates |
| --- | --- | --- | --- |
| thermal transport matrix | `RHSMode = 2` | 3×3 | density gradient, temperature gradient and inductive parallel field to particle flux, heat flux and parallel flow |
| monoenergetic matrix | `RHSMode = 3` | 2×2 | one speed, pitch-angle scattering only; gives $D_{11}^*$, $D_{31}^*$, $D_{33}^*$ |

The monoenergetic form drops the species and speed structure and solves one
pitch-angle problem per point, so a whole database over collisionality and
$E^*$ costs about what one profile solve does. It is the form neoclassical codes
are compared on (Beidler et al. 2011; {doc}`../benchmarks/cross_code`). The
normalizations are in {doc}`../physics/normalizations` and the reduced models in
{doc}`../physics/reduced_models`.

## A monoenergetic database

`examples/04_monoenergetic_scan` sweeps $\nu'$ over five decades at two values
of $E^*$ on a circular tokamak deck:

```python
import numpy as np
import dkx

database = dkx.run_monoenergetic_database(
    "examples/04_monoenergetic_scan/input.namelist",
    (1.0e-3, 1.0e-2, 1.0e-1, 1.0e0, 1.0e1),   # nu' = nu R / (iota v)
    (0.0, 1.0e-1),                             # E* = Er / (v B)
    output_path="monoenergetic.npz",
    solve_method="auto",
    tol=1.0e-10,
)
d11 = np.asarray(database.d11_star)   # shape (n_nu, n_estar)
d31 = np.asarray(database.d31_star)
d33 = np.asarray(database.d33_star)
```

The deck supplies the geometry and the pitch grid; the driver sets the rest.
The same scan from the command line:

```console
dkx sfincs monoenergetic-database --input examples/04_monoenergetic_scan/input.namelist \
    --nu-prime 1e-3 1e-2 1e-1 1 10 --e-star 0 0.1 --out monoenergetic.npz
```

Two checks come free with the physics, and the example script asserts both:

- $D_{33}^*$ is normalized by its collisional value, so it must approach 1 from
  below as $\nu'$ grows.
- At $E^* = 0$ the star-normalized $D_{11}^*$ must rise monotonically with
  $\nu'$. A database that is not monotone is under-resolved in pitch angle,
  whatever its residuals say: raise `Nxi` in the deck.

$D_{31}^*$ is the coefficient that changes sign, so plot it on a linear axis; a
log axis of $\lvert D_{31}^*\rvert$ hides the zero crossing.

At low collisionality the pitch-angle boundary layer narrows and `Nxi` must
grow with $1/\nu'$. The Shaing–Callen benchmark raises it point by point
({doc}`../benchmarks/analytic_limits`).

## A thermal transport matrix

`dkx.run.run_transport_matrix` runs a deck with `RHSMode = 2` or `3` end to end:

```python
from dkx.run import run_transport_matrix

run = run_transport_matrix(
    "examples/sfincs_examples/transportMatrix_geometryScheme2/input.namelist",
    tol=1e-10,
    out_path="sfincsOutput.h5",
)
print(run.transport_matrix)            # 3x3 for RHSMode=2, 2x2 for RHSMode=3
print(run.solve_result.method)         # the route that solved it
```

All right-hand sides are solved together against one operator, so on a direct
route they share one factorization ({doc}`../numerics/factor_reuse`).
`run.moments` holds the diagnostic table under the `sfincsOutput.h5` names, and
for `RHSMode = 3` `run.d11_bounds` carries variational bounds on $D_{11}$. The
command-line form:

```console
dkx sfincs transport-matrix-v3 --input examples/sfincs_examples/transportMatrix_geometryScheme2/input.namelist \
    --out-matrix transportMatrix.npy --out sfincsOutput.h5
```

`examples/transport/transport_matrix_rhsmode2_and_rhsmode3.py` computes both
forms on small grids and draws each matrix as a heat map;
`examples/transport/transport_coefficients.py` and
`examples/transport/transport_matrix_rhsmode2_scheme11_and_scheme5.py` extend it to
more geometry schemes.

## Checks on a transport matrix

- **Onsager symmetry.** The thermal matrix satisfies $L_{ij} = L_{ji}$ in the
  continuum. The discrete asymmetry falls under refinement, at least 3× per
  rung to 7e-5 in `tests/test_transport_limits.py`, so its size is a direct
  measure of discretization error.
- **Collision operator.** Pitch-angle scattering lacks momentum restoration and
  overestimates the bootstrap coefficient. Compare with
  `collisionOperator = 0` (full Fokker–Planck) before quoting a current.

```{figure} ../_static/figures/paper/dkx_fig2_w7x_collisionality.png
:alt: W7-X ion transport matrix elements against collisionality, full Fokker-Planck and pitch-angle scattering.
:width: 85%

Ion transport-matrix elements against collisionality on W7-X, full
Fokker–Planck against pitch-angle scattering
(`tools/publication_figures/generate_sfincs_paper_figs.py`).
```
