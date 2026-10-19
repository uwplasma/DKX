# 7. Transport matrices: RHSMode 2/3 and monoenergetic coefficients

A profile solve gives the fluxes for one set of gradients. A transport matrix
gives the response to any gradients with one set of solves, and the
monoenergetic coefficients separate geometry from profiles altogether.

## Monoenergetic coefficients

Freezing the speed gives one pitch-angle equation with PAS collisions,
parameterized by two numbers only: the collisionality $\nu' = \nu R/(\iota v)$
and the normalized field $E^* = E_r/(vB)$:

$$
\xi\,\mathbf{b}\cdot\nabla f
- \frac{1-\xi^2}{2}\,(\mathbf{b}\cdot\nabla\ln B)\,\frac{\partial f}{\partial\xi}
+ \frac{\mathbf{E}\times\mathbf{B}}{vB^2}\cdot\nabla f
- \frac{\nu}{2v}\,\frac{\partial}{\partial\xi}\Bigl[(1-\xi^2)\frac{\partial f}{\partial\xi}\Bigr] = s .
$$

The radial flux and parallel flow respond to the radial gradient force and the
parallel (inductive) force through

$$
\begin{pmatrix} \text{flux} \\ \text{flow} \end{pmatrix}
= -\begin{pmatrix} D_{11} & D_{13} \\ D_{31} & D_{33} \end{pmatrix}
\begin{pmatrix} \text{radial force} \\ \text{parallel force} \end{pmatrix},
\qquad D_{13} = -D_{31}\ \text{(Onsager)}.
$$

$D_{11}$ is the radial diffusion ($1/\nu$, $\sqrt\nu$ and plateau regimes),
$D_{31}$ the bootstrap coefficient and $D_{33}$ the parallel conductivity. DKX
reports $D^*$ in the Beidler et al. (2011) normalization: $D_{11}^*$ against
the plateau value, $D_{31}^*$ against the banana-regime bootstrap coefficient,
and $D_{33}^*$ against the Spitzer conductivity, so $D_{33}^*\to1$ at high
$\nu'$.

`examples/tutorials/09_monoenergetic.py` on a circular-tokamak deck:

```python
db = dkx.run_monoenergetic_database(
    "examples/tutorials/09_monoenergetic.namelist",
    np.logspace(-3, 1, 9), (0.0, 0.1), output_path="database.npz")
```

CLI:
`dkx sfincs monoenergetic-database --input DECK --nu-prime ... --e-star ... --out db.npz`.
Five collisionalities at $E^*=0$ (5 s for the full scan):

```text
  nu_prime    E_star          D11*          D31*          D33*
     0.001         0   2.8459e-01   5.5342e-01   8.2917e-01
      0.01         0   1.9163e+00   3.6224e-01   8.8818e-01
       0.1         0   3.1830e+00   3.3623e-02   9.8962e-01
         1         0   1.5707e+01   4.6646e-04   9.9986e-01
        10         0   1.5483e+02   4.6850e-06   1.0000e+00
  D33* at the largest nu' (collisional limit is 1): 1.0000
  D11* monotone in nu' at E*=0: True
```

```{figure} ../_static/figures/tutorials/monoenergetic.png
:alt: D11, D31 and D33 against collisionality at two values of E*.
:width: 90%

$D_{11}^*$, $D_{31}^*$, $D_{33}^*$ against $\nu'$ at $E^*=0$ and $0.1$.
```

Read it as a tokamak behaves: $D_{11}^*$ rises from the banana regime through
the plateau into Pfirsch-Schlueter; $D_{31}^*$ is largest at low
collisionality and falls away; $D_{33}^*$ approaches 1 from below. A
non-monotone $D_{11}^*$ at $E^*=0$ means the pitch grid is too coarse whatever
the residual says: the boundary layer at the trapped-passing boundary narrows
as $\nu'$ falls and `Nxi` must grow like $1/\sqrt{\nu'}$ or faster. A finite
$E^*$ suppresses $D_{11}^*$ at high $\nu'$ by $E\times B$ precession. Since
2.9.0 the database uses spectral angular derivatives (MONKES's discretization).

## The thermal transport matrix, RHSMode 2

Across all speeds the fluxes of one species are linear in three forces: a
density-gradient force, a temperature-gradient force and the inductive
$\langle E_\parallel B\rangle$. `RHSMode = 2` solves one unit of each, and the
$3\times3$ matrix $L_{ij}$ relates them to particle flux, heat flux and
parallel flow in SFINCS's normalization ({doc}`../physics/drives_and_rhs_modes`).
Any gradients follow without another solve. `RHSMode = 3` is the
$2\times2$ monoenergetic form. `10_transport_matrix.py` runs both (10 s):

```python
matrix = dkx.run(RHSMode=2, geometryScheme=2, Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
                 nu_n=0.15, collisionOperator=1, ..., Ntheta=15, Nzeta=15, Nxi=24, NL=4, Nx=5).transport_matrix
```

```text
  RHSMode=2 (3x3, LHD model):
    [[-0.0458 -0.101  -0.0154]
     [-0.101  -0.3732 -0.0861]
     [-0.0154 -0.0861  2.954 ]]
    Onsager check |L12 - L21| / |L12| = 1.59e-14
  RHSMode=3 (2x2, monoenergetic):
    [[-0.1029  0.0039]
     [ 0.004   0.8773]]
    Onsager check |L12 - L21| / |L12| = 3.33e-03
```

```{figure} ../_static/figures/tutorials/transport_matrix.png
:alt: Heat maps of the 3x3 and 2x2 transport matrices.
:width: 70%

The two matrices as heat maps.
```

The symmetry $L_{ij}=L_{ji}$ is exact in the continuum, so the Onsager check
is a resolution diagnostic: $10^{-14}$ for the thermal matrix here, $3\times10^{-3}$
for the monoenergetic one, which falls under refinement. The three right-hand
sides share one factorization on the structured direct route
({doc}`../numerics/factor_reuse`). CLI:
`dkx sfincs transport-matrix-v3 --input DECK --out-matrix transportMatrix.npy`.

## Next

{doc}`08_phi1_bootstrap_gradients`: $\Phi_1$, the bootstrap current and
gradients.
