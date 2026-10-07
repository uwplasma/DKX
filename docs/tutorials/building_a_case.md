# 2. Building a case: geometry, collisions, species

The first part used a built-in tokamak, one species and pitch-angle
scattering. This part changes one table of the case at a time: the magnetic
geometry, the collision operator, and the species list, including an impurity.

Scripts: `examples/tutorials/04_vmec_geometry.py` and `.toml`,
`05_boozer_geometry.py` and `.toml`, `06_fokker_planck_vs_pas.py`,
`07_species_and_impurity.py`.

## Geometry

The solve needs, on each flux surface, $B(\theta,\zeta)$ and its angular
derivatives, the Jacobian, the rotational transform $\iota$ and the covariant
components $B_\theta$, $B_\zeta$ (in Boozer coordinates the flux functions
$I = \langle B_\theta\rangle$ and $G = \langle B_\zeta\rangle$). The parallel
gradient that streaming needs is then

$$
\mathbf{b}\cdot\nabla = \frac{B^\theta\,\partial_\theta + B^\zeta\,\partial_\zeta}{B},
$$

and the radial magnetic drift that drives transport is, for DKES trajectories,

$$
\mathbf{v}_{ms}\cdot\nabla\psi = \frac{m_s v^2 (1+\xi^2)}{2 Z_s e B^3}\,\mathbf{B}\times\nabla B\cdot\nabla\psi
\;\propto\; G\,\partial_\theta B - I\,\partial_\zeta B \quad\text{(Boozer coordinates)}:
$$

it is set by how $B$ varies on the surface. That is why the geometry is the
whole difference between a tokamak and a stellarator here
({doc}`../physics/geometry`). The choice is one field of the case:

| `geometry.format` | `geometry.file` | Source |
| --- | --- | --- |
| `"analytic"` | `tokamak`, `lhd_standard`, `lhd_inward`, `w7x_standard` | built-in model fields, for tests and teaching |
| `"vmec"` | a VMEC `wout_*.nc` | a fixed-boundary equilibrium, including non-stellarator-symmetric (`lasym`) ones |
| `"boozer"` | a Boozer `.bc` file | the format stellarator neoclassical work circulates in |

The analytic `tokamak` is SFINCS geometry scheme 1 at its defaults, which keeps
a small $l = 2$, $n = 10$ helical term: it is close to, not exactly,
axisymmetric.

### VMEC

`examples/tutorials/04_vmec_geometry.toml` differs from the first case in two
tables:

```toml
[geometry]
format = "vmec"
file = "../../tests/ref/wout_up_down_asymmetric_tokamak.nc"
surfaces = [0.16, 0.25, 0.36]

[resolution]
theta = 9
zeta = 9
pitch = 8
speed = 4
```

DKX interpolates the VMEC Fourier coefficients to the requested $\psi_N$ and
transforms to the coordinates the solver uses. `zeta` becomes 9 because a VMEC
field is resolved on a toroidal grid even when it is nearly axisymmetric.

```console
python examples/tutorials/04_vmec_geometry.py
dkx run examples/tutorials/04_vmec_geometry.toml --out vmec.nc
```

```text
  psi_N=0.16  Gamma = +2.1862e+19 m^-2 s^-1  Q = +9.1724e+03 W m^-2  <j.B> = -2.9154e+05 A T m^-2
  psi_N=0.25  Gamma = +1.7973e+19 m^-2 s^-1  Q = +6.6725e+03 W m^-2  <j.B> = -2.5557e+05 A T m^-2
  psi_N=0.36  Gamma = +1.5047e+19 m^-2 s^-1  Q = +4.7415e+03 W m^-2  <j.B> = -2.0819e+05 A T m^-2
  converged: True
  solver route: block_tridiagonal_truncated
  residual norm: 2.106e-14
```

The script also prints the SHA-256 of the geometry, so a result records which
equilibrium it came from. The test equilibria in `tests/ref/` are stored as
`*.nc.xz`; the script decompresses the one it needs, and for the CLI route
`xz -dk tests/ref/wout_up_down_asymmetric_tokamak.nc.xz` does it once. For your
own equilibrium, point `file` at its `wout` and keep the surfaces inside the
plasma: the axis is singular for radial derivatives and the last surface has
no outer neighbour.

### Boozer

```toml
[geometry]
format = "boozer"
file = "../../tests/ref/nonStelSym_tiny_geometryScheme12.bc"
surfaces = [0.20, 0.30]
```

A `.bc` file already holds $|B|$ as a Boozer spectrum, so nothing is
transformed. DKX detects the stellarator-symmetric (cosine-only) or asymmetric
column layout itself; the bundled file is the harder, asymmetric one.

```console
python examples/tutorials/05_boozer_geometry.py
```

```text
  psi_N=0.20  Gamma = +2.4396e+07 m^-2 s^-1  Q = +8.5286e-09 W m^-2  <j.B> = +7.0729e-02 A T m^-2
  psi_N=0.30  Gamma = +2.1454e+07 m^-2 s^-1  Q = +5.9739e-09 W m^-2  <j.B> = +5.2741e-02 A T m^-2
  converged: True
  solver route: gcrot
  residual norm: 4.402e-14
```

The fixture is a tiny test field, so the magnitudes mean nothing; the route
does. This field's operator is not block tridiagonal in the way the structured
route needs, so `auto` took the recycled Krylov route (GCROT) and still reports
the true residual of the original equation.

```{figure} ../_static/figures/docs/bmag_contour_w7x.png
:alt: Contour of the magnetic field strength on a W7-X flux surface in the two angles.
:width: 70%

$|B|$ on a W7-X flux surface. The angular grid must resolve this structure,
and the number of toroidal points grows with the number of field periods and
the harmonic content of $|B|$: in a stellarator, refine `theta`, `zeta` and
`pitch` together.
```

A SFINCS deck names its geometry with `geometryScheme`; DKX reads schemes 1–5,
11, 12 and 13, and `dkx convert` maps a deck onto `format`/`file`. The
differentiable path builds the geometry from Fourier amplitudes with
`FluxSurfaceGeometry.from_fourier` ({doc}`bootstrap_gradients_optimization`).

## Collision operators

The collision term $C_{sb}[f_{s1}, f_{b1}]$ is what sets the transport regime.
DKX has three linearized operators ({doc}`../physics/collisions`):

| Operator | Case field | SFINCS `collisionOperator` | Conserves |
| --- | --- | --- | --- |
| Pitch-angle scattering (PAS) | `collisions = "pitch_angle_scattering"` | 1 | particles |
| Linearized Fokker–Planck (FP) | `collisions = "linearized_fokker_planck"` | 0 | particles, momentum, energy |
| Improved Sugama model | namelist only | 3 | particles, momentum, energy |

Pitch-angle scattering is the Lorentz operator,

$$
C^{\mathrm{PAS}}_s[f_{s1}] = \frac{\nu_D^s(v)}{2}\,\frac{\partial}{\partial\xi}\left[(1 - \xi^2)\frac{\partial f_{s1}}{\partial\xi}\right],
\qquad
C^{\mathrm{PAS}}_s[P_L] = -\frac{L(L+1)}{2}\,\nu_D^s\,P_L ,
$$

diagonal in Legendre modes and in speed. The full linearized Fokker–Planck
operator adds energy scattering and the field-particle terms built from the
Rosenbluth potentials of $f_{b1}$; it conserves momentum, couples all speeds
and species, and is what a bootstrap current needs. `06_fokker_planck_vs_pas.py`
runs the same deuterium–electron case with both:

```text
pitch_angle_scattering     route=block_tridiagonal_truncated      <j.B>=['-5.360e+05', '-3.585e+05', '-2.346e+05']
linearized_fokker_planck   route=block_tridiagonal                <j.B>=['-5.035e+05', '-3.344e+05', '-2.170e+05']
```

Two things to read here:

- **Physics.** PAS overestimates the bootstrap current (here by 6–8%) because
  it has no momentum restoring term: the friction between species that should
  return momentum is missing. Quote currents from FP. When FP is too
  expensive, `dkx.momentum_correction` restores momentum to a PAS solve with
  the Sugama–Nishimura moment method, using DKX's own FP friction coefficients;
  it is `collision_model="pas+momentum_correction"` in the bootstrap objective
  and `collision_operator="pas+momentum_correction"` in
  `run_representative` ({doc}`../physics/reduced_models`).
- **Route.** FP couples all species and speeds inside each Legendre block, but
  with DKES trajectories, no tangential drifts and no $\Phi_1$ it is still
  block tridiagonal in $L$. `auto` then takes the speed-coupled structured
  direct route: an exact block elimination from the highest Legendre mode down,
  which lifts the momentum null space of the $L = 1$ block with a low-rank
  correction ({doc}`../numerics/solver_routes`). It takes it when the estimated
  memory fits and the cost is below 150 GFlop per right-hand side. Larger FP
  and Sugama decks go to recycled Krylov, preconditioned by the same
  speed-coupled elimination (`preconditioner="coupled"`) when it fits half the
  available memory and by the cheaper `coarse` operator otherwise. The Krylov
  basis grows with a memory-aware restart up to 1,000 vectors
  ({doc}`../numerics/krylov_and_preconditioners`).

```{figure} ../_static/figures/paper/dkx_fig2_w7x_collisionality.png
:alt: W7-X ion transport matrix elements against collisionality, full Fokker-Planck and pitch-angle scattering.
:width: 85%

Ion transport-matrix elements against collisionality on W7-X, full
Fokker–Planck against pitch-angle scattering
(`tools/publication_figures/generate_sfincs_paper_figs.py`).
```

## Species and impurities

Each `[[species]]` table is one kinetic species with its charge in units of
$e$, its mass in amu, and its density and temperature profiles. Species must be
quasineutral on every surface, $\sum_s Z_s n_s = 0$. With FP collisions every
species collides with every other; with PAS each scatters off all of them.
`07_species_and_impurity.py` adds 1% fully ionized carbon to a deuterium
plasma, keeping $n_e = n_D + 6 n_C$:

```python
species = [
    {"name": "deuterium", "charge": 1, "mass_amu": 2.014,
     "density_m3": [7.52e19, 6.58e19, 5.452e19], "temperature_keV": [1.0, 0.8, 0.6]},
    {"name": "carbon", "charge": 6, "mass_amu": 12.011,
     "density_m3": [8.0e17, 7.0e17, 5.8e17], "temperature_keV": [1.0, 0.8, 0.6]},
    {"name": "electron", "charge": -1, "mass_amu": 0.000548579909,
     "density_m3": [8.0e19, 7.0e19, 5.8e19], "temperature_keV": [1.0, 0.8, 0.6]},
]
```

With FP collisions on the analytic tokamak:

```text
route block_tridiagonal
deuterium  Gamma(psi_N=0.16)=+9.385e+19 m^-2 s^-1
carbon     Gamma(psi_N=0.16)=-1.264e+18 m^-2 s^-1
electron   Gamma(psi_N=0.16)=+1.062e+19 m^-2 s^-1
```

The carbon flux is **inward**. This is the classic neoclassical impurity
accumulation: friction with the main ions pushes a high-$Z$ impurity up the
main-ion density gradient. The sum $\sum_s Z_s\Gamma_s$ is not zero here: $E_r$
was prescribed, not solved for. The next part makes it ambipolar. With more
than a trace of impurity, the in-surface potential $\Phi_1$ can matter too;
that is also in the next part.

The classical (gyro-motion plus friction) impurity flux, which adds to the
neoclassical one, is algebraic and is in `dkx.impurity`
({doc}`../physics/phi1_and_impurities`).

## Next

{doc}`transport_physics`: the radial electric field, transport coefficients
and $\Phi_1$.
