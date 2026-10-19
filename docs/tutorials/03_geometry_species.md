# 3. Geometry and species

This page changes two tables of the case: the magnetic geometry and the
species list.

## Geometry

On each flux surface the solve needs $B(\theta,\zeta)$ and its derivatives, the
Jacobian, the rotational transform $\iota$ and the covariant components
$B_\theta$, $B_\zeta$ (in Boozer coordinates the flux functions $I$ and $G$).
Streaming uses

$$
\mathbf{b}\cdot\nabla = \frac{B^\theta\,\partial_\theta + B^\zeta\,\partial_\zeta}{B},
$$

and the radial drift that drives transport is set by how $B$ varies on the
surface: for DKES trajectories it is proportional to
$G\,\partial_\theta B - I\,\partial_\zeta B$. That is the whole difference
between a tokamak and a stellarator here ({doc}`../physics/geometry`). The
choice is one field of the case:

| `geometry.format` | `geometry.file` | Source |
| --- | --- | --- |
| `"analytic"` | `tokamak`, `lhd_standard`, `lhd_inward`, `w7x_standard` | built-in model fields |
| `"vmec"` | a VMEC `wout_*.nc` | fixed-boundary equilibrium, including `lasym` |
| `"boozer"` | a Boozer `.bc` file | the format stellarator neoclassical work circulates in |

The analytic `tokamak` is SFINCS geometry scheme 1 at its defaults, which keeps
a small $l = 2$, $n = 10$ helical term: close to, not exactly, axisymmetric.

### VMEC

`examples/tutorials/04_vmec_geometry.toml` differs from page 1 in two tables:

```toml
[geometry]
format = "vmec"
file = "../../tests/ref/wout_vmex_seed_nfp2_ns13.nc"
surfaces = [0.25, 0.36, 0.49]
```

with `[resolution]` `theta` and `zeta` both above 1, since a VMEC field is
resolved on a toroidal grid. It is a two-field-period equilibrium. In Python,
`dkx.Case.from_file` loads the file and `dkx.run` runs it; from the shell:

```console
python examples/tutorials/04_vmec_geometry.py
dkx run examples/tutorials/04_vmec_geometry.toml --out vmec.nc
```

```text
  psi_N=0.25  Gamma=+1.727e+22 m^-2 s^-1  Q=+1.045e+07 W m^-2  <j.B>=+5.555e+06 A T m^-2
  psi_N=0.36  Gamma=+1.799e+22 m^-2 s^-1  Q=+8.892e+06 W m^-2  <j.B>=+4.697e+06 A T m^-2
  psi_N=0.49  Gamma=+1.692e+22 m^-2 s^-1  Q=+6.539e+06 W m^-2  <j.B>=+3.537e+06 A T m^-2
  geometry sha256: fd2d2691de90ed7d  converged: True  route: block_tridiagonal_truncated
```

```{figure} ../_static/figures/tutorials/vmec.png
:alt: Fluxes and current against normalized toroidal flux for the VMEC case.
:width: 60%

The VMEC case.
```

The fluxes are two orders of magnitude above the tokamak because this
equilibrium has strong helical ripple. The geometry SHA-256 records which
equilibrium was used. For your own `wout`, keep the surfaces inside the plasma:
the axis is singular for radial derivatives, and the last surface has no outer
neighbour. The test equilibria are stored as `*.nc.xz`; the script decompresses
the one it needs (`xz -dk file.nc.xz` for the CLI).

### W7-X, HSX and Boozer files

The same fields serve any stellarator: `format = "vmec"` with a W7-X or HSX
`wout`, or `format = "boozer"` with a `.bc` file (`05_boozer_geometry.toml`),
whose $|B|$ spectrum is read directly. DKX detects the symmetric or asymmetric
column layout itself. `08_ambipolar_er_w7x.toml` is a production-scale W7-X
case to `dkx validate`; it needs the release-hosted equilibrium to run. The
Boozer test field is a tiny fixture and its magnitudes mean nothing, but its
route does:

```text
[dkx.solve] the structured direct solve missed the tolerance (residuals=[0.00128906]); falling back to the recycled Krylov route.
  solver route: gcrot  converged: True
```

The first attempt, the cheap direct elimination, did not meet the tolerance, so
DKX fell back to Krylov and still reports the residual of the original
equation. A fallback line like this is information, not an error (page 4).
A SFINCS deck names its geometry with `geometryScheme`; DKX reads schemes 1-5,
11, 12 and 13.

```{figure} ../_static/figures/docs/bmag_contour_w7x.png
:alt: Contour of the magnetic field strength on a W7-X flux surface in the two angles.
:width: 55%

$|B|$ on a W7-X flux surface. In a stellarator, refine `theta`, `zeta` and
`pitch` together: the toroidal points grow with the field periods and the
harmonic content of $|B|$.
```

## Species and impurities

Each `[[species]]` table is one kinetic species: charge in units of $e$, mass
in amu, density and temperature profiles. Species must be quasineutral on every
surface, $\sum_s Z_s n_s = 0$. `07_species_and_impurity.py` adds 1% fully
ionized carbon to deuterium and electrons, keeping $n_e = n_D + 6 n_C$:

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

In TOML this is three `[[species]]` blocks with the same keys. With Fokker-Planck
collisions on the analytic tokamak (12 s):

```text
  psi_N=0.09  Gamma: deuterium=+1.903e+20  carbon=-1.088e+18  electron=+1.904e+19  |sum_s Z_s Gamma_s| / Gamma_D = 8.66e-01
  psi_N=0.16  Gamma: deuterium=+1.240e+20  carbon=-1.318e+18  electron=+1.194e+19  |sum_s Z_s Gamma_s| / Gamma_D = 8.40e-01
  psi_N=0.25  Gamma: deuterium=+8.476e+19  carbon=-1.175e+18  electron=+8.051e+18  |sum_s Z_s Gamma_s| / Gamma_D = 8.22e-01
  carbon flux is inward (accumulating) on every surface: True
```

```{figure} ../_static/figures/tutorials/species.png
:alt: Particle flux of deuterium, carbon and electrons against normalized toroidal flux.
:width: 55%

Per-species particle flux.
```

The carbon flux is **inward**: neoclassical impurity accumulation, friction
with the main ions pushing a high-$Z$ impurity up the main-ion density
gradient. In an exactly axisymmetric tokamak the fluxes are ambipolar at any
$E_r$; the built-in field's small helical ripple makes
$\sum_s Z_s\Gamma_s$ about 0.8 of $\Gamma_D$, so the next steps find the
$E_r$ that zeroes it (page 6). The classical impurity flux is algebraic and is
in `dkx.impurity`; with more than a trace of impurity, $\Phi_1$ matters
(page 8).

## Next

{doc}`04_collisions_solvers`: the collision operator and the solver route.
