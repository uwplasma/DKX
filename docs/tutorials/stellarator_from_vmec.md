# A stellarator from a VMEC or Boozer file

The drift-kinetic solve needs the magnetic field on each flux surface: $B$ and
its variation in two angles, the Jacobian, the rotational transform and the
covariant field components. DKX reads these from three kinds of source, and the
choice is one field of the case. This tutorial moves rung 01 onto a VMEC
equilibrium (`examples/02_vmec_stellarator`) and a Boozer file
(`examples/03_boozer_stellarator`). The geometry model itself is described in
{doc}`../physics/geometry`.

| `geometry.format` | `geometry.file` | Typical source |
| --- | --- | --- |
| `"analytic"` | a configuration name: `tokamak`, `lhd_standard`, `lhd_inward`, `w7x_standard` | tests, tutorials, reference cases |
| `"vmec"` | a VMEC `wout_*.nc` file | a fixed-boundary equilibrium, including non-stellarator-symmetric (`lasym`) ones |
| `"boozer"` | a Boozer `.bc` file | the format most stellarator neoclassical work circulates in |

Relative paths resolve beside the case file, not from the shell's working
directory, so a case directory is portable.

## VMEC

The case differs from rung 01 in the geometry table and in `zeta`:

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

Species, physics and solver tables are unchanged. The surfaces are normalized
toroidal flux, as for every format; DKX interpolates the VMEC Fourier
coefficients to them and transforms to the coordinates the solver uses.
`zeta` is 9 because a VMEC field is resolved on a toroidal grid even when the
configuration is nearly axisymmetric.

```console
python examples/02_vmec_stellarator/run.py
dkx run examples/02_vmec_stellarator/case.toml --out examples/output/02_vmec_stellarator/result.nc
dkx inspect examples/output/02_vmec_stellarator/result.nc
```

```{note}
The test equilibria under `tests/ref/` are stored compressed as `*.nc.xz`. If
`wout_up_down_asymmetric_tokamak.nc` is missing in a fresh checkout,
`examples/02_vmec_stellarator/run.py` decompresses it for you on first run;
for the CLI route decompress it once with `xz -dk tests/ref/wout_up_down_asymmetric_tokamak.nc.xz`.
```

To use your own equilibrium, point `file` at its `wout` and choose surfaces
inside the plasma: the magnetic axis is singular for the radial derivatives,
and the last surface has no outer neighbour.

## Boozer

```toml
[geometry]
format = "boozer"
file = "../../tests/ref/nonStelSym_tiny_geometryScheme12.bc"
surfaces = [0.20, 0.30]

[resolution]
theta = 5
zeta = 5
```

A `.bc` file already carries $\lvert B\rvert$ as a Boozer spectrum, so nothing
is transformed on the way in. DKX detects whether the file uses the
stellarator-symmetric (cosine-only) or the asymmetric column layout, so the case
never names a SFINCS geometry-scheme number. The bundled fixture is
non-stellarator-symmetric, which exercises the harder of the two.

```console
python examples/03_boozer_stellarator/run.py
```

## Resolution in a stellarator

Both angles matter. A stellarator field varies along the field line as well
as across it, and the number of toroidal points needed grows with the number of
field periods and the harmonic content of $\lvert B\rvert$. The example grids
are chosen to run in seconds. Before trusting any number:

```console
dkx converge examples/02_vmec_stellarator/case.toml
```

and raise `theta`, `zeta` and `pitch` together until the observables stop
moving. At low collisionality `pitch` dominates the cost
({doc}`../user_guide/convergence`).

```{figure} ../_static/figures/docs/bmag_contour_w7x.png
:alt: Contour of the magnetic field strength on a W7-X flux surface in the two angles.
:width: 70%

$\lvert B\rvert$ on a W7-X flux surface. The solver's angular grid must resolve
this structure.
```

## SFINCS geometry schemes

A SFINCS deck names its geometry with `geometryScheme`. DKX reads schemes 1–5,
11, 12 and 13 from a namelist; `dkx convert` maps a deck onto the native
`format`/`file` fields ({doc}`sfincs_migration`,
{doc}`../user_guide/sfincs_namelist`). The differentiable path builds the
geometry from Fourier amplitudes with `FluxSurfaceGeometry.from_fourier`
({doc}`gradients`, {doc}`vmex_optimization`).
