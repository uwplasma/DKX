# 5. Scans and resolution

Two kinds of scan recur: over a physical parameter such as $E_r$, and over the
grid. The second decides whether the first means anything.

## Choosing and checking a grid

A small residual says the discrete system was solved, not that the grid is fine
enough. `dkx converge` measures it:

```console
dkx converge examples/tutorials/13_convergence.toml
python examples/tutorials/13_convergence.py
```

It solves the case at its stated resolution, again with each of `theta`,
`zeta`, `pitch` and `speed` refined by a factor (1.5 here), and once with all
refined together. It exits zero only when the worst relative change of particle
flux, heat flux and current is below the tolerance (default 0.02) **and**
every solve passed its residual check. In Python it is
`dkx.workflows.converge.converge_case(case, axes=..., factor=1.5, tolerance=0.02)`.

The tutorial case is a dense, cold deuterium plasma (`n = 2e20 m^-3`,
`T = 0.2 keV`, plateau/Pfirsch-Schlüter side of the banana regime), where the
distribution is smooth and a modest grid converges. Its stated grid
(`theta=9, pitch=8, speed=4`) is deliberately coarse. The report (about 10 s):

```text
  theta     pitch=8 speed=4 theta=14 zeta=1      worst change    19.9%
  pitch     pitch=12 speed=4 theta=9 zeta=1      worst change     0.0%
  speed     pitch=8 speed=6 theta=9 zeta=1       worst change     4.4%
  all axes  pitch=12 speed=6 theta=14 zeta=1     worst change    18.1%
  converged at 2%: False
```

`pitch` moves nothing here only because `theta` is the limiting axis: the
per-axis rows alone would be a poor guide, which is why the joint row exists.

## A resolution ladder

Refining one axis at a time is misleading because the axes couple. The ladder
refines everything together and records the outputs on the first surface:

```python
LADDER = [(9, 8, 4), (13, 12, 5), (17, 16, 5), (21, 20, 6), (25, 24, 6), (33, 32, 6)]  # theta, pitch, speed
for n_theta, n_pitch, n_speed in LADDER:
    grid = replace(case.resolution, theta=n_theta, pitch=n_pitch, speed=n_speed)
    arrays = dkx.run(replace(case, resolution=grid)).arrays
```

```text
  ladder (theta/pitch/speed): particle flux, heat flux, <j.B> at the first surface
    9/ 8/4  +8.1087e+20  +1.0841e+05  -4.0834e+04
   13/12/5  +8.6168e+20  +1.1216e+05  -4.6574e+04
   17/16/5  +8.6140e+20  +1.1314e+05  -4.7402e+04
   21/20/6  +8.4898e+20  +1.1258e+05  -4.7379e+04
   25/24/6  +8.4754e+20  +1.1246e+05  -4.7453e+04
   33/32/6  +8.4623e+20  +1.1230e+05  -4.7487e+04
```

```{figure} ../_static/figures/tutorials/convergence.png
:alt: Worst relative change under refinement per axis, and flux, heat flux and current along the resolution ladder.
:width: 85%

Left: worst relative change per refinement against the tolerance. Right: each
output along the ladder, relative to the finest rung.
```

What to read: from `13/12/5` on, no rung moves the three outputs by 2% or more,
and from `21/20/6` on none moves them by more than 0.2%. `21/20/6` is the first
rung whose own `dkx converge` passes:

```text
  theta=21 pitch=20 speed=6: worst change 0.8%, converged at 2%: True
```

A number may be quoted only from a rung where the next rung changes it by less
than the precision you need, and where the refinement was joint. At 62 times
lower collisionality (`n = 8e19 m^-3`, `T = 1 keV`) `theta` alone still moves
the particle flux by 17% from 25 to 33 points and by 8% to 41; such cases need
the finer grids of {doc}`../user_guide/convergence`. For a production grid:

- start from the smallest grid that runs, then walk up the ladder until
  `dkx converge` exits zero, or add `--check-grid` to `dkx run` for a one-solve,
  per-radius verdict;
- in a stellarator refine `theta`, `zeta` and `pitch` together;
- the bootstrap current needs more `pitch` than the fluxes, especially at low
  collisionality (the VMEX objective defaults to 48 modes);
- `speed` converges spectrally at high collisionality, but not at low: see the
  table in the guide.

How to choose a production resolution: {doc}`../user_guide/convergence`.

## Scans over $E_r$

`08_ambipolar_er.py` begins with a prescribed-field scan: the same case run at
nine values of $E_r$, recording $J_r = e\sum_s Z_s\Gamma_s$. The loop in the
script replaces `electric_field` and calls `dkx.run`. For many fields on one
surface use the batched path, which builds geometry, grids and collisions once
and evaluates all fields in one compiled call:

```python
problem = dkx.prepare_er_scan(case, surface_index=0)
scan = dkx.batched_er_scan(problem, er_values)
```

For a full scan from a SFINCS deck, `dkx scan-er` reproduces the
`sfincsScan` workflow ({doc}`../user_guide/scans_and_parallelism`). The figure
and numbers of the $J_r(E_r)$ scan are on the next page.

## Next

{doc}`06_ambipolar_er`: turning the scan into a root search.
