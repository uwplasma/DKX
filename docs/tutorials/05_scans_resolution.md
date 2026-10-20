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

The tutorial case is deliberately under-resolved (`theta=9, pitch=8, speed=4`).
The report (18 s):

```text
  theta     pitch=8 speed=4 theta=14 zeta=1      worst change   1144.5%
  pitch     pitch=12 speed=4 theta=9 zeta=1      worst change     26.3%
  speed     pitch=8 speed=6 theta=9 zeta=1       worst change     27.4%
  all axes  pitch=12 speed=6 theta=14 zeta=1     worst change    670.1%
  converged at 2%: False
```

## A resolution ladder

Refining one axis at a time is misleading because the axes couple. The ladder
refines everything together and records the outputs on the first surface:

```python
LADDER = [(9, 8, 4), (17, 16, 5), (25, 24, 6), (33, 32, 6), (41, 40, 7)]  # theta, pitch, speed
for n_theta, n_pitch, n_speed in LADDER:
    grid = replace(case.resolution, theta=n_theta, pitch=n_pitch, speed=n_speed)
    arrays = dkx.run(replace(case, resolution=grid)).arrays
```

```text
  ladder (theta/pitch/speed): particle flux, heat flux, <j.B> at the first surface
    9/ 8/4  +1.5606e+20  +7.6246e+04  -2.1238e+05
   17/16/5  +1.7111e+20  +9.3474e+04  -2.0991e+05
   25/24/6  +2.3093e+20  +1.2670e+05  -2.1712e+05
   33/32/6  +2.7462e+20  +1.5271e+05  -2.2754e+05
   41/40/7  +3.0332e+20  +1.7257e+05  -2.3336e+05
```

```{figure} ../_static/figures/tutorials/convergence.png
:alt: Worst relative change under refinement per axis, and flux, heat flux and current along the resolution ladder.
:width: 85%

Left: worst relative change per refinement against the tolerance. Right: each
output along the ladder, relative to the finest rung.
```

What to read: the current moves by about 10% along the ladder, but the fluxes
keep rising by 10-35% per rung and have not flattened at the finest rung. This
case has a helical $n=10$ component of $|B|$ on a grid with `zeta = 1`, which
the angular grid cannot represent, so it is a poor choice for a convergence
claim. That is the point: **a number may be quoted only from a rung where the
next rung changes it by less than the precision you need**, and where the
refinement was joint. For a production grid:

- start from the smallest grid that runs, then walk up the ladder until
  `dkx converge` exits zero;
- in a stellarator refine `theta`, `zeta` and `pitch` together;
- the bootstrap current needs more `pitch` than the fluxes, especially at low
  collisionality (the VMEX objective defaults to 48 modes);
- `speed` converges spectrally: a few nodes are usually enough.

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
