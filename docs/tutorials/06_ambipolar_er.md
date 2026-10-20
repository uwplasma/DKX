# 6. Ambipolar $E_r$ and radial profiles

In a stellarator the electron and ion particle fluxes are not automatically
equal. Each depends on $E_r$, and $E_r$ adjusts until no net charge leaves the
surface:

$$
J_r(E_r) = \sum_s Z_s e\,\Gamma_s(E_r) = 0 .
$$

In a tokamak the fluxes are intrinsically ambipolar at any $E_r$, so this
workflow is for stellarators ({doc}`../physics/electric_field`).

## Roots

$J_r(E_r)$ can have one root or three: an **ion root** at small or negative
$E_r$, an **electron root** at large positive $E_r$, and an unstable root
between them. A root is stable when a small increase of $E_r$ drives an outward
current that pushes it back, i.e. $dJ_r/dE_r > 0$; DKX reads this from the sign
of the slope and labels each root `ion`, `electron` or `unstable`.

## The case

Two fields turn a profile case into an ambipolar one
(`examples/tutorials/08_ambipolar_er.toml`, deuterium and electrons, full
Fokker-Planck, analytic W7-X field):

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

DKX samples $J_r$ at `search_points` fields across `search_kV_m`, brackets every
sign change, refines each bracket and classifies the root. In Python it is the
same case through `dkx.run`; the script first scans nine prescribed fields to
draw $J_r$, then runs the root search (34 s):

```console
python examples/tutorials/08_ambipolar_er.py
dkx run examples/tutorials/08_ambipolar_er.toml --out er.nc
dkx roots er.nc
```

```text
  psi_N=0.09: 1 root(s): -3.672 kV/m [ion]; selected Er = -3.672 kV/m
  psi_N=0.16: 1 root(s): -3.516 kV/m [ion]; selected Er = -3.516 kV/m
  selection rule: continue the selected branch identity; nearest zero on the first surface and nearest prior field after branch loss
  all surfaces bracketed: True
```

`dkx roots` prints the table from the stored result:

```text
┃ surface ┃ root ┃      E_r ┃ J_r [A/m^2] ┃ type ┃  bracket ┃ selected ┃
│       0 │    0 │ -3.67188 │   2.716e-03 │ ion  │ 3.91e-02 │ yes      │
│       1 │    0 │ -3.51562 │   4.691e-03 │ ion  │ 3.91e-02 │ yes      │
```

```{figure} ../_static/figures/tutorials/ambipolar.png
:alt: Radial current against radial electric field on two surfaces, with the ambipolar roots marked.
:width: 60%

$J_r(E_r)$ on two surfaces; the crosses are the roots.
```

## Reading the roots

- With equal ion and electron temperatures at these collisionalities each
  surface has a single ion root, a few kV/m negative.
- `J_r` at the root is not exactly zero: it is within
  `root_tolerance_kV_m` times the slope. Tighten the tolerance to shrink it.
- The fluxes reported per surface are those at the selected root; `E_r` per
  surface is in `electric_field_kV_m` and the full list in
  `ambipolar_root_kV_m`, `ambipolar_root_type`, `selected_ambipolar_root`.
- With several roots, `continue_branches` follows one branch across surfaces so
  the profile does not jump between ion and electron roots.

For an electron root to appear the hot-electron, low-density conditions of
W7-X are needed; the benchmark-scale figure is
{doc}`../benchmarks/validation_matrix` and
`tools/paper_benchmarks/w7x_ambipolar_er.py`.

```{figure} ../_static/figures/docs/ambipolar_er_roots.png
:alt: Radial current against radial electric field with ion, unstable and electron roots.
:width: 55%

A three-root $J_r(E_r)$: ion, unstable and electron roots.
```

## Failures to expect

| Symptom | Cause | Remedy |
| --- | --- | --- |
| `all surfaces bracketed: False` | no sign change inside `search_kV_m` | widen the window |
| a root is missing | two roots fell between consecutive samples | raise `search_points` |
| a solve inside the search did not pass its residual check | that field is excluded from bracketing and reported, not silently used | tighten the solver or refine |
| $E_r$ moves on refinement | $E_r$ is resolution-sensitive | `dkx converge` the ambipolar case |
| root jumps between surfaces | branch loss | read `selection rule` and the root list |

Single-surface alternatives: `dkx.find_ambipolar_er` (Brent search, reports
failures instead of raising, `all_roots=True` for every root) and
`dkx ambipolar-solve` for SFINCS decks.

## Next

{doc}`07_transport_matrices`: transport coefficients that do not depend on
the profiles.
