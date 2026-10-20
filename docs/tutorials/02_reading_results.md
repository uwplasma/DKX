# 2. Reading results

A result holds arrays, a metadata record and a certificate. This page reads
each one for the case of page 1 and for a SFINCS namelist, the other input
format.

## Outputs and their units

A native case is written in SI units (m$^{-3}$, keV, amu, kV/m) and results
come back in SI. Arrays indexed `[surface, species]` are per species; the
current is summed over species.

| Array | Shape | Unit | Definition |
| --- | --- | --- | --- |
| `particle_flux_m2_s` | (surface, species) | m$^{-2}$ s$^{-1}$ | $\Gamma_s = \langle\int d^3v\, f_{s1}\,\mathbf{v}_{ms}\cdot\nabla\psi\rangle$ in the $|\nabla r|$ normalization |
| `heat_flux_W_m2` | (surface, species) | W m$^{-2}$ | same with weight $\tfrac12 m_s v^2$ |
| `parallel_current_A_T_m2` | (surface) | A T m$^{-2}$ | $\langle j_\parallel B\rangle = \sum_s Z_s e\langle B\int d^3v\, v_\parallel f_{s1}\rangle$ |
| `electric_field_kV_m` | (surface) | kV/m | $E_r = -d\Phi_0/dr$ |
| `primal_residual` | (surface) | | residual of the original linear system |

$\langle\cdot\rangle$ is the flux-surface average. With no inductive field the
current is the bootstrap current. Inside the solver everything is a ratio to
$\bar n = 10^{20}\,$m$^{-3}$, $\bar T = 1\,$keV, the proton mass, $\bar B = 1\,$T
and $\bar R = 1\,$m; the conversion lives in one place, `dkx.units`
({doc}`../physics/normalizations`).

## The certificate

```python
cert = result.certificate()
```

For `02_cli_case.toml` the relevant entries are:

```text
converged: True
solver_route: block_tridiagonal_truncated
residual_norm: 4.274e-15
primal_residual: [1.85e-15, 3.22e-15, 4.27e-15]    # one per surface
primal_rhs_norm: [2.44e-03, 2.30e-03, 2.40e-03]
precision: float64    device: cpu:cpu
```

It also records the DKX, JAX and Python versions, the geometry SHA-256 and the
case ID, so a stored result says what produced it.

## What "converged" means

`converged: True` says one thing: the discretized linear system was solved to
the requested tolerance, and the residual was measured on the **original**
equation, not on a preconditioned or truncated one. It does not say the grid is
fine enough. Two separate questions:

| Question | Answered by |
| --- | --- |
| Was the linear system solved? | `converged`, `primal_residual` against `primal_rhs_norm` |
| Is the answer resolved in $\theta$, $\zeta$, pitch, speed? | `dkx converge`, a resolution ladder (page 5) |

A direct solve gives a residual near round-off ($10^{-15}$ here) on any grid,
including a useless one.

## A SFINCS namelist, the other input format

DKX reads SFINCS v3 `input.namelist` decks unchanged and writes
`sfincsOutput.h5` with SFINCS's names, so existing decks and post-processing
keep working. Decks are written in the hatted variables (`nHats`, `THats`,
`dNHatdrHats`, `Delta`, `nu_n`, `Er`, `Ntheta`, `Nzeta`, `Nxi`, `Nx`).

```console
dkx examples/tutorials/03_sfincs_namelist.namelist
```

The bundled deck is the `geometryScheme = 4` (W7-X-like) field at $r/a = 0.5$,
protons and electrons, pitch-angle scattering, `Er = 0`. The script
`03_sfincs_namelist.py` runs it and summarizes (5 s):

```text
  protons    particleFlux_vm_psiHat=-4.7081e-08  heatFlux_vm_psiHat=-1.0635e-07  FSABFlow=+1.2531e-04
  electrons  particleFlux_vm_psiHat=-1.2250e-09  heatFlux_vm_psiHat=-2.8062e-09  FSABFlow=+1.3038e-04
  radial current sum_s Z_s Gamma_s = -4.5856e-08 (non-zero: Er=0 is not ambipolar)
  FSABjHat (bootstrap current) = -5.0729e-06
```

`FSABjHat` is $\langle j_\parallel B\rangle$ in the hatted units; the fluxes
carry the `_vm_psiHat` suffix (magnetic drift, $\hat\psi$ derivative).

```{figure} ../_static/figures/tutorials/sfincs_namelist.png
:alt: Particle and heat flux of protons and electrons from the SFINCS deck.
:width: 55%

Fluxes of the namelist case.
```

The ion flux is 40 times the electron flux, and the radial current
$\sum_s Z_s\Gamma_s$ is not zero: that is the cue to find the ambipolar $E_r$
(page 6). To leave the namelist world, `dkx convert input.namelist case.toml`
translates the units and fails, naming the key, on anything a native case
cannot carry. `dkx compare a.h5 b.h5 --rtol 1e-9` compares DKX NetCDF and
SFINCS HDF5 outputs ({doc}`../user_guide/sfincs_namelist`).

Upstream SFINCS plotting scripts read DKX output too
(`dkx sfincs postprocess-upstream --case-dir DIR --util sfincsScanPlot_1 -- pdf`):

```{figure} ../_static/figures/utils/sfincsScanPlot_1.png
:alt: Upstream sfincsScanPlot_1 output generated from DKX sfincsOutput.h5 files.
:width: 60%

An upstream `sfincsScanPlot_1` figure drawn from DKX output.
```

## Next

{doc}`03_geometry_species`: real magnetic geometries and more species.
