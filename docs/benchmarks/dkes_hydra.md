# DKX against DKES on the HELIAS reactor hydra_Np04

The Institute for Plasma Physics (IPP) DKES coefficients of the 4-period HELIAS
reactor `hydra_Np04_20190108` (R = 23.6 m, a = 2.0 m, <beta> = 5.4%) at seven
radii, rho = 0.12247 to 0.875, against DKX's monoenergetic route
(`geometryScheme = 11`, spectral angles, `Nx = 1`). Sealed record:
`validation/dkes_hydra_np04_v1.json` (registry `dkes_hydra_np04`, audit
`tools/paper_benchmarks/audit_dkes_hydra_np04.py`).

The DKES results (`DKES_rho_<rho>.h5`, the `.dk` output, IPP) and the Boozer file
(J. Geiger, IPP) cannot be redistributed. The repository holds DKX values and the agreement
statistics below, never a DKES value; the tests and the audit read the private data from
`DKX_DKES_HYDRA_DIR` and `DKX_HYDRA_BOOZER` and skip when they are unset.

## Normalization of the DKES files

| DKES / file quantity | Meaning | DKX |
| --- | --- | --- |
| `cmul` (the `.dk` output column) | $\nu_D/v$ in 1/m | `nuPrime = cmul |G + iota I| / (B0 nuD(1))`, `nuD(1) = 0.836028`; `nuprime / nuprimefac = cmul` |
| `efield` (the `.dk` column) | $E_r/(v B_{00})$ | `v_E = efield`; `EStar = efield |G| / (B0 |iota| r_hat)` |
| `nuprimefac` | $|G|/(B0\,\nu_D(1))$ in the `B_00 = 1` units of the `.dk` | 29.38 at rho = 0.5 |
| `Estarfac` | $G/(r\,\iota)$ with $r = \rho a$, $a = 2.01879$ m | `Erstar = Estarfac * efield` |
| `D11_S, D13_S, D31_S, D33_S` | SFINCS `transportMatrix` entries (hat units, psi-based, $x_0 = 1$) | `dkx.monoenergetic.monoenergetic_dstar_from_transport_matrix` gives Beidler $D^*$ directly |
| `Gamma11/31/33`, `xnu`, `xEr` | the same coefficients on a 15 x 2 (nu, E) grid at rho = 0.5 | not used |

The raw DKES `D33` is $-\tfrac23 \langle B^2\rangle/\mathrm{cmul}$ in the collisional limit, and
`D33_S = -D33_dkes 2/(sqrt(pi) R0)` with $R_0 = |G|/B_0 = 24.56$ m. The check that
fixes the whole chain is the collisional limit: at `cmul = 10` every radius and every
$E$ gives $D_{33}^* = 1$ to 1e-4 (`tests/test_dkes_hydra_np04.py`). The raw `D11` of the
`.dk` file differs from `D11_S` by the DKES radial convention (a constant factor 0.875
against DKX at every radius), which is why the comparison uses the psi-based `_S` values.
The Boozer file has to be read with `VMECRadialOption = 0` (interpolation); the default
snaps to the nearest surface, 1% away in rho.

## Result

DKX 17 x 41 x 96 ($N_\theta \times N_\zeta \times N_\xi$, per period), tolerance 1e-8,
office workstation. Relative difference of $D^*$ against the DKES value (median and maximum over the points of the regime, rounded to 0.1%), per radius and regime;
`E=0` is `efield = 0`, `E>0` pools `efield = 0.01` and `0.03`. The file carries no
variational bounds, so the single stored value is the reference.

| rho | regime | D11* E=0: median / max | D11* E>0: median / max | D33* max | D31* max abs (E>0) |
|---|---|---|---|---|---|
| 0.12247 | collisional, cmul >= 0.1 | 3.9 / 7.3% | 5.2 / 6.5% | 0.1% | 0.000 |
| 0.12247 | plateau, 1e-3 to 3e-2 | 3.0 / 10.3% | 6.7 / 19.2% | 0.1% | 0.000 |
| 0.12247 | low nu, cmul <= 3e-4 | 2.9 / 6.7% | 21.7 / 45.3% | 21.8% | 0.003 |
| 0.25 | collisional, cmul >= 0.1 | 0.2 / 1.7% | 2.4 / 4.1% | 0.1% | 0.000 |
| 0.25 | plateau, 1e-3 to 3e-2 | 1.4 / 7.8% | 0.8 / 7.1% | 0.4% | 0.000 |
| 0.25 | low nu, cmul <= 3e-4 | 2.1 / 6.0% | 5.9 / 9.5% | 8.8% | 0.015 |
| 0.375 | collisional, cmul >= 0.1 | 0.1 / 7.9% | 2.4 / 4.1% | 0.1% | 0.000 |
| 0.375 | plateau, 1e-3 to 3e-2 | 0.5 / 8.7% | 1.8 / 6.4% | 0.1% | 0.000 |
| 0.375 | low nu, cmul <= 3e-4 | 4.3 / 6.3% | 7.4 / 12.1% | 9.4% | 0.038 |
| 0.5 | collisional, cmul >= 0.1 | n/a | 2.6 / 4.4% | 0.1% | 0.000 |
| 0.5 | plateau, 1e-3 to 3e-2 | n/a | 1.3 / 2.2% | 0.1% | 0.000 |
| 0.5 | low nu, cmul <= 3e-4 | n/a | 4.6 / 16.3% | 1.7% | 0.016 |
| 0.625 | collisional, cmul >= 0.1 | 0.5 / 4.4% | 2.8 / 5.0% | 0.1% | 0.000 |
| 0.625 | plateau, 1e-3 to 3e-2 | 3.6 / 7.2% | 1.2 / 7.4% | 0.1% | 0.001 |
| 0.625 | low nu, cmul <= 3e-4 | 2.0 / 6.9% | 5.7 / 42.4% | 37.4% | 0.155 |
| 0.75 | collisional, cmul >= 0.1 | 2.0 / 2.6% | 3.3 / 6.1% | 0.1% | 0.000 |
| 0.75 | plateau, 1e-3 to 3e-2 | 2.4 / 19.3% | 1.4 / 9.0% | 0.1% | 0.001 |
| 0.75 | low nu, cmul <= 3e-4 | 10.9 / 13.7% | 5.3 / 61.7% | 9.4% | 0.236 |
| 0.875 | collisional, cmul >= 0.1 | 27.0 / 27.2% | 4.0 / 7.6% | 0.1% | 0.000 |
| 0.875 | plateau, 1e-3 to 3e-2 | 19.9 / 24.3% | 3.9 / 8.1% | 0.1% | 0.001 |
| 0.875 | low nu, cmul <= 3e-4 | 20.1 / 24.3% | 13.6 / 50.6% | 11.2% | 0.680 |

Gates (cmul >= 1e-3, all three `efield`): $D_{33}^*$ 1%, $D_{11}^*$ 25%, $D_{31}^*$ 0.003 absolute
at `efield > 0`; measured 0.37%, 19.3% and 0.0013. Not gated and listed above:

- `cmul <= 3e-4`. Raising the grid to 25 x 61 x 160 moves $D_{11}^*$ by at most 6% (cmul 1e-6,
  rho 0.12247) and by 2% or less elsewhere, so the 45% deficit at rho = 0.12247, `efield = 0.03`
  is not DKX resolution. At rho = 0.5 DKX is 3% to 6% above DKES. The largest differences at
  rho = 0.625 come from single DKES points between consistent neighbours.
- rho = 0.875, `efield = 0`: the stored `D11_S` looks inconsistent with the `.dk` output of the same
  run: the `.dk` output agrees with DKX at the common factor of every other radius and the stored
  `_S` value does not, while the same radius agrees to a few percent at `efield > 0`. Excluded from
  the gate and reported to the file's owner.
- $D_{31}^*$ at `efield = 0` is DKES roundoff noise of either sign below 1e-5.
- rho = 0.5, `efield = 0` has no DKES reference.
- NTX (uwplasma/NTX#40) reads this Boozer format since that pull request and reproduces the same
  agreement statistics (`D33*` within 0.3%, `D11*` within 19.3%, `D31*` within 1e-3 for
  `cmul >= 1e-3`) with DKX's normalization; it also found that harmonics above the grid Nyquist
  must be dropped, which DKX already does.

## Running it

Point `DKX_HYDRA_BOOZER` at `boozer.hydra_Np04_20190108.fix001.03.890_12x12.data` and
`DKX_DKES_HYDRA_DIR` at the directory with `DKES_rho_<rho>.h5`; the tests that read them skip without.

```bash
export DKX_HYDRA_BOOZER=/path/to/boozer.hydra_Np04_20190108.fix001.03.890_12x12.data
export DKX_DKES_HYDRA_DIR=/path/to/DKES_data
python tools/paper_benchmarks/run_dkes_hydra_np04.py run raw.json --grid 17 41 96 --rho 0.25
python tools/paper_benchmarks/audit_dkes_hydra_np04.py
```
