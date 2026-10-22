# 4. Collisions and solvers

The collision term $C_{sb}$ sets the transport regime, and the operator's
structure decides which linear solver is used. Both matter for whether a run
can be trusted.

## Three collision operators

| Operator | Case field | SFINCS `collisionOperator` | Conserves |
| --- | --- | --- | --- |
| Pitch-angle scattering (PAS) | `collisions = "pitch_angle_scattering"` | 1 | particles |
| Linearized Fokker-Planck (FP) | `collisions = "linearized_fokker_planck"` | 0 | particles, momentum, energy |
| Improved Sugama model | namelist only | 3 | particles, momentum, energy |

The Sugama model's l = 1 friction is corrected toward the Landau operator on the first five Laguerre flow moments (Sugama et al. 2019, Eqs. 40-43). Its Spitzer-Harm conductivity is within 1% of the Fokker-Planck value for all Z, and it is valid for unequal species temperatures. With two moments (`flow_order = 2`) the conductivity is within 2.1%.

PAS is the Lorentz operator,

$$
C^{\mathrm{PAS}}_s[P_L] = -\frac{L(L+1)}{2}\,\nu_D^s(v)\,P_L ,
$$

diagonal in Legendre modes $P_L(\xi)$ and in speed. FP adds energy scattering
and the field-particle terms built from the Rosenbluth potentials of $f_{b1}$:
it conserves momentum and couples all speeds and species. It is one field in
the case, `physics.collisions`.

## PAS against FP

`06_fokker_planck_vs_pas.py` solves a deuterium-electron tokamak with each
operator and nothing else changed:

```python
results = {}
for operator in ("pitch_angle_scattering", "linearized_fokker_planck"):
    results[operator] = dkx.run(dkx.Case.from_mapping({**CASE, "physics": {"collisions": operator}}))
```

FP/PAS ratio on the three surfaces (17 s):

```text
  particle_flux_m2_s         [0.566 0.551 0.541]
  heat_flux_W_m2             [1.39  1.339 1.301]
  parallel_current_A_T_m2    [0.95  0.947 0.937]
```

```{figure} ../_static/figures/tutorials/operators.png
:alt: Ion particle flux, heat flux and parallel current for PAS and Fokker-Planck.
:width: 85%

Ion particle flux, heat flux and $\langle j\cdot B\rangle$ for both operators.
```

PAS puts the ion particle flux almost a factor of two too high: like-particle
collisions that conserve momentum drive no net flux in a tokamak, and PAS does
not conserve momentum. It overestimates the bootstrap current by 5-7% for the
same reason. Quote currents from FP. When FP is too expensive,
`dkx.momentum_correction` restores momentum to a PAS solve
(`collision_operator="pas+momentum_correction"` in `run_representative`;
{doc}`../physics/reduced_models`).

## The solver route

`solver.method = "auto"` chooses a route from the operator's structure and the
memory estimate ({doc}`../numerics/solver_routes`). The route is printed by the
run and stored in the certificate (`result.metadata["solver_route"]`):

| Printed route | When `auto` chooses it |
| --- | --- |
| `block_tridiagonal_truncated` | PAS with DKES trajectories: each Legendre mode couples only to $L\pm1$ and every (species, speed) pair is independent, so exact block elimination keeps only the lowest blocks the moments need. Pages 1, 3 |
| speed-coupled structured direct | FP with DKES trajectories, no $\Phi_1$: still block tridiagonal in $L$; eliminated from the highest mode down when memory fits and the cost is under 150 GFlop per right-hand side |
| `gcrot` (recycled Krylov) | operators that are not block tridiagonal (page 3's Boozer fixture), or large FP and Sugama decks. Preconditioned by the speed-coupled elimination when it fits half the memory, a cheaper `coarse` operator otherwise |
| `phi1_newton_krylov` | $\Phi_1$ on (page 8) |

Force a route only to test one: `[solver] method = "gcrot"` (see
{doc}`../numerics/krylov_and_preconditioners`).

## How to tell whether a run is trustworthy

1. **The route line.** A `falling back to the recycled Krylov route` message
   means the first route missed the tolerance; the answer is still checked.
2. **`converged` and the residual.** `converged: True` with
   `primal_residual` near the tolerance relative to `primal_rhs_norm`. For
   a direct route it is round-off ($10^{-15}$); for Krylov it is bounded by the
   requested tolerance (default $10^{-8}$). A `warnings` list in the
   certificate must be empty.
3. **Resolution.** The residual cannot tell you this; page 5 does.
4. **Operator.** FP for currents; PAS only for a cheap look at trends.

## Next

{doc}`05_scans_resolution`: scans over $E_r$ and resolution ladders.
