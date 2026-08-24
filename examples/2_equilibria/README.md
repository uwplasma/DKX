# 2. Equilibria — DKX on real geometry

Same solver, real magnetic fields. What separates this from
[`../1_basics`](../1_basics) is that the field is no longer a two-parameter
analytic model.

| script | what it shows |
|---|---|
| `run_stellarator.py` | add helical ripple to the tokamak and the 1/ν regime appears. **Start here.** |
| `run_from_vmec.py` | VMEX solves an equilibrium, writes a `wout`, `geometryScheme=5` reads it — one script |

`run_from_vmec.py` needs the optional companion `vmex` (`pip install vmex`) and
exits with a clear message when it is absent. `run_stellarator.py` needs
nothing beyond `dkx`.

## No equilibrium files ship with DKX

The repository is slimmed to ~22 MB, which leaves no room for a `wout` or `.bc`
large enough to be interesting. So the folder earns its geometry two ways:
analytically, by turning on `epsilon_h`, and by generating a real equilibrium
in-process with VMEX. Both are self-contained.

To use your own equilibrium instead, point `equilibriumFile` at it:

```python
run = dkx.run(geometryScheme=5, equilibriumFile="wout_mydevice.nc", rN_wish=0.5, ...)
```

`geometryScheme=11`/`12` read a Boozer `.bc` file the same way.

Next: [`../3_gradients`](../3_gradients) differentiates through the solve.
