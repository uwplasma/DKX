# Optimization (Optax / JAX-native)

Optimization examples that leverage differentiability:
- fitting geometry harmonics
- calibrating parameters against frozen Fortran fixtures
- end-to-end objective optimization (with publication-style plots in some scripts)
- bounded optional ecosystem gates for differentiable objective wrappers

Examples:
- `fit_geometry_harmonics_with_optax.py`
- `calibrate_nu_n_to_fortran_residual_fixture.py`
- `benchmark_optional_eqx_jaxopt_scheme4_gate.py` — optional Equinox/JAXopt gate on a real `geometryScheme=4` harmonic-fit objective; it verifies gradient agreement for an `equinox.Module` wrapper and bounded loss reduction for `jaxopt.GradientDescent`, and it skips cleanly when those packages are not installed.
