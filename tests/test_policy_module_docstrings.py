from __future__ import annotations

import importlib


POLICY_MODULES = (
    "sfincs_jax.rhs1_handoff",
    "sfincs_jax.rhs1_pas_policy",
    "sfincs_jax.rhs1_preconditioner_dispatch",
    "sfincs_jax.rhs1_sparse_polish_policy",
    "sfincs_jax.rhs1_sparse_rescue_policy",
    "sfincs_jax.rhs1_stage2_policy",
    "sfincs_jax.rhs1_strong_auto_kind",
    "sfincs_jax.rhs1_strong_control",
    "sfincs_jax.rhs1_strong_fallback",
    "sfincs_jax.rhs1_strong_policy",
    "sfincs_jax.transport_dense_lu",
    "sfincs_jax.transport_handoff_policy",
    "sfincs_jax.transport_host_gmres",
    "sfincs_jax.transport_preconditioner_dispatch",
    "sfincs_jax.transport_solve_policy",
)


def test_refactored_policy_modules_expose_real_module_docstrings() -> None:
    """Keep split policy modules discoverable in API docs and introspection."""

    for module_name in POLICY_MODULES:
        module = importlib.import_module(module_name)
        assert module.__doc__ is not None, module_name
        assert module.__doc__.strip(), module_name
