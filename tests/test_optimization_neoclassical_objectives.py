from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from sfincs_jax.optimization_objectives import (
    bootstrap_current_objective,
    electron_root_penalty,
    find_ambipolar_roots,
    flux_selectivity_objective,
    kinetic_validation_gate,
    qa_proxy_gradient_gate,
    qa_proxy_neoclassical_objective,
    symmetry_proxy_neoclassical_components,
    symmetry_proxy_neoclassical_objective,
)


_REPO = Path(__file__).resolve().parents[1]


def test_ambipolar_root_summary_identifies_electron_root() -> None:
    summary = find_ambipolar_roots(
        [-2.0, -1.0, 0.5, 2.0],
        [-3.0, -1.0, 0.25, 2.0],
    )

    assert summary.bracketed
    assert summary.has_electron_root
    assert len(summary.roots) == 1
    assert summary.roots[0].root_type == "electron"
    assert electron_root_penalty(summary, min_positive_er=0.05) == 0.0


def test_ambipolar_root_penalty_rejects_unbracketed_scan() -> None:
    summary = find_ambipolar_roots([-2.0, 0.0, 2.0], [1.0, 2.0, 3.0])

    assert not summary.bracketed
    assert not summary.has_electron_root
    assert electron_root_penalty(summary) > 1.0


def test_bootstrap_current_objective_uses_surface_weights() -> None:
    value = bootstrap_current_objective(
        [2.0, -1.0],
        normalizer=2.0,
        surface_weights=[1.0, 4.0],
    )

    assert value == 2.0


def test_flux_selectivity_rewards_outward_impurity_flux() -> None:
    poor = flux_selectivity_objective(
        particle_flux=[[0.1, -0.2, 0.01]],
        heat_flux=[[0.3, 0.4, 0.0]],
        impurity_species_index=2,
        target_impurity_flux=0.05,
    )
    good = flux_selectivity_objective(
        particle_flux=[[0.01, -0.02, 0.08]],
        heat_flux=[[0.03, 0.04, 0.0]],
        impurity_species_index=2,
        target_impurity_flux=0.05,
    )

    assert good["impurity_penalty"] == 0.0
    assert good["total"] < poor["total"]


def test_kinetic_validation_gate_requires_residual_and_cpu_gpu_agreement() -> None:
    assert kinetic_validation_gate(residual_norm=1.0e-10, residual_target=1.0e-8)["status"] == "pass"

    bad = kinetic_validation_gate(
        residual_norm=1.0e-4,
        residual_target=1.0e-8,
        cpu_gpu_relative_difference=1.0e-3,
    )
    assert bad["status"] == "fail"
    assert len(bad["failures"]) == 2


def test_qa_proxy_gradient_gate_passes() -> None:
    gate = qa_proxy_gradient_gate(n_theta=20, n_zeta=16)

    assert gate["status"] == "pass"
    assert gate["gradient_norm"] > 0.0
    assert "not a kinetic SFINCS solve" in gate["claim"]


def test_qa_proxy_objective_decreases_along_negative_gradient() -> None:
    jax.config.update("jax_enable_x64", True)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 20, endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, jnp.pi, 16, endpoint=False, dtype=jnp.float64)
    active = jnp.asarray([0.080, 0.030, -0.024, 0.018, -0.010], dtype=jnp.float64)
    ixm_b = jnp.asarray([0, 1, 1, 2, 2, 3], dtype=jnp.int32)
    ixn_b = jnp.asarray([0, 0, 2, 0, -2, 4], dtype=jnp.int32)

    def objective(x):
        return qa_proxy_neoclassical_objective(x, ixm_b, ixn_b, theta=theta, zeta=zeta)

    value, grad = jax.value_and_grad(objective)(active)
    candidate = active - 0.1 * grad

    assert float(objective(candidate)) < float(value)


def test_qi_proxy_keeps_allowed_nfp2_toroidal_well_unpenalized() -> None:
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 20, endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, jnp.pi, 16, endpoint=False, dtype=jnp.float64)
    active = jnp.asarray([0.06, 0.025], dtype=jnp.float64)
    ixm_b = jnp.asarray([0, 0, 1], dtype=jnp.int32)
    ixn_b = jnp.asarray([0, 2, 2], dtype=jnp.int32)

    components = symmetry_proxy_neoclassical_components(
        active,
        ixm_b,
        ixn_b,
        theta=theta,
        zeta=zeta,
        symmetry="qi",
        nfp=2,
    )

    assert float(components["symmetry_regularization"]) == pytest.approx(float(active[1] ** 2))
    assert float(components["electron_root_drive"]) > 0.0


def test_qi_proxy_objective_decreases_along_negative_gradient() -> None:
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 20, endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, jnp.pi, 16, endpoint=False, dtype=jnp.float64)
    active = jnp.asarray([0.085, -0.030, 0.014, 0.020, -0.012], dtype=jnp.float64)
    ixm_b = jnp.asarray([0, 0, 0, 1, 1, 2], dtype=jnp.int32)
    ixn_b = jnp.asarray([0, 2, 4, 2, -2, 4], dtype=jnp.int32)

    def objective(x):
        return symmetry_proxy_neoclassical_objective(
            x,
            ixm_b,
            ixn_b,
            theta=theta,
            zeta=zeta,
            symmetry="qi",
            nfp=2,
        )

    value, grad = jax.value_and_grad(objective)(active)
    candidate = active - 0.05 * grad

    assert float(objective(candidate)) < float(value)


def test_public_qa_nfp2_optimization_example_runs(tmp_path: Path) -> None:
    script = _REPO / "examples" / "optimization" / "qa_nfp2_sfincs_jax_objectives.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--steps",
            "4",
            "--out-dir",
            str(tmp_path),
            "--stem",
            "qa_proxy_test",
        ],
        cwd=_REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads((tmp_path / "qa_proxy_test.json").read_text(encoding="utf-8"))
    assert summary["workflow"] == "qa_nfp2_sfincs_jax_neoclassical_optimization_proxy"
    assert summary["autodiff_gradient_gate"]["status"] == "pass"
    assert summary["history"][-1]["objective"] <= summary["history"][0]["objective"]
    assert (tmp_path / "qa_proxy_test.png").exists()
    assert (tmp_path / "qa_proxy_test.pdf").exists()
