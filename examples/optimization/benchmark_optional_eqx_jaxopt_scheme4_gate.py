"""Optional Equinox/JAXopt objective-wrapper gate on a differentiable geometry task.

This script intentionally stays outside the production solver path. It checks that
the broader JAX ecosystem can wrap a real ``sfincs_jax`` objective cleanly before
any dependency is considered for a supported workflow.

The bounded gate uses the differentiable ``geometryScheme=4`` harmonic-fit task:

- ``equinox`` provides a small callable objective wrapper,
- ``jaxopt`` runs a short gradient-descent solve on that wrapped objective.

This is a fast, meaningful gate because it exercises a real repository objective
with nontrivial gradients while remaining cheap enough for focused tests.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.geometry import boozer_geometry_scheme4  # noqa: E402


_AUTO_IMPORT = object()


@dataclass(frozen=True)
class ObjectiveGateResult:
    case: str
    backend: str
    status: str
    initial_loss: float | None
    final_loss: float | None
    loss_ratio: float | None
    directional_grad: float | None
    finite_difference_grad: float | None
    directional_grad_abs_error: float | None
    final_param_error: float | None
    elapsed_s: float | None
    error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_gate_results(results: list[ObjectiveGateResult]) -> dict[str, Any]:
    """Summarize optional objective-wrapper evidence without promoting deps."""
    rows = [result.to_json_dict() for result in results]
    counts = Counter(str(row["status"]) for row in rows)
    measured_rows = [
        row for row in rows if row.get("status") == "ok" and row.get("elapsed_s") is not None
    ]
    by_backend = {str(row["backend"]): row for row in rows}
    eqx = by_backend.get("equinox_wrapper")
    jaxopt = by_backend.get("jaxopt_gradient_descent")

    if eqx is None:
        eqx_decision = "not_evaluated"
        eqx_reason = "No Equinox row was requested."
    elif eqx["status"] == "skipped":
        eqx_decision = "not_evaluated_missing_optional_dependency"
        eqx_reason = str(eqx.get("error"))
    elif eqx["status"] == "ok" and float(eqx.get("directional_grad_abs_error") or 1.0) < 1.0e-6:
        eqx_decision = "candidate_objective_wrapper_only"
        eqx_reason = "Equinox wrapped the real scheme-4 objective and matched finite difference."
    else:
        eqx_decision = "defer_gradient_gate_not_clean"
        eqx_reason = "Equinox did not satisfy the directional-gradient gate."

    if jaxopt is None:
        jaxopt_decision = "not_evaluated"
        jaxopt_reason = "No JAXopt row was requested."
    elif jaxopt["status"] == "skipped":
        jaxopt_decision = "not_evaluated_missing_optional_dependency"
        jaxopt_reason = str(jaxopt.get("error"))
    elif (
        jaxopt["status"] == "ok"
        and float(jaxopt.get("loss_ratio") or 1.0) < 1.0e-6
        and float(jaxopt.get("final_param_error") or 1.0) < 1.0e-4
    ):
        jaxopt_decision = "candidate_for_bounded_optimization_examples"
        jaxopt_reason = "JAXopt reduced the real scheme-4 objective and recovered target parameters."
    else:
        jaxopt_decision = "defer_optimization_gate_not_clean"
        jaxopt_reason = "JAXopt did not satisfy the bounded optimization gate."

    return {
        "gate": "optional_equinox_jaxopt_scheme4",
        "rows": len(rows),
        "status_counts": dict(counts),
        "measured_rows": len(measured_rows),
        "backends": sorted({str(row["backend"]) for row in rows}),
        "case": "scheme4_geometry_fit",
        "evidence": {
            "equinox_directional_grad_abs_error": None if eqx is None else eqx.get("directional_grad_abs_error"),
            "jaxopt_loss_ratio": None if jaxopt is None else jaxopt.get("loss_ratio"),
            "jaxopt_final_param_error": None if jaxopt is None else jaxopt.get("final_param_error"),
        },
        "adoption_decision": {
            "equinox": eqx_decision,
            "equinox_reason": eqx_reason,
            "jaxopt": jaxopt_decision,
            "jaxopt_reason": jaxopt_reason,
            "production_solver_dependency": "do_not_promote_from_objective_wrapper_gate",
            "hard_dependency": False,
        },
    }


def make_scheme4_problem(
    *,
    n_theta: int = 21,
    n_zeta: int = 21,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(n_theta), endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / 5.0, int(n_zeta), endpoint=False, dtype=jnp.float64)
    amps_target = jnp.asarray([0.05645, -0.05351, -0.01402], dtype=jnp.float64)
    bhat_target = boozer_geometry_scheme4(theta=theta, zeta=zeta, harmonics_amp0=amps_target).b_hat
    return theta, zeta, amps_target, bhat_target


def _directional_fd(objective, amps0: jnp.ndarray, direction: jnp.ndarray, eps: float = 1.0e-6) -> float:
    f_plus = float(objective(amps0 + eps * direction))
    f_minus = float(objective(amps0 - eps * direction))
    return (f_plus - f_minus) / (2.0 * eps)


def _import_optional_modules():
    eqx = None
    jaxopt = None
    eqx_error: Exception | None = None
    jaxopt_error: Exception | None = None
    try:
        import equinox as _eqx  # type: ignore[import-not-found]

        eqx = _eqx
    except Exception as exc:  # noqa: BLE001
        eqx_error = exc
    try:
        import jaxopt as _jaxopt  # type: ignore[import-not-found]

        jaxopt = _jaxopt
    except Exception as exc:  # noqa: BLE001
        jaxopt_error = exc
    return eqx, jaxopt, eqx_error, jaxopt_error


def _build_eqx_objective(
    *,
    eqx,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    bhat_target: jnp.ndarray,
):
    class Scheme4HarmonicObjective(eqx.Module):
        theta: jnp.ndarray
        zeta: jnp.ndarray
        bhat_target: jnp.ndarray

        def __call__(self, amps: jnp.ndarray) -> jnp.ndarray:
            bhat = boozer_geometry_scheme4(theta=self.theta, zeta=self.zeta, harmonics_amp0=amps).b_hat
            return jnp.mean((bhat - self.bhat_target) ** 2)

    return Scheme4HarmonicObjective(theta=theta, zeta=zeta, bhat_target=bhat_target)


def run_equinox_gate(
    *,
    n_theta: int = 21,
    n_zeta: int = 21,
    eqx_module: Any = _AUTO_IMPORT,
) -> ObjectiveGateResult:
    if eqx_module is _AUTO_IMPORT:
        eqx, _jaxopt, eqx_error, _jaxopt_error = _import_optional_modules()
    elif eqx_module is None:
        eqx, eqx_error = None, ImportError("equinox was not provided")
    else:
        eqx, eqx_error = eqx_module, None
    if eqx is None:
        return ObjectiveGateResult(
            case="scheme4_geometry_fit",
            backend="equinox_wrapper",
            status="skipped",
            initial_loss=None,
            final_loss=None,
            loss_ratio=None,
            directional_grad=None,
            finite_difference_grad=None,
            directional_grad_abs_error=None,
            final_param_error=None,
            elapsed_s=None,
            error=f"Equinox unavailable: {eqx_error}",
        )

    theta, zeta, _amps_target, bhat_target = make_scheme4_problem(n_theta=n_theta, n_zeta=n_zeta)
    objective = _build_eqx_objective(eqx=eqx, theta=theta, zeta=zeta, bhat_target=bhat_target)
    amps0 = jnp.zeros(3, dtype=jnp.float64)
    direction = jnp.asarray([0.3, -0.2, 0.1], dtype=jnp.float64)

    wrapped = eqx.filter_jit(eqx.filter_value_and_grad(objective))
    t0 = time.perf_counter()
    loss0, grad0 = wrapped(amps0)
    loss0.block_until_ready()
    grad0.block_until_ready()
    elapsed_s = time.perf_counter() - t0
    dir_grad = float(jnp.vdot(grad0, direction))
    fd = _directional_fd(objective, amps0, direction)
    return ObjectiveGateResult(
        case="scheme4_geometry_fit",
        backend="equinox_wrapper",
        status="ok",
        initial_loss=float(loss0),
        final_loss=float(loss0),
        loss_ratio=1.0,
        directional_grad=dir_grad,
        finite_difference_grad=fd,
        directional_grad_abs_error=abs(dir_grad - fd),
        final_param_error=float(jnp.linalg.norm(amps0)),
        elapsed_s=float(elapsed_s),
        error=None,
    )


def run_jaxopt_gate(
    *,
    n_theta: int = 21,
    n_zeta: int = 21,
    maxiter: int = 5,
    stepsize: float = 0.1,
    eqx_module: Any = _AUTO_IMPORT,
    jaxopt_module: Any = _AUTO_IMPORT,
) -> ObjectiveGateResult:
    if eqx_module is _AUTO_IMPORT or jaxopt_module is _AUTO_IMPORT:
        eqx, jaxopt, eqx_error, jaxopt_error = _import_optional_modules()
    else:
        eqx = None if eqx_module is None else eqx_module
        jaxopt = None if jaxopt_module is None else jaxopt_module
        eqx_error = None if eqx is not None else ImportError("equinox was not provided")
        jaxopt_error = None if jaxopt is not None else ImportError("jaxopt was not provided")
    if eqx is None:
        return ObjectiveGateResult(
            case="scheme4_geometry_fit",
            backend="jaxopt_gradient_descent",
            status="skipped",
            initial_loss=None,
            final_loss=None,
            loss_ratio=None,
            directional_grad=None,
            finite_difference_grad=None,
            directional_grad_abs_error=None,
            final_param_error=None,
            elapsed_s=None,
            error=f"Equinox unavailable: {eqx_error}",
        )
    if jaxopt is None:
        return ObjectiveGateResult(
            case="scheme4_geometry_fit",
            backend="jaxopt_gradient_descent",
            status="skipped",
            initial_loss=None,
            final_loss=None,
            loss_ratio=None,
            directional_grad=None,
            finite_difference_grad=None,
            directional_grad_abs_error=None,
            final_param_error=None,
            elapsed_s=None,
            error=f"JAXopt unavailable: {jaxopt_error}",
        )

    theta, zeta, amps_target, bhat_target = make_scheme4_problem(n_theta=n_theta, n_zeta=n_zeta)
    objective = _build_eqx_objective(eqx=eqx, theta=theta, zeta=zeta, bhat_target=bhat_target)
    amps0 = jnp.zeros_like(amps_target)
    loss0 = float(objective(amps0))

    solver = jaxopt.GradientDescent(
        fun=objective,
        stepsize=float(stepsize),
        maxiter=int(maxiter),
        tol=1.0e-12,
        acceleration=False,
        implicit_diff=False,
        jit=True,
    )

    t0 = time.perf_counter()
    result = solver.run(amps0)
    params = result.params
    params.block_until_ready()
    elapsed_s = time.perf_counter() - t0
    loss1 = float(objective(params))
    return ObjectiveGateResult(
        case="scheme4_geometry_fit",
        backend="jaxopt_gradient_descent",
        status="ok",
        initial_loss=loss0,
        final_loss=loss1,
        loss_ratio=loss1 / max(loss0, 1.0e-300),
        directional_grad=None,
        finite_difference_grad=None,
        directional_grad_abs_error=None,
        final_param_error=float(jnp.linalg.norm(params - amps_target)),
        elapsed_s=float(elapsed_s),
        error=None,
    )


def run_gate(args: argparse.Namespace) -> list[ObjectiveGateResult]:
    backends = ["equinox", "jaxopt"] if args.backend == "all" else [str(args.backend)]
    results: list[ObjectiveGateResult] = []
    if "equinox" in backends:
        results.append(run_equinox_gate(n_theta=int(args.n_theta), n_zeta=int(args.n_zeta)))
    if "jaxopt" in backends:
        results.append(
            run_jaxopt_gate(
                n_theta=int(args.n_theta),
                n_zeta=int(args.n_zeta),
                maxiter=int(args.maxiter),
                stepsize=float(args.stepsize),
            )
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("equinox", "jaxopt", "all"), default="all")
    parser.add_argument("--n-theta", type=int, default=21)
    parser.add_argument("--n-zeta", type=int, default=21)
    parser.add_argument("--maxiter", type=int, default=5)
    parser.add_argument("--stepsize", type=float, default=0.1)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Write measured summary and adoption decision JSON without changing --out-json rows.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    results = run_gate(args)
    payload = [result.to_json_dict() for result in results]
    summary = summarize_gate_results(results)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n")
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
