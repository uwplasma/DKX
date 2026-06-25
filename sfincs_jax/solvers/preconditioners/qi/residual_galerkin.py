"""Residual-derived Galerkin coarse primitive for RHSMode=1 QI hard seeds.

This module is intentionally independent of the existing device preconditioner
lane.  It builds a small coarse correction space from the remaining operator
residual itself, caches ``A @ Q``, and fails closed unless the measured setup
residual is reduced by the candidate coarse action.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


ArrayLike = Any
LinearOperator = ArrayLike | Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True)
class RHS1QIResidualGalerkinConfig:
    """Static controls for residual-derived coarse-space construction."""

    max_stages: int = 3
    max_stage_rank: int | None = 4
    max_rank: int | None = None
    min_rank: int = 1
    rank_rtol: float = 1.0e-10
    rank_atol: float = 0.0
    regularization_rcond: float = 1.0e-12
    damping: float = 1.0
    solver: str = "action_lstsq"
    min_relative_improvement: float = 0.0
    acceptance_atol: float = 0.0
    include_global_residual: bool = False
    include_block_residuals: bool = True
    include_operator_images: bool = False
    include_operator_preimages: bool = False
    sort_blocks_by_residual_norm: bool = True


@dataclass(frozen=True)
class RHS1QIResidualGalerkinMetadata:
    """JSON-friendly setup diagnostics for the residual Galerkin state."""

    rank: int
    stage_count: int
    stage_ranks: tuple[int, ...]
    candidate_count: int
    residual_before: float
    residual_after: float
    labels: tuple[str, ...]
    solver: str
    condition_estimate: float
    accepted: bool
    reason: str
    candidate_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return plain Python containers suitable for traces or JSON."""

        return asdict(self)


@dataclass(frozen=True)
class RHS1QIResidualGalerkinState:
    """Reusable pure-JAX residual coarse correction.

    ``basis`` stores the accepted solution-space basis ``Q`` and
    ``operator_on_basis`` stores the cached action ``A @ Q``.  Applying the
    state to any residual uses only those cached arrays and a small JAX dense
    solve, so the apply path is safe to close over in ``jax.jit``.
    """

    basis: ArrayLike
    operator_on_basis: ArrayLike
    coarse_operator: ArrayLike
    regularization_rcond: float
    damping: float
    solver: str
    metadata: RHS1QIResidualGalerkinMetadata

    def solve_coefficients(self, residual: ArrayLike) -> ArrayLike:
        """Solve the cached coarse problem for ``residual``."""

        residual_vec = jnp.asarray(residual, dtype=self.basis.dtype).reshape((-1,))
        rank = int(self.metadata.rank)
        if rank <= 0:
            return jnp.zeros((0,), dtype=residual_vec.dtype)
        if self.solver == "action_lstsq":
            return _regularized_least_squares(
                self.operator_on_basis,
                residual_vec,
                rcond=float(self.regularization_rcond),
            )
        if self.solver == "galerkin":
            projected = jnp.conjugate(self.basis).T @ residual_vec
            return _regularized_least_squares(
                self.coarse_operator,
                projected,
                rcond=float(self.regularization_rcond),
            )
        raise ValueError(f"unsupported residual Galerkin solver: {self.solver!r}")

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Return the coarse correction for ``residual``."""

        residual_vec = jnp.asarray(residual, dtype=self.basis.dtype).reshape((-1,))
        if int(self.metadata.rank) <= 0:
            return jnp.zeros_like(residual_vec)
        coefficients = self.solve_coefficients(residual_vec)
        return jnp.asarray(self.damping, dtype=residual_vec.dtype) * (self.basis @ coefficients)

    def residual_after_apply(self, residual: ArrayLike) -> ArrayLike:
        """Return the residual remaining after applying the cached coarse action."""

        residual_vec = jnp.asarray(residual, dtype=self.basis.dtype).reshape((-1,))
        if int(self.metadata.rank) <= 0:
            return residual_vec
        coefficients = self.solve_coefficients(residual_vec)
        action = jnp.asarray(self.damping, dtype=residual_vec.dtype) * (self.operator_on_basis @ coefficients)
        return residual_vec - action

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        """Return ``apply`` for preconditioner call sites."""

        return self.apply


def setup_rhs1_qi_residual_galerkin(
    operator: LinearOperator,
    residual: ArrayLike,
    *,
    block_sizes: Sequence[int] | None = None,
    config: RHS1QIResidualGalerkinConfig | None = None,
) -> RHS1QIResidualGalerkinState:
    """Build a residual-derived, fail-closed Galerkin coarse state.

    ``residual`` is the remaining true operator residual after any outer setup
    step, for example ``rhs - A @ x_smoother``.  The returned state is accepted
    only when its cached action reduces this setup residual.
    """

    cfg = config or RHS1QIResidualGalerkinConfig()
    solver = str(cfg.solver)
    if solver not in {"action_lstsq", "galerkin"}:
        raise ValueError("solver must be 'action_lstsq' or 'galerkin'")

    residual_vec = jnp.asarray(residual).reshape((-1,))
    total_size = int(residual_vec.shape[0])
    blocks = _normalize_block_sizes(block_sizes, total_size)
    residual_before = float(jnp.linalg.norm(residual_vec))
    empty_state = _empty_state(
        total_size,
        residual_vec.dtype,
        cfg=cfg,
        residual_before=residual_before,
        reason="zero_residual" if residual_before == 0.0 else "empty_basis",
        candidate_count=0,
        candidate_labels=(),
    )
    if residual_before == 0.0:
        return empty_state

    stage_residual = residual_vec
    accepted_columns: list[ArrayLike] = []
    accepted_actions: list[ArrayLike] = []
    accepted_labels: list[str] = []
    all_candidate_labels: list[str] = []
    stage_ranks: list[int] = []
    candidate_count = 0
    rejection_reason = "empty_basis"

    max_stages = max(0, int(cfg.max_stages))
    max_total_rank = None if cfg.max_rank is None else max(0, int(cfg.max_rank))
    for stage_index in range(max_stages):
        if max_total_rank is not None and len(accepted_columns) >= max_total_rank:
            break

        candidates, labels = _build_stage_candidates(
            operator,
            stage_residual,
            blocks=blocks,
            stage_index=stage_index,
            cfg=cfg,
        )
        candidate_count += int(candidates.shape[1])
        all_candidate_labels.extend(labels)
        if int(candidates.shape[1]) == 0:
            rejection_reason = "empty_basis"
            break

        remaining_rank = None
        if max_total_rank is not None:
            remaining_rank = max(0, max_total_rank - len(accepted_columns))
        stage_rank_limit = _stage_rank_limit(cfg.max_stage_rank, remaining_rank, int(candidates.shape[1]))
        q_stage, labels_stage = _orthonormalize_against(
            candidates,
            labels=labels,
            existing_columns=accepted_columns,
            rtol=float(cfg.rank_rtol),
            atol=float(cfg.rank_atol),
            max_rank=stage_rank_limit,
        )
        stage_rank = int(q_stage.shape[1])
        if stage_rank <= 0:
            rejection_reason = "rank_deficient"
            break

        aq_stage = _apply_operator_to_basis(operator, q_stage)
        stage_coefficients = _solve_cached(
            q_stage,
            aq_stage,
            stage_residual,
            solver=solver,
            rcond=float(cfg.regularization_rcond),
        )
        stage_action = jnp.asarray(float(cfg.damping), dtype=stage_residual.dtype) * (aq_stage @ stage_coefficients)
        trial_residual = stage_residual - stage_action
        trial_norm = float(jnp.linalg.norm(trial_residual))
        stage_norm = float(jnp.linalg.norm(stage_residual))
        if not _is_reduced(
            before=stage_norm,
            after=trial_norm,
            min_relative_improvement=float(cfg.min_relative_improvement),
            acceptance_atol=float(cfg.acceptance_atol),
        ):
            rejection_reason = "not_reduced"
            break

        for column_index in range(stage_rank):
            accepted_columns.append(q_stage[:, column_index])
            accepted_actions.append(aq_stage[:, column_index])
        accepted_labels.extend(labels_stage)
        stage_ranks.append(stage_rank)
        stage_residual = trial_residual

    if not accepted_columns:
        return _empty_state(
            total_size,
            residual_vec.dtype,
            cfg=cfg,
            residual_before=residual_before,
            reason=rejection_reason,
            candidate_count=candidate_count,
            candidate_labels=tuple(all_candidate_labels),
            stage_ranks=tuple(stage_ranks),
        )

    q = jnp.stack(tuple(accepted_columns), axis=1)
    aq = jnp.stack(tuple(accepted_actions), axis=1)
    rank = int(q.shape[1])
    if rank < max(0, int(cfg.min_rank)):
        return _empty_state(
            total_size,
            residual_vec.dtype,
            cfg=cfg,
            residual_before=residual_before,
            reason="rank_deficient",
            candidate_count=candidate_count,
            candidate_labels=tuple(all_candidate_labels),
            stage_ranks=tuple(stage_ranks),
        )

    coarse_operator = jnp.conjugate(q).T @ aq
    coefficients = _solve_cached(
        q,
        aq,
        residual_vec,
        solver=solver,
        rcond=float(cfg.regularization_rcond),
    )
    residual_after_vec = residual_vec - jnp.asarray(float(cfg.damping), dtype=residual_vec.dtype) * (aq @ coefficients)
    residual_after = float(jnp.linalg.norm(residual_after_vec))
    if not _is_reduced(
        before=residual_before,
        after=residual_after,
        min_relative_improvement=float(cfg.min_relative_improvement),
        acceptance_atol=float(cfg.acceptance_atol),
    ):
        return _empty_state(
            total_size,
            residual_vec.dtype,
            cfg=cfg,
            residual_before=residual_before,
            reason="not_reduced",
            candidate_count=candidate_count,
            candidate_labels=tuple(all_candidate_labels),
            stage_ranks=tuple(stage_ranks),
        )

    conditioning_matrix = aq if solver == "action_lstsq" else coarse_operator
    metadata = RHS1QIResidualGalerkinMetadata(
        rank=rank,
        stage_count=len(stage_ranks),
        stage_ranks=tuple(stage_ranks),
        candidate_count=candidate_count,
        residual_before=residual_before,
        residual_after=residual_after,
        labels=tuple(accepted_labels),
        solver=solver,
        condition_estimate=_condition_estimate(conditioning_matrix),
        accepted=True,
        reason="residual_reduced",
        candidate_labels=tuple(all_candidate_labels),
    )
    return RHS1QIResidualGalerkinState(
        basis=q,
        operator_on_basis=aq,
        coarse_operator=coarse_operator,
        regularization_rcond=float(cfg.regularization_rcond),
        damping=float(cfg.damping),
        solver=solver,
        metadata=metadata,
    )


build_rhs1_qi_residual_galerkin = setup_rhs1_qi_residual_galerkin


def _normalize_block_sizes(block_sizes: Sequence[int] | None, total_size: int) -> tuple[int, ...]:
    if block_sizes is None:
        return (int(total_size),)
    blocks = tuple(int(size) for size in block_sizes)
    if not blocks:
        raise ValueError("block_sizes must contain at least one block")
    if any(size <= 0 for size in blocks):
        raise ValueError("block_sizes must be positive")
    if sum(blocks) != int(total_size):
        raise ValueError("block_sizes must sum to the residual length")
    return blocks


def _empty_matrix(n_rows: int, dtype: Any) -> ArrayLike:
    return jnp.zeros((int(n_rows), 0), dtype=dtype)


def _empty_state(
    total_size: int,
    dtype: Any,
    *,
    cfg: RHS1QIResidualGalerkinConfig,
    residual_before: float,
    reason: str,
    candidate_count: int,
    candidate_labels: tuple[str, ...],
    stage_ranks: tuple[int, ...] = (),
) -> RHS1QIResidualGalerkinState:
    basis = _empty_matrix(total_size, dtype)
    coarse_operator = jnp.zeros((0, 0), dtype=dtype)
    metadata = RHS1QIResidualGalerkinMetadata(
        rank=0,
        stage_count=len(stage_ranks),
        stage_ranks=stage_ranks,
        candidate_count=int(candidate_count),
        residual_before=float(residual_before),
        residual_after=float(residual_before),
        labels=(),
        solver=str(cfg.solver),
        condition_estimate=float("inf"),
        accepted=False,
        reason=str(reason),
        candidate_labels=tuple(candidate_labels),
    )
    return RHS1QIResidualGalerkinState(
        basis=basis,
        operator_on_basis=basis,
        coarse_operator=coarse_operator,
        regularization_rcond=float(cfg.regularization_rcond),
        damping=float(cfg.damping),
        solver=str(cfg.solver),
        metadata=metadata,
    )


def _block_offsets(blocks: Sequence[int]) -> tuple[int, ...]:
    offsets = [0]
    for size in blocks:
        offsets.append(offsets[-1] + int(size))
    return tuple(offsets)


def _build_stage_candidates(
    operator: LinearOperator,
    residual: ArrayLike,
    *,
    blocks: Sequence[int],
    stage_index: int,
    cfg: RHS1QIResidualGalerkinConfig,
) -> tuple[ArrayLike, tuple[str, ...]]:
    residual_vec = jnp.asarray(residual).reshape((-1,))
    total_size = int(residual_vec.shape[0])
    columns: list[ArrayLike] = []
    labels: list[str] = []

    if bool(cfg.include_global_residual):
        columns.append(residual_vec)
        labels.append(f"stage:{stage_index}:residual")

    if bool(cfg.include_block_residuals):
        block_columns: list[tuple[float, int, ArrayLike, str]] = []
        offsets = _block_offsets(blocks)
        for block_index in range(len(blocks)):
            start = int(offsets[block_index])
            stop = int(offsets[block_index + 1])
            values = jnp.zeros((total_size,), dtype=residual_vec.dtype).at[start:stop].set(residual_vec[start:stop])
            block_norm = float(jnp.linalg.norm(values))
            block_columns.append(
                (
                    block_norm,
                    block_index,
                    values,
                    f"stage:{stage_index}:block:{block_index}:residual",
                )
            )
        if bool(cfg.sort_blocks_by_residual_norm):
            block_columns = sorted(block_columns, key=lambda item: (-item[0], item[1]))
        for _, _, values, label in block_columns:
            columns.append(values)
            labels.append(label)

    base_count = len(columns)
    if bool(cfg.include_operator_images):
        for column_index in range(base_count):
            image = _apply_operator(operator, columns[column_index])
            if int(image.shape[0]) == total_size:
                columns.append(image)
                labels.append(f"{labels[column_index]}:operator_image")

    if bool(cfg.include_operator_preimages) and not callable(operator):
        operator_matrix = jnp.asarray(operator)
        if operator_matrix.ndim == 2 and tuple(int(v) for v in operator_matrix.shape) == (total_size, total_size):
            for column_index in range(base_count):
                preimage = _regularized_least_squares(
                    operator_matrix,
                    columns[column_index],
                    rcond=float(cfg.regularization_rcond),
                )
                columns.append(preimage)
                labels.append(f"{labels[column_index]}:operator_preimage")

    if not columns:
        return _empty_matrix(total_size, residual_vec.dtype), ()
    return jnp.stack(tuple(columns), axis=1), tuple(labels)


def _stage_rank_limit(max_stage_rank: int | None, remaining_rank: int | None, candidate_count: int) -> int:
    limit = int(candidate_count)
    if max_stage_rank is not None:
        limit = min(limit, max(0, int(max_stage_rank)))
    if remaining_rank is not None:
        limit = min(limit, max(0, int(remaining_rank)))
    return limit


def _orthonormalize_against(
    candidates: ArrayLike,
    *,
    labels: Sequence[str],
    existing_columns: Sequence[ArrayLike],
    rtol: float,
    atol: float,
    max_rank: int,
) -> tuple[ArrayLike, tuple[str, ...]]:
    matrix = jnp.asarray(candidates)
    if matrix.ndim == 1:
        matrix = matrix.reshape((-1, 1))
    if matrix.ndim != 2:
        raise ValueError("candidates must be a vector or matrix")
    if len(labels) != int(matrix.shape[1]):
        raise ValueError("labels must match candidate columns")
    if int(max_rank) <= 0:
        return _empty_matrix(int(matrix.shape[0]), matrix.dtype), ()

    candidate_norms = tuple(float(jnp.linalg.norm(matrix[:, idx])) for idx in range(int(matrix.shape[1])))
    reference_norm = max(candidate_norms, default=0.0)
    threshold = max(float(atol), float(rtol) * reference_norm)

    q_columns: list[ArrayLike] = []
    accepted_labels: list[str] = []
    for column_index in range(int(matrix.shape[1])):
        if len(q_columns) >= int(max_rank):
            break
        vector = matrix[:, column_index]
        if float(jnp.linalg.norm(vector)) <= threshold:
            continue
        residual = vector
        for _ in range(2):
            for existing in existing_columns:
                residual = residual - existing * jnp.vdot(existing, residual)
            for q_col in q_columns:
                residual = residual - q_col * jnp.vdot(q_col, residual)
        residual_norm = float(jnp.linalg.norm(residual))
        if residual_norm <= threshold:
            continue
        q_columns.append(residual / residual_norm)
        accepted_labels.append(str(labels[column_index]))

    if not q_columns:
        return _empty_matrix(int(matrix.shape[0]), matrix.dtype), ()
    return jnp.stack(tuple(q_columns), axis=1), tuple(accepted_labels)


def _apply_operator(operator: LinearOperator, vector: ArrayLike) -> ArrayLike:
    vec = jnp.asarray(vector).reshape((-1,))
    if callable(operator):
        return jnp.asarray(operator(vec)).reshape((-1,))
    return (jnp.asarray(operator) @ vec).reshape((-1,))


def _apply_operator_to_basis(operator: LinearOperator, basis: ArrayLike) -> ArrayLike:
    q = jnp.asarray(basis)
    if q.ndim != 2:
        raise ValueError("basis must be a matrix")
    if int(q.shape[1]) == 0:
        return _empty_matrix(int(q.shape[0]), q.dtype)
    if not callable(operator):
        return jnp.asarray(operator) @ q
    columns = tuple(_apply_operator(operator, q[:, idx]) for idx in range(int(q.shape[1])))
    return jnp.stack(columns, axis=1)


def _regularized_least_squares(matrix: ArrayLike, rhs: ArrayLike, *, rcond: float) -> ArrayLike:
    a = jnp.asarray(matrix)
    rhs_vec = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if a.ndim != 2:
        raise ValueError("matrix must be 2D")
    if int(rhs_vec.shape[0]) != int(a.shape[0]):
        raise ValueError("rhs length must match matrix rows")
    n_cols = int(a.shape[1])
    if n_cols == 0:
        return jnp.zeros((0,), dtype=a.dtype)
    a_h = jnp.conjugate(a).T
    gram = a_h @ a
    projected_rhs = a_h @ rhs_vec
    row_sums = jnp.sum(jnp.abs(gram), axis=1)
    scale = jnp.maximum(jnp.max(row_sums), jnp.asarray(1.0, dtype=row_sums.dtype))
    ridge = jnp.asarray(max(0.0, float(rcond)), dtype=row_sums.dtype) * scale
    return jnp.linalg.solve(gram + ridge * jnp.eye(n_cols, dtype=gram.dtype), projected_rhs)


def _solve_cached(
    basis: ArrayLike,
    operator_on_basis: ArrayLike,
    residual: ArrayLike,
    *,
    solver: str,
    rcond: float,
) -> ArrayLike:
    if int(jnp.asarray(basis).shape[1]) == 0:
        return jnp.zeros((0,), dtype=jnp.asarray(residual).dtype)
    if solver == "action_lstsq":
        return _regularized_least_squares(operator_on_basis, residual, rcond=rcond)
    if solver == "galerkin":
        q = jnp.asarray(basis)
        aq = jnp.asarray(operator_on_basis)
        return _regularized_least_squares(jnp.conjugate(q).T @ aq, jnp.conjugate(q).T @ residual, rcond=rcond)
    raise ValueError(f"unsupported residual Galerkin solver: {solver!r}")


def _is_reduced(
    *,
    before: float,
    after: float,
    min_relative_improvement: float,
    acceptance_atol: float,
) -> bool:
    if not (math.isfinite(float(before)) and math.isfinite(float(after))):
        return False
    required_drop = max(float(acceptance_atol), float(before) * max(0.0, float(min_relative_improvement)))
    return float(after) < float(before) - required_drop


def _condition_estimate(matrix: ArrayLike) -> float:
    a = jnp.asarray(matrix)
    if a.ndim != 2 or min(int(a.shape[0]), int(a.shape[1])) == 0:
        return float("inf")
    singular_values = jnp.linalg.svd(a, compute_uv=False)
    if int(singular_values.shape[0]) == 0:
        return float("inf")
    largest = float(jnp.max(singular_values))
    smallest = float(jnp.min(singular_values))
    if largest == 0.0 or smallest <= 0.0:
        return float("inf")
    return largest / smallest


# Galerkin probe-policy helpers share the residual-Galerkin owner.
VALID_QI_GALERKIN_MODES = ("additive", "multiplicative")


@dataclass(frozen=True)
class RHS1QIGalerkinProbeCandidate:
    """Measured residual for one Galerkin preconditioner candidate."""

    mode: str
    damping: float
    residual_norm: float
    improvement_ratio: float | None
    reduced: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "damping": float(self.damping),
            "residual_norm": float(self.residual_norm),
            "improvement_ratio": None if self.improvement_ratio is None else float(self.improvement_ratio),
            "reduced": bool(self.reduced),
        }


@dataclass(frozen=True)
class RHS1QIGalerkinProbeSelection:
    """Fail-closed selection result for candidate Galerkin preconditioners."""

    accepted: bool
    reason: str
    selected_index: int | None
    selected_mode: str | None
    selected_damping: float | None
    residual_before_norm: float
    residual_after_norm: float | None
    improvement_ratio: float | None
    candidates: tuple[RHS1QIGalerkinProbeCandidate, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "selected_index": self.selected_index,
            "selected_mode": self.selected_mode,
            "selected_damping": None if self.selected_damping is None else float(self.selected_damping),
            "residual_before_norm": float(self.residual_before_norm),
            "residual_after_norm": None if self.residual_after_norm is None else float(self.residual_after_norm),
            "improvement_ratio": None if self.improvement_ratio is None else float(self.improvement_ratio),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def parse_rhs1_qi_galerkin_modes(raw: str | None, *, default: str = "auto") -> tuple[str, ...]:
    """Parse a mode list for the QI Galerkin probe.

    ``auto`` tries both supported compositions.  Explicit lists such as
    ``additive,multiplicative`` are accepted so bounded GPU campaigns can use
    one environment variable instead of rerunning the whole process per mode.
    Invalid tokens are ignored; if nothing valid remains the result is
    ``("additive",)``.
    """

    text = (default if raw is None or not str(raw).strip() else str(raw)).strip().lower().replace("-", "_")
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens or "auto" in tokens:
        return VALID_QI_GALERKIN_MODES
    modes: list[str] = []
    for token in tokens:
        if token in VALID_QI_GALERKIN_MODES and token not in modes:
            modes.append(token)
    return tuple(modes) if modes else ("additive",)


def parse_rhs1_qi_galerkin_dampings(
    raw: str | None,
    *,
    default: float = 1.0,
    auto_defaults: Sequence[float] = (1.0, 0.5, 0.25),
) -> tuple[float, ...]:
    """Parse positive damping candidates for the QI Galerkin probe."""

    if raw is None or not str(raw).strip():
        values = tuple(float(value) for value in auto_defaults) if float(default) == 1.0 else (float(default),)
    else:
        parsed: list[float] = []
        for token in str(raw).replace(";", ",").split(","):
            if not token.strip():
                continue
            try:
                parsed.append(float(token))
            except ValueError:
                continue
        values = tuple(parsed)
    cleaned: list[float] = []
    for value in values:
        if np.isfinite(value) and value >= 0.0 and value not in cleaned:
            cleaned.append(float(value))
    return tuple(cleaned) if cleaned else (max(0.0, float(default)),)


def select_rhs1_qi_galerkin_probe_candidate(
    residual_before_norm: float,
    candidates: Sequence[dict[str, Any] | RHS1QIGalerkinProbeCandidate],
    *,
    min_relative_improvement: float = 0.0,
    acceptance_atol: float = 0.0,
) -> RHS1QIGalerkinProbeSelection:
    """Select the best candidate only if it reduces the true residual.

    The gate is intentionally conservative.  Non-finite residuals are recorded
    but cannot be selected; a candidate must beat ``residual_before_norm`` by at
    least the requested relative or absolute margin.
    """

    before = float(residual_before_norm)
    records: list[RHS1QIGalerkinProbeCandidate] = []
    for candidate in candidates:
        if isinstance(candidate, RHS1QIGalerkinProbeCandidate):
            record = candidate
        else:
            residual = float(candidate.get("residual_norm", np.inf))
            ratio = residual / before if before > 0.0 and np.isfinite(residual) else None
            record = RHS1QIGalerkinProbeCandidate(
                mode=str(candidate.get("mode", "additive")),
                damping=float(candidate.get("damping", 1.0)),
                residual_norm=residual,
                improvement_ratio=ratio,
                reduced=bool(np.isfinite(residual) and residual < before),
            )
        records.append(record)

    if not records:
        return RHS1QIGalerkinProbeSelection(
            accepted=False,
            reason="no_probe_candidates",
            selected_index=None,
            selected_mode=None,
            selected_damping=None,
            residual_before_norm=before,
            residual_after_norm=None,
            improvement_ratio=None,
            candidates=(),
        )

    finite = [(idx, record) for idx, record in enumerate(records) if np.isfinite(float(record.residual_norm))]
    if not finite:
        return RHS1QIGalerkinProbeSelection(
            accepted=False,
            reason="no_finite_probe_candidates",
            selected_index=None,
            selected_mode=None,
            selected_damping=None,
            residual_before_norm=before,
            residual_after_norm=None,
            improvement_ratio=None,
            candidates=tuple(records),
        )

    best_index, best = min(finite, key=lambda item: float(item[1].residual_norm))
    required_drop = max(float(acceptance_atol), before * max(0.0, float(min_relative_improvement)))
    accepted = bool(float(best.residual_norm) < before - required_drop)
    return RHS1QIGalerkinProbeSelection(
        accepted=accepted,
        reason="probe_reduced" if accepted else "probe_not_reduced",
        selected_index=int(best_index) if accepted else None,
        selected_mode=best.mode if accepted else None,
        selected_damping=float(best.damping) if accepted else None,
        residual_before_norm=before,
        residual_after_norm=float(best.residual_norm),
        improvement_ratio=best.improvement_ratio,
        candidates=tuple(records),
    )


__all__ = [
    "RHS1QIGalerkinProbeCandidate",
    "RHS1QIGalerkinProbeSelection",
    "VALID_QI_GALERKIN_MODES",
    "parse_rhs1_qi_galerkin_dampings",
    "parse_rhs1_qi_galerkin_modes",
    "select_rhs1_qi_galerkin_probe_candidate",
]
