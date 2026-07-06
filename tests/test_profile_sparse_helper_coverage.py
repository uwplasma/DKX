from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
import pytest
from scipy import sparse

import sfincs_jax.problems.profile_policies as profile_policies
import sfincs_jax.problems.profile_sparse_direct as sparse_direct
import sfincs_jax.problems.profile_sparse_policy as profile_sparse_policy
import sfincs_jax.problems.profile_solve as profile_solve
import sfincs_jax.solvers.explicit_sparse as explicit_sparse
import sfincs_jax.solvers.path_policy as path_policy
import sfincs_jax.solvers.preconditioner_xblock_tz_sparse as xblock_tz_sparse


class _HalfSolve:
    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return 0.5 * np.asarray(rhs, dtype=np.float64)


class _FakeNamelist:
    def __init__(self) -> None:
        self._groups = {
            "geometryParameters": {"geometryScheme": 1},
            "physicsParameters": {},
            "preconditionerOptions": {},
        }

    def group(self, name: str) -> dict[str, object]:
        return self._groups.setdefault(name, {})


def test_rhsmode1_xblock_sparse_lu_default_max_targets_full_fp_host_path() -> None:
    full_fp_op = SimpleNamespace(fblock=SimpleNamespace(fp=object(), pas=None))
    pas_op = SimpleNamespace(fblock=SimpleNamespace(fp=None, pas=object()))
    generic_op = SimpleNamespace(fblock=SimpleNamespace(fp=None, pas=None))

    assert xblock_tz_sparse.rhsmode1_xblock_sparse_lu_default_max(full_fp_op, build_jax_factors=False) == 30000
    assert xblock_tz_sparse.rhsmode1_xblock_sparse_lu_default_max(full_fp_op, build_jax_factors=True) == 2000
    assert xblock_tz_sparse.rhsmode1_xblock_sparse_lu_default_max(pas_op, build_jax_factors=False) == 2000
    assert xblock_tz_sparse.rhsmode1_xblock_sparse_lu_default_max(generic_op, build_jax_factors=False) == 2000


def test_rhsmode1_fp_xblock_host_species_decoupling_equivalence() -> None:
    one_species = SimpleNamespace(n_species=1)
    two_species = SimpleNamespace(n_species=2)

    assert xblock_tz_sparse.rhsmode1_fp_xblock_species_decoupled_for_host_assembly(
        op=one_species,
        preconditioner_species=0,
    )
    assert xblock_tz_sparse.rhsmode1_fp_xblock_species_decoupled_for_host_assembly(
        op=two_species,
        preconditioner_species=1,
    )
    assert not xblock_tz_sparse.rhsmode1_fp_xblock_species_decoupled_for_host_assembly(
        op=two_species,
        preconditioner_species=0,
    )


def test_host_sparse_direct_allowed_and_sparse_pc_rescue_policy(monkeypatch) -> None:
    assert not profile_policies.rhs1_host_sparse_direct_allowed(sparse_exact_lu=False)
    assert not profile_policies.rhs1_host_sparse_direct_allowed(sparse_exact_lu=True, use_implicit=True)

    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_HOST", raising=False)
    assert profile_policies.rhs1_host_sparse_direct_allowed(sparse_exact_lu=True)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_HOST", "0")
    assert not profile_policies.rhs1_host_sparse_direct_allowed(sparse_exact_lu=True)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_HOST", "1")
    assert profile_policies.rhs1_host_sparse_direct_allowed(sparse_exact_lu=True)

    op = SimpleNamespace(
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=1,
        fblock=SimpleNamespace(fp=object(), pas=None),
    )
    monkeypatch.setattr("sfincs_jax.problems.profile_policies.jax.default_backend", lambda: "cpu")
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES", raising=False)
    assert profile_policies.rhsmode1_sparse_operator_preconditioned_rescue_allowed_current_backend(
        op=op, sparse_exact_lu=True, host_sparse_direct_wanted=True
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES", "off")
    assert not profile_policies.rhsmode1_sparse_operator_preconditioned_rescue_allowed_current_backend(
        op=op, sparse_exact_lu=True, host_sparse_direct_wanted=True
    )


def test_host_sparse_factor_dtype_and_cache_key(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_HOST_SPARSE_FACTOR_FLOAT32_MIN", raising=False)
    monkeypatch.setattr("sfincs_jax.problems.profile_policies.jax.default_backend", lambda: "cpu")
    assert profile_policies.host_sparse_factor_dtype_current_backend(size=20_000, factorization="lu", use_implicit=False) == np.dtype(np.float32)
    assert profile_policies.host_sparse_factor_dtype_current_backend(size=1_000, factorization="ilu", use_implicit=False) == np.dtype(np.float64)
    assert profile_policies.host_sparse_factor_dtype_current_backend(size=20_000, factorization="lu", use_implicit=True) == np.dtype(np.float64)

    monkeypatch.setenv("SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE", "fp64")
    assert profile_policies.host_sparse_factor_dtype_current_backend(size=20_000, factorization="lu", use_implicit=False) == np.dtype(np.float64)
    monkeypatch.setenv("SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE", "32")
    assert profile_policies.host_sparse_factor_dtype_current_backend(size=1, factorization="lu", use_implicit=False) == np.dtype(np.float32)

    key = sparse_direct.sparse_factor_cache_key(("a", 1), np.dtype(np.float32))
    assert key == ("a", 1, np.dtype(np.float32).str)


def test_host_sparse_refine_step_parsing_and_skip_dense_ratio(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_SKIP_DENSE_RATIO", raising=False)
    assert profile_policies.rhs1_host_sparse_skip_dense_ratio() == pytest.approx(1.0e4)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_SKIP_DENSE_RATIO", "bad")
    assert profile_policies.rhs1_host_sparse_skip_dense_ratio() == pytest.approx(1.0e4)

    monkeypatch.delenv("MY_REFINE_STEPS", raising=False)
    assert profile_policies.host_sparse_direct_refine_steps("MY_REFINE_STEPS", default=3) == 3
    monkeypatch.setenv("MY_REFINE_STEPS", "bad")
    assert profile_policies.host_sparse_direct_refine_steps("MY_REFINE_STEPS", default=2) == 2
    monkeypatch.setenv("MY_REFINE_STEPS", "-4")
    assert profile_policies.host_sparse_direct_refine_steps("MY_REFINE_STEPS", default=2) == 0


def test_rhs1_residual_rescue_uses_small_target_slack(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_RESCUE_TARGET_SLACK", raising=False)
    assert not path_policy.rhs1_residual_needs_rescue(1.006e-12, 1.0e-12)
    assert path_policy.rhs1_residual_needs_rescue(1.02e-12, 1.0e-12)
    assert path_policy.rhs1_residual_needs_rescue(1.006e-12, 1.0e-12, force=True)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_RESCUE_TARGET_SLACK", "0")
    assert path_policy.rhs1_residual_needs_rescue(1.006e-12, 1.0e-12)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_RESCUE_TARGET_SLACK", "bad")
    assert not path_policy.rhs1_residual_needs_rescue(1.006e-12, 1.0e-12)


def test_host_direct_refinement_helpers_improve_residual() -> None:
    rhs = jnp.asarray([2.0, -4.0], dtype=jnp.float64)
    ident = np.eye(2, dtype=np.float64)

    x_direct, rn_direct = explicit_sparse.host_direct_solve_with_refinement(
        factor_solve=lambda v: 0.5 * np.asarray(v, dtype=np.float64),
        operator_matrix=ident,
        rhs_vec=rhs,
        factor_dtype=np.dtype(np.float64),
        refine_steps=3,
    )
    np.testing.assert_allclose(x_direct, np.asarray([1.875, -3.75]))
    assert rn_direct < np.linalg.norm(np.asarray(rhs) - 0.5 * np.asarray(rhs))

    x_sparse, rn_sparse = explicit_sparse.host_sparse_direct_solve_with_refinement(
        ilu=_HalfSolve(),
        a_csr_full=sparse.csr_matrix(ident),
        rhs_vec=rhs,
        factor_dtype=np.dtype(np.float64),
        refine_steps=3,
    )
    np.testing.assert_allclose(x_sparse, x_direct)
    assert rn_sparse == pytest.approx(rn_direct)


def test_host_sparse_direct_polish_uses_preconditioner_and_reports_residual(monkeypatch) -> None:
    seen = {}

    def fake_gmres(**kwargs):
        probe = kwargs["preconditioner"](jnp.asarray([4.0, -2.0], dtype=jnp.float64))
        seen["probe"] = np.asarray(probe)
        return np.asarray([1.0, -2.0]), 0.0, [0.0]

    monkeypatch.setattr(sparse_direct, "gmres_solve_with_history_scipy", fake_gmres)
    x, rn = sparse_direct.host_sparse_direct_polish(
        matvec_fn=lambda x: x,
        rhs_vec=jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        x0_np=np.zeros(2, dtype=np.float64),
        ilu=_HalfSolve(),
        factor_dtype=np.dtype(np.float64),
        tol=1e-12,
        atol=0.0,
        restart=4,
        maxiter=4,
        precondition_side="left",
    )
    np.testing.assert_allclose(seen["probe"], np.asarray([2.0, -1.0]))
    np.testing.assert_allclose(x, np.asarray([1.0, -2.0]))
    assert rn == 0.0


def test_explicit_sparse_host_direct_allowed_and_env_bounds(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_HELPER", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_HELPER_MAX", raising=False)
    assert profile_policies.rhs1_explicit_sparse_host_direct_allowed(
        sparse_exact_lu=True,
        use_implicit=False,
        active_size=10_000,
    )
    assert not profile_policies.rhs1_explicit_sparse_host_direct_allowed(
        sparse_exact_lu=False,
        use_implicit=False,
        active_size=10_000,
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_HELPER_MAX", "bad")
    assert profile_policies.rhs1_explicit_sparse_host_direct_allowed(
        sparse_exact_lu=True,
        use_implicit=False,
        active_size=10_000,
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_HELPER", "off")
    assert not profile_policies.rhs1_explicit_sparse_host_direct_allowed(
        sparse_exact_lu=True,
        use_implicit=False,
        active_size=10_000,
    )


def test_profile_sparse_direct_pattern_probe_and_cache_key(monkeypatch) -> None:
    assert not sparse_direct.rhsmode1_explicit_sparse_pattern_probe_enabled(env={})
    assert sparse_direct.rhsmode1_explicit_sparse_pattern_probe_enabled(
        env={"SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_PATTERN": "pattern"}
    )

    assert sparse_direct.maybe_rhsmode1_full_sparse_pattern(SimpleNamespace(), env={}) is None

    emits: list[str] = []
    pattern = SimpleNamespace(shape=(2, 2), nnz=3)
    summary = SimpleNamespace(shape=(2, 2), nnz=3, avg_row_nnz=1.5, max_row_nnz=2)
    monkeypatch.setattr(
        sparse_direct,
        "v3_full_system_conservative_sparsity_pattern",
        lambda _op: pattern,
    )
    monkeypatch.setattr(sparse_direct, "summarize_v3_sparse_pattern", lambda _op, _pattern: summary)
    assert sparse_direct.maybe_rhsmode1_full_sparse_pattern(
        SimpleNamespace(),
        emit=lambda _level, message: emits.append(message),
        env={"SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_PATTERN": "on"},
    ) is pattern
    assert any("explicit_sparse_pattern" in message for message in emits)

    monkeypatch.setattr(
        sparse_direct,
        "_rhsmode1_precond_cache_key",
        lambda _op, kind: ("operator", kind),
    )
    key = sparse_direct.rhsmode1_sparse_cache_key(
        SimpleNamespace(),
        kind="ilu",
        active_size=17,
        use_active_dof_mode=True,
        use_pas_projection=False,
        drop_tol=1.0e-3,
        drop_rel=2.0e-3,
        ilu_drop_tol=3.0e-3,
        fill_factor=4.0,
    )
    assert key == ("operator", "ilu", 17, 1, 0, 1.0e-3, 2.0e-3, 3.0e-3, 4.0)


def test_build_host_sparse_direct_factor_from_matvec_falls_back_on_invalid_env(monkeypatch) -> None:
    seen: dict[str, object] = {}
    operator_bundle = SimpleNamespace(metadata=SimpleNamespace(storage_kind="csr", reason="fallback"))
    factor_bundle = SimpleNamespace(kind="lu")

    def fake_build_operator_from_matvec(matvec, **kwargs):
        seen["kwargs"] = kwargs
        np.testing.assert_allclose(matvec(np.asarray([1.0, -2.0])), np.asarray([2.0, -4.0]))
        np.testing.assert_allclose(
            kwargs["matmat"](np.asarray([[1.0, 0.0], [0.0, 1.0]])),
            2.0 * np.eye(2),
        )
        return operator_bundle

    def fake_factorize_host_sparse_operator(bundle, *, kind, **kwargs):
        seen["factor"] = (bundle, kind, kwargs)
        return factor_bundle

    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_BLOCK_COLS", "bad")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_DENSE_MAX_MB", "bad")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB", "bad")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_DROP_TOL", "bad")
    monkeypatch.setattr(sparse_direct, "build_operator_from_matvec", fake_build_operator_from_matvec)
    monkeypatch.setattr(sparse_direct, "factorize_host_sparse_operator", fake_factorize_host_sparse_operator)
    monkeypatch.setattr("sfincs_jax.problems.profile_sparse_direct.jax.default_backend", lambda: "cpu")

    messages: list[tuple[int, str]] = []
    op, fac = sparse_direct.build_host_sparse_direct_factor_from_matvec(
        matvec=lambda x: 2.0 * x,
        n=2,
        dtype=jnp.float32,
        factor_dtype=np.dtype(np.float64),
        emit=lambda level, msg: messages.append((level, msg)),
    )

    assert op is operator_bundle
    assert fac is factor_bundle
    factor_bundle_seen, factor_kind_seen, factor_kwargs_seen = seen["factor"]
    assert factor_bundle_seen is operator_bundle
    assert factor_kind_seen == "lu"
    assert factor_kwargs_seen["fill_factor"] == pytest.approx(10.0)
    assert factor_kwargs_seen["drop_tol"] == pytest.approx(1.0e-4)
    assert factor_kwargs_seen["permc_spec"] == "COLAMD"
    assert factor_kwargs_seen["diag_pivot_thresh"] == pytest.approx(1.0)
    kwargs = seen["kwargs"]
    assert kwargs["dtype"] == np.dtype(np.float64)
    assert kwargs["block_cols"] == 32
    assert kwargs["dense_max_mb"] == pytest.approx(128.0)
    assert kwargs["csr_max_mb"] == pytest.approx(512.0)
    assert kwargs["drop_tol"] == pytest.approx(0.0)
    assert kwargs["prefer_sparse_on_gpu"] is True
    assert kwargs["allow_operator_only"] is False
    assert messages[0][0] == 1
    assert messages[0][1].startswith(
        "explicit_sparse: storage=csr reason=fallback factor_kind=lu "
        "factor_dtype=float64 permc=COLAMD diag_pivot=1"
    )
    assert "operator_nnz=None operator_csr_mb=unknown" in messages[0][1]
    assert messages[1] == (
        1,
        "explicit_sparse: factorization start factor_kind=lu permc=COLAMD shape=(2, 2)",
    )
    assert messages[2][0] == 1
    assert messages[2][1].startswith(
        "explicit_sparse: factorization complete factor_kind=lu elapsed_s="
    )
    assert messages[2][1].endswith(" factor_nnz=None factor_mb=unknown")


def test_build_host_sparse_direct_factor_from_matvec_respects_env_overrides(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_build_operator_from_matvec(matvec, **kwargs):
        seen["kwargs"] = kwargs
        np.testing.assert_allclose(matvec(np.asarray([3.0, -1.0])), np.asarray([9.0, -3.0]))
        return SimpleNamespace(metadata=SimpleNamespace(storage_kind="dense", reason="forced"))

    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_BLOCK_COLS", "7")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_DENSE_MAX_MB", "3.5")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB", "9.5")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_DROP_TOL", "1e-3")
    monkeypatch.setattr(sparse_direct, "build_operator_from_matvec", fake_build_operator_from_matvec)
    monkeypatch.setattr(
        sparse_direct,
        "factorize_host_sparse_operator",
        lambda bundle, *, kind, **kwargs: SimpleNamespace(bundle=bundle, kind=kind, kwargs=kwargs),
    )
    monkeypatch.setattr("sfincs_jax.problems.profile_sparse_direct.jax.default_backend", lambda: "gpu")

    sparse_direct.build_host_sparse_direct_factor_from_matvec(
        matvec=lambda x: 3.0 * x,
        n=2,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float32),
    )

    kwargs = seen["kwargs"]
    assert kwargs["dtype"] == np.dtype(np.float32)
    assert kwargs["backend"] == "gpu"
    assert kwargs["block_cols"] == 7
    assert kwargs["dense_max_mb"] == pytest.approx(3.5)
    assert kwargs["csr_max_mb"] == pytest.approx(9.5)
    assert kwargs["drop_tol"] == pytest.approx(1.0e-3)


def test_build_host_sparse_direct_factor_from_matvec_can_use_pattern_probe(monkeypatch) -> None:
    seen: dict[str, object] = {}
    pattern = sparse.eye(2, format="csr")

    def fake_build_operator_from_pattern(matvec, **kwargs):
        seen["kwargs"] = kwargs
        np.testing.assert_allclose(matvec(np.asarray([4.0, -2.0])), np.asarray([8.0, -4.0]))
        return SimpleNamespace(metadata=SimpleNamespace(storage_kind="csr", reason="pattern"))

    monkeypatch.setattr(sparse_direct, "build_operator_from_pattern", fake_build_operator_from_pattern)
    def fake_factorize_host_sparse_operator(bundle, *, kind, **kwargs):
        seen["factor_kwargs"] = kwargs
        return SimpleNamespace(bundle=bundle, kind=kind, kwargs=kwargs)

    monkeypatch.setattr(sparse_direct, "factorize_host_sparse_operator", fake_factorize_host_sparse_operator)
    monkeypatch.setattr("sfincs_jax.problems.profile_sparse_direct.jax.default_backend", lambda: "cpu")

    sparse_direct.build_host_sparse_direct_factor_from_matvec(
        matvec=lambda x: 2.0 * x,
        n=2,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float64),
        pattern=pattern,
        default_permc_spec="MMD_ATA",
    )

    kwargs = seen["kwargs"]
    assert kwargs["pattern"] is pattern
    assert kwargs["dtype"] == np.dtype(np.float64)
    assert kwargs["backend"] == "cpu"
    assert kwargs["allow_operator_only"] is False
    assert kwargs["color_batch"] == 1
    assert callable(kwargs["matmat"])
    assert seen["factor_kwargs"]["permc_spec"] == "MMD_ATA"


def test_build_host_sparse_direct_factor_from_matvec_default_ilu_and_env_override(monkeypatch) -> None:
    seen_kinds: list[str] = []

    monkeypatch.setattr(
        sparse_direct,
        "build_operator_from_matvec",
        lambda _matvec, **_kwargs: SimpleNamespace(metadata=SimpleNamespace(storage_kind="csr", reason="fake")),
    )

    def fake_factorize_host_sparse_operator(bundle, *, kind, **kwargs):
        seen_kinds.append(kind)
        return SimpleNamespace(bundle=bundle, kind=kind, kwargs=kwargs)

    monkeypatch.setattr(sparse_direct, "factorize_host_sparse_operator", fake_factorize_host_sparse_operator)
    monkeypatch.setattr("sfincs_jax.problems.profile_sparse_direct.jax.default_backend", lambda: "cpu")
    monkeypatch.delenv("SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND", raising=False)

    sparse_direct.build_host_sparse_direct_factor_from_matvec(
        matvec=lambda x: x,
        n=2,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float64),
        default_factor_kind="ilu",
    )

    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND", "lu")
    sparse_direct.build_host_sparse_direct_factor_from_matvec(
        matvec=lambda x: x,
        n=2,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float64),
        default_factor_kind="ilu",
    )

    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND", "jacobi")
    sparse_direct.build_host_sparse_direct_factor_from_matvec(
        matvec=lambda x: x,
        n=2,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float64),
        default_factor_kind="ilu",
    )

    assert seen_kinds == ["ilu", "lu", "jacobi"]


def test_rhsmode1_sparse_factor_policy_auto_avoids_large_monolithic_default() -> None:
    setup = profile_sparse_policy.resolve_sparse_pc_factor_policy(
        env={},
        constrained_pas_pc=True,
        tokamak_fp_pc=False,
        fortran_reduced_sparse_pc=False,
        sparse_pc_linear_size=469_321,
        pc_maxiter=8,
        default_permc_spec="COLAMD",
        host_sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float64),
    )

    assert setup.default_factor_kind == "symbolic_block_lu_coarse"
    assert setup.factorization == "symbolic_block_lu_coarse"


def test_rhsmode1_sparse_factor_policy_keeps_explicit_large_factor_override() -> None:
    setup = profile_sparse_policy.resolve_sparse_pc_factor_policy(
        env={"SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND": "native_nd_frontal_schur_lu"},
        constrained_pas_pc=True,
        tokamak_fp_pc=False,
        fortran_reduced_sparse_pc=False,
        sparse_pc_linear_size=469_321,
        pc_maxiter=8,
        default_permc_spec="COLAMD",
        host_sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float64),
    )

    assert setup.default_factor_kind == "lu"
    assert setup.factorization == "symbolic_nd_frontal_schur_lu"


def test_build_host_sparse_direct_factor_from_matvec_rejects_large_monolithic_factor(monkeypatch) -> None:
    operator_bundle = SimpleNamespace(
        metadata=SimpleNamespace(
            storage_kind="csr",
            reason="unit-large",
            shape=(4, 4),
            nnz_estimate=4,
            csr_nbytes_estimate=128,
        )
    )
    factorize_called = False

    monkeypatch.setattr(sparse_direct, "build_operator_from_matvec", lambda _matvec, **_kwargs: operator_bundle)

    def fake_factorize_host_sparse_operator(*_args, **_kwargs):
        nonlocal factorize_called
        factorize_called = True
        return SimpleNamespace()

    monkeypatch.setattr(sparse_direct, "factorize_host_sparse_operator", fake_factorize_host_sparse_operator)
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_MONOLITHIC_MAX_SIZE", "3")
    monkeypatch.setattr("sfincs_jax.problems.profile_sparse_direct.jax.default_backend", lambda: "cpu")
    messages: list[tuple[int, str]] = []

    with pytest.raises(MemoryError, match="monolithic factor preflight rejected"):
        sparse_direct.build_host_sparse_direct_factor_from_matvec(
            matvec=lambda x: x,
            n=4,
            dtype=jnp.float64,
            factor_dtype=np.dtype(np.float64),
            emit=lambda level, msg: messages.append((level, msg)),
        )

    assert not factorize_called
    assert any("monolithic factor preflight rejected factor_kind=lu" in msg for _, msg in messages)
    assert not any("factorization start" in msg for _, msg in messages)


def test_large_fortran_reduced_sparse_pc_defaults_to_ilu_but_env_override_wins(monkeypatch) -> None:
    size = 100_000
    op = SimpleNamespace(
        total_size=size,
        rhs_mode=1,
        include_phi1=False,
        constraint_scheme=1,
        phi1_size=0,
        n_xi=4,
        n_zeta=1,
        n_species=1,
        fblock=SimpleNamespace(
            fp=object(),
            pas=None,
            collisionless=SimpleNamespace(n_xi_for_x=np.asarray([4, 4], dtype=np.int32)),
        ),
    )
    factor_defaults: list[str] = []
    color_batch_defaults: list[int] = []

    monkeypatch.setattr(profile_solve, "rhs_v3_full_system", lambda _op: jnp.zeros((size,), dtype=jnp.float64))
    monkeypatch.setattr(
        profile_solve,
        "build_rhs1_active_dof_state",
        lambda **kwargs: SimpleNamespace(
            active_idx_jnp=None,
            full_to_active_jnp=None,
            active_size=int(kwargs["op"].total_size),
        ),
    )
    monkeypatch.setattr(
        profile_solve,
        "_build_rhsmode1_preconditioner_operator_fortran_reduced",
        lambda op_arg, **_kwargs: op_arg,
    )
    monkeypatch.setattr(
        profile_solve,
        "v3_full_system_fortran_reduced_preconditioner_sparsity_pattern",
        lambda *_args, **_kwargs: sparse.eye(1, format="csr"),
    )
    monkeypatch.setattr(
        profile_solve,
        "summarize_v3_sparse_pattern",
        lambda _op, _pattern: SimpleNamespace(nnz=1, avg_row_nnz=1.0, max_row_nnz=1),
    )
    monkeypatch.setattr(
        profile_solve,
        "apply_v3_full_system_operator_cached",
        lambda _op, x: jnp.zeros_like(x),
    )

    def fake_build_host_sparse_direct_factor_from_matvec(**kwargs):
        factor_defaults.append(kwargs["default_factor_kind"])
        color_batch_defaults.append(kwargs["default_pattern_color_batch"])
        return (
            SimpleNamespace(metadata=SimpleNamespace(storage_kind="csr", reason="fake")),
            SimpleNamespace(
                solve=lambda rhs: np.asarray(rhs, dtype=np.float64),
                factor_nbytes_estimate=None,
                factor_nnz_estimate=None,
            ),
        )

    def fake_gmres_solve_with_history_scipy(**kwargs):
        return np.zeros_like(np.asarray(kwargs["b"], dtype=np.float64)), 0.0, [0.0]

    monkeypatch.setattr(
        profile_solve,
        "_build_host_sparse_direct_factor_from_matvec",
        fake_build_host_sparse_direct_factor_from_matvec,
    )
    monkeypatch.setattr(profile_solve, "gmres_solve_with_history_scipy", fake_gmres_solve_with_history_scipy)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.delenv("SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND", raising=False)

    default_result = profile_solve.solve_v3_full_system_linear_gmres(
        nml=_FakeNamelist(),
        op=op,
        solve_method="fortran_reduced_pc_gmres",
        restart=2,
        maxiter=2,
    )

    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND", "lu")
    override_result = profile_solve.solve_v3_full_system_linear_gmres(
        nml=_FakeNamelist(),
        op=op,
        solve_method="fortran_reduced_pc_gmres",
        restart=2,
        maxiter=2,
    )

    assert factor_defaults == ["ilu", "ilu"]
    assert color_batch_defaults == [16, 16]
    assert default_result.metadata["sparse_pc_default_factorization"] == "ilu"
    assert default_result.metadata["sparse_pc_factorization"] == "ilu"
    assert default_result.metadata["sparse_pc_default_pattern_color_batch"] == 16
    assert default_result.metadata["sparse_pc_linear_size"] == size
    assert override_result.metadata["sparse_pc_default_factorization"] == "ilu"
    assert override_result.metadata["sparse_pc_factorization"] == "lu"
