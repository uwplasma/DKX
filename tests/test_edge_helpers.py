"""Degenerate inputs and error messages of the batch, direct-solve and memory helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkx import batch, solve_direct
from dkx.solve_direct import DirectFactors


class _Lu:
    closed = False

    def solve(self, b, trans="N"):
        return b / (2.0 if trans == "N" else 4.0)

    def close(self):
        self.closed = True


def _factors(**kw) -> DirectFactors:
    return DirectFactors(_Lu(), np.full(3, 2.0), np.full(3, 3.0), 3, 0, 3, **kw)


def test_memory_budget_falls_back_from_device_to_host_to_constant(monkeypatch) -> None:
    gb = batch._BYTES_PER_GB
    assert batch.resolve_memory_budget_bytes(2.0) == 2.0 * gb
    dev = SimpleNamespace(memory_stats=lambda: {"bytes_limit": 8 * gb})
    monkeypatch.setattr(jax, "devices", lambda: [dev])
    assert batch._device_memory_bytes() == 8 * gb
    assert batch.resolve_memory_budget_bytes() == batch._DEFAULT_BUDGET_FRACTION * 8 * gb
    for stats in ({}, {"bytes_limit": 0}, None):  # a backend without a limit
        monkeypatch.setattr(jax, "devices", lambda s=stats: [SimpleNamespace(memory_stats=lambda: s)])
        assert batch._device_memory_bytes() is None
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(memory_stats=lambda: 1 / 0)])
    assert batch._device_memory_bytes() is None
    monkeypatch.setattr("os.sysconf", lambda name: 1 << 30 if name == "SC_PAGE_SIZE" else 4)
    assert batch._host_memory_bytes() == 4.0 * gb
    assert batch.resolve_memory_budget_bytes() == batch._DEFAULT_BUDGET_FRACTION * 4.0 * gb
    monkeypatch.setattr("os.sysconf", lambda name: (_ for _ in ()).throw(ValueError(name)))
    assert batch._host_memory_bytes() is None
    assert batch.resolve_memory_budget_bytes() == batch._DEFAULT_BUDGET_FRACTION * batch._FALLBACK_BUDGET_GB * gb


def test_batch_axis_validation_messages() -> None:
    with pytest.raises(ValueError, match="empty"):
        batch._batch_size_of({})
    with pytest.raises(ValueError, match="no array leaves with a batch axis"):
        batch._batch_size_of({"a": 1.0})
    with pytest.raises(ValueError, match=r"sizes \[2, 3\]"):
        batch._batch_size_of({"a": jnp.zeros(2), "b": jnp.zeros(3)})
    assert batch._batch_size_of({"a": jnp.zeros((4, 2)), "b": 1.0}) == 4
    with pytest.raises(TypeError, match="mapping"):
        batch._validate_batch_leaves(None, [1])
    with pytest.raises(ValueError, match="not a batchable"):
        batch._validate_batch_leaves(None, {"no_such_leaf": 1})
    shared = sorted(batch._DISCRETIZATION_FIELDS & set(batch.KineticOperator._CHILD_FIELDS))[0]
    with pytest.raises(ValueError, match="must stay shared"):
        batch._validate_batch_leaves(None, {shared: 1})


def test_direct_factors_apply_scaling_and_release_storage() -> None:
    with _factors() as f:
        x = f.solve(np.ones((3, 2)))
        xt = f.solve(np.ones(3), transpose=True)
        assert x.shape == (3, 2) and xt.shape == (3,)
        np.testing.assert_allclose(x, 3.0 * (2.0 / 2.0) * np.ones((3, 2)))  # D_c * (D_r b / 2)
        np.testing.assert_allclose(xt, 2.0 * (3.0 / 4.0) * np.ones(3))  # D_r * (D_c b / 4)
        with pytest.raises(ValueError, match="size 3; this right-hand side has 5 rows"):
            f.solve(np.ones(5))
    assert f.lu.closed
    bare = DirectFactors(object(), np.ones(1), np.ones(1), 1, 0, 1)  # no close()/memory attributes
    bare.close()
    assert bare.symbolic_memory_bytes is None and bare.effective_memory_bytes is None


def test_rss_evidence_falls_back_to_the_peak_then_to_nothing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr("dkx.profiling._peak_rss_mb", lambda: 2.0)
    assert solve_direct._process_rss_bytes() == (2 << 20, "peak RSS fallback")
    monkeypatch.setattr("dkx.profiling._peak_rss_mb", lambda: None)
    assert solve_direct._process_rss_bytes() == (None, "RSS unavailable")
    with pytest.raises(RuntimeError, match="requires process RSS evidence"):
        solve_direct._admit_mumps_reuse(rhs_nbytes=8, memory_budget_bytes=1 << 30)


def test_direct_solve_refuses_tracing_and_mismatched_factors() -> None:
    kw = dict(tol=1e-10, atol=0.0, max_dense_size=10)
    op = SimpleNamespace(total_size=3)
    with pytest.raises(RuntimeError, match="non-differentiable"):
        jax.jit(lambda r: solve_direct._solve_tier3(op, r, **kw))(jnp.ones(3))
    with pytest.raises(ValueError, match="of size 3 and this operator is 4"):
        solve_direct._solve_tier3(SimpleNamespace(total_size=4), np.ones(4), factors=_factors(), **kw)
    with pytest.raises(ValueError, match="not requested direct_backend='mumps'"):
        solve_direct._solve_tier3(op, np.ones(3), factors=_factors(), direct_backend="mumps", **kw)
    with pytest.raises(ValueError, match="retains its original memory budget"):
        solve_direct._solve_tier3(op, np.ones(3), factors=_factors(memory_budget_bytes=100), direct_memory_budget_gb=1.0, **kw)


def test_solve_device_routing_on_an_accelerator_host(monkeypatch, capsys) -> None:
    from dkx import solve as solve_module

    op = SimpleNamespace(total_size=1)
    cpu = jax.local_devices(backend="cpu")[0]
    route = solve_module._resolve_solve_device
    assert route("cpu", "gmres", op, traced=True) is None  # inert under tracing
    assert route(cpu, "gmres", op, traced=False) is cpu
    assert route("default", "gmres", op, traced=False) is None
    assert route("cpu", "gmres", op, traced=False) is None  # already the default backend
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(solve_module, "_cpu_device_or_none", lambda: cpu)
    assert route("cpu", "gmres", op, traced=False) is cpu
    assert route("gpu", "gmres", op, traced=False) is None
    assert route("auto", "gmres", op, traced=False) is None  # Krylov stays on the accelerator by default
    monkeypatch.setenv(solve_module._SOLVE_CPU_MAX_TIER2_ENV, "10")
    assert route("auto", "gmres", op, traced=False) is cpu and "device route" in capsys.readouterr().out
    monkeypatch.setenv(solve_module._SOLVE_CPU_MAX_TIER1_ENV, "0")
    assert route("auto", "block_tridiagonal", op, traced=False) is None  # too big for the host route
    monkeypatch.setenv(solve_module._SOLVE_CPU_MAX_TIER1_ENV, "not-a-number")
    assert solve_module._env_size(solve_module._SOLVE_CPU_MAX_TIER1_ENV, 7) == 7
    with pytest.raises(ValueError, match="unknown solve device 'tpu2'"):
        route("tpu2", "gmres", op, traced=False)
    monkeypatch.setattr(solve_module, "_cpu_device_or_none", lambda: None)
    assert route("auto", "gmres", op, traced=False) is None
    with pytest.raises(ValueError, match="no CPU backend"):
        route("cpu", "gmres", op, traced=False)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(ValueError, match="default JAX backend is CPU"):
        route("gpu", "gmres", op, traced=False)


def test_device_probes_tolerate_missing_backends(monkeypatch) -> None:
    from dkx import solve as solve_module

    monkeypatch.setattr(jax, "local_devices", lambda backend=None: (_ for _ in ()).throw(RuntimeError(backend)))
    assert solve_module._cpu_device_or_none() is None
    assert solve_module._single_device_of(SimpleNamespace(devices=lambda: 1 / 0)) is None
    assert solve_module._single_device_of(SimpleNamespace(devices=lambda: {1, 2})) is None


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"method": "bogus"}, ValueError, "unknown method 'bogus'"),
        ({"tol": float("nan")}, ValueError, "tol must be finite"),
        ({"atol": -1.0}, ValueError, "atol must be finite"),
        ({"direct_backend": "pardiso"}, ValueError, "direct_backend must be"),
        ({"restart": 0}, ValueError, "restart must be None or a positive integer"),
        ({"direct_memory_budget_gb": 0.0}, ValueError, "finite and positive"),
        ({"transpose": True, "differentiable": True}, ValueError, "pass one or the other"),
        ({"method": "gmres", "transpose": True}, ValueError, "belong to the direct routes"),
        ({"method": "direct", "factors": object()}, TypeError, "DirectFactors"),
    ],
)
def test_solve_rejects_malformed_options_before_any_work(kwargs, error, message) -> None:
    from dkx.solve import solve

    op = SimpleNamespace(total_size=3)
    with pytest.raises(error, match=message):
        solve(op, np.ones(3), **kwargs)


def test_solve_rejects_a_right_hand_side_of_the_wrong_length() -> None:
    from dkx.solve import solve

    with pytest.raises(ValueError, match="rhs has 2 rows; operator expects 3"):
        solve(SimpleNamespace(total_size=3), np.ones(2), method="direct")


def _export_inputs(n_theta=4, n_zeta=3, **group):
    from dkx import writer

    grids = SimpleNamespace(theta=np.linspace(0, 2 * np.pi, n_theta, endpoint=False),
                            zeta=np.linspace(0, np.pi, n_zeta, endpoint=False), x=np.array([0.5, 1.0, 2.0]), n_xi=3)
    groups = {"export_f": {"EXPORT_FULL_F": True, **group}, "otherNumericalParameters": {}}
    return writer, SimpleNamespace(group=groups.__getitem__), grids, SimpleNamespace(n_periods=2)


def test_export_f_axis_maps_wrap_interpolate_and_select() -> None:
    writer, raw, grids, geom = _export_inputs(EXPORT_F_THETA_OPTION=1, EXPORT_F_ZETA_OPTION=1, EXPORT_F_THETA=[-1e-20, 1.0],
                                              EXPORT_F_ZETA=[0.5, 1.4], EXPORT_F_X_OPTION=2, EXPORT_F_X=[0.6, 1.9], EXPORT_F_XI_OPTION=0)
    cfg = writer._export_f_config(raw=raw, grids=grids, geom=geom)
    for name in ("map_theta", "map_zeta"):
        np.testing.assert_allclose(getattr(cfg, name).sum(axis=1), 1.0)  # interpolation weights partition unity
    assert cfg.map_x.shape == (2, 3) and cfg.map_xi.shape == (3, 3)
    nearest = _export_inputs(EXPORT_F_THETA_OPTION=2, EXPORT_F_ZETA_OPTION=2, EXPORT_F_THETA=[0.1, 3.0], EXPORT_F_ZETA=[0.2])
    cfg = writer._export_f_config(raw=nearest[1], grids=nearest[2], geom=nearest[3])
    assert cfg.n_export_theta == 2 and cfg.n_export_zeta == 1 and cfg.n_export_xi == 1
    assert writer._export_f_config(raw=SimpleNamespace(group=lambda n: {}), grids=grids, geom=geom) is None
    assert writer._export_f_int({"K": []}, "k", 7) == 7 and writer._export_f_float({"K": []}, "k", 2.5) == 2.5


@pytest.mark.parametrize(
    ("group", "error", "match"),
    [
        ({"EXPORT_F_THETA_OPTION": 9}, ValueError, "export_f_theta_option"),
        ({"EXPORT_F_ZETA_OPTION": 9}, ValueError, "export_f_zeta_option"),
        ({"EXPORT_F_X_OPTION": 9}, ValueError, "export_f_x_option"),
        ({"EXPORT_F_XI_OPTION": 9}, ValueError, "export_f_xi_option"),
    ],
)
def test_export_f_rejects_unknown_axis_options(group, error, match) -> None:
    writer, raw, grids, geom = _export_inputs(**group)
    with pytest.raises(error, match=match):
        writer._export_f_config(raw=raw, grids=grids, geom=geom)
    with pytest.raises(ValueError, match="n_l must be >= 1"):
        writer._export_f_legendre_matrix(np.zeros(2), n_l=0)
    assert writer._export_f_legendre_matrix(np.zeros(2), n_l=1).shape == (2, 1)


def test_export_f_x_interpolation_needs_a_supported_speed_grid() -> None:
    writer, raw, grids, geom = _export_inputs(EXPORT_F_X_OPTION=1)
    raw = SimpleNamespace(group=lambda n: {"EXPORT_F_X_OPTION": 1, "EXPORT_FULL_F": True} if n == "export_f" else {"XGRIDSCHEME": 3})
    with pytest.raises(NotImplementedError, match=r"xGridScheme in \{1,2,5,6\} \(got 3\)"):
        writer._export_f_config(raw=raw, grids=grids, geom=geom)


def test_boozer_bc_readers_name_the_problem(tmp_path) -> None:
    from dkx import boozer_bc as bc

    assert bc._try_parse_floats(["1", "x"], 2) is None and bc._try_parse_floats(["1"], 2) is None
    assert bc._try_parse_ints(["1", "x"], 2) is None and bc._try_parse_ints(["1"], 2) is None
    assert bc._try_parse_floats(["1D2", "3d0"], 2) == [100.0, 3.0]
    with pytest.raises(ValueError, match="too short"):
        bc._parse_bc_header_line("1 2 3")
    with pytest.raises(ValueError, match="no psi_a and a"):
        bc._parse_bc_header_line("1 2 3 4 5 words")
    assert bc._parse_bc_header_line("1 2 3 4 5.0 6.0 trailing free text")[1] == [5.0, 6.0]

    with pytest.raises(ValueError, match="geometry_scheme must be 11 or 12"):
        bc.read_boozer_bc(tmp_path, geometry_scheme=3)
    with pytest.raises(ValueError, match="Could not read Boozer equilibrium"):
        bc.read_boozer_bc(tmp_path, geometry_scheme=11)  # a directory exists but cannot be read
    with pytest.raises(ValueError, match="Could not read Boozer equilibrium"):
        bc.read_native_boozer(tmp_path)
    junk = tmp_path / "junk.bc"
    junk.write_text("not a boozer file\n")
    with pytest.raises(ValueError, match="Could not detect Boozer .bc column convention"):
        bc.read_native_boozer(junk)


def test_bracketing_surfaces_and_the_snap_to_surface_radius(monkeypatch) -> None:
    from dkx import boozer_bc as bc

    surf = lambda r: SimpleNamespace(r_n=r)  # noqa: E731
    with pytest.raises(ValueError, match="bracketing rN_wish=0.9"):
        bc._bracketing_surfaces((surf(0.2), surf(0.5)), 0.9)
    monkeypatch.setattr(bc, "read_boozer_bc", lambda *a, **kw: (None, (surf(0.5), surf(0.5))))
    assert bc.selected_r_n_from_bc(path="x", geometry_scheme=11, r_n_wish=0.4) == 0.5


_PHI1_NEEDS_KINETIC = {"INCLUDEPHI1": True, "INCLUDEPHI1INKINETICEQUATION": False, "INCLUDEPHI1INCOLLISIONOPERATOR": True}


@pytest.mark.parametrize(
    ("group", "settings", "error", "match"),
    [
        ("physicsparameters", _PHI1_NEEDS_KINETIC, NotImplementedError, "requires includePhi1InKineticEquation"),
        ("physicsparameters", {"INCLUDEPHI1": True, "QUASINEUTRALITYOPTION": 3}, NotImplementedError, "quasineutralityOption=3"),
        ("physicsparameters", {"COLLISIONOPERATOR": 2}, NotImplementedError, "collisionOperator=2 is not supported"),
        ("physicsparameters", {"CONSTRAINTSCHEME": 7}, NotImplementedError, "constraintScheme=7 is not supported"),
        ("othernumericalparameters", {"XGRIDSCHEME": 9}, ValueError, "xGridScheme must be between 1 and 8"),
        ("geometryparameters", {"GEOMETRYSCHEME": 99}, NotImplementedError, "geometryScheme in"),
        ("geometryparameters", {"GEOMETRYSCHEME": 13}, ValueError, "at least one boozer_bmnc"),
        ("geometryparameters", {"GEOMETRYSCHEME": 11}, ValueError, "requires equilibriumFile"),
    ],
)
def test_unsupported_namelist_choices_are_refused_by_name(group, settings, error, match) -> None:
    from pathlib import Path

    from dkx.drift_kinetic import KineticOperator
    from dkx.namelist import read_sfincs_input

    nml = read_sfincs_input(Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    nml.groups[group].update(settings)
    with pytest.raises(error, match=match):
        KineticOperator.from_namelist(nml)


def test_namelist_value_helpers_treat_an_empty_list_as_the_default() -> None:
    from dkx import operator_build as ob

    empty = {"K": []}
    assert ob._get_int(empty, "k", 3) == 3 and ob._get_float(empty, "k", 1.5) == 1.5 and ob._get_bool(empty, "k", True) is True
    assert ob._get_str_or_none(empty, "k") is None and ob._get_str_or_none({"K": [" a "]}, "k") == "a"
    assert ob._get_str_or_none({"K": "  "}, "k") is None
