import jax.numpy as jnp

from sfincs_jax.problems.transport_diagnostics import transport_matrix_size_from_rhs_mode
from sfincs_jax.problems.transport_setup import (
    resolve_transport_maxiter_setup,
    resolve_transport_parallel_request,
    resolve_transport_state_setup,
    resolve_transport_which_rhs_setup,
)


def test_resolve_transport_maxiter_setup_defaults_and_overrides() -> None:
    default = resolve_transport_maxiter_setup(400, maxiter_env="")
    assert default.maxiter == 400
    assert default.notes == ()

    clamped = resolve_transport_maxiter_setup(None, maxiter_env="0")
    assert clamped.maxiter == 1
    assert clamped.notes
    assert "maxiter override=1" in clamped.notes[0][1]

    invalid = resolve_transport_maxiter_setup(250, maxiter_env="not-an-int")
    assert invalid.maxiter == 250
    assert invalid.notes
    assert "ignoring invalid" in invalid.notes[0][1]


def test_resolve_transport_state_setup_uses_state_when_no_explicit_guess() -> None:
    x_full = jnp.array([1.0, 2.0, 3.0])
    x_rhs = {1: jnp.array([4.0, 5.0])}

    def load_state(**kwargs):
        assert kwargs["path"] == "state.npz"
        assert kwargs["op"] == "op"
        return {"x_full": x_full, "x_by_rhs": x_rhs}

    setup = resolve_transport_state_setup(
        op="op",
        x0=None,
        x0_by_rhs=None,
        state_in_env="state.npz",
        state_out_env="state-out.npz",
        load_state=load_state,
    )

    assert setup.state_in_path == "state.npz"
    assert setup.state_out_path == "state-out.npz"
    assert setup.x0 is x_full
    assert setup.x0_by_rhs is x_rhs
    assert setup.state_x_by_rhs is x_rhs


def test_resolve_transport_state_setup_preserves_explicit_guesses_and_ignores_loader_failures() -> None:
    x_full = jnp.array([1.0])
    x_rhs = {2: jnp.array([2.0])}
    state_rhs = {1: jnp.array([3.0])}

    def load_state(**_kwargs):
        return {"x_full": jnp.array([4.0]), "x_by_rhs": state_rhs}

    setup = resolve_transport_state_setup(
        op=object(),
        x0=x_full,
        x0_by_rhs=x_rhs,
        state_in_env="state.npz",
        load_state=load_state,
    )

    assert setup.x0 is x_full
    assert setup.x0_by_rhs is x_rhs
    assert setup.state_x_by_rhs is state_rhs

    def broken_loader(**_kwargs):
        raise RuntimeError("bad checkpoint")

    broken = resolve_transport_state_setup(
        op=object(),
        x0=None,
        x0_by_rhs=None,
        state_in_env="broken.npz",
        state_out_env="out.npz",
        load_state=broken_loader,
    )
    assert broken.state_in_path == "broken.npz"
    assert broken.state_out_path == "out.npz"
    assert broken.x0 is None
    assert broken.x0_by_rhs is None
    assert broken.state_x_by_rhs is None


def test_resolve_transport_which_rhs_setup_normalizes_subset() -> None:
    n_rhs = transport_matrix_size_from_rhs_mode(2)

    all_rhs = resolve_transport_which_rhs_setup(rhs_mode=2, which_rhs_values=None)
    assert all_rhs.rhs_mode == 2
    assert all_rhs.n_rhs == n_rhs
    assert all_rhs.which_rhs_values == list(range(1, n_rhs + 1))
    assert not all_rhs.subset_mode

    subset = resolve_transport_which_rhs_setup(
        rhs_mode=2,
        which_rhs_values=[n_rhs, -1, 1, n_rhs + 1, 1],
    )
    assert subset.which_rhs_values == [1, n_rhs]
    assert subset.subset_mode == (n_rhs > 2)


def test_resolve_transport_parallel_request_cpu_policy() -> None:
    disabled = resolve_transport_parallel_request(
        which_rhs_count=3,
        n_rhs=3,
        parallel_workers=None,
        parallel_backend="cpu",
        visible_gpu_ids=lambda _workers: [],
        parallel_env="",
        workers_env="",
        cpu_count=8,
    )
    assert disabled.parallel_workers == 1
    assert disabled.parallel_backend == "cpu"

    auto = resolve_transport_parallel_request(
        which_rhs_count=2,
        n_rhs=3,
        parallel_workers=None,
        parallel_backend="cpu",
        visible_gpu_ids=lambda _workers: [],
        parallel_env="auto",
        workers_env="",
        cpu_count=8,
    )
    assert auto.parallel_workers == 2

    invalid_workers = resolve_transport_parallel_request(
        which_rhs_count=3,
        n_rhs=3,
        parallel_workers=None,
        parallel_backend="cpu",
        visible_gpu_ids=lambda _workers: [],
        parallel_env="true",
        workers_env="bad",
        cpu_count=2,
    )
    assert invalid_workers.parallel_workers == 2

    capped = resolve_transport_parallel_request(
        which_rhs_count=2,
        n_rhs=3,
        parallel_workers=99,
        parallel_backend="cpu",
        visible_gpu_ids=lambda _workers: [],
    )
    assert capped.parallel_workers == 2


def test_resolve_transport_parallel_request_gpu_policy_and_child_flag() -> None:
    one_gpu = resolve_transport_parallel_request(
        which_rhs_count=3,
        n_rhs=3,
        parallel_workers=3,
        parallel_backend="gpu",
        visible_gpu_ids=lambda _workers: ["0"],
        parallel_child_env="yes",
    )
    assert one_gpu.parallel_child
    assert one_gpu.parallel_workers == 1
    assert one_gpu.parallel_backend == "cpu"

    two_gpus = resolve_transport_parallel_request(
        which_rhs_count=3,
        n_rhs=3,
        parallel_workers=3,
        parallel_backend="gpu",
        visible_gpu_ids=lambda _workers: ["0", "1"],
    )
    assert two_gpus.parallel_workers == 2
    assert two_gpus.parallel_backend == "gpu"
