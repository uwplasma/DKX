"""Differentiable neoclassical transport solvers and SFINCS-style outputs in JAX.

The public CLI and Python APIs are maintained as standalone research tools while
retaining release-gated comparisons against SFINCS Fortran v3 for trust building.
"""

from __future__ import annotations

# Enable host-device parallelism and a default JAX compilation cache for repeated
# CLI invocations unless the user explicitly disables it. This improves cold-start
# performance without requiring environment configuration.
import os
import tempfile

# Suppress low-value XLA/PjRt C++ warning chatter by default. Users can still
# override this before importing dkx if they need backend debug logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_distributed_runtime_initialized = False


def initialize_distributed_runtime_from_env() -> bool:
    """Best-effort JAX multi-host bootstrap from DKX_* env vars.

    This helper is called at import time for env-driven workflows and again by the
    CLI after parsing explicit multi-host flags. Repeated calls are safe.
    """
    global _distributed_runtime_initialized
    if _distributed_runtime_initialized:
        return True

    distributed_env = os.environ.get("DKX_DISTRIBUTED", "").strip().lower()
    if distributed_env not in {"1", "true", "yes", "on"}:
        return False

    try:
        import jax.distributed as _jax_distributed  # noqa: PLC0415

        process_id_env = os.environ.get("DKX_PROCESS_ID", "").strip()
        process_count_env = os.environ.get("DKX_PROCESS_COUNT", "").strip()
        coord_addr = os.environ.get("DKX_COORDINATOR_ADDRESS", "").strip()
        coord_port_env = os.environ.get("DKX_COORDINATOR_PORT", "").strip()

        process_id = int(process_id_env) if process_id_env else 0
        process_count = int(process_count_env) if process_count_env else 1
        coord_port = int(coord_port_env) if coord_port_env else 1234

        if not coord_addr:
            return False

        _jax_distributed.initialize(
            coordinator_address=coord_addr,
            coordinator_port=coord_port,
            num_processes=process_count,
            process_id=process_id,
        )
        _distributed_runtime_initialized = True
        return True
    except Exception:
        # Best-effort: avoid hard failures when distributed runtime is unavailable.
        return False


# Optional JAX multi-host bootstrap (must run before any JAX device use).
initialize_distributed_runtime_from_env()

# High-level cores knob (DKX_CORES / CLI --cores): pin the XLA host CPU
# threadpool.  XLA sizes its eigen threadpool from the NPROC environment
# variable, read once when the CPU backend initializes, so this must run before
# the first jax import (imports below).  Semantics:
#
#   DKX_CORES=N (N > 0)  pin the solver threadpool to N threads (NPROC), and
#                        default the host BLAS pools (OMP/OpenBLAS) to match;
#   DKX_CORES=0          let XLA size the threadpool itself (full width);
#   unset                clamp to min(8, os.cpu_count()) unless NPROC is
#                        already set: the measured optimum is 4-8 threads on
#                        8-36-core hosts, and a full-width threadpool on a
#                        many-core box is several times slower than 8 threads
#                        (docs/performance.rst).
#
# Forcing multiple host *devices* is a separate, test-oriented concern: it is
# available only through an explicit DKX_CPU_DEVICES (below) and has no
# measured benefit for solves (all forced host devices share one threadpool).
_cores_env = os.environ.get("DKX_CORES", "").strip()
if _cores_env:
    try:
        _cores_val = int(_cores_env)
    except ValueError:
        _cores_val = None  # invalid value: fail closed, change nothing
    if _cores_val is not None and _cores_val > 0:
        os.environ["NPROC"] = str(_cores_val)
        os.environ.setdefault("OMP_NUM_THREADS", str(_cores_val))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_cores_val))
        os.environ.pop("_DKX_NPROC_DEFAULTED", None)
    elif _cores_val == 0:
        # Explicit "let XLA size the threadpool": undo a default clamp
        # inherited from a parent dkx process (the sentinel marks our own
        # clamp, never a user-set NPROC).
        if os.environ.pop("_DKX_NPROC_DEFAULTED", None):
            os.environ.pop("NPROC", None)
else:
    if "NPROC" not in os.environ:
        os.environ["NPROC"] = str(min(8, os.cpu_count() or 1))
        os.environ["_DKX_NPROC_DEFAULTED"] = "1"

# Explicit opt-in: force multiple host CPU devices (JAX SPMD / multi-device
# tests).  Must be set before importing JAX.  This is never derived from
# DKX_CORES — forced host devices share one threadpool, so device forcing does
# not speed up solves; thread control is the DKX_CORES/NPROC path above.
_cpu_devices_env = os.environ.get("DKX_CPU_DEVICES", "").strip()
if _cpu_devices_env:
    try:
        _cpu_devices = int(_cpu_devices_env)
    except ValueError:
        _cpu_devices = 0
    if _cpu_devices > 0:
        _xla_flags = os.environ.get("XLA_FLAGS", "")
        if "--xla_force_host_platform_device_count" not in _xla_flags:
            flag = f"--xla_force_host_platform_device_count={_cpu_devices}"
            os.environ["XLA_FLAGS"] = f"{_xla_flags} {flag}".strip()

_disable_cache = os.environ.get("DKX_DISABLE_COMPILATION_CACHE", "").strip().lower()
if _disable_cache not in {"1", "true", "yes", "on"}:
    if not os.environ.get("JAX_COMPILATION_CACHE_DIR", "").strip():
        def _is_writable_dir(path: str) -> bool:
            try:
                test_path = os.path.join(path, ".dkx_write_test")
                with open(test_path, "wb") as f:
                    f.write(b"")
                os.remove(test_path)
                return True
            except OSError:
                return False

        cache_override = os.environ.get("DKX_COMPILATION_CACHE_DIR", "").strip()
        if cache_override:
            default_cache_dir = cache_override
        else:
            xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
            if xdg_cache:
                default_cache_dir = os.path.join(xdg_cache, "dkx", "jax_compilation_cache")
            else:
                default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "dkx", "jax_compilation_cache")
        try:
            os.makedirs(default_cache_dir, exist_ok=True)
        except OSError:
            default_cache_dir = os.path.join(tempfile.gettempdir(), "dkx", "jax_compilation_cache")
            try:
                os.makedirs(default_cache_dir, exist_ok=True)
            except OSError:
                default_cache_dir = ""
        if default_cache_dir and (not _is_writable_dir(default_cache_dir)):
            # Some environments (CI sandboxes, read-only homes) can create the directory but
            # cannot write compilation entries. Fall back to a tempdir cache to avoid noisy
            # warnings and degraded cold-start performance.
            default_cache_dir = os.path.join(tempfile.gettempdir(), "dkx", "jax_compilation_cache")
            try:
                os.makedirs(default_cache_dir, exist_ok=True)
            except OSError:
                default_cache_dir = ""
            if default_cache_dir and (not _is_writable_dir(default_cache_dir)):
                default_cache_dir = ""
        if default_cache_dir:
            os.environ["JAX_COMPILATION_CACHE_DIR"] = default_cache_dir
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

# SFINCS parity fixtures and most scientific use-cases rely on float64 accuracy.
# Set this as early as possible on package import.
try:
    from jax import config as _jax_config  # noqa: PLC0415

    _jax_config.update("jax_enable_x64", True)
    # Enable the persistent compilation cache via the current jax config API.
    # The JAX_COMPILATION_CACHE_DIR env var set above only takes effect if jax
    # reads its config for the first time here; when the user imported jax
    # before dkx that ordering is already lost, so set the flags
    # explicitly (works regardless of import order).  The retired
    # jax.experimental.compilation_cache.set_cache_dir was removed in recent jax
    # (e.g. 0.10.x) and silently no-ops, so it must not be relied on.  Forcing
    # the min-compile-time / min-entry-size thresholds to zero makes even the
    # tiny fast-compiling kernels cacheable.
    _cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "").strip()
    if _cache_dir:
        _jax_config.update("jax_compilation_cache_dir", _cache_dir)
        # Mirror the thresholds set above as env-var defaults (respecting any
        # explicit user override) via config so they also apply when jax was
        # imported before dkx and never read the env vars.
        try:
            _jax_config.update(
                "jax_persistent_cache_min_compile_time_secs",
                float(os.environ.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")),
            )
            _jax_config.update(
                "jax_persistent_cache_min_entry_size_bytes",
                int(os.environ.get("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")),
            )
        except ValueError:
            pass
except Exception:
    # Keep import lightweight for tooling that inspects the package without JAX.
    pass

from .api import (  # noqa: E402
    BenchmarkReport,
    GeometryState,
    GridState,
    OperatorState,
    OutputSchema,
    PreconditionerState,
    SolveInputs,
    SolverOptions,
    SolverResult,
    TransportResult,
    batched_er_scan,
    read_output,
    run_ambipolar_brent,
    run_monoenergetic_database,
    write_output,
)
from .inputs import SfincsInput, load_sfincs_input  # noqa: E402

# Heavy flagship entry points (they import the JAX solve stack) are exported
# lazily via PEP 562 module __getattr__ so `import dkx` stays cheap.
_LAZY_EXPORTS = {
    "run_profile": ("dkx.run", "run_profile"),
    "run_transport_matrix": ("dkx.run", "run_transport_matrix"),
    "run_from_namelist": ("dkx.run", "run_from_namelist"),
    "batched_solve": ("dkx.batch", "batched_solve"),
    "monoenergetic_database": ("dkx.monoenergetic", "monoenergetic_database"),
    "ambipolar_er": ("dkx.er", "ambipolar_er"),
    "find_ambipolar_er": ("dkx.er", "find_ambipolar_er"),
    "classical_impurity_flux": ("dkx.impurity", "classical_impurity_flux"),
    "build_impurity_plasma": ("dkx.impurity", "build_impurity_plasma"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib  # noqa: PLC0415

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value  # cache: subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "BenchmarkReport",
    "GeometryState",
    "GridState",
    "OperatorState",
    "OutputSchema",
    "PreconditionerState",
    "SfincsInput",
    "SolveInputs",
    "SolverOptions",
    "SolverResult",
    "TransportResult",
    "__version__",
    "ambipolar_er",
    "batched_er_scan",
    "batched_solve",
    "build_impurity_plasma",
    "classical_impurity_flux",
    "find_ambipolar_er",
    "initialize_distributed_runtime_from_env",
    "load_sfincs_input",
    "monoenergetic_database",
    "read_output",
    "run_ambipolar_brent",
    "run_from_namelist",
    "run_monoenergetic_database",
    "run_profile",
    "run_transport_matrix",
    "write_output",
]

__version__ = "2.0.0"
