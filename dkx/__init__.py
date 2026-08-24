"""Differentiable neoclassical transport solvers and SFINCS-style outputs in JAX.

The public CLI and Python APIs are maintained as standalone research tools while
retaining release-gated comparisons against SFINCS Fortran v3 for trust building.
"""

from __future__ import annotations

# Enable host-device parallelism and a default JAX compilation cache for repeated
# CLI invocations unless the user explicitly disables it. This improves cold-start
# performance without requiring environment configuration.
import os
import sys as _sys
import types as _types
import tempfile

# Suppress low-value XLA/PjRt C++ warning chatter by default. Users can still
# override this before importing dkx if they need backend debug logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _check_numpy() -> None:
    """Fail with the actual problem instead of a NameError from numpy internals.

    ``jax`` requires ``numpy>=2.1``.  Installing dkx into an environment that
    holds a conda-managed numpy 1.x -- the Anaconda base environment is the
    common case -- resolves without complaint and then dies on the first
    ``import jax`` with::

        File numpy/core/getlimits.py, line 606, in smallest_normal
        NameError: name 'isnan' is not defined

    raised while ``ml_dtypes`` probes ``bfloat16``'s ``finfo``.  That traceback
    names neither dkx, nor jax, nor numpy's version as the cause, so it costs a
    user real time.  Checking here is cheap: numpy is imported either way, and
    this runs before any jax import.
    """
    import numpy  # noqa: PLC0415

    major = int(numpy.__version__.split(".", 1)[0])
    if major < 2:
        raise ImportError(
            f"dkx needs numpy>=2.1 (jax's own requirement); found {numpy.__version__}. "
            "numpy 1.x crashes inside ml_dtypes at 'import jax'. Fix the "
            "environment with:\n"
            "    pip install -U 'numpy>=2.1' jax\n"
            "or, better, install into a clean environment rather than the "
            "Anaconda base one:\n"
            "    conda create -n dkx python=3.11 -y && conda activate dkx\n"
            "    pip install dkx"
        )


_check_numpy()

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

        # Versioned because the cache is bounded now (see the cap below) and
        # JAX's LRU keeps a sidecar "-atime" file per entry.  A directory
        # filled in before the cap existed has none, so every eviction pass
        # tries to touch a file that was never written and warns -- measured,
        # 12 warnings on a single small solve against a legacy cache, and 0
        # against a fresh one.  Starting a new directory is what makes the
        # bound work; the old one simply stops being written to.
        _CACHE_DIR_NAME = "jax_compilation_cache_v2"
        _LEGACY_CACHE_DIR_NAME = "jax_compilation_cache"

        cache_override = os.environ.get("DKX_COMPILATION_CACHE_DIR", "").strip()
        if cache_override:
            default_cache_dir = cache_override
        else:
            xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
            if xdg_cache:
                default_cache_dir = os.path.join(xdg_cache, "dkx", _CACHE_DIR_NAME)
            else:
                default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "dkx", _CACHE_DIR_NAME)
        try:
            os.makedirs(default_cache_dir, exist_ok=True)
        except OSError:
            default_cache_dir = os.path.join(tempfile.gettempdir(), "dkx", _CACHE_DIR_NAME)
            try:
                os.makedirs(default_cache_dir, exist_ok=True)
            except OSError:
                default_cache_dir = ""
        if default_cache_dir and (not _is_writable_dir(default_cache_dir)):
            # Some environments (CI sandboxes, read-only homes) can create the directory but
            # cannot write compilation entries. Fall back to a tempdir cache to avoid noisy
            # warnings and degraded cold-start performance.
            default_cache_dir = os.path.join(tempfile.gettempdir(), "dkx", _CACHE_DIR_NAME)
            try:
                os.makedirs(default_cache_dir, exist_ok=True)
            except OSError:
                default_cache_dir = ""
            if default_cache_dir and (not _is_writable_dir(default_cache_dir)):
                default_cache_dir = ""
        if default_cache_dir:
            os.environ["JAX_COMPILATION_CACHE_DIR"] = default_cache_dir
            # The pre-cap directory is orphaned by the rename above.  It is
            # only a cache, but it can be large (782 MB / 64k entries on the
            # machine that prompted the cap), so say so once with the path
            # instead of leaving it to be found by accident.
            _legacy = os.path.join(
                os.path.dirname(default_cache_dir), _LEGACY_CACHE_DIR_NAME
            )
            if os.path.isdir(_legacy):
                try:
                    _legacy_bytes = sum(
                        entry.stat().st_size
                        for entry in os.scandir(_legacy)
                        if entry.is_file()
                    )
                except OSError:
                    _legacy_bytes = 0
                if _legacy_bytes > 100 * 1024**2:
                    import warnings as _warnings  # noqa: PLC0415

                    _warnings.warn(
                        f"dkx's compilation cache moved to {default_cache_dir}; "
                        f"the old unbounded one is still on disk "
                        f"({_legacy_bytes / 1024**3:.1f} GB) and is safe to "
                        f"delete:  rm -rf {_legacy}",
                        stacklevel=2,
                    )
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

# float64 is a correctness requirement, not a preference: the block
# eliminations and the parity fixtures both depend on it, and a solver that
# quietly ran in single precision would be worse than one that refused to
# start.  This is the *only* place in the package that sets it -- sixteen other
# modules used to do it at module scope, which made a global, invisible,
# import-order-dependent change to any process that merely touched dkx.
#
# ``DKX_NO_X64_SETUP=1`` opts out, for a caller who manages JAX precision
# themselves.  Opting out does not opt into wrong answers:
# :func:`require_float64` is called by the solve entry points and raises with
# the fix in the message.
try:
    from jax import config as _jax_config  # noqa: PLC0415

    if os.environ.get("DKX_NO_X64_SETUP", "").strip() not in {"1", "true", "yes"}:
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
        # Bound the cache.  The thresholds above are deliberately zero so even
        # tiny kernels are cached, which means every distinct grid a scan
        # touches leaves an entry behind and nothing ever removes one.  Left
        # uncapped this reached 782 MB across 64k files on a development
        # machine -- a slow leak into the user's home directory that no run
        # ever reports.  A cap makes JAX evict least-recently-used entries
        # instead.  Set DKX_COMPILATION_CACHE_MAX_BYTES=0 to disable the bound.
        try:
            _cache_cap = int(
                os.environ.get("DKX_COMPILATION_CACHE_MAX_BYTES", str(4 * 1024**3))
            )
            if _cache_cap > 0:
                # jax raises "Please install the `filelock` package to set
                # jax_compilation_cache_max_size" -- and it raises it later, on
                # every cache read, not here.  Setting the cap without filelock
                # therefore disables the cache instead of bounding it, which is
                # strictly worse than leaving it unbounded.  filelock is a
                # declared dependency; this guard is for an environment that
                # somehow lacks it.
                import filelock  # noqa: F401, PLC0415

                _jax_config.update("jax_compilation_cache_max_size", _cache_cap)
        except (ValueError, AttributeError, ImportError):
            # Older jax without the knob: an unbounded cache still works, it
            # just grows, so this must not stop dkx from importing.
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
    "plot": ("dkx.plotting", "plot"),
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
    if name == "run":
        return _lazy_run_module()
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


# ``dkx/run.py`` is a module and ``dkx.run(case)`` is a call.  Rather than pick
# one -- the function breaks ``dkx.run.run_profile`` and every monkeypatch that
# targets it by path, the module breaks the call -- ``dkx/run.py`` makes itself
# callable, so ``dkx.run`` is always the module and always invocable.  The name
# is resolved here rather than listed in _LAZY_EXPORTS because what it yields
# is the module itself, not an attribute of one.
def _lazy_run_module():
    import importlib  # noqa: PLC0415

    module = importlib.import_module(f"{__name__}.run")
    globals()["run"] = module
    return module



def require_float64() -> None:
    """Raise unless JAX is in float64 mode.

    Importing :mod:`dkx` enables it; a caller who set ``DKX_NO_X64_SETUP`` has
    taken that job on, and this is where they find out if they dropped it.  The
    check is a dtype probe rather than a config read because the config can be
    set and then overridden, and what matters is the dtype arrays actually get.
    """
    import jax.numpy as _jnp  # noqa: PLC0415

    if _jnp.zeros(1).dtype != _jnp.float64:
        raise RuntimeError(
            "dkx requires JAX float64: the block eliminations and every parity "
            "fixture depend on it, and single precision changes which results "
            "are trustworthy rather than merely how accurate they are. "
            "Enable it with jax.config.update('jax_enable_x64', True) before "
            "the first array is created, or JAX_ENABLE_X64=1 in the "
            "environment, or unset DKX_NO_X64_SETUP and let dkx set it."
        )

__all__ = [
    "require_float64",
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
    "plot",
    "run",
    "run_profile",
    "run_transport_matrix",
    "write_output",
]

__version__ = "2.3.0"
