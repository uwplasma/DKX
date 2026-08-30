"""Explicit runtime configuration for DKX.

Importing :mod:`dkx` used to do all of this: validate numpy, mutate seven
environment variables, bootstrap a JAX distributed runtime, size the XLA host
threadpool, create a compilation-cache directory, import JAX, and switch on
float64. That made ``import dkx`` an irreversible, order-dependent change to
any process that merely touched the package, and it happened whether or not the
caller ever solved anything.

The same work now lives here behind one idempotent call:

    import dkx
    dkx.runtime.configure()

The CLI calls it during bootstrap. The solve entry points call it too, so a
caller who only ever writes ``dkx.run(case)`` gets identical numerics to before
without knowing this module exists -- what changed is that *importing* the
package no longer does it. An embedded user who manages JAX themselves can pass
``jax_x64=False`` or simply never call this, and
:func:`dkx.require_float64` still refuses to let a solve run in single
precision rather than silently returning worse answers.

Every knob keeps the environment-variable name it had, because those names are
in user scripts, job files, and the CI workflows.

**Why the call sites look the way they do.** Every module that imports the JAX
backend calls ``configure()`` immediately above its ``import jax``. That is not
decoration: XLA reads ``NPROC``, ``XLA_FLAGS``, and the compilation-cache
directory once, when the CPU backend initialises, and ``jax_enable_x64`` has to
be set before the first array exists. Configuring after the import would be a
no-op that looked like it worked. ``configure()`` is idempotent, so the cost at
the twenty-third call site is one boolean check.
"""

from __future__ import annotations

import os
import tempfile

#: Set once ``configure()`` has run to completion. The call is idempotent; this
#: is what makes calling it from every solve entry point free.
_configured = False
_distributed_runtime_initialized = False


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


# Deliberately at import of *this* module, not inside configure().
#
# Everything else here is deferred because it changes process-global state.
# This is different: it only reads numpy's version and raises. Deferring it
# would mean `import dkx` succeeded on a numpy 1.x environment and then died
# later inside ml_dtypes with a traceback naming neither dkx nor numpy's
# version -- the exact user-facing bug the check was written for
# (uwplasma/DKX, Anaconda base environment). numpy is imported two lines later
# by dkx.result regardless, so the check costs nothing that was not already
# being paid, and it is not one of the seven things plan.md section 6.4
# forbids at import.
_check_numpy()


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



def configure(*, jax_x64: bool | None = None) -> None:
    """Apply the DKX runtime environment. Safe to call repeatedly.

    ``jax_x64`` overrides the ``DKX_NO_X64_SETUP`` environment variable: pass
    ``False`` to leave JAX precision alone entirely.
    """
    global _configured
    if _configured:
        return
    _configured = True

    _check_numpy()

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
            _CACHE_DIR_NAME = "jax_compilation_cache"

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

        if jax_x64 is None:
            enable_x64 = os.environ.get("DKX_NO_X64_SETUP", "").strip() not in {
                "1",
                "true",
                "yes",
            }
        else:
            enable_x64 = jax_x64
        if enable_x64:
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
            # The cache is deliberately NOT size-capped.  Setting
            # jax_compilation_cache_max_size turns on jax's LRU, which writes a
            # sidecar "-atime" file per entry and does size bookkeeping on every
            # write.  With the zero thresholds above -- which exist so even small
            # kernels are cached -- that is measurably expensive: on
            # tests/test_monoenergetic_database.py, 32 s and 1792 files capped
            # against 20 s and 896 uncapped, a 60% penalty on every run.  CI proved
            # it at scale, nine of ten coverage shards crossing a 10-minute timeout
            # they had been finishing in four to eight.
            #
            # So the cache grows without bound.  That is a disk-space cost -- it
            # reached 782 MB over 64k entries on a development machine -- and the
            # remedy is to delete the directory, which loses nothing but compile
            # time.  Paying 60% on every run to avoid it is the worse trade.
    except ImportError:
        # A caller can legitimately configure the environment on a machine with
        # no JAX -- packaging checks and documentation builds do. The env-var
        # half above has already been applied; only the config half is skipped,
        # and require_float64() is what stops a solve from running anyway.
        pass



def is_configured() -> bool:
    """Whether :func:`configure` has run in this process."""
    return _configured
