"""RHSMode=1 host dense/sparse-direct policy helpers.

The functions in this module decide when the driver may leave the default JAX
Krylov path for host dense or host sparse direct work.  They intentionally depend
only on environment variables, backend strings, and small operator metadata so
they can be tested without assembling a kinetic operator.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_bool(name: str) -> bool | None:
    env = str(os.environ.get(name, "")).strip().lower()
    if env in _TRUE_VALUES:
        return True
    if env in _FALSE_VALUES:
        return False
    return None


def _env_int(name: str, default: int) -> int:
    env = str(os.environ.get(name, "")).strip()
    try:
        return int(env) if env else int(default)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    env = str(os.environ.get(name, "")).strip()
    try:
        return float(env) if env else float(default)
    except ValueError:
        return float(default)


def rhs1_dense_backend_allowed(*, backend: str) -> bool:
    """Return whether RHSMode=1 dense linear algebra may run on the active backend."""
    env = _env_bool("SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR")
    if env is not None:
        return bool(env)
    return str(backend).strip().lower() == "cpu"


def rhs1_host_dense_fallback_allowed(*, backend: str) -> bool:
    """Return whether host dense LU fallback is allowed for RHSMode=1."""
    if str(backend).strip().lower() == "cpu":
        return True
    env = _env_bool("SFINCS_JAX_RHSMODE1_DENSE_HOST_LU")
    return bool(env)


def rhs1_host_dense_shortcut_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    dense_fallback_max: int,
) -> bool:
    """Allow the small accelerator FP branch to use host dense LU directly."""
    env = _env_bool("SFINCS_JAX_RHSMODE1_HOST_DENSE_SHORTCUT")
    if env is False:
        return False
    if bool(use_implicit):
        return False
    if str(backend).strip().lower() == "cpu":
        return False
    if str(solve_method_kind).strip().lower() in {"dense", "dense_ksp"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if op.fblock.fp is None:
        return False
    host_dense_env = _env_bool("SFINCS_JAX_RHSMODE1_DENSE_HOST_LU")
    if host_dense_env is False:
        return False
    shortcut_max = _env_int("SFINCS_JAX_RHSMODE1_HOST_DENSE_SHORTCUT_MAX", 900)
    dense_cap = min(max(0, int(shortcut_max)), max(0, int(dense_fallback_max)))
    if dense_cap <= 0:
        return False
    return int(active_size) <= dense_cap


def rhs1_dense_fallback_max(op: Any) -> int:
    """Resolve the RHSMode=1 dense fallback active-size ceiling.

    Full Fokker-Planck systems use a larger conservative default because dense
    fallback is often the cheapest robust path for small/medium FP systems. PAS
    systems are stricter: dense fallback can drift away from PETSc-style
    approximate branches, so PAS is disabled by default except for
    ``constraintScheme=0`` or explicit user opt-in.
    """
    base_max = _env_int("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX", 400)
    if op.fblock.fp is None:
        dense_pas_raw = str(os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_PAS_MAX", "")).strip()
        if dense_pas_raw:
            dense_pas_max = _env_int("SFINCS_JAX_RHSMODE1_DENSE_PAS_MAX", base_max)
            if dense_pas_max <= 0:
                return 0
            return max(base_max, dense_pas_max)
        if int(op.constraint_scheme) != 0:
            return 0
        dense_pas_max = 5000
        if dense_pas_max <= 0:
            return base_max
        return max(base_max, dense_pas_max)

    dense_fp_raw = str(os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FP_MAX", "")).strip()
    dense_fp_cutoff_raw = str(os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF", "")).strip()
    if dense_fp_raw:
        dense_fp_max = _env_int("SFINCS_JAX_RHSMODE1_DENSE_FP_MAX", base_max)
    elif dense_fp_cutoff_raw:
        dense_fp_max = _env_int("SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF", base_max)
    else:
        dense_fp_max = 6000
    if dense_fp_max <= 0:
        return base_max
    return max(base_max, dense_fp_max)


def rhs1_dense_auto_fp_cutoff(*, dense_active_cutoff: int) -> int:
    """Resolve the initial dense-solve cutoff for full-FP RHSMode=1 systems.

    This is the pre-Krylov auto-selection threshold used by the CLI/output
    writer. It intentionally matches the default full-FP dense fallback budget
    (6000 active unknowns) so moderate FP systems do not first run through the
    expensive Krylov/strong/sparse rescue ladder. Users may still disable the
    initial dense path with ``SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF=0`` or lower it
    for memory-constrained hosts.
    """
    raw = str(os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF", "")).strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return min(max(0, int(dense_active_cutoff)), 6000)


def rhs1_dense_auto_fp_accelerator_min() -> int:
    """Minimum active size for default accelerator dense auto-selection.

    Tiny GPU full-FP systems are usually faster on the existing matrix-free path
    because dense assembly/solver setup dominates. Moderate systems can avoid the
    expensive Krylov/preconditioner ladder, so enable accelerator dense auto only
    above this floor unless the user explicitly overrides the solve method.
    """
    return max(0, _env_int("SFINCS_JAX_RHSMODE1_DENSE_FP_ACCELERATOR_MIN", 1000))


def rhs1_dense_auto_fp_allowed(
    *,
    backend: str,
    active_size: int,
    dense_active_cutoff: int,
) -> bool:
    """Return whether full-FP RHSMode=1 auto mode should start with dense LU.

    CPU defaults use the dense path for all systems below the FP cutoff. On
    accelerators, keep tiny fixtures on the lower-overhead matrix-free path but
    use dense LU for moderate systems that otherwise pay a long Krylov/probe
    ladder before falling back.
    """
    cutoff = rhs1_dense_auto_fp_cutoff(dense_active_cutoff=dense_active_cutoff)
    if cutoff <= 0 or int(active_size) > int(cutoff):
        return False
    if str(backend).strip().lower() == "cpu":
        return True
    return int(active_size) >= int(rhs1_dense_auto_fp_accelerator_min())


def rhs1_dense_krylov_allowed() -> bool:
    """Return whether dense Krylov fallback is enabled."""
    env = _env_bool("SFINCS_JAX_RHSMODE1_DENSE_KRYLOV")
    if env is not None:
        return bool(env)
    return True


def rhs1_host_sparse_direct_allowed(*, sparse_exact_lu: bool, use_implicit: bool = False) -> bool:
    """Return whether exact sparse LU may be built and solved on the host."""
    if not bool(sparse_exact_lu):
        return False
    if bool(use_implicit):
        return False
    env = _env_bool("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_HOST")
    if env is not None:
        return bool(env)
    return True


def rhs1_sparse_operator_preconditioned_rescue_allowed(
    *,
    op: Any,
    sparse_exact_lu: bool,
    host_sparse_direct_wanted: bool,
    backend: str,
) -> bool:
    """Allow sparse-preconditioned GMRES before exact sparse LU.

    This branch is kept narrow because it is a parity-preserving rescue for CPU
    full-FP constraint-scheme-1 systems, not a general sparse solve replacement.
    """
    if not bool(sparse_exact_lu) or not bool(host_sparse_direct_wanted):
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 1:
        return False
    if op.fblock.fp is None or op.fblock.pas is not None:
        return False
    env = _env_bool("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES")
    if env is False:
        return False
    return True


def rhs1_constrained_pas_sparse_pc_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
) -> bool:
    """Return whether large constrained-PAS RHSMode=1 should start sparse-PC GMRES.

    The matrix-free PAS path is robust for small examples, but production-sized
    finite-beta profile-current decks can spend many minutes in Krylov fallback
    and still stall at a large true residual.  The host sparse-PC branch builds
    the same explicit operator sparsity used for diagnostics, factors the
    RHSMode=1 preconditioner, and then polishes the true residual with GMRES.

    Keep this as a narrow non-differentiable policy: it is a CLI/production
    solve path, not the JAX-native autodiff route.
    """
    env = _env_bool("SFINCS_JAX_RHSMODE1_CONSTRAINED_PAS_SPARSE_PC")
    if env is False:
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default", "incremental"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 2:
        return False
    if op.fblock.fp is not None or op.fblock.pas is None:
        return False

    min_size = _env_int("SFINCS_JAX_RHSMODE1_CONSTRAINED_PAS_SPARSE_PC_MIN", 30_000)
    max_size = _env_int("SFINCS_JAX_RHSMODE1_CONSTRAINED_PAS_SPARSE_PC_MAX", 300_000)
    if env is True:
        min_size = 0
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    return int(active_size) >= max(0, int(min_size))


def rhs1_tokamak_pas_er_sparse_pc_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    er_abs: float,
    use_dkes: bool,
    include_xdot: bool,
    include_electric_field_xi: bool,
) -> bool:
    """Return whether tokamak PAS+Er should start the host sparse-PC lane.

    Production-floor tokamak PAS+Er full-trajectory cases at
    ``25 x 1 x 8 x 100`` stall in the matrix-free PAS-ILU/Schur fallback ladder
    but are parity-clean with the non-differentiable sparse-PC GMRES route using
    a tiny diagonal shift and relaxed SuperLU pivoting. Keep this policy narrow:
    CPU/GPU only, no Phi1, pure PAS, axisymmetric, electric-field trajectory
    terms enabled, and active size in the measured window. The sparse
    factorization is still hosted, but the matrix-vector probes follow the
    active JAX backend; keep accelerators outside CPU/GPU opt-in until tested.
    """
    env = _env_bool("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_ER_SPARSE_PC")
    if env is False:
        return False
    backend_norm = str(backend).strip().lower()
    if backend_norm not in {"cpu", "gpu", "cuda"}:
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default", "incremental"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 2:
        return False
    if op.fblock.fp is not None or op.fblock.pas is None:
        return False
    if int(getattr(op, "n_zeta", 1)) != 1:
        return False
    has_er_drive = abs(float(er_abs)) > 0.0 and (
        bool(use_dkes) or bool(include_xdot) or bool(include_electric_field_xi)
    )
    if not has_er_drive:
        return False

    min_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_ER_SPARSE_PC_MIN", 10_000)
    max_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_ER_SPARSE_PC_MAX", 60_000)
    if env is True:
        min_size = 0
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    return int(active_size) >= max(0, int(min_size))


def rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    er_abs: float,
) -> bool:
    """Return whether tokamak PAS no-Er should start the host sparse-PC lane.

    The measured production-floor ``tokamak_*species_PASCollisions_noEr`` cases
    at ``25 x 1 x (4|8) x 100`` spend most of their default matrix-free memory
    or GPU wall time inside the Krylov solve even though a host sparse-PC solve
    reaches the same Fortran output. Keep this policy scoped to the validated
    non-differentiable axisymmetric no-Er PAS window so geometry-rich and Phi1
    systems continue to use their own policies.
    """

    env = _env_bool("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_NOER_SPARSE_PC")
    if env is False:
        return False
    backend_norm = str(backend).strip().lower()
    if backend_norm not in {"cpu", "gpu", "cuda"}:
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default", "incremental"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 2:
        return False
    if op.fblock.fp is not None or op.fblock.pas is None:
        return False
    if int(getattr(op, "n_zeta", 1)) != 1:
        return False
    if abs(float(er_abs)) > 0.0:
        return False

    min_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_NOER_SPARSE_PC_MIN", 5_000)
    max_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_NOER_SPARSE_PC_MAX", 60_000)
    if env is True:
        min_size = 0
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    return int(active_size) >= max(0, int(min_size))


def rhs1_tokamak_fp_er_sparse_pc_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    er_abs: float,
    use_dkes: bool,
    include_xdot: bool,
    include_electric_field_xi: bool,
) -> bool:
    """Return whether tokamak full-FP + Er should start sparse-PC GMRES.

    Production-floor CPU/GPU probes at ``25 x 1 x 8 x 100`` show the
    matrix-free FP+Er routes can either stall before the strong fallback or pay
    for large generic XLA solves. The x-block sparse-PC route is parity-clean
    and materially faster in this measured axisymmetric window, so it is the
    default for both CPU and GPU non-differentiable output/CLI lanes.
    """

    env = _env_bool("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_ER_SPARSE_PC")
    if env is False:
        return False
    backend_norm = str(backend).strip().lower()
    allowed_backends = {"cpu", "gpu", "cuda"}
    if backend_norm not in allowed_backends:
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default", "incremental"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 1:
        return False
    if op.fblock.fp is None or op.fblock.pas is not None:
        return False
    if int(getattr(op, "n_zeta", 1)) != 1:
        return False
    has_er_drive = abs(float(er_abs)) > 0.0 and (
        bool(use_dkes) or bool(include_xdot) or bool(include_electric_field_xi)
    )
    if not has_er_drive:
        return False

    min_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_ER_SPARSE_PC_MIN", 10_000)
    max_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_ER_SPARSE_PC_MAX", 60_000)
    if env is True:
        min_size = 0
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    return int(active_size) >= max(0, int(min_size))


def rhs1_tokamak_fp_noer_sparse_pc_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    er_abs: float,
    use_dkes: bool,
    include_xdot: bool,
    include_electric_field_xi: bool,
) -> bool:
    """Return whether tokamak full-FP no-Er should start sparse-PC GMRES.

    The production-floor ``tokamak_1species_FPCollisions_noEr`` GPU row at
    ``25 x 1 x 8 x 100`` can exit the matrix-free XMG/strong-preconditioner
    ladder with a small-but-physics-visible residual.  Sparse-PC GMRES is slower
    than the memory-heavy theta-line route, but it is parity-clean against the
    Fortran direct solve and has substantially lower peak memory.  Keep this
    default GPU-only and constrained to the measured axisymmetric no-Er window.
    """

    env = _env_bool("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_NOER_SPARSE_PC")
    if env is False:
        return False
    backend_norm = str(backend).strip().lower()
    allowed_backends = {"gpu", "cuda"} if env is not True else {"cpu", "gpu", "cuda"}
    if backend_norm not in allowed_backends:
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default", "incremental"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 0:
        return False
    if op.fblock.fp is None or op.fblock.pas is not None:
        return False
    if int(getattr(op, "n_zeta", 1)) != 1:
        return False
    has_er_drive = abs(float(er_abs)) > 0.0 and (
        bool(use_dkes) or bool(include_xdot) or bool(include_electric_field_xi)
    )
    if has_er_drive:
        return False

    min_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_NOER_SPARSE_PC_MIN", 10_000)
    max_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_NOER_SPARSE_PC_MAX", 60_000)
    if env is True:
        min_size = 0
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    return int(active_size) >= max(0, int(min_size))


def rhs1_fp_3d_sparse_pc_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    eparallel_abs: float = 0.0,
) -> bool:
    """Return whether 3D full-FP RHSMode=1 should start sparse-PC GMRES.

    Bounded HSX and geometryScheme11 FP probes show that the host sparse-PC
    branch can beat the dense FP shortcut on runtime and memory for some
    geometry-rich systems while preserving Fortran parity. Keep the promotion
    narrow and CPU-only, and do not take it for small or low-pitch-resolution
    systems by default: dense LU is more robust for QI-like smoke decks and
    avoids sparse-factorization failures before a solver trace is written.
    """
    env = _env_bool("SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC")
    if env is False:
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default", "incremental"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 1:
        return False
    if op.fblock.fp is None or op.fblock.pas is not None:
        return False
    if int(getattr(op, "n_zeta", 1)) <= 1:
        return False
    if abs(float(eparallel_abs)) > 0.0:
        return False

    min_nxi = _env_int("SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC_MIN_NXI", 50)
    if int(getattr(op, "n_xi", max(0, int(min_nxi)))) < max(0, int(min_nxi)):
        return False

    min_size = _env_int("SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC_MIN", 5001)
    max_size = _env_int("SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC_MAX", 20000)
    if env is True:
        min_size = 0
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    return int(active_size) >= max(0, int(min_size))


def rhs1_fp_3d_xblock_sparse_pc_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    eparallel_abs: float = 0.0,
) -> bool:
    """Return whether 3D full-FP RHSMode=1 should use x-block sparse-PC GMRES.

    The scale-0.50 QI CPU/GPU ladder is too large for dense fallback and too
    stiff for the active-DOF XMG/strong-preconditioner route, but it converges
    quickly with host-assembled x-block sparse LU as a right preconditioner.
    Keep this as a bounded non-differentiable output/CLI route until larger QI
    ladders are checked.
    """

    env = _env_bool("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC")
    if env is False:
        return False
    if str(backend).strip().lower() not in {"cpu", "gpu", "cuda"}:
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default", "incremental"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 1:
        return False
    if op.fblock.fp is None or op.fblock.pas is not None:
        return False
    if int(getattr(op, "n_species", 1)) != 1:
        return False
    if int(getattr(op, "n_zeta", 1)) <= 1:
        return False
    if bool(getattr(op, "point_at_x0", False)):
        return False
    if abs(float(eparallel_abs)) > 0.0:
        return False

    min_nxi = _env_int("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MIN_NXI", 50)
    if int(getattr(op, "n_xi", max(0, int(min_nxi)))) < max(0, int(min_nxi)):
        return False

    min_size = _env_int("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MIN", 30_000)
    max_size = _env_int("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MAX", 45_000)
    if env is True:
        min_size = 0
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    return int(active_size) >= max(0, int(min_size))


def rhs1_tokamak_er_dense_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    use_dkes: bool,
    er_abs: float,
    include_xdot: bool,
    include_electric_field_xi: bool,
) -> bool:
    """Return whether bounded tokamak electric-field RHSMode=1 may use dense LU.

    Production-resolution tokamak Er probes show that the matrix-free
    Krylov/strong/sparse-rescue ladder can spend O(100 s) on systems just above
    the generic dense cutoff, while dense LU solves the same algebraic problem in
    a few seconds with Fortran-clean diagnostics. Keep this CPU-only and
    size-bounded because dense LU has a larger transient RSS footprint.
    """
    env = _env_bool("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE")
    if env is False:
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default", "incremental"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(getattr(op, "n_zeta", 1)) != 1:
        return False
    if op.fblock.fp is None and op.fblock.pas is None:
        return False
    has_er_drive = abs(float(er_abs)) > 0.0 and (
        bool(use_dkes) or bool(include_xdot) or bool(include_electric_field_xi)
    )
    if not has_er_drive:
        return False

    min_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE_MIN", 5000)
    max_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE_MAX", 6500)
    max_dense_bytes = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE_MAX_BYTES", 350_000_000)
    if env is True:
        min_size = 0
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    dense_bytes = int(active_size) * int(active_size) * 8
    if int(max_dense_bytes) > 0 and dense_bytes > int(max_dense_bytes):
        return False
    return int(active_size) >= max(0, int(min_size))


def host_sparse_factor_dtype(
    *,
    size: int,
    factorization: str,
    use_implicit: bool,
    backend: str,
) -> np.dtype:
    """Resolve the dtype used for host sparse factorization."""
    env = str(os.environ.get("SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE", "")).strip().lower()
    if env in {"float64", "fp64", "64"}:
        return np.dtype(np.float64)
    if env in {"float32", "fp32", "32"}:
        return np.dtype(np.float32)
    if bool(use_implicit):
        return np.dtype(np.float64)
    if str(backend).strip().lower() != "cpu":
        return np.dtype(np.float64)
    if str(factorization).strip().lower() != "lu":
        return np.dtype(np.float64)
    min_size = _env_int("SFINCS_JAX_HOST_SPARSE_FACTOR_FLOAT32_MIN", 12000)
    if int(size) >= max(1, int(min_size)):
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def host_sparse_direct_refine_steps(env_name: str, default: int = 2) -> int:
    """Parse nonnegative iterative-refinement step count for host direct solves."""
    return max(0, _env_int(env_name, int(default)))


def rhs1_host_sparse_skip_dense_ratio() -> float:
    """Residual ratio above which sparse direct paths may skip dense fallback."""
    return _env_float("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_SKIP_DENSE_RATIO", 1.0e4)


def rhs1_explicit_sparse_host_direct_allowed(
    *,
    sparse_exact_lu: bool,
    use_implicit: bool,
    active_size: int,
) -> bool:
    """Return whether the explicit sparse helper may build a host sparse operator."""
    env = _env_bool("SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_HELPER")
    if env is False:
        return False
    if bool(use_implicit) or (not bool(sparse_exact_lu)):
        return False
    max_size = _env_int("SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_HELPER_MAX", 20000)
    return int(active_size) <= max(1, int(max_size))


__all__ = [
    "host_sparse_direct_refine_steps",
    "host_sparse_factor_dtype",
    "rhs1_dense_backend_allowed",
    "rhs1_dense_auto_fp_cutoff",
    "rhs1_dense_auto_fp_allowed",
    "rhs1_dense_auto_fp_accelerator_min",
    "rhs1_dense_fallback_max",
    "rhs1_dense_krylov_allowed",
    "rhs1_explicit_sparse_host_direct_allowed",
    "rhs1_host_dense_fallback_allowed",
    "rhs1_host_dense_shortcut_allowed",
    "rhs1_host_sparse_direct_allowed",
    "rhs1_host_sparse_skip_dense_ratio",
    "rhs1_constrained_pas_sparse_pc_auto_allowed",
    "rhs1_fp_3d_sparse_pc_auto_allowed",
    "rhs1_fp_3d_xblock_sparse_pc_auto_allowed",
    "rhs1_tokamak_er_dense_auto_allowed",
    "rhs1_tokamak_fp_er_sparse_pc_auto_allowed",
    "rhs1_tokamak_fp_noer_sparse_pc_auto_allowed",
    "rhs1_tokamak_pas_er_sparse_pc_auto_allowed",
    "rhs1_tokamak_pas_noer_sparse_pc_auto_allowed",
    "rhs1_sparse_operator_preconditioned_rescue_allowed",
]
