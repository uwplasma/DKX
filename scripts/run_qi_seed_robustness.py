#!/usr/bin/env python
"""Build and optionally run deterministic QI seed-robustness cases.

The checked-in quasi-isodynamic VMEC example is expensive at authored
resolution, so this lane creates reproducible neighboring smoke decks around
that input. By default it only writes inputs and a manifest; pass ``--execute``
to run each seed through ``sfincs_jax write-output``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QI_INPUT = REPO_ROOT / "examples" / "additional_examples" / "input.namelist"
DEFAULT_OUT_ROOT = REPO_ROOT / "tests" / "qi_seed_robustness"
DEFAULT_EVIDENCE_ARTIFACTS = (
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_smoke.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_multiseed.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_multiseed_gpu.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale035_cpu_gpu.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_multiseed5_cpu.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale045_cpu_probe.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale050_cpu_probe.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale050_solver_matrix_2026_05_12.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale050_xblock_lu_right_cpu.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale050_xblock_lu_right_gpu.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale050_xblock_lu_right_multiseed5_cpu.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale050_xblock_lu_right_multiseed5_gpu.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale055_auto_cpu_blocker.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale055_xblock_lu_right_cpu.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale055_xblock_auto_side_seed3_cpu.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale055_xblock_auto_side_multiseed5_cpu.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale055_xblock_auto_side_multiseed5_gpu.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale060_xblock_auto_side_seed0_cpu.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale060_xblock_auto_side_seed0_gpu.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale060_xblock_lgmres_rescue_multiseed5_cpu.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale060_probe_coarse_seed3_cpu.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale060_qi_coarse_seed3_cpu_2026_05_14.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_probe_coarse_angular_residual_seed3_cpu_2026_05_14.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_device_host_fallback_seed3_cpu_2026_05_15.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_two_level_smoothed_load_seed3_cpu_2026_05_16.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_moment_schur_probe_seed3_cpu_2026_05_16.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_deflated_seed3_cpu_2026_05_16.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_deflated_cycles8_seed3_cpu_2026_05_17.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_deflated_minres8_seed3_cpu_2026_05_17.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_deflated_minres8_seed3_gpu0_2026_05_17.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_deflated_minres8_lgmres_forced_seed3_gpu0_2026_05_17.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_preconditioner_seed3_cpu_2026_05_19.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_matrixfree_seed3_cpu_2026_05_19.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_early_qi_skipstrong_skipglobal_seed3_cpu_2026_05_19.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_early_seed3_gpu1_reduced_not_output_2026_05_19.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_xblock_hostfallback_seed3_gpu1_timeout_2026_05_19.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_rank48_depth2_seed3_gpu1_timeout_2026_05_19.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_minres_cycles4_rank27_seed3_gpu0_2026_05_19.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_fixed_cycles4_rank27_seed3_gpu0_2026_05_19.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_krylov_rank27_seed3_gpu0_timeout_2026_05_19.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_recycle_cycles2_rank32_seed3_gpu0_2026_05_19.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_mfminres_sweeps2_seed3_gpu1_2026_05_19.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_mfminres_sweeps8_seed3_gpu1_2026_05_19.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_blockminres_sweeps1_seed3_cpu_2026_05_19.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_blockminres_groups32_sweeps1_seed3_cpu_2026_05_19.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_blockminres_hybrid_groups32_sweeps1_seed3_cpu_2026_05_19.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_blockminres_groups32_sweeps1_seed3_gpu0_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_blockminres_hybrid_groups32_sweeps1_seed3_gpu0_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_blockminres_hybrid_groups48_sweeps2_cycles8_seed3_gpu0_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_blockminres_hybrid_groups48_sweeps4_cycles12_minres_seed3_gpu0_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_operator_krylov_depth64_blockminres_hybrid_seed3_gpu0_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_gpu0_public_auto_after_transpose_fixes_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_operator_krylov_device_qi_gpu0_fgmres_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_operator_krylov_multilevel_device_qi_gpu0_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_operator_krylov_pitch_multilevel_device_qi_gpu0_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_current_constraint_device_qi_gpu0_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_augmented_krylov_device_qi_cpu_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_augmented_krylov_device_qi_gpu0_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_coarse_residual_device_qi_cpu_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_residual_snapshot_device_qi_cpu_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_residual_snapshot_equation_device_qi_cpu_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_global_moment_closure_device_qi_cpu_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_global_moment_closure_device_qi_gpu0_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_residual_galerkin_device_qi_cpu_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_residual_galerkin_device_qi_gpu1_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_block_schur_device_qi_cpu_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_block_schur_bestof_device_qi_gpu0_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_composite_closure_device_qi_gpu1_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_adjoint_krylov_device_qi_cpu_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_operator_krylov_composite_device_qi_gpu0_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_device_global_coupling_gpu0_2026_05_20.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale060_rankdef_schur_gpu0_2026_05_20.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_qi_device_krylov_nohost_recycle_seed3_gpu0_2026_05_19.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_enriched_qi_coarse_seed3_cpu_rejected_2026_05_14.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_lower_fill_seed3_cpu_rejected_2026_05_14.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_lower_fill8_seed3_cpu_rejected_2026_05_14.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale060_probe_coarse_seed3_gpu0_timeout.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale060_xblock_lgmres_rescue_seed3_gpu_timeout.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale060_xblock_right_gmres_seed3_gpu_timeout.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale060_qi_coarse_seed3_gpu0_heartbeat_timeout_2026_05_14.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale060_qi_coarse_seed3_gpu0_no_lgmres_timeout_2026_05_14.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_enriched_angular_seed3_gpu0_no_lgmres_timeout_2026_05_14.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_device_krylov_enriched_seed3_gpu1_timeout_2026_05_14.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_device_krylov_skip_side_probe_seed3_gpu0_2026_05_15.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_device_krylov_compact_no_moment_seed3_gpu1_2026_05_15.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_device_krylov_compact_right_restart20_seed3_gpu0_2026_05_15.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_device_cycle_jit_restart20_seed3_gpu0_2026_05_15.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_device_cycle_jit_restart4_diag_seed3_gpu0_2026_05_15.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_device_cycle_jit_diag_factor_seed3_gpu0_2026_05_15.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_device_cycle_jit_diag_exact_lu_seed3_gpu0_2026_05_15.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_device_cycle_jit_exact_cap16_seed3_gpu0_2026_05_15.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_device_cycle_jit_diag_left_seed3_gpu0_2026_05_15.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_device_cycle_jit_diag_left_gmres_seed3_gpu0_2026_05_15.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_galerkin_failclosed_seed3_gpu0_2026_05_15.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_galerkin_forced_xblock_seed3_gpu1_2026_05_15.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "qi_seed_robustness_scale060_xblock_lgmres_rescue_seed3_gpu0_2026_05_15_retry.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale060_gpu_rejected_solver_probes_2026_05_13.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale060_global_coupling_rejected_2026_05_13.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale060_device_krylov_rejected_2026_05_13.json",
    REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_scale060_device_operator_rejected_2026_05_13.json",
)
RESOLUTION_KEYS = ("NTHETA", "NZETA", "NX", "NXI")
LOG_TAIL_LINES = 16
OPERATOR_KRYLOV_DEVICE_QI_PROBE_PRESET = "operator-krylov-device-qi"
CURRENT_CONSTRAINT_DEVICE_QI_PROBE_PRESET = "current-constraint-device-qi"
ADJOINT_KRYLOV_DEVICE_QI_PROBE_PRESET = "adjoint-krylov-device-qi"
AUGMENTED_KRYLOV_DEVICE_QI_PROBE_PRESET = "augmented-krylov-device-qi"
COARSE_RESIDUAL_DEVICE_QI_PROBE_PRESET = "coarse-residual-device-qi"
RESIDUAL_SNAPSHOT_DEVICE_QI_PROBE_PRESET = "residual-snapshot-device-qi"
RESIDUAL_SNAPSHOT_EQUATION_DEVICE_QI_PROBE_PRESET = "residual-snapshot-equation-device-qi"
ASSEMBLED_REUSE_DEVICE_QI_PROBE_PRESET = "assembled-reuse-device-qi"
COMPOSITE_CLOSURE_DEVICE_QI_PROBE_PRESET = "composite-closure-device-qi"
GLOBAL_MOMENT_CLOSURE_DEVICE_QI_PROBE_PRESET = "global-moment-closure-device-qi"
RESIDUAL_GALERKIN_DEVICE_QI_PROBE_PRESET = "residual-galerkin-device-qi"
BLOCK_SCHUR_DEVICE_QI_PROBE_PRESET = "block-schur-device-qi"
OPERATOR_KRYLOV_DEVICE_QI_SOLVE_METHOD = "xblock_sparse_pc_gmres"
OPERATOR_KRYLOV_DEVICE_QI_ENV = {
    "SFINCS_JAX_GMRES_PRECONDITION_SIDE": "right",
    "SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER": "160",
    "SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART": "40",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV": "fgmres-jax",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_DEPTH": "64",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_REUSE_COARSE_OPERATOR": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_COARSE": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_LEVELS": "3",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_AGGREGATE_FACTOR": "2",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_RANK": "64",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_ANGULAR_MODE": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_RADIAL_DEGREE": "2",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_PITCH_DEGREE": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER": (
        "matrix_free_block_minres_hybrid"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_SWEEPS": "4",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_STEP_POLICY": (
        "residual_minimizing"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_MAX_GROUPS": "48",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_CYCLES": "12",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_STEP_POLICY": "minres",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MAX_RANK": "128",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_SKIP_STRONG": "1",
}
CURRENT_CONSTRAINT_DEVICE_QI_ENV = {
    **OPERATOR_KRYLOV_DEVICE_QI_ENV,
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MOMENTS": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MAX_PITCH_DEGREE": "1",
}
ADJOINT_KRYLOV_DEVICE_QI_ENV = {
    **OPERATOR_KRYLOV_DEVICE_QI_ENV,
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_ENRICHMENT": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_DEPTH": "2",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_TRANSPOSE": "autodiff",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MAX_RANK": "132",
}
AUGMENTED_KRYLOV_DEVICE_QI_ENV = {
    **OPERATOR_KRYLOV_DEVICE_QI_ENV,
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_MODE": "cycle",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_OUTER_K": "0",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV_MODE": "combined",
}
COARSE_RESIDUAL_DEVICE_QI_ENV = {
    **OPERATOR_KRYLOV_DEVICE_QI_ENV,
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_MAX_LEVEL_RANK": (
        "16"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_ORDER": (
        "coarse_to_fine"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_INCLUDE_GLOBAL": (
        "1"
    ),
}
RESIDUAL_SNAPSHOT_DEVICE_QI_ENV = {
    **OPERATOR_KRYLOV_DEVICE_QI_ENV,
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_ENRICHMENT": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_MAX_RANK": "48",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_INCLUDE_PRIMAL": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_USE_ADJOINT": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_INCLUDE_GLOBAL": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_INCLUDE_BLOCKS": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_INCLUDE_AGGREGATES": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MAX_RANK": "192",
}
RESIDUAL_SNAPSHOT_EQUATION_DEVICE_QI_ENV = {
    **OPERATOR_KRYLOV_DEVICE_QI_ENV,
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_MAX_RANK": (
        "48"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_SOLVER": (
        "action_lstsq"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_INCLUDE_GLOBAL": (
        "1"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_INCLUDE_PRIMAL": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_USE_ADJOINT": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_INCLUDE_GLOBAL": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_INCLUDE_BLOCKS": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_INCLUDE_AGGREGATES": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MAX_RANK": "192",
}
ASSEMBLED_REUSE_DEVICE_QI_ENV = {
    **RESIDUAL_SNAPSHOT_DEVICE_QI_ENV,
    "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB": "6144",
    "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS": "4096",
    "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_VALIDATE": "1",
}
COMPOSITE_CLOSURE_DEVICE_QI_ENV = {
    **RESIDUAL_SNAPSHOT_DEVICE_QI_ENV,
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_MAX_STAGES": (
        "3"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_MAX_STAGE_RANK": (
        "8"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_MAX_RANK": (
        "48"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_SOLVER": (
        "action_lstsq"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_INCLUDE_GLOBAL_RESIDUAL": (
        "1"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_INCLUDE_BLOCK_RESIDUALS": (
        "1"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_INCLUDE_OPERATOR_IMAGES": (
        "1"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_MAX_RANK": "64",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_INCLUDE_GLOBAL": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_INCLUDE_BLOCKS": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_INCLUDE_AGGREGATES": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MAX_RANK": "256",
}
GLOBAL_MOMENT_CLOSURE_DEVICE_QI_ENV = {
    **OPERATOR_KRYLOV_DEVICE_QI_ENV,
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_MAX_RANK": (
        "64"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_SOLVER": (
        "galerkin"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_INCLUDE_PROFILE": (
        "1"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_INCLUDE_CURRENT": (
        "1"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_INCLUDE_TAIL": (
        "1"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MOMENTS": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MAX_PITCH_DEGREE": "2",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MAX_RANK": "224",
}
RESIDUAL_GALERKIN_DEVICE_QI_ENV = {
    **OPERATOR_KRYLOV_DEVICE_QI_ENV,
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_MAX_STAGES": (
        "3"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_MAX_STAGE_RANK": (
        "8"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_MAX_RANK": (
        "48"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_SOLVER": (
        "action_lstsq"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_INCLUDE_GLOBAL_RESIDUAL": (
        "1"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_INCLUDE_BLOCK_RESIDUALS": (
        "1"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_INCLUDE_OPERATOR_IMAGES": (
        "0"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MAX_RANK": "192",
}
BLOCK_SCHUR_DEVICE_QI_ENV = {
    **OPERATOR_KRYLOV_DEVICE_QI_ENV,
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_MAX_RANK": "64",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_INCLUDE_GLOBAL": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_INCLUDE_BLOCKS": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_INCLUDE_AGGREGATES": "1",
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_SOLVER": (
        "galerkin"
    ),
    "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MAX_RANK": "192",
}
QI_TWO_LEVEL_TRACE_KEYS = (
    "xblock_qi_two_level_preconditioner_enabled",
    "xblock_qi_two_level_preconditioner_built",
    "xblock_qi_two_level_preconditioner_used",
    "xblock_qi_two_level_preconditioner_reason",
    "xblock_qi_two_level_preconditioner_rank",
    "xblock_qi_two_level_preconditioner_candidate_count",
    "xblock_qi_two_level_preconditioner_coarse_solver",
    "xblock_qi_two_level_preconditioner_residual_before",
    "xblock_qi_two_level_preconditioner_residual_after",
    "xblock_qi_two_level_preconditioner_improvement_ratio",
    "xblock_qi_two_level_preconditioner_probe_candidates",
    "xblock_qi_two_level_preconditioner_residual_augmented",
    "xblock_qi_two_level_preconditioner_smoothed_load_basis",
    "xblock_qi_two_level_preconditioner_smoothed_load_metadata",
    "xblock_qi_two_level_preconditioner_setup_s",
)
QI_DEFLATED_TRACE_KEYS = (
    "xblock_qi_deflated_preconditioner_enabled",
    "xblock_qi_deflated_preconditioner_built",
    "xblock_qi_deflated_preconditioner_used",
    "xblock_qi_deflated_preconditioner_reason",
    "xblock_qi_deflated_preconditioner_rank",
    "xblock_qi_deflated_preconditioner_candidate_count",
    "xblock_qi_deflated_preconditioner_residual_before",
    "xblock_qi_deflated_preconditioner_residual_after",
    "xblock_qi_deflated_preconditioner_improvement_ratio",
    "xblock_qi_deflated_preconditioner_metadata",
    "xblock_qi_deflated_preconditioner_setup_s",
    "xblock_qi_deflated_preconditioner_applies",
    "xblock_qi_deflated_preconditioner_local_applies",
    "xblock_qi_deflated_preconditioner_cycles",
    "xblock_qi_deflated_preconditioner_seed_solver",
    "xblock_qi_deflated_preconditioner_cycle_residual_history",
    "xblock_qi_deflated_preconditioner_cycle_coefficients",
    "xblock_qi_deflated_preconditioner_use_in_krylov",
)
QI_DEVICE_TRACE_KEYS = (
    "xblock_qi_device_preconditioner_enabled",
    "xblock_qi_device_preconditioner_built",
    "xblock_qi_device_preconditioner_used",
    "xblock_qi_device_preconditioner_used_in_krylov",
    "xblock_qi_device_preconditioner_reason",
    "xblock_qi_device_preconditioner_rank",
    "xblock_qi_device_preconditioner_candidate_count",
    "xblock_qi_device_preconditioner_coarse_operator_shape",
    "xblock_qi_device_preconditioner_operator_on_basis_shape",
    "xblock_qi_device_preconditioner_coarse_operator_norm",
    "xblock_qi_device_preconditioner_operator_on_basis_norm",
    "xblock_qi_device_preconditioner_residual_before",
    "xblock_qi_device_preconditioner_residual_after",
    "xblock_qi_device_preconditioner_improvement_ratio",
    "xblock_qi_device_preconditioner_metadata",
    "xblock_qi_device_preconditioner_setup_s",
    "xblock_qi_device_preconditioner_min_improvement",
    "xblock_qi_device_preconditioner_use_in_krylov",
    "xblock_qi_device_preconditioner_applies",
    "xblock_qi_device_preconditioner_seed_only",
    "xblock_qi_device_preconditioner_operator_krylov_enrichment",
    "xblock_qi_device_preconditioner_coarse_reuse",
    "xblock_qi_device_preconditioner_residual_snapshot_enrichment",
    "xblock_qi_device_preconditioner_residual_snapshot_residual_equation",
    "xblock_qi_device_preconditioner_residual_snapshot_residual_equation_candidate_count",
    "xblock_qi_device_preconditioner_residual_snapshot_residual_equation_rank",
    "xblock_qi_device_preconditioner_residual_snapshot_residual_equation_group_count",
    "xblock_qi_device_preconditioner_block_schur_residual_enrichment",
    "xblock_qi_device_preconditioner_block_schur_residual_candidate_count",
    "xblock_qi_device_preconditioner_block_schur_residual_rank",
    "xblock_qi_device_preconditioner_block_schur_residual_group_count",
    "xblock_qi_device_preconditioner_multilevel_residual_equation",
    "xblock_qi_device_preconditioner_multilevel_residual_equation_stage_rank",
    "xblock_qi_device_preconditioner_multilevel_residual_equation_order",
    "xblock_qi_device_preconditioner_multilevel_residual_equation_solver",
    "xblock_qi_device_preconditioner_multilevel_residual_equation_include_global",
    "xblock_qi_device_preconditioner_global_moment_residual_equation",
    "xblock_qi_device_preconditioner_global_moment_residual_equation_max_rank",
    "xblock_qi_device_preconditioner_global_moment_residual_equation_candidate_count",
    "xblock_qi_device_preconditioner_global_moment_residual_equation_rank",
    "xblock_qi_device_preconditioner_global_moment_residual_equation_solver",
    "xblock_qi_device_preconditioner_global_moment_residual_equation_condition_estimate",
    "xblock_qi_device_preconditioner_residual_galerkin_equation",
    "xblock_qi_device_preconditioner_residual_galerkin_equation_max_stages",
    "xblock_qi_device_preconditioner_residual_galerkin_equation_max_stage_rank",
    "xblock_qi_device_preconditioner_residual_galerkin_equation_max_rank",
    "xblock_qi_device_preconditioner_residual_galerkin_equation_candidate_count",
    "xblock_qi_device_preconditioner_residual_galerkin_equation_rank",
    "xblock_qi_device_preconditioner_residual_galerkin_equation_stage_count",
    "xblock_qi_device_preconditioner_residual_galerkin_equation_solver",
    "xblock_qi_device_preconditioner_residual_galerkin_equation_condition_estimate",
    "xblock_qi_device_preconditioner_block_schur_residual_equation",
    "xblock_qi_device_preconditioner_block_schur_residual_equation_candidate_count",
    "xblock_qi_device_preconditioner_block_schur_residual_equation_rank",
    "xblock_qi_device_preconditioner_block_schur_residual_equation_group_count",
)
MOMENT_SCHUR_TRACE_KEYS = (
    "xblock_moment_schur_enabled",
    "xblock_moment_schur_built",
    "xblock_moment_schur_used",
    "xblock_moment_schur_reason",
    "xblock_moment_schur_rank",
    "xblock_moment_schur_extra_size",
    "xblock_moment_schur_probe_residual_before",
    "xblock_moment_schur_probe_residual_after",
    "xblock_moment_schur_probe_improvement_ratio",
    "xblock_moment_schur_seed_used",
    "xblock_moment_schur_seed_residual_ratio",
)
PROGRESS_EVENT_LIMIT = 24
PROGRESS_MARKERS = (
    "active matrix size=",
    "active-DOF mode enabled",
    "RHSMode=1 BiCGStab",
    "building RHSMode=1 preconditioner",
    "strong preconditioner fallback",
    "targeted sparse",
    "xblock factorization",
    "QI residual-deflated preconditioner",
    "QI device preconditioner",
    "explicit FP x-block seed",
    "fallback",
    "QI coarse seed",
    "probe-coarse",
    "side probe",
    "lgmres",
    "LGMRES",
    "rescue",
    "solve method forced",
    "sparse_host pattern",
    "sparse_lsmr complete",
    "sparse_ilu:",
    "sparse_lu:",
    "post-minres",
    "post-coarse",
    "solve start",
    "device-cycle",
    "assembled operator",
    "assembled_device_matvecs",
    "gmres complete",
    "GMRES complete",
    "residual=",
    "residual_norm=",
    "Refusing to write nonconverged",
    "Host sparse factorization failed",
    "timed out",
    "CUDA_ERROR",
)
_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?"
_ACTIVE_DOF_SIZE_RE = re.compile(r"active-DOF mode enabled\s+\(size=(?P<active>\d+)/(?P<total>\d+)\)")
_ACTIVE_SIZE_KEY_RE = re.compile(r"\bactive_size=(?P<active>\d+)\b")
_TOTAL_SIZE_KEY_RE = re.compile(r"\btotal_size=(?P<total>\d+)\b")
_MATRIX_SIZE_RE = re.compile(r"The matrix is (?P<size>\d+) x (?P=size) elements\.")
_KEY_VALUE_RE = re.compile(
    r"\b(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^,\s)]+)"
)
_RESIDUAL_ARROW_RE = re.compile(
    rf"\bresidual(?:[_\s-]*(?:norm|after|before|seed))?\s+(?P<before>{_FLOAT_PATTERN})\s*->\s*(?P<after>{_FLOAT_PATTERN})",
    re.IGNORECASE,
)
_SIDE_PROBE_RE = re.compile(
    r"\bside probe\s+(?P<decision>[A-Za-z0-9_]+)"
    r"(?:\s+side=(?P<side_from>[A-Za-z0-9_.-]+)->(?P<side_to>[A-Za-z0-9_.-]+)"
    r"|\s+(?P<side_from_short>left|right)->(?P<side_to_short>left|right))"
    r"(?:\s+method=(?P<method_from>[A-Za-z0-9_.-]+)->(?P<method_to>[A-Za-z0-9_.-]+)"
    r"|\s+(?P<method_from_short>[A-Za-z0-9_.-]+)->(?P<method_to_short>[A-Za-z0-9_.-]+))",
    re.IGNORECASE,
)
_QI_DEFLATED_PROGRESS_RE = re.compile(
    rf"\bQI residual-deflated preconditioner\s+(?P<decision>accepted|rejected)\s+"
    rf"(?:reason=(?P<reason>[A-Za-z0-9_.:-]+)\s+)?"
    rf"(?:residual\s+(?P<before>{_FLOAT_PATTERN})\s*->\s*(?P<after>{_FLOAT_PATTERN})"
    rf"|residual=(?P<residual>{_FLOAT_PATTERN}))",
    re.IGNORECASE,
)
_QI_DEVICE_PROGRESS_RE = re.compile(
    rf"\bQI device preconditioner\s+(?P<decision>accepted|rejected)\s+"
    rf"(?:reason=(?P<reason>[A-Za-z0-9_.:-]+)\s+)?"
    rf"(?:residual\s+(?P<before>{_FLOAT_PATTERN})\s*->\s*(?P<after>{_FLOAT_PATTERN})"
    rf"|residual=(?P<residual>{_FLOAT_PATTERN}))",
    re.IGNORECASE,
)
_TRUTHY_LOG_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSY_LOG_VALUES = {"0", "false", "no", "off", "disabled"}


def _read_resolution(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in RESOLUTION_KEYS:
        value = _read_number_parameter(text, key)
        if value is not None:
            out[key] = int(round(float(value)))
    return out


def _read_number_parameter(text: str, key: str) -> float | None:
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*=\s*([-+0-9.eEdD]+)", text)
    if match is None:
        return None
    try:
        return float(match.group(1).replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def _read_string_parameter(text: str, key: str) -> str | None:
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*=\s*([^!\n]+)", text)
    if match is None:
        return None
    value = match.group(1).strip().rstrip(",").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _replace_or_append_parameter(text: str, *, group: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?im)^(\s*{re.escape(key)}\s*=\s*)([^!\n]*?)(\s*(?:!.*)?)$")
    if pattern.search(text):
        return pattern.sub(rf"\g<1>{value}\3", text, count=1)

    group_pattern = re.compile(rf"(?ims)(^\s*&{re.escape(group)}\b.*?)(^\s*/\s*$)")
    group_match = group_pattern.search(text)
    if group_match is not None:
        return text[: group_match.start(2)] + f"  {key} = {value}\n" + text[group_match.start(2) :]

    return text.rstrip() + f"\n\n&{group}\n  {key} = {value}\n/\n"


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n"


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _probe_env_for_preset(preset: str) -> dict[str, str]:
    normalized = str(preset or "").strip().lower().replace("_", "-")
    if normalized in {"", "none", "default"}:
        return {}
    if normalized == OPERATOR_KRYLOV_DEVICE_QI_PROBE_PRESET:
        return dict(OPERATOR_KRYLOV_DEVICE_QI_ENV)
    if normalized == CURRENT_CONSTRAINT_DEVICE_QI_PROBE_PRESET:
        return dict(CURRENT_CONSTRAINT_DEVICE_QI_ENV)
    if normalized == ADJOINT_KRYLOV_DEVICE_QI_PROBE_PRESET:
        return dict(ADJOINT_KRYLOV_DEVICE_QI_ENV)
    if normalized == AUGMENTED_KRYLOV_DEVICE_QI_PROBE_PRESET:
        return dict(AUGMENTED_KRYLOV_DEVICE_QI_ENV)
    if normalized == COARSE_RESIDUAL_DEVICE_QI_PROBE_PRESET:
        return dict(COARSE_RESIDUAL_DEVICE_QI_ENV)
    if normalized == RESIDUAL_SNAPSHOT_DEVICE_QI_PROBE_PRESET:
        return dict(RESIDUAL_SNAPSHOT_DEVICE_QI_ENV)
    if normalized == RESIDUAL_SNAPSHOT_EQUATION_DEVICE_QI_PROBE_PRESET:
        return dict(RESIDUAL_SNAPSHOT_EQUATION_DEVICE_QI_ENV)
    if normalized == ASSEMBLED_REUSE_DEVICE_QI_PROBE_PRESET:
        return dict(ASSEMBLED_REUSE_DEVICE_QI_ENV)
    if normalized == COMPOSITE_CLOSURE_DEVICE_QI_PROBE_PRESET:
        return dict(COMPOSITE_CLOSURE_DEVICE_QI_ENV)
    if normalized == GLOBAL_MOMENT_CLOSURE_DEVICE_QI_PROBE_PRESET:
        return dict(GLOBAL_MOMENT_CLOSURE_DEVICE_QI_ENV)
    if normalized == RESIDUAL_GALERKIN_DEVICE_QI_PROBE_PRESET:
        return dict(RESIDUAL_GALERKIN_DEVICE_QI_ENV)
    if normalized == BLOCK_SCHUR_DEVICE_QI_PROBE_PRESET:
        return dict(BLOCK_SCHUR_DEVICE_QI_ENV)
    raise ValueError(f"Unknown QI probe preset: {preset!r}")


def _solve_method_for_probe_preset(*, solve_method: str, probe_preset: str) -> str:
    """Resolve the concrete solver path required by an explicit probe preset."""

    requested = str(solve_method or "").strip()
    normalized_preset = str(probe_preset or "").strip().lower().replace("_", "-")
    qi_device_presets = {
        OPERATOR_KRYLOV_DEVICE_QI_PROBE_PRESET,
        CURRENT_CONSTRAINT_DEVICE_QI_PROBE_PRESET,
        ADJOINT_KRYLOV_DEVICE_QI_PROBE_PRESET,
        AUGMENTED_KRYLOV_DEVICE_QI_PROBE_PRESET,
        COARSE_RESIDUAL_DEVICE_QI_PROBE_PRESET,
        RESIDUAL_SNAPSHOT_DEVICE_QI_PROBE_PRESET,
        RESIDUAL_SNAPSHOT_EQUATION_DEVICE_QI_PROBE_PRESET,
        ASSEMBLED_REUSE_DEVICE_QI_PROBE_PRESET,
        COMPOSITE_CLOSURE_DEVICE_QI_PROBE_PRESET,
        GLOBAL_MOMENT_CLOSURE_DEVICE_QI_PROBE_PRESET,
        RESIDUAL_GALERKIN_DEVICE_QI_PROBE_PRESET,
        BLOCK_SCHUR_DEVICE_QI_PROBE_PRESET,
    }
    if normalized_preset in qi_device_presets and requested.lower() in {"", "auto", "default"}:
        return OPERATOR_KRYLOV_DEVICE_QI_SOLVE_METHOD
    return requested or "auto"


def _command_with_env(command: list[str], env: dict[str, str]) -> list[str]:
    if not env:
        return command
    return ["env", *[f"{key}={value}" for key, value in sorted(env.items())], *command]


def _shell_command(command: list[str], env: dict[str, str] | None = None) -> str:
    env_parts = [f"{key}={value}" for key, value in sorted((env or {}).items())]
    return " ".join([*env_parts, *command])


def _total_size_from_resolution(resolution: dict[str, object]) -> int | None:
    try:
        product = 1
        for key in RESOLUTION_KEYS:
            product *= int(resolution[key])
    except (KeyError, TypeError, ValueError):
        return None
    return product + 2


def _canonical_resolution(resolution: object) -> dict[str, int]:
    """Return resolution with canonical SFINCS upper-case keys when possible."""
    if not isinstance(resolution, dict):
        return {}
    key_aliases = {
        "NTHETA": "NTHETA",
        "NTHEETA": "NTHETA",
        "Ntheta": "NTHETA",
        "ntheta": "NTHETA",
        "NZETA": "NZETA",
        "Nzeta": "NZETA",
        "nzeta": "NZETA",
        "NX": "NX",
        "Nx": "NX",
        "nx": "NX",
        "NXI": "NXI",
        "Nxi": "NXI",
        "nxi": "NXI",
    }
    out: dict[str, int] = {}
    for key, value in resolution.items():
        canonical = key_aliases.get(str(key), str(key).upper())
        if canonical not in RESOLUTION_KEYS:
            continue
        try:
            out[canonical] = int(round(float(value)))
        except (TypeError, ValueError):
            continue
    return out


def _resolution_fractions(resolution: dict[str, object], production_resolution: dict[str, int]) -> dict[str, float]:
    fractions: dict[str, float] = {}
    for key in RESOLUTION_KEYS:
        denominator = int(production_resolution.get(key, 0))
        if denominator <= 0:
            continue
        try:
            fractions[key] = float(resolution[key]) / float(denominator)
        except (KeyError, TypeError, ValueError):
            continue
    return fractions


def _hash_unit(seed: int, label: str) -> float:
    digest = hashlib.sha256(f"{int(seed)}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _signed_jitter(seed: int, label: str) -> float:
    return 2.0 * _hash_unit(seed, label) - 1.0


def _scaled_resolution(
    resolution: dict[str, int],
    *,
    scale: float,
    min_ntheta: int,
    min_nzeta: int,
    min_nx: int,
    min_nxi: int,
) -> dict[str, int]:
    def scaled_value(key: str, minimum: int) -> int:
        source = int(resolution.get(key, minimum))
        return max(int(minimum), int(round(source * float(scale))))

    out = {
        "NTHETA": scaled_value("NTHETA", min_ntheta),
        "NZETA": scaled_value("NZETA", min_nzeta),
        "NX": scaled_value("NX", min_nx),
        "NXI": scaled_value("NXI", min_nxi),
    }
    for key in ("NTHETA", "NZETA"):
        if int(resolution.get(key, out[key])) % 2 == 1 and out[key] % 2 == 0:
            out[key] += 1
    return out


def _resolve_equilibrium(input_path: Path, text: str) -> Path | None:
    raw = _read_string_parameter(text, "equilibriumFile")
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    relative = (input_path.parent / candidate).resolve()
    if relative.exists():
        return relative
    by_basename = input_path.parent / candidate.name
    if by_basename.exists():
        return by_basename.resolve()
    return None


def _case_command(case_dir: Path, *, solve_method: str, env: dict[str, str] | None = None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sfincs_jax",
        "write-output",
        "--input",
        str(case_dir / "input.namelist"),
        "--out",
        str(case_dir / "sfincsOutput_jax.h5"),
        "--solver-trace",
        str(case_dir / "sfincsOutput_jax.solver_trace.json"),
    ]
    if str(solve_method).strip().lower() not in {"", "auto", "default"}:
        command.extend(["--solve-method", str(solve_method)])
    return _command_with_env(command, env or {})


def _materialize_case(
    *,
    seed: int,
    source_input: Path,
    source_text: str,
    source_resolution: dict[str, int],
    source_equilibrium: Path | None,
    out_root: Path,
    resolution_scale: float,
    min_ntheta: int,
    min_nzeta: int,
    min_nx: int,
    min_nxi: int,
    nu_jitter: float,
    er_jitter: float,
    solve_method: str,
    probe_preset: str,
    probe_env: dict[str, str],
) -> dict[str, object]:
    case_name = f"qi_seed_{int(seed):04d}"
    case_dir = out_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    text = source_text
    resolution = _scaled_resolution(
        source_resolution,
        scale=resolution_scale,
        min_ntheta=min_ntheta,
        min_nzeta=min_nzeta,
        min_nx=min_nx,
        min_nxi=min_nxi,
    )
    for key, value in resolution.items():
        text = _replace_or_append_parameter(text, group="resolutionParameters", key=key, value=str(int(value)))

    base_nu = _read_number_parameter(source_text, "nu_n")
    base_er = _read_number_parameter(source_text, "Er")
    nu_factor = 1.0 + float(nu_jitter) * _signed_jitter(seed, "nu_n")
    er_delta = float(er_jitter) * _signed_jitter(seed, "Er")
    nu_value = None if base_nu is None else float(base_nu) * nu_factor
    er_value = None if base_er is None else float(base_er) + er_delta
    if nu_value is not None:
        text = _replace_or_append_parameter(text, group="physicsParameters", key="nu_n", value=f"{nu_value:.12g}")
    if er_value is not None:
        text = _replace_or_append_parameter(text, group="physicsParameters", key="Er", value=f"{er_value:.12g}")

    copied_equilibrium = None
    if source_equilibrium is not None:
        copied_equilibrium = case_dir / source_equilibrium.name
        if source_equilibrium.resolve() != copied_equilibrium.resolve():
            shutil.copy2(source_equilibrium, copied_equilibrium)
        text = _replace_or_append_parameter(
            text,
            group="geometryParameters",
            key="equilibriumFile",
            value=f"'{copied_equilibrium.name}'",
        )

    input_path = case_dir / "input.namelist"
    input_path.write_text(_normalize_text(text), encoding="utf-8")
    (case_dir / "input.source.namelist").write_text(_normalize_text(source_text), encoding="utf-8")
    command = _case_command(case_dir, solve_method=solve_method, env=probe_env)
    return {
        "case": case_name,
        "seed": int(seed),
        "input": str(input_path.relative_to(out_root)),
        "output": str((case_dir / "sfincsOutput_jax.h5").relative_to(out_root)),
        "solver_trace": str((case_dir / "sfincsOutput_jax.solver_trace.json").relative_to(out_root)),
        "source_input": str(source_input),
        "source_equilibrium": str(source_equilibrium) if source_equilibrium is not None else None,
        "copied_equilibrium": str(copied_equilibrium.relative_to(out_root)) if copied_equilibrium is not None else None,
        "solve_method": str(solve_method),
        "probe_preset": str(probe_preset),
        "env": dict(sorted(probe_env.items())),
        "resolution": resolution,
        "perturbations": {
            "nu_n": nu_value,
            "nu_factor": nu_factor if nu_value is not None else None,
            "Er": er_value,
            "Er_delta": er_delta if er_value is not None else None,
        },
        "command": command,
    }


def _finite_float_or_none(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if out != out or out in {float("inf"), float("-inf")}:
        return None
    return out


def _solver_trace_summary(trace_path: Path) -> dict[str, object] | None:
    if not trace_path.exists():
        return None
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"path": str(trace_path), "readable": False}

    residual_norm = _finite_float_or_none(payload.get("residual_norm"))
    residual_target = _finite_float_or_none(payload.get("residual_target"))
    residual_ratio = None
    if residual_norm is not None and residual_target is not None and residual_target > 0.0:
        residual_ratio = residual_norm / residual_target
    metadata = payload.get("metadata")
    solver_metadata = metadata.get("solver_metadata", {}) if isinstance(metadata, dict) else {}
    summary = {
        "path": str(trace_path),
        "readable": True,
        "solve_method": payload.get("solve_method"),
        "selected_path": payload.get("selected_path"),
        "backend": payload.get("backend"),
        "active_size": _finite_float_or_none(payload.get("active_size")),
        "total_size": _finite_float_or_none(payload.get("total_size")),
        "elapsed_s": _finite_float_or_none(payload.get("elapsed_s")),
        "residual_norm": residual_norm,
        "residual_target": residual_target,
        "residual_ratio": residual_ratio,
        "converged": payload.get("converged"),
        "accepted_converged": solver_metadata.get("accepted_converged"),
        "acceptance_criterion": solver_metadata.get("acceptance_criterion"),
        "precondition_side": solver_metadata.get("precondition_side"),
        "default_right_preconditioned": solver_metadata.get("default_right_preconditioned"),
        "gmres_restart": solver_metadata.get("gmres_restart"),
        "iterations": solver_metadata.get("iterations"),
        "matvecs": solver_metadata.get("matvecs"),
        "xblock_side_probe_used": solver_metadata.get("xblock_side_probe_used"),
        "xblock_side_probe_switched": solver_metadata.get("xblock_side_probe_switched"),
        "xblock_side_probe_initial_side": solver_metadata.get("xblock_side_probe_initial_side"),
        "xblock_side_probe_selected_side": solver_metadata.get("xblock_side_probe_selected_side"),
        "xblock_side_probe_initial_method": solver_metadata.get("xblock_side_probe_initial_method"),
        "xblock_side_probe_selected_method": solver_metadata.get("xblock_side_probe_selected_method"),
        "xblock_side_probe_lgmres_rescue": solver_metadata.get("xblock_side_probe_lgmres_rescue"),
        "xblock_lgmres_rescue_outer_k": solver_metadata.get("xblock_lgmres_rescue_outer_k"),
        "xblock_side_probe_residual_ratio": solver_metadata.get("xblock_side_probe_residual_ratio"),
        "xblock_side_probe_iterations": solver_metadata.get("xblock_side_probe_iterations"),
        "xblock_side_probe_matvecs": solver_metadata.get("xblock_side_probe_matvecs"),
        "xblock_device_host_fallback_used": solver_metadata.get("xblock_device_host_fallback_used"),
        "xblock_device_host_fallback_mode": solver_metadata.get("xblock_device_host_fallback_mode"),
        "xblock_device_host_fallback_reason": solver_metadata.get("xblock_device_host_fallback_reason"),
        "xblock_device_host_fallback_requested_method": solver_metadata.get(
            "xblock_device_host_fallback_requested_method"
        ),
        "xblock_device_host_fallback_effective_krylov_env_value": solver_metadata.get(
            "xblock_device_host_fallback_effective_krylov_env_value"
        ),
        "xblock_device_host_fallback_non_autodiff": solver_metadata.get(
            "xblock_device_host_fallback_non_autodiff"
        ),
        "solver_kind": solver_metadata.get("solver_kind"),
    }
    summary.update({key: solver_metadata.get(key) for key in QI_TWO_LEVEL_TRACE_KEYS})
    summary.update({key: solver_metadata.get(key) for key in QI_DEFLATED_TRACE_KEYS})
    summary.update({key: solver_metadata.get(key) for key in QI_DEVICE_TRACE_KEYS})
    qi_device_metadata = solver_metadata.get("xblock_qi_device_preconditioner_metadata")
    if isinstance(qi_device_metadata, dict):
        nested_block_schur_keys = {
            "xblock_qi_device_preconditioner_global_moment_residual_equation": (
                "global_moment_residual_equation_enabled"
            ),
            "xblock_qi_device_preconditioner_global_moment_residual_equation_candidate_count": (
                "global_moment_residual_equation_candidate_count"
            ),
            "xblock_qi_device_preconditioner_global_moment_residual_equation_rank": (
                "global_moment_residual_equation_rank"
            ),
            "xblock_qi_device_preconditioner_global_moment_residual_equation_solver": (
                "global_moment_residual_equation_solver"
            ),
            "xblock_qi_device_preconditioner_global_moment_residual_equation_condition_estimate": (
                "global_moment_residual_equation_condition_estimate"
            ),
            "xblock_qi_device_preconditioner_residual_galerkin_equation": (
                "residual_galerkin_equation_enabled"
            ),
            "xblock_qi_device_preconditioner_residual_galerkin_equation_candidate_count": (
                "residual_galerkin_equation_candidate_count"
            ),
            "xblock_qi_device_preconditioner_residual_galerkin_equation_rank": (
                "residual_galerkin_equation_rank"
            ),
            "xblock_qi_device_preconditioner_residual_galerkin_equation_stage_count": (
                "residual_galerkin_equation_stage_count"
            ),
            "xblock_qi_device_preconditioner_residual_galerkin_equation_solver": (
                "residual_galerkin_equation_solver"
            ),
            "xblock_qi_device_preconditioner_residual_galerkin_equation_condition_estimate": (
                "residual_galerkin_equation_condition_estimate"
            ),
            "xblock_qi_device_preconditioner_residual_snapshot_residual_equation": (
                "residual_snapshot_residual_equation_enabled"
            ),
            "xblock_qi_device_preconditioner_residual_snapshot_residual_equation_candidate_count": (
                "residual_snapshot_residual_equation_candidate_count"
            ),
            "xblock_qi_device_preconditioner_residual_snapshot_residual_equation_rank": (
                "residual_snapshot_residual_equation_rank"
            ),
            "xblock_qi_device_preconditioner_residual_snapshot_residual_equation_group_count": (
                "residual_snapshot_residual_equation_group_count"
            ),
            "xblock_qi_device_preconditioner_block_schur_residual_equation": (
                "block_schur_residual_equation_enabled"
            ),
            "xblock_qi_device_preconditioner_block_schur_residual_equation_candidate_count": (
                "block_schur_residual_equation_candidate_count"
            ),
            "xblock_qi_device_preconditioner_block_schur_residual_equation_rank": (
                "block_schur_residual_equation_rank"
            ),
            "xblock_qi_device_preconditioner_block_schur_residual_equation_group_count": (
                "block_schur_residual_equation_group_count"
            ),
            "xblock_qi_device_preconditioner_block_schur_residual_enrichment": (
                "block_schur_residual_enrichment_enabled"
            ),
            "xblock_qi_device_preconditioner_block_schur_residual_candidate_count": (
                "block_schur_residual_candidate_count"
            ),
            "xblock_qi_device_preconditioner_block_schur_residual_rank": "block_schur_residual_rank",
            "xblock_qi_device_preconditioner_block_schur_residual_group_count": (
                "block_schur_residual_group_count"
            ),
        }
        for summary_key, metadata_key in nested_block_schur_keys.items():
            if summary.get(summary_key) is None:
                summary[summary_key] = qi_device_metadata.get(metadata_key)
    summary.update({key: solver_metadata.get(key) for key in MOMENT_SCHUR_TRACE_KEYS})
    return summary


def _tail_lines(path: Path, *, max_lines: int = LOG_TAIL_LINES) -> list[str]:
    """Return a compact tail from a runner log file."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [line.rstrip() for line in lines[-max(1, int(max_lines)) :]]


def _extract_progress_events(*paths: Path, max_events: int = PROGRESS_EVENT_LIMIT) -> list[str]:
    """Extract solver-stage breadcrumbs from stdout/stderr without preserving bulky logs."""
    events: list[str] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            clean = line.strip()
            if not clean:
                continue
            if any(marker in clean for marker in PROGRESS_MARKERS):
                events.append(clean)
    if len(events) > int(max_events):
        return events[-int(max_events) :]
    return events


def _log_key_values(text: object) -> dict[str, str]:
    """Return simple key=value pairs from one progress line."""
    return {match.group("key").lower(): match.group("value").rstrip(".,;") for match in _KEY_VALUE_RE.finditer(str(text))}


def _float_from_log_value(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().rstrip(".,;")
    if "->" in text:
        text = text.split("->")[-1]
    return _finite_float_or_none(text.replace("D", "E").replace("d", "e"))


def _int_from_log_value(value: object) -> int | None:
    parsed = _float_from_log_value(value)
    if parsed is None:
        return None
    return int(parsed)


def _bool_from_log_value(value: object) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower().rstrip(".,;")
    if text in _TRUTHY_LOG_VALUES:
        return True
    if text in _FALSY_LOG_VALUES:
        return False
    return None


def _arrow_before_after(value: object) -> tuple[float | None, float | None]:
    text = str(value).strip().rstrip(".,;")
    if "->" not in text:
        parsed = _float_from_log_value(text)
        return None, parsed
    before, after = text.split("->", 1)
    return _float_from_log_value(before), _float_from_log_value(after)


def _infer_side_probe_progress(events: Iterable[object]) -> dict[str, object]:
    """Infer selected x-block side/method from compact side-probe progress lines."""
    summary: dict[str, object] = {}
    for event in events:
        text = str(event)
        match = _SIDE_PROBE_RE.search(text)
        if match is None:
            continue
        key_values = _log_key_values(text)
        initial_side = match.group("side_from") or match.group("side_from_short")
        selected_side = match.group("side_to") or match.group("side_to_short")
        initial_method = match.group("method_from") or match.group("method_from_short")
        selected_method = match.group("method_to") or match.group("method_to_short")
        initial_side = initial_side.lower() if initial_side else None
        selected_side = selected_side.lower() if selected_side else None
        initial_method = initial_method.lower() if initial_method else None
        selected_method = selected_method.lower() if selected_method else None
        summary = {
            "precondition_side": selected_side,
            "xblock_side_probe_used": True,
            "xblock_side_probe_decision": match.group("decision").lower(),
            "xblock_side_probe_switched": (
                selected_side != initial_side if selected_side is not None and initial_side is not None else None
            ),
            "xblock_side_probe_initial_side": initial_side,
            "xblock_side_probe_selected_side": selected_side,
            "xblock_side_probe_initial_method": initial_method,
            "xblock_side_probe_selected_method": selected_method,
            "xblock_side_probe_lgmres_rescue": (
                selected_method == "lgmres" if selected_method is not None else None
            ),
            "xblock_side_probe_iterations": _int_from_log_value(
                key_values.get("iters") or key_values.get("iterations")
            ),
            "xblock_side_probe_matvecs": _int_from_log_value(key_values.get("matvecs")),
            "xblock_side_probe_residual_norm": _float_from_log_value(
                key_values.get("residual") or key_values.get("residual_norm")
            ),
            "xblock_side_probe_residual_ratio": _float_from_log_value(
                key_values.get("ratio") or key_values.get("residual_ratio")
            ),
        }
    return summary


def _infer_qi_deflated_progress(events: Iterable[object]) -> dict[str, object]:
    """Infer residual-deflated QI probe metadata from progress lines.

    Long GPU attempts can time out before the solver trace is flushed. The
    progress stream still carries the fail-closed acceptance line, so preserve
    that evidence in compact artifacts instead of dropping it on timeout.
    """

    summary: dict[str, object] = {}
    for event in events:
        text = str(event)
        match = _QI_DEFLATED_PROGRESS_RE.search(text)
        if match is None:
            continue
        key_values = _log_key_values(text)
        decision = match.group("decision").lower()
        residual_before = _float_from_log_value(match.group("before"))
        residual_after = _float_from_log_value(match.group("after") or match.group("residual"))
        ratio = _float_from_log_value(key_values.get("ratio"))
        if ratio is None and residual_before is not None and residual_after is not None and residual_before > 0.0:
            ratio = residual_after / residual_before
        rank = _int_from_log_value(key_values.get("rank"))
        cycles = _int_from_log_value(key_values.get("cycles"))
        use_in_krylov_raw = key_values.get("use_in_krylov")
        seed_solver = key_values.get("seed_solver")
        summary = {
            "xblock_qi_deflated_preconditioner_enabled": True,
            "xblock_qi_deflated_preconditioner_built": True,
            "xblock_qi_deflated_preconditioner_used": decision == "accepted",
            "xblock_qi_deflated_preconditioner_reason": (
                "residual_reduced" if decision == "accepted" else match.group("reason") or "residual_not_reduced"
            ),
            "xblock_qi_deflated_preconditioner_rank": rank,
            "xblock_qi_deflated_preconditioner_residual_before": residual_before,
            "xblock_qi_deflated_preconditioner_residual_after": residual_after,
            "xblock_qi_deflated_preconditioner_improvement_ratio": ratio,
            "xblock_qi_deflated_preconditioner_cycles": cycles,
            "xblock_qi_deflated_preconditioner_seed_solver": seed_solver,
            "xblock_qi_deflated_preconditioner_use_in_krylov": (
                None if use_in_krylov_raw is None else str(use_in_krylov_raw).strip().lower() not in {"0", "false"}
            ),
        }
    return summary


def _infer_qi_device_progress(events: Iterable[object]) -> dict[str, object]:
    """Infer true-device QI probe metadata from compact progress lines."""

    summary: dict[str, object] = {}
    for event in events:
        text = str(event)
        if "QI device preconditioner residual-snapshot coarse enrichment" in text:
            key_values = _log_key_values(text)
            summary.update(
                {
                    "xblock_qi_device_preconditioner_residual_snapshot_enrichment": True,
                    "xblock_qi_device_preconditioner_residual_snapshot_max_rank": (
                        _int_from_log_value(key_values.get("max_rank"))
                    ),
                    "xblock_qi_device_preconditioner_residual_snapshot_include_primal": (
                        _bool_from_log_value(key_values.get("include_primal"))
                    ),
                    "xblock_qi_device_preconditioner_residual_snapshot_use_adjoint": (
                        _bool_from_log_value(key_values.get("use_adjoint"))
                    ),
                    "xblock_qi_device_preconditioner_residual_snapshot_include_global": (
                        _bool_from_log_value(key_values.get("include_global"))
                    ),
                    "xblock_qi_device_preconditioner_residual_snapshot_include_blocks": (
                        _bool_from_log_value(key_values.get("include_blocks"))
                    ),
                    "xblock_qi_device_preconditioner_residual_snapshot_include_aggregates": (
                        _bool_from_log_value(key_values.get("include_aggregates"))
                    ),
                }
            )
        if "QI device preconditioner residual-snapshot residual equation" in text:
            key_values = _log_key_values(text)
            summary.update(
                {
                    "xblock_qi_device_preconditioner_residual_snapshot_residual_equation": True,
                    "xblock_qi_device_preconditioner_residual_snapshot_residual_equation_max_rank": (
                        _int_from_log_value(key_values.get("max_rank"))
                    ),
                    "xblock_qi_device_preconditioner_residual_snapshot_residual_equation_solver": (
                        key_values.get("solver")
                    ),
                    "xblock_qi_device_preconditioner_residual_snapshot_residual_equation_include_global": (
                        _bool_from_log_value(key_values.get("include_global"))
                    ),
                    "xblock_qi_device_preconditioner_residual_snapshot_include_primal": (
                        _bool_from_log_value(key_values.get("include_primal"))
                    ),
                    "xblock_qi_device_preconditioner_residual_snapshot_use_adjoint": (
                        _bool_from_log_value(key_values.get("use_adjoint"))
                    ),
                    "xblock_qi_device_preconditioner_residual_snapshot_include_blocks": (
                        _bool_from_log_value(key_values.get("include_blocks"))
                    ),
                    "xblock_qi_device_preconditioner_residual_snapshot_include_aggregates": (
                        _bool_from_log_value(key_values.get("include_aggregates"))
                    ),
                }
            )
        if "QI device preconditioner global moment residual equation" in text:
            key_values = _log_key_values(text)
            summary.update(
                {
                    "xblock_qi_device_preconditioner_global_moment_residual_equation": True,
                    "xblock_qi_device_preconditioner_global_moment_residual_equation_max_rank": (
                        _int_from_log_value(key_values.get("max_rank"))
                    ),
                    "xblock_qi_device_preconditioner_global_moment_residual_equation_solver": (
                        key_values.get("solver")
                    ),
                    "xblock_qi_device_preconditioner_global_moment_residual_equation_include_profile": (
                        _bool_from_log_value(key_values.get("profile"))
                    ),
                    "xblock_qi_device_preconditioner_global_moment_residual_equation_include_current": (
                        _bool_from_log_value(key_values.get("current"))
                    ),
                    "xblock_qi_device_preconditioner_global_moment_residual_equation_include_tail": (
                        _bool_from_log_value(key_values.get("tail"))
                    ),
                }
            )
        if "QI device preconditioner residual Galerkin equation" in text:
            key_values = _log_key_values(text)
            summary.update(
                {
                    "xblock_qi_device_preconditioner_residual_galerkin_equation": True,
                    "xblock_qi_device_preconditioner_residual_galerkin_equation_max_stages": (
                        _int_from_log_value(key_values.get("max_stages"))
                    ),
                    "xblock_qi_device_preconditioner_residual_galerkin_equation_max_stage_rank": (
                        _int_from_log_value(key_values.get("stage_rank"))
                    ),
                    "xblock_qi_device_preconditioner_residual_galerkin_equation_max_rank": (
                        _int_from_log_value(key_values.get("max_rank"))
                    ),
                    "xblock_qi_device_preconditioner_residual_galerkin_equation_solver": (
                        key_values.get("solver")
                    ),
                    "xblock_qi_device_preconditioner_residual_galerkin_equation_include_global_residual": (
                        _bool_from_log_value(key_values.get("global"))
                    ),
                    "xblock_qi_device_preconditioner_residual_galerkin_equation_include_block_residuals": (
                        _bool_from_log_value(key_values.get("blocks"))
                    ),
                    "xblock_qi_device_preconditioner_residual_galerkin_equation_include_operator_images": (
                        _bool_from_log_value(key_values.get("images"))
                    ),
                }
            )
        if "QI device preconditioner block-Schur residual equation" in text:
            key_values = _log_key_values(text)
            summary.update(
                {
                    "xblock_qi_device_preconditioner_block_schur_residual_equation": True,
                    "xblock_qi_device_preconditioner_block_schur_residual_equation_max_rank": (
                        _int_from_log_value(key_values.get("max_rank"))
                    ),
                    "xblock_qi_device_preconditioner_block_schur_residual_equation_include_global": (
                        _bool_from_log_value(key_values.get("include_global"))
                    ),
                    "xblock_qi_device_preconditioner_block_schur_residual_equation_include_blocks": (
                        _bool_from_log_value(key_values.get("include_blocks"))
                    ),
                    "xblock_qi_device_preconditioner_block_schur_residual_equation_include_aggregates": (
                        _bool_from_log_value(key_values.get("include_aggregates"))
                    ),
                }
            )
        if "QI device preconditioner multilevel residual equation" in text:
            key_values = _log_key_values(text)
            summary.update(
                {
                    "xblock_qi_device_preconditioner_multilevel_residual_equation": True,
                    "xblock_qi_device_preconditioner_multilevel_residual_equation_stage_rank": (
                        _int_from_log_value(key_values.get("stage_rank"))
                    ),
                    "xblock_qi_device_preconditioner_multilevel_residual_equation_order": key_values.get(
                        "order"
                    ),
                    "xblock_qi_device_preconditioner_multilevel_residual_equation_solver": key_values.get(
                        "solver"
                    ),
                    "xblock_qi_device_preconditioner_multilevel_residual_equation_include_global": (
                        _bool_from_log_value(key_values.get("include_global"))
                    ),
                }
            )
        match = _QI_DEVICE_PROGRESS_RE.search(text)
        if match is None:
            continue
        key_values = _log_key_values(text)
        decision = match.group("decision").lower()
        residual_before = _float_from_log_value(match.group("before"))
        residual_after = _float_from_log_value(match.group("after") or match.group("residual"))
        ratio = _float_from_log_value(key_values.get("ratio"))
        if ratio is None and residual_before is not None and residual_after is not None and residual_before > 0.0:
            ratio = residual_after / residual_before
        use_in_krylov_raw = key_values.get("use_in_krylov")
        seed_only = _bool_from_log_value(key_values.get("seed_only"))
        if use_in_krylov_raw is None and seed_only is not None:
            use_in_krylov = not seed_only
        else:
            use_in_krylov = _bool_from_log_value(use_in_krylov_raw)
        summary.update({
            "xblock_qi_device_preconditioner_enabled": True,
            "xblock_qi_device_preconditioner_built": True,
            "xblock_qi_device_preconditioner_used": decision == "accepted",
            "xblock_qi_device_preconditioner_reason": (
                "residual_reduced" if decision == "accepted" else match.group("reason") or "residual_not_reduced"
            ),
            "xblock_qi_device_preconditioner_rank": _int_from_log_value(key_values.get("rank")),
            "xblock_qi_device_preconditioner_residual_before": residual_before,
            "xblock_qi_device_preconditioner_residual_after": residual_after,
            "xblock_qi_device_preconditioner_improvement_ratio": ratio,
            "xblock_qi_device_preconditioner_use_in_krylov": use_in_krylov,
            "xblock_qi_device_preconditioner_seed_only": seed_only,
            "xblock_qi_device_preconditioner_operator_krylov_enrichment": _bool_from_log_value(
                key_values.get("operator_krylov")
                or key_values.get("operator_krylov_enrichment")
                or key_values.get("operator_enrichment")
            ),
            "xblock_qi_device_preconditioner_coarse_reuse": _bool_from_log_value(
                key_values.get("coarse_reuse") or key_values.get("reuse_coarse")
            ),
            "xblock_qi_device_preconditioner_residual_snapshot_enrichment": _bool_from_log_value(
                key_values.get("residual_snapshot")
            ),
            "xblock_qi_device_preconditioner_residual_snapshot_residual_equation": _bool_from_log_value(
                key_values.get("residual_snapshot_equation")
            ),
            "xblock_qi_device_preconditioner_global_moment_residual_equation": _bool_from_log_value(
                key_values.get("global_moment_equation")
            ),
            "xblock_qi_device_preconditioner_global_moment_residual_equation_candidate_count": (
                _int_from_log_value(key_values.get("global_moment_candidates"))
            ),
            "xblock_qi_device_preconditioner_global_moment_residual_equation_rank": (
                _int_from_log_value(key_values.get("global_moment_rank"))
            ),
            "xblock_qi_device_preconditioner_global_moment_residual_equation_condition_estimate": (
                _float_from_log_value(key_values.get("global_moment_cond"))
            ),
            "xblock_qi_device_preconditioner_residual_galerkin_equation": _bool_from_log_value(
                key_values.get("residual_galerkin_equation")
            ),
            "xblock_qi_device_preconditioner_residual_galerkin_equation_candidate_count": (
                _int_from_log_value(key_values.get("residual_galerkin_candidates"))
            ),
            "xblock_qi_device_preconditioner_residual_galerkin_equation_rank": (
                _int_from_log_value(key_values.get("residual_galerkin_rank"))
            ),
            "xblock_qi_device_preconditioner_block_schur_residual_equation": _bool_from_log_value(
                key_values.get("block_schur_equation")
            ),
            "xblock_qi_device_preconditioner_block_schur_residual_enrichment": _bool_from_log_value(
                key_values.get("block_schur")
            ),
        })
    return summary


def _infer_lgmres_rescue_status(events: Iterable[object], side_probe: dict[str, object]) -> str | None:
    """Infer whether LGMRES rescue was visibly forced, disabled, used, or skipped."""
    status: str | None = None
    for event in events:
        text = str(event).lower()
        text_for_tokens = text.replace("-", "_")
        if "lgmres" not in text_for_tokens:
            continue
        if any(
            token in text_for_tokens
            for token in (
                "disabled",
                "disable",
                "no_lgmres",
                "without_lgmres",
                "without lgmres",
                "lgmres_rescue=0",
                "rescue disabled",
            )
        ):
            status = "disabled"
        if any(token in text_for_tokens for token in ("forced", "force", "opt_in", "optin")):
            status = "forced"

    selected_method = side_probe.get("xblock_side_probe_selected_method")
    if isinstance(selected_method, str):
        if selected_method.lower() == "lgmres":
            return status or "used"
        return status or "not_selected"
    return status


def _infer_last_residual_progress(events: Iterable[object]) -> dict[str, object] | None:
    """Infer the last residual-like progress record without treating it as convergence."""
    latest: dict[str, object] | None = None
    for event in events:
        text = str(event)
        if "residual" not in text.lower():
            continue
        key_values = _log_key_values(text)
        residual_before: float | None = None
        residual_norm: float | None = None
        residual_kind: str | None = None

        arrow_match = _RESIDUAL_ARROW_RE.search(text)
        if arrow_match is not None:
            residual_before = _float_from_log_value(arrow_match.group("before"))
            residual_norm = _float_from_log_value(arrow_match.group("after"))
            residual_kind = "residual"

        for key in ("residual", "residual_norm", "relative_residual", "ksp_residual"):
            if key not in key_values:
                continue
            value_before, value_after = _arrow_before_after(key_values[key])
            residual_before = value_before if value_before is not None else residual_before
            residual_norm = value_after
            residual_kind = key
            break

        target = _float_from_log_value(key_values.get("target") or key_values.get("residual_target"))
        ratio = _float_from_log_value(key_values.get("ratio") or key_values.get("residual_ratio"))
        if ratio is None and residual_norm is not None and target is not None and target > 0.0:
            ratio = residual_norm / target

        latest = {
            "event": text,
            "kind": residual_kind,
            "residual_before": residual_before,
            "residual_norm": residual_norm,
            "residual_target": target,
            "residual_ratio": ratio,
        }
    return latest


def _infer_augmented_krylov_progress(events: Iterable[object]) -> dict[str, object]:
    """Infer whether the augmented-FGMRES QI operator-reuse hook was reached."""

    summary: dict[str, object] = {}
    for event in events:
        text = str(event)
        if "QI augmented Krylov" not in text:
            continue
        lower = text.lower()
        key_values = _log_key_values(text)
        summary = {
            "xblock_device_fgmres_qi_augmented_krylov_used": "enabled" in lower,
            "xblock_device_fgmres_qi_augmented_krylov_reason": (
                "enabled" if "enabled" in lower else "disabled"
            ),
            "xblock_device_fgmres_qi_augmented_krylov_rank": _int_from_log_value(
                key_values.get("rank")
            ),
            "xblock_device_fgmres_qi_augmented_krylov_mode": key_values.get("mode"),
        }
    return summary


def _infer_sizes_from_progress_events(events: Iterable[object]) -> tuple[int | None, int | None]:
    """Infer matrix sizes from preserved progress breadcrumbs when traces are absent."""
    active_size: int | None = None
    total_size: int | None = None
    for event in events:
        text = str(event)
        active_match = _ACTIVE_DOF_SIZE_RE.search(text)
        if active_match is not None:
            active_size = int(active_match.group("active"))
            total_size = int(active_match.group("total"))
            continue
        active_key_match = _ACTIVE_SIZE_KEY_RE.search(text)
        if active_key_match is not None and active_size is None:
            active_size = int(active_key_match.group("active"))
        total_key_match = _TOTAL_SIZE_KEY_RE.search(text)
        if total_key_match is not None and total_size is None:
            total_size = int(total_key_match.group("total"))
        matrix_match = _MATRIX_SIZE_RE.search(text)
        if matrix_match is not None and active_size is None:
            active_size = int(matrix_match.group("size"))
    return active_size, total_size


def _infer_last_matvec_progress(events: Iterable[object]) -> tuple[int | None, float | None]:
    """Infer the latest reported Krylov matvec progress from compact logs."""
    last_matvecs: int | None = None
    last_elapsed_s: float | None = None
    for event in events:
        key_values = _log_key_values(event)
        matvecs = key_values.get("matvecs") or key_values.get("assembled_device_matvecs")
        elapsed_s = key_values.get("elapsed_s")
        if matvecs is None or elapsed_s is None:
            continue
        parsed_matvecs = _int_from_log_value(matvecs)
        parsed_elapsed_s = _float_from_log_value(elapsed_s)
        if parsed_matvecs is None or parsed_elapsed_s is None:
            continue
        last_matvecs = parsed_matvecs
        last_elapsed_s = parsed_elapsed_s
    return last_matvecs, last_elapsed_s


def _append_heartbeat(path: Path, payload: dict[str, object]) -> None:
    """Append one runner heartbeat event as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _run_command_with_heartbeat(
    command: list[str],
    *,
    cwd: Path,
    stdout,
    stderr,
    timeout_s: float,
    heartbeat_s: float,
    heartbeat_path: Path,
) -> tuple[int, bool, int]:
    """Run a command with JSONL liveness events and hard process-group timeout."""
    timeout_use = max(0.0, float(timeout_s))
    heartbeat_use = max(0.0, float(heartbeat_s))
    start = time.perf_counter()
    deadline = start + timeout_use
    heartbeat_count = 0
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )

    def _heartbeat(event: str, **extra: object) -> None:
        nonlocal heartbeat_count
        now = time.perf_counter()
        payload: dict[str, object] = {
            "event": event,
            "elapsed_s": now - start,
            "pid": int(proc.pid),
            "returncode": proc.returncode,
        }
        payload.update(extra)
        _append_heartbeat(heartbeat_path, payload)
        heartbeat_count += 1

    _heartbeat("started")
    next_heartbeat = start + heartbeat_use if heartbeat_use > 0.0 else float("inf")
    while True:
        now = time.perf_counter()
        if timeout_use > 0.0 and now >= deadline:
            _heartbeat("timeout", timeout_s=timeout_use)
            stderr.write(f"\nQI seed execution timed out after {timeout_use:.3f} s.\n")
            stderr.flush()
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10.0)
            _heartbeat("terminated", timeout_s=timeout_use)
            return 124, True, heartbeat_count

        wait_until = min(deadline if timeout_use > 0.0 else float("inf"), next_heartbeat)
        wait_s = max(0.05, min(1.0, wait_until - now if wait_until != float("inf") else 1.0))
        try:
            returncode = proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            if heartbeat_use > 0.0 and time.perf_counter() >= next_heartbeat:
                stdout.flush()
                stderr.flush()
                _heartbeat("running", timeout_s=timeout_use)
                next_heartbeat += heartbeat_use
            continue
        _heartbeat("completed", timeout_s=timeout_use)
        return int(returncode), False, heartbeat_count


def _execute_cases(
    out_root: Path,
    cases: Iterable[dict[str, object]],
    *,
    timeout_s: float,
    fail_fast: bool,
    heartbeat_s: float = 0.0,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    heartbeat_use = max(0.0, float(heartbeat_s))
    for case in cases:
        command = [str(part) for part in case["command"]]  # type: ignore[index]
        case_dir = out_root / str(case["case"])
        stdout_path = case_dir / "sfincs_jax.stdout.log"
        stderr_path = case_dir / "sfincs_jax.stderr.log"
        heartbeat_path = case_dir / "runner_heartbeat.jsonl"
        trace_path = case_dir / "sfincsOutput_jax.solver_trace.json"
        start = time.perf_counter()
        heartbeat_count = 0
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            if heartbeat_use > 0.0:
                returncode, timed_out, heartbeat_count = _run_command_with_heartbeat(
                    command,
                    cwd=REPO_ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    timeout_s=float(timeout_s),
                    heartbeat_s=heartbeat_use,
                    heartbeat_path=heartbeat_path,
                )
            else:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=float(timeout_s),
                        check=False,
                    )
                    returncode = int(completed.returncode)
                    timed_out = False
                except subprocess.TimeoutExpired:
                    stderr.write(f"\nQI seed execution timed out after {float(timeout_s):.3f} s.\n")
                    returncode = 124
                    timed_out = True
        elapsed_s = time.perf_counter() - start
        result = {
            "case": case["case"],
            "seed": case["seed"],
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_s": elapsed_s,
            "stdout": str(stdout_path.relative_to(out_root)),
            "stderr": str(stderr_path.relative_to(out_root)),
            "heartbeat": str(heartbeat_path.relative_to(out_root)) if heartbeat_path.exists() else None,
            "heartbeat_count": int(heartbeat_count),
            "output_exists": (case_dir / "sfincsOutput_jax.h5").exists(),
            "solver_trace_exists": trace_path.exists(),
            "solver_trace_summary": _solver_trace_summary(trace_path),
            "progress_events": _extract_progress_events(stdout_path, stderr_path),
            "stdout_tail": _tail_lines(stdout_path),
            "stderr_tail": _tail_lines(stderr_path),
        }
        results.append(result)
        if returncode != 0 and fail_fast:
            break
    return results


def _execution_summary(results: Iterable[dict[str, object]]) -> dict[str, object]:
    """Return compact aggregate diagnostics for an executed seed ladder."""
    result_list = list(results)
    trace_summaries = [
        result.get("solver_trace_summary")
        for result in result_list
        if isinstance(result.get("solver_trace_summary"), dict)
    ]
    residual_ratios = [
        float(summary["residual_ratio"])
        for summary in trace_summaries
        if _finite_float_or_none(summary.get("residual_ratio")) is not None
    ]
    elapsed_values = [
        float(result["elapsed_s"])
        for result in result_list
        if _finite_float_or_none(result.get("elapsed_s")) is not None
    ]
    return {
        "attempted": len(result_list),
        "process_passed": sum(1 for result in result_list if int(result["returncode"]) == 0),
        "process_failed": sum(1 for result in result_list if int(result["returncode"]) != 0),
        "timed_out": sum(1 for result in result_list if bool(result.get("timed_out"))),
        "outputs_written": sum(1 for result in result_list if bool(result.get("output_exists"))),
        "solver_traces_written": sum(1 for result in result_list if bool(result.get("solver_trace_exists"))),
        "converged": sum(1 for summary in trace_summaries if summary.get("converged") is True),
        "accepted_converged": sum(1 for summary in trace_summaries if summary.get("accepted_converged") is True),
        "max_residual_ratio": max(residual_ratios) if residual_ratios else None,
        "max_elapsed_s": max(elapsed_values) if elapsed_values else None,
        "backends": sorted({str(summary.get("backend")) for summary in trace_summaries if summary.get("backend")}),
        "solve_methods": sorted(
            {str(summary.get("solve_method")) for summary in trace_summaries if summary.get("solve_method")}
        ),
        "selected_paths": sorted(
            {str(summary.get("selected_path")) for summary in trace_summaries if summary.get("selected_path")}
        ),
    }


def _evaluate_execution_gates(
    results: Iterable[dict[str, object]],
    *,
    max_residual_ratio: float | None,
    require_converged: bool,
    require_accepted_converged: bool,
) -> dict[str, object]:
    """Evaluate optional seed-ladder promotion gates against executed cases."""
    failures: list[dict[str, object]] = []
    for result in results:
        case_name = str(result.get("case"))
        returncode = int(result.get("returncode", 1))
        if returncode != 0:
            failures.append({"case": case_name, "reason": "process_failed", "returncode": returncode})
            continue

        summary = result.get("solver_trace_summary")
        if (max_residual_ratio is not None or require_converged or require_accepted_converged) and not isinstance(
            summary, dict
        ):
            failures.append({"case": case_name, "reason": "missing_solver_trace_summary"})
            continue

        if isinstance(summary, dict) and max_residual_ratio is not None:
            residual_ratio = _finite_float_or_none(summary.get("residual_ratio"))
            if residual_ratio is None:
                failures.append({"case": case_name, "reason": "missing_residual_ratio"})
            elif residual_ratio > float(max_residual_ratio):
                failures.append(
                    {
                        "case": case_name,
                        "reason": "residual_ratio_exceeded",
                        "residual_ratio": residual_ratio,
                        "max_residual_ratio": float(max_residual_ratio),
                    }
                )

        if isinstance(summary, dict) and bool(require_converged) and summary.get("converged") is not True:
            failures.append({"case": case_name, "reason": "not_converged"})

        if (
            isinstance(summary, dict)
            and bool(require_accepted_converged)
            and summary.get("accepted_converged") is not True
        ):
            failures.append({"case": case_name, "reason": "not_accepted_converged"})

    return {
        "passed": not failures,
        "failures": failures,
        "max_residual_ratio": max_residual_ratio,
        "require_converged": bool(require_converged),
        "require_accepted_converged": bool(require_accepted_converged),
    }


def _env_flag(env: object, key: str) -> bool | None:
    if not isinstance(env, dict):
        return None
    return _bool_from_log_value(env.get(key))


def _seed_evidence_classification(seed: dict[str, object], probe_env: object) -> dict[str, object]:
    """Classify one seed summary without treating failed attempts as promotion evidence."""
    tags: set[str] = set()
    returncode = int(seed.get("returncode", 1) or 0)
    timed_out = bool(seed.get("timed_out"))
    solver_trace_exists = bool(seed.get("solver_trace_exists"))
    output_exists = bool(seed.get("output_exists"))
    converged = seed.get("converged") is True
    accepted_converged = seed.get("accepted_converged")
    if returncode == 0:
        if solver_trace_exists and not converged:
            outcome = "not_converged"
        elif solver_trace_exists and accepted_converged is False:
            outcome = "not_accepted_converged"
        else:
            outcome = "process_passed"
    elif timed_out:
        outcome = "timed_out"
    else:
        outcome = "process_failed"
    if not solver_trace_exists:
        tags.add("failed_before_solver_trace_summary")
    if not output_exists:
        tags.add("no_hdf5_output")

    requested_device_qi = _env_flag(
        probe_env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER",
    )
    requested_installed_krylov = _env_flag(
        probe_env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV",
    )
    requested_operator_krylov = _env_flag(
        probe_env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT",
    )
    requested_coarse_reuse = _env_flag(
        probe_env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_REUSE_COARSE_OPERATOR",
    )
    requested_augmented_krylov = _env_flag(
        probe_env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV",
    )
    requested_coarse_residual_equation = _env_flag(
        probe_env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION",
    )
    requested_residual_snapshot_enrichment = _env_flag(
        probe_env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_ENRICHMENT",
    )
    requested_residual_snapshot_equation = _env_flag(
        probe_env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION",
    )
    requested_residual_snapshot = requested_residual_snapshot_enrichment or requested_residual_snapshot_equation
    requested_global_moment = _env_flag(
        probe_env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION",
    )
    requested_residual_galerkin = _env_flag(
        probe_env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION",
    )
    requested_block_schur = _env_flag(
        probe_env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_ENRICHMENT",
    ) or _env_flag(
        probe_env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION",
    )

    if requested_device_qi:
        tags.add("requested_device_qi")
    if requested_installed_krylov:
        tags.add("requested_installed_krylov")
    if requested_operator_krylov:
        tags.add("requested_operator_krylov")
    if requested_coarse_reuse:
        tags.add("requested_coarse_reuse")
    if requested_augmented_krylov:
        tags.add("requested_augmented_krylov")
    if requested_coarse_residual_equation:
        tags.add("requested_multilevel_residual_equation")
    if requested_residual_snapshot:
        tags.add("requested_residual_snapshot")
    if requested_residual_snapshot_equation:
        tags.add("requested_residual_snapshot_residual_equation")
    if requested_global_moment:
        tags.add("requested_global_moment_residual_equation")
    if requested_residual_galerkin:
        tags.add("requested_residual_galerkin_equation")
    if requested_block_schur:
        tags.add("requested_block_schur_residual")
    if returncode == 0 and solver_trace_exists and not converged:
        tags.add("not_converged")
    if returncode == 0 and solver_trace_exists and accepted_converged is False:
        tags.add("not_accepted_converged")

    observed_device_qi = seed.get("xblock_qi_device_preconditioner_used") is True
    observed_installed_krylov = seed.get("xblock_qi_device_preconditioner_use_in_krylov") is True
    observed_seed_only = seed.get("xblock_qi_device_preconditioner_seed_only") is True
    observed_operator_krylov = seed.get("xblock_qi_device_preconditioner_operator_krylov_enrichment") is True
    observed_coarse_reuse = seed.get("xblock_qi_device_preconditioner_coarse_reuse") is True
    observed_augmented_krylov = seed.get("xblock_device_fgmres_qi_augmented_krylov_used") is True
    observed_coarse_residual_equation = (
        seed.get("xblock_qi_device_preconditioner_multilevel_residual_equation") is True
    )
    observed_residual_snapshot_enrichment = (
        seed.get("xblock_qi_device_preconditioner_residual_snapshot_enrichment") is True
    )
    observed_residual_snapshot_equation = (
        seed.get("xblock_qi_device_preconditioner_residual_snapshot_residual_equation") is True
    )
    observed_residual_snapshot = observed_residual_snapshot_enrichment or observed_residual_snapshot_equation
    observed_global_moment = (
        seed.get("xblock_qi_device_preconditioner_global_moment_residual_equation") is True
    )
    observed_residual_galerkin = (
        seed.get("xblock_qi_device_preconditioner_residual_galerkin_equation") is True
    )
    observed_block_schur = (
        seed.get("xblock_qi_device_preconditioner_block_schur_residual_enrichment") is True
        or seed.get("xblock_qi_device_preconditioner_block_schur_residual_equation") is True
    )
    if seed.get("xblock_qi_device_preconditioner_operator_on_basis_shape") is not None:
        observed_operator_krylov = True
    if seed.get("xblock_qi_device_preconditioner_coarse_operator_shape") is not None:
        observed_coarse_reuse = True

    if observed_device_qi:
        tags.add("observed_device_qi_residual_probe")
    if observed_seed_only:
        tags.add("observed_seed_only")
    if observed_installed_krylov:
        tags.add("observed_installed_krylov")
    if observed_operator_krylov:
        tags.add("observed_operator_krylov")
    if observed_coarse_reuse:
        tags.add("observed_coarse_reuse")
    if observed_augmented_krylov:
        tags.add("observed_augmented_krylov")
    if observed_coarse_residual_equation:
        tags.add("observed_multilevel_residual_equation")
    if observed_residual_snapshot:
        tags.add("observed_residual_snapshot")
    if observed_residual_snapshot_equation:
        tags.add("observed_residual_snapshot_residual_equation")
    if observed_global_moment:
        tags.add("observed_global_moment_residual_equation")
    if observed_residual_galerkin:
        tags.add("observed_residual_galerkin_equation")
    if observed_block_schur:
        tags.add("observed_block_schur_residual")

    if observed_installed_krylov and observed_coarse_reuse and observed_residual_galerkin:
        classification = "device_qi_residual_galerkin_equation_coarse_reuse"
    elif observed_installed_krylov and observed_coarse_reuse and observed_global_moment:
        classification = "device_qi_global_moment_residual_equation_coarse_reuse"
    elif observed_installed_krylov and observed_coarse_reuse and observed_block_schur:
        classification = "device_qi_block_schur_residual_coarse_reuse"
    elif observed_installed_krylov and observed_coarse_reuse and observed_residual_snapshot_equation:
        classification = "device_qi_residual_snapshot_residual_equation_coarse_reuse"
    elif observed_installed_krylov and observed_coarse_reuse and observed_residual_snapshot:
        classification = "device_qi_residual_snapshot_coarse_reuse"
    elif observed_installed_krylov and observed_coarse_reuse and observed_coarse_residual_equation:
        classification = "device_qi_multilevel_residual_equation"
    elif observed_installed_krylov and observed_coarse_reuse and observed_augmented_krylov:
        classification = "device_qi_augmented_krylov_coarse_reuse"
    elif observed_installed_krylov and observed_coarse_reuse:
        classification = "device_qi_installed_krylov_coarse_reuse"
    elif observed_installed_krylov:
        classification = "device_qi_installed_krylov"
    elif observed_device_qi and observed_seed_only:
        classification = "device_qi_seed_only_probe"
    elif requested_residual_galerkin:
        classification = "requested_residual_galerkin_equation_device_qi"
    elif requested_global_moment:
        classification = "requested_global_moment_residual_equation_device_qi"
    elif requested_block_schur:
        classification = "requested_block_schur_residual_device_qi"
    elif requested_operator_krylov:
        classification = "requested_operator_krylov_device_qi"
    elif requested_device_qi:
        classification = "requested_device_qi"
    else:
        classification = "public_auto_or_legacy"

    return {
        "classification": classification,
        "outcome": outcome,
        "promotion_eligible": (
            returncode == 0
            and solver_trace_exists
            and output_exists
            and converged
            and accepted_converged is not False
        ),
        "failed_before_summary_json": not solver_trace_exists,
        "requested_device_qi": requested_device_qi,
        "requested_installed_krylov": requested_installed_krylov,
        "requested_operator_krylov": requested_operator_krylov,
        "requested_coarse_reuse": requested_coarse_reuse,
        "requested_augmented_krylov": requested_augmented_krylov,
        "requested_coarse_residual_equation": requested_coarse_residual_equation,
        "requested_residual_snapshot": requested_residual_snapshot,
        "requested_residual_snapshot_equation": requested_residual_snapshot_equation,
        "requested_global_moment": requested_global_moment,
        "requested_residual_galerkin": requested_residual_galerkin,
        "requested_block_schur": requested_block_schur,
        "observed_device_qi": observed_device_qi,
        "observed_installed_krylov": observed_installed_krylov,
        "observed_operator_krylov": observed_operator_krylov,
        "observed_coarse_reuse": observed_coarse_reuse,
        "observed_augmented_krylov": observed_augmented_krylov,
        "observed_coarse_residual_equation": observed_coarse_residual_equation,
        "observed_residual_snapshot": observed_residual_snapshot,
        "observed_residual_snapshot_equation": observed_residual_snapshot_equation,
        "observed_global_moment": observed_global_moment,
        "observed_residual_galerkin": observed_residual_galerkin,
        "observed_block_schur": observed_block_schur,
        "observed_seed_only": observed_seed_only,
        "tags": sorted(tags),
    }


def _aggregate_seed_classifications(seed_summaries: Iterable[dict[str, object]]) -> dict[str, object]:
    seed_list = list(seed_summaries)
    tags = sorted({tag for seed in seed_list for tag in seed.get("evidence_tags", []) if isinstance(tag, str)})
    classes = sorted(
        {
            str(seed["evidence_class"])
            for seed in seed_list
            if isinstance(seed.get("evidence_class"), str) and seed.get("evidence_class")
        }
    )
    outcomes = sorted(
        {
            str(seed["run_outcome"])
            for seed in seed_list
            if isinstance(seed.get("run_outcome"), str) and seed.get("run_outcome")
        }
    )
    return {
        "classes": classes,
        "outcomes": outcomes,
        "tags": tags,
        "has_failed_before_summary_json": any(seed.get("failed_before_summary_json") is True for seed in seed_list),
        "has_observed_installed_krylov": any(
            seed.get("observed_qi_device_installed_krylov") is True for seed in seed_list
        ),
        "has_observed_coarse_reuse": any(seed.get("observed_qi_device_coarse_reuse") is True for seed in seed_list),
        "has_observed_augmented_krylov": any(
            seed.get("observed_qi_device_augmented_krylov") is True for seed in seed_list
        ),
        "has_observed_multilevel_residual_equation": any(
            seed.get("observed_qi_device_coarse_residual_equation") is True for seed in seed_list
        ),
        "has_observed_residual_snapshot": any(
            seed.get("observed_qi_device_residual_snapshot") is True for seed in seed_list
        ),
        "has_observed_residual_snapshot_residual_equation": any(
            seed.get("observed_qi_device_residual_snapshot_equation") is True for seed in seed_list
        ),
        "has_observed_global_moment_residual_equation": any(
            seed.get("observed_qi_device_global_moment_residual_equation") is True for seed in seed_list
        ),
        "has_observed_residual_galerkin_equation": any(
            seed.get("observed_qi_device_residual_galerkin_equation") is True for seed in seed_list
        ),
        "has_observed_block_schur_residual": any(
            seed.get("observed_qi_device_block_schur_residual") is True for seed in seed_list
        ),
        "promotion_eligible_seed_count": sum(1 for seed in seed_list if seed.get("promotion_eligible") is True),
    }


def _compact_execution_artifact(manifest: dict[str, object]) -> dict[str, object]:
    """Return a docs-friendly execution summary from a generated manifest."""
    execution = manifest.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("--summary-output requires --execute so execution results exist")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest has no cases to summarize")

    case_by_name = {str(case.get("case")): case for case in cases if isinstance(case, dict)}
    results = execution.get("results")
    if not isinstance(results, list):
        raise ValueError("manifest execution has no result list")

    seed_summaries: list[dict[str, object]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        trace = result.get("solver_trace_summary")
        trace_summary = trace if isinstance(trace, dict) else {}
        case = case_by_name.get(str(result.get("case")), {})
        progress_events = result.get("progress_events")
        stdout_tail = result.get("stdout_tail")
        stderr_tail = result.get("stderr_tail")
        log_context = [
            *(progress_events if isinstance(progress_events, list) else []),
            *(stdout_tail if isinstance(stdout_tail, list) else []),
            *(stderr_tail if isinstance(stderr_tail, list) else []),
        ]
        side_probe_progress = _infer_side_probe_progress(log_context)
        qi_deflated_progress = _infer_qi_deflated_progress(log_context)
        qi_device_progress = _infer_qi_device_progress(log_context)
        lgmres_rescue_status = _infer_lgmres_rescue_status(log_context, side_probe_progress)
        last_residual_progress = _infer_last_residual_progress(log_context) or {}
        augmented_krylov_progress = _infer_augmented_krylov_progress(log_context)
        last_matvecs, last_matvec_elapsed_s = _infer_last_matvec_progress(log_context)
        inferred_active_size, inferred_total_size = _infer_sizes_from_progress_events(log_context)
        active_size = trace_summary.get("active_size")
        total_size = trace_summary.get("total_size")

        def trace_or_progress(key: str) -> object:
            trace_value = trace_summary.get(key)
            if trace_value is not None:
                return trace_value
            return side_probe_progress.get(key)

        def trace_or_qi_deflated_progress(key: str) -> object:
            trace_value = trace_summary.get(key)
            if trace_value is not None:
                return trace_value
            return qi_deflated_progress.get(key)

        def trace_or_qi_device_progress(key: str) -> object:
            trace_value = trace_summary.get(key)
            if trace_value is not None:
                return trace_value
            return qi_device_progress.get(key)

        trace_lgmres_rescue = trace_summary.get("xblock_side_probe_lgmres_rescue")
        if lgmres_rescue_status is None and isinstance(trace_lgmres_rescue, bool):
            lgmres_rescue_status = "used" if trace_lgmres_rescue else "not_selected"

        seed_summary = {
            "case": result.get("case"),
            "seed": result.get("seed"),
            "returncode": result.get("returncode"),
            "timed_out": result.get("timed_out"),
            "output_exists": result.get("output_exists"),
            "solver_trace_exists": result.get("solver_trace_exists"),
            "elapsed_s": result.get("elapsed_s"),
            "solver_elapsed_s": trace_summary.get("elapsed_s"),
            "backend": trace_summary.get("backend"),
            "active_size": active_size if active_size is not None else inferred_active_size,
            "total_size": total_size if total_size is not None else inferred_total_size,
            "solve_method": trace_summary.get("solve_method"),
            "selected_path": trace_summary.get("selected_path"),
            "converged": trace_summary.get("converged"),
            "accepted_converged": trace_summary.get("accepted_converged"),
            "residual_norm": trace_summary.get("residual_norm"),
            "residual_target": trace_summary.get("residual_target"),
            "residual_ratio": trace_summary.get("residual_ratio"),
            "precondition_side": trace_or_progress("precondition_side"),
            "default_right_preconditioned": trace_summary.get("default_right_preconditioned"),
            "gmres_restart": trace_summary.get("gmres_restart"),
            "iterations": trace_summary.get("iterations"),
            "matvecs": trace_summary.get("matvecs"),
            "xblock_side_probe_used": trace_or_progress("xblock_side_probe_used"),
            "xblock_side_probe_decision": trace_or_progress("xblock_side_probe_decision"),
            "xblock_side_probe_switched": trace_or_progress("xblock_side_probe_switched"),
            "xblock_side_probe_initial_side": trace_or_progress("xblock_side_probe_initial_side"),
            "xblock_side_probe_selected_side": trace_or_progress("xblock_side_probe_selected_side"),
            "xblock_side_probe_initial_method": trace_or_progress("xblock_side_probe_initial_method"),
            "xblock_side_probe_selected_method": trace_or_progress("xblock_side_probe_selected_method"),
            "xblock_side_probe_lgmres_rescue": trace_or_progress("xblock_side_probe_lgmres_rescue"),
            "xblock_lgmres_rescue_status": lgmres_rescue_status,
            "xblock_lgmres_rescue_outer_k": trace_summary.get("xblock_lgmres_rescue_outer_k"),
            "xblock_side_probe_residual_ratio": trace_or_progress("xblock_side_probe_residual_ratio"),
            "xblock_side_probe_residual_norm": trace_or_progress("xblock_side_probe_residual_norm"),
            "xblock_side_probe_iterations": trace_or_progress("xblock_side_probe_iterations"),
            "xblock_side_probe_matvecs": trace_or_progress("xblock_side_probe_matvecs"),
            "xblock_device_host_fallback_used": trace_summary.get("xblock_device_host_fallback_used"),
            "xblock_device_host_fallback_mode": trace_summary.get("xblock_device_host_fallback_mode"),
            "xblock_device_host_fallback_reason": trace_summary.get("xblock_device_host_fallback_reason"),
            "xblock_device_host_fallback_requested_method": trace_summary.get(
                "xblock_device_host_fallback_requested_method"
            ),
            "xblock_device_host_fallback_effective_krylov_env_value": trace_summary.get(
                "xblock_device_host_fallback_effective_krylov_env_value"
            ),
            "xblock_device_host_fallback_non_autodiff": trace_summary.get(
                "xblock_device_host_fallback_non_autodiff"
            ),
            "resolution": case.get("resolution") if isinstance(case, dict) else None,
            "progress_events": progress_events,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "last_progress_residual_event": last_residual_progress.get("event"),
            "last_progress_residual_norm": last_residual_progress.get("residual_norm"),
            "last_progress_residual_target": last_residual_progress.get("residual_target"),
            "last_progress_residual_ratio": last_residual_progress.get("residual_ratio"),
            "last_progress_residual_before": last_residual_progress.get("residual_before"),
            "xblock_device_fgmres_qi_augmented_krylov_used": (
                trace_summary.get("xblock_device_fgmres_qi_augmented_krylov_used")
                if trace_summary.get("xblock_device_fgmres_qi_augmented_krylov_used") is not None
                else augmented_krylov_progress.get("xblock_device_fgmres_qi_augmented_krylov_used")
            ),
            "xblock_device_fgmres_qi_augmented_krylov_rank": (
                trace_summary.get("xblock_device_fgmres_qi_augmented_krylov_rank")
                if trace_summary.get("xblock_device_fgmres_qi_augmented_krylov_rank") is not None
                else augmented_krylov_progress.get("xblock_device_fgmres_qi_augmented_krylov_rank")
            ),
            "xblock_device_fgmres_qi_augmented_krylov_reason": (
                trace_summary.get("xblock_device_fgmres_qi_augmented_krylov_reason")
                if trace_summary.get("xblock_device_fgmres_qi_augmented_krylov_reason") is not None
                else augmented_krylov_progress.get("xblock_device_fgmres_qi_augmented_krylov_reason")
            ),
            "xblock_device_fgmres_qi_augmented_krylov_mode": (
                trace_summary.get("xblock_device_fgmres_qi_augmented_krylov_mode")
                if trace_summary.get("xblock_device_fgmres_qi_augmented_krylov_mode") is not None
                else augmented_krylov_progress.get("xblock_device_fgmres_qi_augmented_krylov_mode")
            ),
            "last_matvecs": last_matvecs,
            "last_matvec_elapsed_s": last_matvec_elapsed_s,
            "heartbeat": result.get("heartbeat"),
            "heartbeat_count": result.get("heartbeat_count"),
        }
        seed_summary.update({key: trace_summary.get(key) for key in QI_TWO_LEVEL_TRACE_KEYS})
        seed_summary.update({key: trace_or_qi_deflated_progress(key) for key in QI_DEFLATED_TRACE_KEYS})
        seed_summary.update({key: trace_or_qi_device_progress(key) for key in QI_DEVICE_TRACE_KEYS})
        seed_summary["xblock_qi_device_preconditioner_seed_only"] = trace_or_qi_device_progress(
            "xblock_qi_device_preconditioner_seed_only"
        )
        seed_summary["xblock_qi_device_preconditioner_operator_krylov_enrichment"] = (
            trace_or_qi_device_progress("xblock_qi_device_preconditioner_operator_krylov_enrichment")
        )
        seed_summary["xblock_qi_device_preconditioner_coarse_reuse"] = trace_or_qi_device_progress(
            "xblock_qi_device_preconditioner_coarse_reuse"
        )
        seed_summary.update({key: trace_summary.get(key) for key in MOMENT_SCHUR_TRACE_KEYS})
        classification = _seed_evidence_classification(seed_summary, manifest.get("probe_env"))
        seed_summary.update(
            {
                "evidence_class": classification["classification"],
                "evidence_tags": classification["tags"],
                "run_outcome": classification["outcome"],
                "promotion_eligible": classification["promotion_eligible"],
                "failed_before_summary_json": classification["failed_before_summary_json"],
                "requested_qi_device_installed_krylov": classification["requested_installed_krylov"],
                "requested_qi_device_operator_krylov": classification["requested_operator_krylov"],
                "requested_qi_device_coarse_reuse": classification["requested_coarse_reuse"],
                "requested_qi_device_augmented_krylov": classification["requested_augmented_krylov"],
                "requested_qi_device_coarse_residual_equation": classification[
                    "requested_coarse_residual_equation"
                ],
                "requested_qi_device_residual_snapshot": classification["requested_residual_snapshot"],
                "requested_qi_device_residual_snapshot_equation": classification[
                    "requested_residual_snapshot_equation"
                ],
                "requested_qi_device_global_moment_residual_equation": classification[
                    "requested_global_moment"
                ],
                "requested_qi_device_residual_galerkin_equation": classification[
                    "requested_residual_galerkin"
                ],
                "requested_qi_device_block_schur_residual": classification["requested_block_schur"],
                "observed_qi_device_installed_krylov": classification["observed_installed_krylov"],
                "observed_qi_device_operator_krylov": classification["observed_operator_krylov"],
                "observed_qi_device_coarse_reuse": classification["observed_coarse_reuse"],
                "observed_qi_device_augmented_krylov": classification["observed_augmented_krylov"],
                "observed_qi_device_coarse_residual_equation": classification[
                    "observed_coarse_residual_equation"
                ],
                "observed_qi_device_residual_snapshot": classification["observed_residual_snapshot"],
                "observed_qi_device_residual_snapshot_equation": classification[
                    "observed_residual_snapshot_equation"
                ],
                "observed_qi_device_global_moment_residual_equation": classification[
                    "observed_global_moment"
                ],
                "observed_qi_device_residual_galerkin_equation": classification[
                    "observed_residual_galerkin"
                ],
                "observed_qi_device_block_schur_residual": classification["observed_block_schur"],
            }
        )
        seed_summaries.append(seed_summary)

    first_case = cases[0]
    resolution = first_case.get("resolution") if isinstance(first_case, dict) else None
    source_input = Path(str(manifest["source_input"]))
    solve_method = str(manifest.get("solve_method", ""))
    probe_env = manifest.get("probe_env")
    has_probe_env = isinstance(probe_env, dict) and bool(probe_env)
    active_sizes = [
        int(seed["active_size"])
        for seed in seed_summaries
        if isinstance(seed.get("active_size"), (int, float)) and seed.get("active_size") is not None
    ]
    return {
        "schema_version": 2,
        "artifact_kind": "qi_seed_execution_summary",
        "lane": "qi_seed_robustness",
        "source_input": _repo_relative(source_input),
        "resolution_scale": manifest.get("resolution_scale"),
        "resolution": resolution,
        "active_size": max(active_sizes) if active_sizes else None,
        "total_size_estimate": _total_size_from_resolution(resolution) if isinstance(resolution, dict) else None,
        "case_count": manifest.get("case_count"),
        "public_cli_default_path": solve_method.strip().lower() in {"auto", "default", ""} and not has_probe_env,
        "solve_method_request": solve_method,
        "probe_preset": manifest.get("probe_preset"),
        "probe_env": manifest.get("probe_env"),
        "nu_jitter": manifest.get("nu_jitter"),
        "er_jitter": manifest.get("er_jitter"),
        "evidence_note": (
            "Bounded QI seed-robustness execution summary generated from the reusable runner "
            "manifest gate. Passing artifacts provide measured evidence at their recorded "
            "resolution; failed or timed-out artifacts are blocker evidence and must not be "
            "used for production promotion."
        ),
        "evidence_classification": _aggregate_seed_classifications(seed_summaries),
        "execution_summary": execution.get("summary"),
        "gates": execution.get("gates"),
        "seeds": seed_summaries,
        "timeout_s": execution.get("timeout_s"),
        "heartbeat_s": execution.get("heartbeat_s"),
        "fail_fast": execution.get("fail_fast"),
    }


def _write_compact_execution_artifact(path: Path, manifest: dict[str, object]) -> dict[str, object]:
    payload = _compact_execution_artifact(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _artifact_passed(payload: dict[str, object]) -> bool:
    gates = payload.get("gates")
    if isinstance(gates, dict) and isinstance(gates.get("passed"), bool):
        return bool(gates["passed"])
    if "passed" in payload and "failed" in payload:
        try:
            return int(payload["passed"]) > 0 and int(payload["failed"]) == 0
        except (TypeError, ValueError):
            return False
    return False


def _artifact_backends(payload: dict[str, object]) -> list[str]:
    summary = payload.get("execution_summary")
    if isinstance(summary, dict) and isinstance(summary.get("backends"), list):
        backends = sorted({str(backend) for backend in summary["backends"] if backend})
        if backends:
            return backends

    if payload.get("backend"):
        return [str(payload["backend"])]

    trace = payload.get("solver_trace_summary")
    if isinstance(trace, dict) and trace.get("backend"):
        return [str(trace["backend"])]

    runs = payload.get("runs")
    if isinstance(runs, dict):
        backends = set()
        for run in runs.values():
            if isinstance(run, dict) and run.get("backend") and run.get("process_passed") is not False:
                backends.add(str(run["backend"]))
        return sorted(backends)

    probes = payload.get("probes")
    if isinstance(probes, list):
        backends = {
            str(probe.get("backend"))
            for probe in probes
            if isinstance(probe, dict) and probe.get("backend")
        }
        return sorted(backends)

    rejected = payload.get("rejected_probes")
    if isinstance(rejected, list):
        backends = {
            str(probe.get("backend"))
            for probe in rejected
            if isinstance(probe, dict) and probe.get("backend")
        }
        if backends:
            return sorted(backends)

    device_operator_probe = payload.get("device_operator_probe")
    if isinstance(device_operator_probe, dict) and device_operator_probe.get("backend"):
        return [str(device_operator_probe["backend"])]
    return []


def _artifact_max_residual_ratio(payload: dict[str, object]) -> float | None:
    candidates: list[float] = []
    summary = payload.get("execution_summary")
    if isinstance(summary, dict):
        value = _finite_float_or_none(summary.get("max_residual_ratio"))
        if value is not None:
            candidates.append(value)

    trace = payload.get("solver_trace_summary")
    if isinstance(trace, dict):
        value = _finite_float_or_none(trace.get("residual_ratio"))
        if value is not None:
            candidates.append(value)

    runs = payload.get("runs")
    if isinstance(runs, dict):
        for run_name, run in runs.items():
            if not isinstance(run, dict) or str(run_name).endswith("before_patch"):
                continue
            value = _finite_float_or_none(run.get("residual_ratio"))
            if value is not None:
                candidates.append(value)
    probes = payload.get("probes")
    if isinstance(probes, list):
        for probe in probes:
            if not isinstance(probe, dict):
                continue
            value = _finite_float_or_none(probe.get("residual_ratio"))
            if value is not None:
                candidates.append(value)
    return max(candidates) if candidates else None


def _artifact_max_elapsed_s(payload: dict[str, object]) -> float | None:
    candidates: list[float] = []
    summary = payload.get("execution_summary")
    if isinstance(summary, dict):
        value = _finite_float_or_none(summary.get("max_elapsed_s"))
        if value is not None:
            candidates.append(value)

    value = _finite_float_or_none(payload.get("execution_elapsed_s"))
    if value is not None:
        candidates.append(value)

    runs = payload.get("runs")
    if isinstance(runs, dict):
        for run_name, run in runs.items():
            if not isinstance(run, dict) or str(run_name).endswith("before_patch"):
                continue
            value = _finite_float_or_none(run.get("elapsed_s"))
            if value is not None:
                candidates.append(value)
    probes = payload.get("probes")
    if isinstance(probes, list):
        for probe in probes:
            if not isinstance(probe, dict):
                continue
            value = _finite_float_or_none(probe.get("elapsed_s"))
            if value is not None:
                candidates.append(value)
    return max(candidates) if candidates else None


def _artifact_last_reported_residual_norm(payload: dict[str, object]) -> float | None:
    candidates: list[float] = []
    seeds = payload.get("seeds")
    if isinstance(seeds, list):
        for seed in seeds:
            if not isinstance(seed, dict):
                continue
            for key in (
                "last_progress_residual_norm",
                "xblock_qi_device_preconditioner_residual_after",
                "residual_norm",
            ):
                value = _finite_float_or_none(seed.get(key))
                if value is not None:
                    candidates.append(value)
                    break
    return max(candidates) if candidates else None


def _artifact_case_count(payload: dict[str, object]) -> int:
    try:
        case_count = int(payload.get("case_count", 0))
    except (TypeError, ValueError):
        case_count = 0
    if case_count > 0:
        return case_count
    return 1 if isinstance(payload.get("runs"), dict) else 0


def _artifact_evidence_classification(path: Path, payload: dict[str, object]) -> dict[str, object]:
    embedded = payload.get("evidence_classification")
    if isinstance(embedded, dict):
        classes = [str(item) for item in embedded.get("classes", []) if item]
        tags = [str(item) for item in embedded.get("tags", []) if item]
        outcomes = [str(item) for item in embedded.get("outcomes", []) if item]
        return {
            "classes": sorted(set(classes)),
            "tags": sorted(set(tags)),
            "outcomes": sorted(set(outcomes)),
            "has_failed_before_summary_json": bool(embedded.get("has_failed_before_summary_json")),
            "has_observed_installed_krylov": bool(embedded.get("has_observed_installed_krylov")),
            "has_observed_coarse_reuse": bool(embedded.get("has_observed_coarse_reuse")),
            "has_observed_multilevel_residual_equation": bool(
                embedded.get("has_observed_multilevel_residual_equation")
            ),
            "has_observed_residual_snapshot": bool(embedded.get("has_observed_residual_snapshot")),
            "has_observed_global_moment_residual_equation": bool(
                embedded.get("has_observed_global_moment_residual_equation")
            ),
            "has_observed_residual_galerkin_equation": bool(
                embedded.get("has_observed_residual_galerkin_equation")
            ),
            "has_observed_block_schur_residual": bool(embedded.get("has_observed_block_schur_residual")),
            "promotion_eligible_seed_count": embedded.get("promotion_eligible_seed_count"),
        }

    tags: set[str] = set()
    classes: set[str] = set()
    outcomes: set[str] = set()
    text = f"{path.name} {json.dumps(payload, sort_keys=True, default=str)}".lower()
    if _artifact_passed(payload):
        outcomes.add("process_passed")
    elif "timeout" in text:
        outcomes.add("timed_out")
    else:
        outcomes.add("process_failed")

    if "operator_krylov" in text or "operator-krylov" in text:
        tags.add("requested_operator_krylov")
    if "use_in_krylov" in text or "installed_krylov" in text or "installed-krylov" in text:
        tags.add("requested_installed_krylov")
    if "coarse_reuse" in text or "reuse_coarse" in text:
        tags.add("requested_coarse_reuse")
    if "multilevel_residual_equation" in text or "coarse-residual" in text:
        tags.add("requested_multilevel_residual_equation")
    if "residual_snapshot_residual_equation" in text or "residual-snapshot-equation" in text:
        tags.add("requested_residual_snapshot_residual_equation")
    if "residual_snapshot" in text or "residual-snapshot" in text:
        tags.add("requested_residual_snapshot")
    if "global_moment" in text or "global-moment" in text:
        tags.add("requested_global_moment_residual_equation")
    if "residual_galerkin" in text or "residual-galerkin" in text:
        tags.add("requested_residual_galerkin_equation")
    if "block_schur" in text or "block-schur" in text:
        tags.add("requested_block_schur_residual")
    if "solver_traces_written\": 0" in text or "solver_trace_exists\": false" in text:
        tags.add("failed_before_solver_trace_summary")

    if "observed_residual_galerkin_equation" in text or "observed_qi_device_residual_galerkin" in text:
        classes.add("device_qi_residual_galerkin_equation_coarse_reuse")
    elif "observed_global_moment_residual_equation" in text or "observed_qi_device_global_moment" in text:
        classes.add("device_qi_global_moment_residual_equation_coarse_reuse")
    elif "observed_block_schur_residual" in text or "observed_qi_device_block_schur_residual" in text:
        classes.add("device_qi_block_schur_residual_coarse_reuse")
    elif (
        "observed_residual_snapshot_residual_equation" in text
        or "observed_qi_device_residual_snapshot_equation" in text
    ):
        classes.add("device_qi_residual_snapshot_residual_equation_coarse_reuse")
    elif "observed_residual_snapshot" in text or "observed_qi_device_residual_snapshot" in text:
        classes.add("device_qi_residual_snapshot_coarse_reuse")
    elif "observed_multilevel_residual_equation" in text or "observed_qi_device_coarse_residual_equation" in text:
        classes.add("device_qi_multilevel_residual_equation")
    elif "observed_installed_krylov" in text:
        classes.add("device_qi_installed_krylov")
    elif "seed_only=1" in text:
        classes.add("device_qi_seed_only_probe")
    elif "residual_galerkin" in text or "residual-galerkin" in text:
        classes.add("requested_residual_galerkin_equation_device_qi")
    elif "global_moment" in text or "global-moment" in text:
        classes.add("requested_global_moment_residual_equation_device_qi")
    elif "block_schur" in text or "block-schur" in text:
        classes.add("requested_block_schur_residual_device_qi")
    elif "operator_krylov" in text or "operator-krylov" in text:
        classes.add("requested_operator_krylov_device_qi")
    elif "qi_device" in text:
        classes.add("requested_device_qi")
    else:
        classes.add("public_auto_or_legacy")

    return {
        "classes": sorted(classes),
        "tags": sorted(tags),
        "outcomes": sorted(outcomes),
        "has_failed_before_summary_json": "failed_before_solver_trace_summary" in tags,
        "has_observed_installed_krylov": (
            "observed_installed_krylov" in text
            or any(str(class_name).startswith("device_qi_") for class_name in classes)
        ),
        "has_observed_coarse_reuse": "observed_coarse_reuse" in text,
        "has_observed_multilevel_residual_equation": "observed_multilevel_residual_equation" in text,
        "has_observed_residual_snapshot": "observed_residual_snapshot" in text,
        "has_observed_global_moment_residual_equation": "observed_global_moment_residual_equation" in text,
        "has_observed_residual_galerkin_equation": "observed_residual_galerkin_equation" in text,
        "has_observed_block_schur_residual": "observed_block_schur_residual" in text,
        "promotion_eligible_seed_count": None,
    }


def _summarize_evidence_artifact(path: Path, payload: dict[str, object], production_resolution: dict[str, int]) -> dict[str, object]:
    resolution_dict = _canonical_resolution(payload.get("resolution"))
    total_size = _finite_float_or_none(payload.get("total_size"))
    if total_size is None:
        total_size = _finite_float_or_none(payload.get("total_size_estimate"))
    if total_size is None:
        estimate = _total_size_from_resolution(resolution_dict)
        total_size = float(estimate) if estimate is not None else None
    active_size = _finite_float_or_none(payload.get("active_size"))
    if active_size is None:
        active_size = _finite_float_or_none(payload.get("active_size_estimate"))
    classification = _artifact_evidence_classification(path, payload)
    return {
        "path": _repo_relative(path),
        "schema_version": payload.get("schema_version"),
        "artifact_kind": payload.get("artifact_kind", "legacy_qi_seed_summary"),
        "passed": _artifact_passed(payload),
        "case_count": _artifact_case_count(payload),
        "backends": _artifact_backends(payload),
        "public_cli_default_path": payload.get("public_cli_default_path"),
        "resolution": resolution_dict or payload.get("resolution"),
        "resolution_fractions": _resolution_fractions(resolution_dict, production_resolution),
        "total_size": int(total_size) if total_size is not None else None,
        "active_size": int(active_size) if active_size is not None else None,
        "max_residual_ratio": _artifact_max_residual_ratio(payload),
        "max_elapsed_s": _artifact_max_elapsed_s(payload),
        "last_reported_residual_norm": _artifact_last_reported_residual_norm(payload),
        "evidence_classification": classification,
        "evidence_classes": classification["classes"],
        "evidence_tags": classification["tags"],
        "run_outcomes": classification["outcomes"],
    }


def build_evidence_manifest(
    *,
    artifact_paths: Iterable[Path],
    source_input: Path,
    production_seed_count: int,
    production_timeout_s: float,
) -> dict[str, object]:
    """Build the QI production-readiness manifest from checked summary artifacts."""
    source_text = source_input.read_text(encoding="utf-8")
    source_resolution = _read_resolution(source_text)
    production_resolution = _scaled_resolution(
        source_resolution,
        scale=1.0,
        min_ntheta=int(source_resolution.get("NTHETA", 25)),
        min_nzeta=int(source_resolution.get("NZETA", 51)),
        min_nx=int(source_resolution.get("NX", 8)),
        min_nxi=int(source_resolution.get("NXI", 100)),
    )
    production_total_size = _total_size_from_resolution(production_resolution)

    artifacts: list[dict[str, object]] = []
    seen_artifacts: set[Path] = set()
    for path in artifact_paths:
        resolved_path = path.resolve()
        if resolved_path in seen_artifacts or not resolved_path.exists():
            continue
        seen_artifacts.add(resolved_path)
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            artifacts.append(_summarize_evidence_artifact(resolved_path, payload, production_resolution))

    passed_artifacts = [artifact for artifact in artifacts if artifact.get("passed") is True]
    nonpassing_artifacts = [artifact for artifact in artifacts if artifact.get("passed") is not True]
    max_total_size = max(
        (int(artifact["total_size"]) for artifact in passed_artifacts if artifact.get("total_size") is not None),
        default=0,
    )
    max_active_size = max(
        (int(artifact["active_size"]) for artifact in passed_artifacts if artifact.get("active_size") is not None),
        default=0,
    )
    max_per_axis_fraction = max(
        (
            min(float(value) for value in artifact["resolution_fractions"].values())
            for artifact in passed_artifacts
            if isinstance(artifact.get("resolution_fractions"), dict) and artifact["resolution_fractions"]
        ),
        default=0.0,
    )
    largest_attempted_total_size = max(
        (int(artifact["total_size"]) for artifact in artifacts if artifact.get("total_size") is not None),
        default=0,
    )
    largest_nonpassing_total_size = max(
        (int(artifact["total_size"]) for artifact in nonpassing_artifacts if artifact.get("total_size") is not None),
        default=0,
    )
    max_total_fraction = (
        float(max_total_size) / float(production_total_size)
        if production_total_size is not None and production_total_size > 0
        else None
    )
    checked_backends = sorted(
        {
            str(backend)
            for artifact in passed_artifacts
            for backend in artifact.get("backends", [])
            if backend
        }
    )
    evidence_class_counts: dict[str, int] = {}
    evidence_tag_counts: dict[str, int] = {}
    failed_before_summary_json_count = 0
    for artifact in artifacts:
        classes = artifact.get("evidence_classes")
        if isinstance(classes, list):
            for class_name in classes:
                evidence_class_counts[str(class_name)] = evidence_class_counts.get(str(class_name), 0) + 1
        tags = artifact.get("evidence_tags")
        if isinstance(tags, list):
            for tag in tags:
                evidence_tag_counts[str(tag)] = evidence_tag_counts.get(str(tag), 0) + 1
        classification = artifact.get("evidence_classification")
        if isinstance(classification, dict) and classification.get("has_failed_before_summary_json") is True:
            failed_before_summary_json_count += 1
    non_autodiff_fallback_artifacts = [
        artifact
        for artifact in passed_artifacts
        if "device_host_fallback" in str(artifact.get("path", ""))
    ]
    non_autodiff_fallback_ready = bool(non_autodiff_fallback_artifacts)
    true_device_qi_blockers = [
        artifact
        for artifact in nonpassing_artifacts
        if (
            "scale060" in str(artifact.get("path", ""))
            and (
                "gpu" in str(artifact.get("path", ""))
                or "device" in str(artifact.get("path", ""))
                or "galerkin" in str(artifact.get("path", ""))
            )
        )
    ]

    cpu_command = [
        "JAX_PLATFORM_NAME=cpu",
        "python",
        "scripts/run_qi_seed_robustness.py",
        "--out-root",
        "tests/qi_seed_robustness_prod_cpu",
        "--seeds",
        *[str(seed) for seed in range(int(production_seed_count))],
        "--resolution-scale",
        "1.0",
        "--min-ntheta",
        str(production_resolution["NTHETA"]),
        "--min-nzeta",
        str(production_resolution["NZETA"]),
        "--min-nx",
        str(production_resolution["NX"]),
        "--min-nxi",
        str(production_resolution["NXI"]),
        "--execute",
        "--timeout-s",
        str(float(production_timeout_s)),
        "--max-residual-ratio",
        "1",
        "--require-converged",
        "--summary-output",
        "docs/_static/qi_seed_robustness_prod_cpu.json",
        "--clean",
    ]
    gpu_command = [
        "python",
        "scripts/run_qi_seed_robustness.py",
        "--out-root",
        "tests/qi_seed_robustness_prod_gpu0",
        "--seeds",
        *[str(seed) for seed in range(int(production_seed_count))],
        "--resolution-scale",
        "1.0",
        "--min-ntheta",
        str(production_resolution["NTHETA"]),
        "--min-nzeta",
        str(production_resolution["NZETA"]),
        "--min-nx",
        str(production_resolution["NX"]),
        "--min-nxi",
        str(production_resolution["NXI"]),
        "--execute",
        "--timeout-s",
        str(float(production_timeout_s)),
        "--max-residual-ratio",
        "1",
        "--require-converged",
        "--summary-output",
        "docs/_static/qi_seed_robustness_prod_gpu0.json",
        "--clean",
    ]
    operator_krylov_probe_command = [
        "python",
        "scripts/run_qi_seed_robustness.py",
        "--out-root",
        "tests/qi_seed_robustness_scale060_operator_krylov_device_qi_gpu0",
        "--seeds",
        "3",
        "--resolution-scale",
        "0.60",
        "--probe-preset",
        OPERATOR_KRYLOV_DEVICE_QI_PROBE_PRESET,
        "--solve-method",
        OPERATOR_KRYLOV_DEVICE_QI_SOLVE_METHOD,
        "--execute",
        "--timeout-s",
        "900",
        "--heartbeat-s",
        "15",
        "--summary-output",
        "docs/_static/qi_seed_robustness_scale060_operator_krylov_device_qi_gpu0.json",
        "--clean",
    ]
    gpu_env = {"CUDA_VISIBLE_DEVICES": "0", "JAX_PLATFORM_NAME": "gpu"}
    operator_krylov_probe_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_PLATFORM_NAME": "gpu",
        **OPERATOR_KRYLOV_DEVICE_QI_ENV,
    }
    current_constraint_probe_command = [
        *operator_krylov_probe_command[:9],
        CURRENT_CONSTRAINT_DEVICE_QI_PROBE_PRESET,
        *operator_krylov_probe_command[10:],
    ]
    current_constraint_probe_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_PLATFORM_NAME": "gpu",
        **CURRENT_CONSTRAINT_DEVICE_QI_ENV,
    }
    adjoint_krylov_probe_command = [
        *operator_krylov_probe_command[:9],
        ADJOINT_KRYLOV_DEVICE_QI_PROBE_PRESET,
        *operator_krylov_probe_command[10:],
    ]
    adjoint_krylov_probe_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_PLATFORM_NAME": "gpu",
        **ADJOINT_KRYLOV_DEVICE_QI_ENV,
    }
    augmented_krylov_probe_command = [
        *operator_krylov_probe_command[:9],
        AUGMENTED_KRYLOV_DEVICE_QI_PROBE_PRESET,
        *operator_krylov_probe_command[10:],
    ]
    augmented_krylov_probe_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_PLATFORM_NAME": "gpu",
        **AUGMENTED_KRYLOV_DEVICE_QI_ENV,
    }
    coarse_residual_probe_command = [
        *operator_krylov_probe_command[:9],
        COARSE_RESIDUAL_DEVICE_QI_PROBE_PRESET,
        *operator_krylov_probe_command[10:],
    ]
    coarse_residual_probe_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_PLATFORM_NAME": "gpu",
        **COARSE_RESIDUAL_DEVICE_QI_ENV,
    }
    residual_snapshot_probe_command = [
        *operator_krylov_probe_command[:9],
        RESIDUAL_SNAPSHOT_DEVICE_QI_PROBE_PRESET,
        *operator_krylov_probe_command[10:],
    ]
    residual_snapshot_probe_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_PLATFORM_NAME": "gpu",
        **RESIDUAL_SNAPSHOT_DEVICE_QI_ENV,
    }
    residual_snapshot_equation_probe_command = [
        *operator_krylov_probe_command[:9],
        RESIDUAL_SNAPSHOT_EQUATION_DEVICE_QI_PROBE_PRESET,
        *operator_krylov_probe_command[10:],
    ]
    residual_snapshot_equation_probe_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_PLATFORM_NAME": "gpu",
        **RESIDUAL_SNAPSHOT_EQUATION_DEVICE_QI_ENV,
    }
    assembled_reuse_probe_command = [
        *operator_krylov_probe_command[:9],
        ASSEMBLED_REUSE_DEVICE_QI_PROBE_PRESET,
        *operator_krylov_probe_command[10:],
    ]
    assembled_reuse_probe_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_PLATFORM_NAME": "gpu",
        **ASSEMBLED_REUSE_DEVICE_QI_ENV,
    }
    composite_closure_probe_command = [
        *operator_krylov_probe_command[:9],
        COMPOSITE_CLOSURE_DEVICE_QI_PROBE_PRESET,
        *operator_krylov_probe_command[10:],
    ]
    composite_closure_probe_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_PLATFORM_NAME": "gpu",
        **COMPOSITE_CLOSURE_DEVICE_QI_ENV,
    }
    global_moment_probe_command = [
        *operator_krylov_probe_command[:9],
        GLOBAL_MOMENT_CLOSURE_DEVICE_QI_PROBE_PRESET,
        *operator_krylov_probe_command[10:],
    ]
    global_moment_probe_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_PLATFORM_NAME": "gpu",
        **GLOBAL_MOMENT_CLOSURE_DEVICE_QI_ENV,
    }
    residual_galerkin_probe_command = [
        *operator_krylov_probe_command[:9],
        RESIDUAL_GALERKIN_DEVICE_QI_PROBE_PRESET,
        *operator_krylov_probe_command[10:],
    ]
    residual_galerkin_probe_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_PLATFORM_NAME": "gpu",
        **RESIDUAL_GALERKIN_DEVICE_QI_ENV,
    }
    block_schur_probe_command = [
        *operator_krylov_probe_command[:9],
        BLOCK_SCHUR_DEVICE_QI_PROBE_PRESET,
        *operator_krylov_probe_command[10:],
    ]
    block_schur_probe_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_PLATFORM_NAME": "gpu",
        **BLOCK_SCHUR_DEVICE_QI_ENV,
    }

    return {
        "schema_version": 1,
        "artifact_kind": "qi_seed_production_gate_manifest",
        "lane": "qi_seed_robustness",
        "source_input": _repo_relative(source_input),
        "release_gate": "bounded_proxy",
        "release_gate_reason": (
            "Scoped production non-autodiff host fallback evidence is release-ready; "
            "the broad public auto QI gate remains a bounded proxy until production-resolution "
            "CPU/GPU ladders pass, and true device-QI is closed deferred post-release."
        ),
        "release_claims": {
            "production_non_autodiff_host_fallback": {
                "claim_status": "release_ready" if non_autodiff_fallback_ready else "bounded_proxy",
                "blocks_current_release": False,
                "scope": (
                    "Explicit large-QI device-Krylov requests may use the non-autodiff host "
                    "x-block sparse-PC fallback. This is production fallback coverage, not "
                    "a differentiable or true device-resident QI solve claim."
                ),
                "evidence": (
                    str(non_autodiff_fallback_artifacts[0].get("path"))
                    if non_autodiff_fallback_artifacts
                    else "missing passing device_host_fallback artifact"
                ),
                "promotion_gate": (
                    "Keep release-ready only while a passing artifact records the host fallback "
                    "metadata, writes output/trace, and preserves residual convergence under "
                    "the QI gate."
                ),
            },
            "true_device_qi": {
                "claim_status": "closed_deferred",
                "blocks_current_release": False,
                "closed_or_deferred_reason": (
                    "Closed post-release: scale-0.60 GPU hard-seed device/Galerkin/x-block "
                    "routes time out or fail residual probes. The next bounded evidence path "
                    "is an operator-Krylov device-QI probe, while production fallback is covered "
                    "by the non-autodiff host path."
                ),
                "evidence": [str(artifact.get("path")) for artifact in true_device_qi_blockers],
                "promotion_gate": (
                    "Promote only after a scale-0.60 GPU hard-seed artifact writes HDF5, "
                    "solver trace, accepted-converged residual metadata, and shows the "
                    "device-resident QI path reduces the true residual before timeout."
                ),
            },
            "public_auto_production_ladder": {
                "claim_status": "bounded_proxy",
                "blocks_current_release": False,
                "scope": (
                    "Checked artifacts are bounded-resolution QI evidence. Production-resolution "
                    "CPU/GPU multi-seed public-auto ladders remain the promotion gate for a "
                    "broad default-policy claim."
                ),
                "evidence": "source_artifacts",
                "promotion_gate": (
                    "Promote only after production CPU and GPU public-auto ladder artifacts "
                    "pass residual, convergence, output, and solver-trace gates."
                ),
            },
        },
        "source_artifacts": artifacts,
        "current_evidence": {
            "artifact_count": len(artifacts),
            "passing_artifact_count": len(passed_artifacts),
            "nonpassing_artifact_count": len(nonpassing_artifacts),
            "checked_backends": checked_backends,
            "max_checked_total_size": max_total_size,
            "max_checked_active_size": max_active_size or None,
            "largest_attempted_total_size": largest_attempted_total_size or None,
            "largest_nonpassing_total_size": largest_nonpassing_total_size or None,
            "max_checked_total_size_fraction": max_total_fraction,
            "max_checked_per_axis_resolution_fraction": max_per_axis_fraction,
            "bounded_lane_completion_estimate_percent": round(100.0 * max_per_axis_fraction, 1),
            "production_total_size_uncovered_percent": (
                round(100.0 * (1.0 - max_total_fraction), 2) if max_total_fraction is not None else None
            ),
            "completion_estimate_basis": "largest passing measured artifact only",
            "evidence_class_counts": dict(sorted(evidence_class_counts.items())),
            "evidence_tag_counts": dict(sorted(evidence_tag_counts.items())),
            "failed_before_summary_json_count": failed_before_summary_json_count,
        },
        "production_target": {
            "resolution": production_resolution,
            "total_size_estimate": production_total_size,
            "seed_count": int(production_seed_count),
            "required_backends": ["cpu", "gpu"],
        },
        "acceptance_gates": {
            "public_cli_default_path": True,
            "solve_method": "auto",
            "process_failed": 0,
            "timed_out": 0,
            "outputs_written": int(production_seed_count),
            "solver_traces_written": int(production_seed_count),
            "converged": int(production_seed_count),
            "max_residual_ratio": 1.0,
            "required_backends": ["cpu", "gpu"],
            "required_artifacts": [
                "docs/_static/qi_seed_robustness_prod_cpu.json",
                "docs/_static/qi_seed_robustness_prod_gpu0.json",
            ],
        },
        "regeneration_commands": {
            "refresh_evidence_manifest": (
                "python scripts/run_qi_seed_robustness.py "
                "--summarize-artifacts-only "
                "--evidence-manifest-output docs/_static/qi_seed_robustness_evidence_manifest.json"
            ),
            "production_cpu_seed_ladder": " ".join(cpu_command),
            "production_gpu0_seed_ladder": _shell_command(gpu_command, gpu_env),
            "operator_krylov_device_qi_gpu0_probe": _shell_command(
                operator_krylov_probe_command,
                operator_krylov_probe_env,
            ),
            "current_constraint_device_qi_gpu0_probe": _shell_command(
                current_constraint_probe_command,
                current_constraint_probe_env,
            ),
            "adjoint_krylov_device_qi_gpu0_probe": _shell_command(
                adjoint_krylov_probe_command,
                adjoint_krylov_probe_env,
            ),
            "augmented_krylov_device_qi_gpu0_probe": _shell_command(
                augmented_krylov_probe_command,
                augmented_krylov_probe_env,
            ),
            "coarse_residual_device_qi_gpu0_probe": _shell_command(
                coarse_residual_probe_command,
                coarse_residual_probe_env,
            ),
            "residual_snapshot_device_qi_gpu0_probe": _shell_command(
                residual_snapshot_probe_command,
                residual_snapshot_probe_env,
            ),
            "residual_snapshot_equation_device_qi_gpu0_probe": _shell_command(
                residual_snapshot_equation_probe_command,
                residual_snapshot_equation_probe_env,
            ),
            "assembled_reuse_device_qi_gpu0_probe": _shell_command(
                assembled_reuse_probe_command,
                assembled_reuse_probe_env,
            ),
            "composite_closure_device_qi_gpu0_probe": _shell_command(
                composite_closure_probe_command,
                composite_closure_probe_env,
            ),
            "global_moment_closure_device_qi_gpu0_probe": _shell_command(
                global_moment_probe_command,
                global_moment_probe_env,
            ),
            "residual_galerkin_device_qi_gpu0_probe": _shell_command(
                residual_galerkin_probe_command,
                residual_galerkin_probe_env,
            ),
            "block_schur_device_qi_gpu0_probe": _shell_command(
                block_schur_probe_command,
                block_schur_probe_env,
            ),
        },
        "probe_presets": {
            OPERATOR_KRYLOV_DEVICE_QI_PROBE_PRESET: {
                "description": (
                    "Bounded scale-0.60 hard-seed GPU0 production probe for true device-QI "
                    "operator-Krylov enrichment. This records a fail-closed evidence path; "
                    "it is not a public default-policy solve."
                ),
                "env": dict(sorted(OPERATOR_KRYLOV_DEVICE_QI_ENV.items())),
                "recommended_command": _shell_command(operator_krylov_probe_command, operator_krylov_probe_env),
            },
            CURRENT_CONSTRAINT_DEVICE_QI_PROBE_PRESET: {
                "description": (
                    "Bounded scale-0.60 hard-seed GPU0 production probe for true device-QI "
                    "current/constraint multilevel coarse moments. The 2026-05-20 evidence "
                    "did not beat the pitch/operator-Krylov baseline, so this remains an "
                    "opt-in fail-closed research probe."
                ),
                "env": dict(sorted(CURRENT_CONSTRAINT_DEVICE_QI_ENV.items())),
                "recommended_command": _shell_command(
                    current_constraint_probe_command,
                    current_constraint_probe_env,
                ),
            },
            ADJOINT_KRYLOV_DEVICE_QI_PROBE_PRESET: {
                "description": (
                    "Bounded scale-0.60 hard-seed GPU0 production probe for true device-QI "
                    "adjoint-normal Krylov enrichment. This targets non-normal left-error "
                    "modes and remains a fail-closed research probe, not a public default."
                ),
                "env": dict(sorted(ADJOINT_KRYLOV_DEVICE_QI_ENV.items())),
                "recommended_command": _shell_command(adjoint_krylov_probe_command, adjoint_krylov_probe_env),
            },
            AUGMENTED_KRYLOV_DEVICE_QI_PROBE_PRESET: {
                "description": (
                    "Bounded scale-0.60 hard-seed GPU0 production probe for the true "
                    "operator-reuse path: a JAX FGMRES solve augmented by the reusable "
                    "QI coarse basis and stored operator-on-basis action."
                ),
                "env": dict(sorted(AUGMENTED_KRYLOV_DEVICE_QI_ENV.items())),
                "recommended_command": _shell_command(
                    augmented_krylov_probe_command,
                    augmented_krylov_probe_env,
                ),
            },
            COARSE_RESIDUAL_DEVICE_QI_PROBE_PRESET: {
                "description": (
                    "Bounded scale-0.60 hard-seed GPU0 production probe for the "
                    "multilevel residual-equation coarse path. This targets coarse "
                    "error components stage-by-stage instead of changing smoother "
                    "restart/projection parameters."
                ),
                "env": dict(sorted(COARSE_RESIDUAL_DEVICE_QI_ENV.items())),
                "recommended_command": _shell_command(
                    coarse_residual_probe_command,
                    coarse_residual_probe_env,
                ),
            },
            RESIDUAL_SNAPSHOT_DEVICE_QI_PROBE_PRESET: {
                "description": (
                    "Bounded scale-0.60 hard-seed GPU0 production probe for the "
                    "residual-snapshot coarse path. This enriches the reusable "
                    "device-QI coarse operator with block/aggregate snapshots of "
                    "the actual current residual."
                ),
                "env": dict(sorted(RESIDUAL_SNAPSHOT_DEVICE_QI_ENV.items())),
                "recommended_command": _shell_command(
                    residual_snapshot_probe_command,
                    residual_snapshot_probe_env,
                ),
            },
            RESIDUAL_SNAPSHOT_EQUATION_DEVICE_QI_PROBE_PRESET: {
                "description": (
                    "Bounded scale-0.60 hard-seed GPU0 production probe for the "
                    "staged residual-snapshot residual-equation path. This turns "
                    "block/aggregate residual snapshots into per-stage cached A Q_l "
                    "coarse solves and remains fail-closed unless output and solver "
                    "trace convergence are observed."
                ),
                "env": dict(sorted(RESIDUAL_SNAPSHOT_EQUATION_DEVICE_QI_ENV.items())),
                "recommended_command": _shell_command(
                    residual_snapshot_equation_probe_command,
                    residual_snapshot_equation_probe_env,
                ),
            },
            ASSEMBLED_REUSE_DEVICE_QI_PROBE_PRESET: {
                "description": (
                    "Bounded scale-0.60 hard-seed GPU0 production probe for the "
                    "assembled/operator-reuse QI path. This uses the residual-snapshot "
                    "device-QI coarse route but requires a device-resident assembled CSR "
                    "operator so Krylov matvecs, setup probes, and cached A Q actions "
                    "reuse the same operator representation."
                ),
                "env": dict(sorted(ASSEMBLED_REUSE_DEVICE_QI_ENV.items())),
                "recommended_command": _shell_command(
                    assembled_reuse_probe_command,
                    assembled_reuse_probe_env,
                ),
            },
            COMPOSITE_CLOSURE_DEVICE_QI_PROBE_PRESET: {
                "description": (
                    "Bounded scale-0.60 hard-seed GPU0 production probe for the "
                    "composite residual-snapshot plus residual-Galerkin and "
                    "block-Schur coarse-closure path. This is the non-smoother "
                    "follow-up after full assembled CSR reuse and individual "
                    "coarse spaces failed to close the hard seed."
                ),
                "env": dict(sorted(COMPOSITE_CLOSURE_DEVICE_QI_ENV.items())),
                "recommended_command": _shell_command(
                    composite_closure_probe_command,
                    composite_closure_probe_env,
                ),
            },
            GLOBAL_MOMENT_CLOSURE_DEVICE_QI_PROBE_PRESET: {
                "description": (
                    "Bounded scale-0.60 hard-seed GPU0 production probe for the "
                    "global moment closure residual-equation path. This builds a "
                    "rank-gated Galerkin Schur closure over profile, current, and "
                    "tail-constraint moments and remains fail-closed unless the "
                    "solver trace reports observed global-moment metadata and convergence."
                ),
                "env": dict(sorted(GLOBAL_MOMENT_CLOSURE_DEVICE_QI_ENV.items())),
                "recommended_command": _shell_command(
                    global_moment_probe_command,
                    global_moment_probe_env,
                ),
            },
            RESIDUAL_GALERKIN_DEVICE_QI_PROBE_PRESET: {
                "description": (
                    "Bounded scale-0.60 hard-seed GPU0 production probe for the "
                    "residual-derived Galerkin coarse path. This builds coarse "
                    "variables from the actual remaining operator residual and "
                    "block residuals, caches A Q, and remains fail-closed unless "
                    "output, solver trace, and convergence are observed."
                ),
                "env": dict(sorted(RESIDUAL_GALERKIN_DEVICE_QI_ENV.items())),
                "recommended_command": _shell_command(
                    residual_galerkin_probe_command,
                    residual_galerkin_probe_env,
                ),
            },
            BLOCK_SCHUR_DEVICE_QI_PROBE_PRESET: {
                "description": (
                    "Bounded scale-0.60 hard-seed GPU0 production probe for the "
                    "block-Schur residual coarse path. This targets off-block "
                    "coupling residuals using setup-time A.T P_off A P_block "
                    "directions while keeping the apply path on cached A Q. "
                    "This is fail-closed unless the solver trace reports "
                    "observed block-Schur residual metadata and convergence."
                ),
                "env": dict(sorted(BLOCK_SCHUR_DEVICE_QI_ENV.items())),
                "recommended_command": _shell_command(
                    block_schur_probe_command,
                    block_schur_probe_env,
                ),
            },
        },
        "open_blockers": [
            "Run and check in production-resolution CPU multi-seed summary artifact before promoting broad public-auto QI.",
            "Run and check in production-resolution GPU0 multi-seed summary artifact before promoting broad public-auto QI.",
            "Keep true device-QI closed deferred until an operator-Krylov device-QI scale-0.60 GPU hard-seed artifact writes output, trace, and accepted-converged residual metadata.",
            "Keep the global-moment closure device-QI probe fail-closed until a converged artifact reports observed global-moment residual-equation metadata.",
            "Keep the residual-Galerkin device-QI probe fail-closed until a converged artifact reports observed residual-derived Galerkin metadata.",
            "Keep the block-Schur residual device-QI probe fail-closed until a converged artifact reports observed block-Schur residual metadata.",
            "Do not use the non-autodiff host fallback as evidence for differentiable or true device-resident QI.",
        ],
    }


def _write_evidence_manifest(
    *,
    output_path: Path,
    artifact_paths: Iterable[Path],
    source_input: Path,
    production_seed_count: int,
    production_timeout_s: float,
) -> dict[str, object]:
    payload = build_evidence_manifest(
        artifact_paths=artifact_paths,
        source_input=source_input,
        production_seed_count=production_seed_count,
        production_timeout_s=production_timeout_s,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_QI_INPUT, help="Base QI input.namelist.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Directory for generated seed cases.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Deterministic seed ids to materialize.")
    parser.add_argument("--resolution-scale", type=float, default=0.25, help="Scale applied to NTHETA/NZETA/NX/NXI.")
    parser.add_argument("--min-ntheta", type=int, default=7)
    parser.add_argument("--min-nzeta", type=int, default=11)
    parser.add_argument("--min-nx", type=int, default=4)
    parser.add_argument("--min-nxi", type=int, default=16)
    parser.add_argument("--nu-jitter", type=float, default=0.05, help="Relative symmetric nu_n jitter per seed.")
    parser.add_argument("--er-jitter", type=float, default=0.02, help="Additive symmetric Er jitter per seed.")
    parser.add_argument(
        "--solve-method",
        default="auto",
        help=(
            "RHSMode=1 solve method passed to sfincs_jax write-output when --execute is set. "
            "The default is auto, which exercises the public CLI solver policy. Pass an explicit "
            "method such as dense or sparse_lsmr only for diagnostic probes."
        ),
    )
    parser.add_argument(
        "--probe-preset",
        choices=(
            "none",
            OPERATOR_KRYLOV_DEVICE_QI_PROBE_PRESET,
            CURRENT_CONSTRAINT_DEVICE_QI_PROBE_PRESET,
            ADJOINT_KRYLOV_DEVICE_QI_PROBE_PRESET,
            AUGMENTED_KRYLOV_DEVICE_QI_PROBE_PRESET,
            COARSE_RESIDUAL_DEVICE_QI_PROBE_PRESET,
            RESIDUAL_SNAPSHOT_DEVICE_QI_PROBE_PRESET,
            RESIDUAL_SNAPSHOT_EQUATION_DEVICE_QI_PROBE_PRESET,
            ASSEMBLED_REUSE_DEVICE_QI_PROBE_PRESET,
            COMPOSITE_CLOSURE_DEVICE_QI_PROBE_PRESET,
            GLOBAL_MOMENT_CLOSURE_DEVICE_QI_PROBE_PRESET,
            RESIDUAL_GALERKIN_DEVICE_QI_PROBE_PRESET,
            BLOCK_SCHUR_DEVICE_QI_PROBE_PRESET,
        ),
        default="none",
        help=(
            "Optional explicit QI evidence probe environment. "
            f"{OPERATOR_KRYLOV_DEVICE_QI_PROBE_PRESET!r} enables the bounded operator-Krylov "
            "device-QI production probe controls and records them in generated manifests."
        ),
    )
    parser.add_argument("--execute", action="store_true", help="Run each generated seed through sfincs_jax write-output.")
    parser.add_argument("--timeout-s", type=float, default=300.0, help="Per-seed execution timeout.")
    parser.add_argument(
        "--heartbeat-s",
        type=float,
        default=0.0,
        help=(
            "Optional per-seed heartbeat interval. When positive, the runner writes "
            "case-local runner_heartbeat.jsonl events and enforces timeout by killing "
            "the whole subprocess process group."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop executing after the first failed seed.")
    parser.add_argument(
        "--max-residual-ratio",
        type=float,
        default=None,
        help="Optional promotion gate: every solver trace residual_norm/residual_target must be at or below this value.",
    )
    parser.add_argument(
        "--require-converged",
        action="store_true",
        help="Optional promotion gate: require every solver trace to report converged=true.",
    )
    parser.add_argument(
        "--require-accepted-converged",
        action="store_true",
        help="Optional promotion gate: require every solver trace metadata to report accepted_converged=true.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional compact JSON artifact written from the executed manifest for docs/_static.",
    )
    parser.add_argument(
        "--evidence-manifest-output",
        type=Path,
        default=None,
        help="Optional production-readiness manifest summarizing checked QI docs/_static artifacts.",
    )
    parser.add_argument(
        "--evidence-artifacts",
        type=Path,
        nargs="+",
        default=None,
        help="QI summary artifacts to include in --evidence-manifest-output.",
    )
    parser.add_argument(
        "--production-seed-count",
        type=int,
        default=5,
        help="Seed count required by generated production-resolution acceptance commands.",
    )
    parser.add_argument(
        "--production-timeout-s",
        type=float,
        default=3600.0,
        help="Per-seed timeout for generated production-resolution acceptance commands.",
    )
    parser.add_argument(
        "--summarize-artifacts-only",
        action="store_true",
        help="Only write --evidence-manifest-output; do not materialize or execute seed cases.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove --out-root before materializing cases.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_input = Path(args.input).resolve()
    if not source_input.exists():
        raise FileNotFoundError(source_input)

    evidence_artifacts = [
        Path(path).resolve()
        for path in (args.evidence_artifacts if args.evidence_artifacts is not None else DEFAULT_EVIDENCE_ARTIFACTS)
    ]
    if bool(args.summarize_artifacts_only):
        if args.evidence_manifest_output is None:
            raise ValueError("--summarize-artifacts-only requires --evidence-manifest-output")
        _write_evidence_manifest(
            output_path=Path(args.evidence_manifest_output).resolve(),
            artifact_paths=evidence_artifacts,
            source_input=source_input,
            production_seed_count=int(args.production_seed_count),
            production_timeout_s=float(args.production_timeout_s),
        )
        print(f"Wrote {Path(args.evidence_manifest_output).resolve()}")
        return 0

    out_root = Path(args.out_root).resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    source_text = source_input.read_text(encoding="utf-8")
    source_resolution = _read_resolution(source_text)
    source_equilibrium = _resolve_equilibrium(source_input, source_text)
    probe_env = _probe_env_for_preset(str(args.probe_preset))
    solve_method = _solve_method_for_probe_preset(
        solve_method=str(args.solve_method),
        probe_preset=str(args.probe_preset),
    )
    cases = [
        _materialize_case(
            seed=int(seed),
            source_input=source_input,
            source_text=source_text,
            source_resolution=source_resolution,
            source_equilibrium=source_equilibrium,
            out_root=out_root,
            resolution_scale=float(args.resolution_scale),
            min_ntheta=int(args.min_ntheta),
            min_nzeta=int(args.min_nzeta),
            min_nx=int(args.min_nx),
            min_nxi=int(args.min_nxi),
            nu_jitter=float(args.nu_jitter),
            er_jitter=float(args.er_jitter),
            solve_method=str(solve_method),
            probe_preset=str(args.probe_preset),
            probe_env=probe_env,
        )
        for seed in args.seeds
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "lane": "qi_seed_robustness",
        "source_input": str(source_input),
        "source_equilibrium": str(source_equilibrium) if source_equilibrium is not None else None,
        "resolution_scale": float(args.resolution_scale),
        "nu_jitter": float(args.nu_jitter),
        "er_jitter": float(args.er_jitter),
        "solve_method": str(solve_method),
        "requested_solve_method": str(args.solve_method),
        "probe_preset": str(args.probe_preset),
        "probe_env": dict(sorted(probe_env.items())),
        "case_count": len(cases),
        "cases": cases,
    }
    if bool(args.execute):
        results = _execute_cases(
            out_root,
            cases,
            timeout_s=float(args.timeout_s),
            fail_fast=bool(args.fail_fast),
            heartbeat_s=float(args.heartbeat_s),
        )
        gates = _evaluate_execution_gates(
            results,
            max_residual_ratio=args.max_residual_ratio,
            require_converged=bool(args.require_converged),
            require_accepted_converged=bool(args.require_accepted_converged),
        )
        manifest["execution"] = {
            "timeout_s": float(args.timeout_s),
            "heartbeat_s": float(args.heartbeat_s),
            "fail_fast": bool(args.fail_fast),
            "results": results,
            "passed": sum(1 for result in results if int(result["returncode"]) == 0),
            "failed": sum(1 for result in results if int(result["returncode"]) != 0),
            "summary": _execution_summary(results),
            "gates": gates,
        }

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    if args.summary_output is not None:
        summary_path = Path(args.summary_output).resolve()
        _write_compact_execution_artifact(summary_path, manifest)
        print(f"Wrote {summary_path}")
    if args.evidence_manifest_output is not None:
        evidence_path = Path(args.evidence_manifest_output).resolve()
        artifact_paths = list(evidence_artifacts)
        if args.summary_output is not None:
            artifact_paths.append(Path(args.summary_output).resolve())
        _write_evidence_manifest(
            output_path=evidence_path,
            artifact_paths=artifact_paths,
            source_input=source_input,
            production_seed_count=int(args.production_seed_count),
            production_timeout_s=float(args.production_timeout_s),
        )
        print(f"Wrote {evidence_path}")
    print(f"Cases: {len(cases)}")
    if bool(args.execute):
        execution = manifest["execution"]  # type: ignore[index]
        print(f"Executed: {execution['passed']} passed, {execution['failed']} failed")
        gates = execution["gates"]
        if not gates["passed"]:
            print(f"Gates failed: {len(gates['failures'])}")
        return 0 if int(execution["failed"]) == 0 and bool(gates["passed"]) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
