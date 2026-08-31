from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field as dataclass_field
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Iterator, Protocol

import numpy as np


LANDREMAN_2014_URL = "https://doi.org/10.1063/1.4870077"
LANDREMAN_2014_OPEN_PDF = "https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf"
SFINCS_FORTRAN_REPO_URL = "https://github.com/landreman/sfincs"
SIMAKOV_HELANDER_HIGH_COLLISIONALITY_URL = "https://doi.org/10.1063/1.3104715"

_TRAPPED_FRACTION_FACTOR = 1.46


def nu_prime_for_nu_over_v(
    nu_over_v: float,
    *,
    g_hat: float,
    i_hat: float,
    iota: float,
    b0_over_bbar: float,
    nu_d_hat_x0: float,
) -> float:
    """Map physical MDKE ``nu/v`` to DKX's ``nuPrime`` input.

    DKX applies ``nuPrime * B0 / |G + iota I| * nuDHat(x0)`` at the
    monoenergetic node. YANCC and MONKES take the applied Lorentz frequency
    divided by speed directly, so matched equations require this inverse map.
    """

    if b0_over_bbar == 0.0 or nu_d_hat_x0 <= 0.0:
        raise ValueError("b0_over_bbar and nu_d_hat_x0 must be nonzero and positive")
    g_plus = abs(float(g_hat) + float(iota) * float(i_hat))
    return g_plus / abs(float(b0_over_bbar)) * float(nu_over_v) / float(nu_d_hat_x0)


def dkes_to_beidler(
    *,
    d11: float,
    d31: float,
    d13: float,
    d33: float,
    nu_over_v: float,
    g_hat: float,
    iota: float,
    b0_over_bbar: float,
    r_hat: float,
    raw_b0_over_bbar: float | None = None,
    raw_fsab_b2: float | None = None,
    d33_spitzer: float | None = None,
    cross_orientation: float = -1.0,
) -> dict[str, float]:
    """Convert MONKES/YANCC DKES-scaled coefficients to Beidler ``D*``.

    ``r_hat`` is the local effective radius, not the LCFS minor radius.
    ``raw_b0_over_bbar`` accounts for a raw reference-field convention that
    differs from DKX's ``B0``. Supply either the raw Spitzer value or
    ``raw_fsab_b2`` for the ``D33`` scale. The default orientation maps the
    handedness of the pinned fixtures to DKX's standardized cross coefficients.
    """

    b0 = abs(float(b0_over_bbar))
    iota_abs = abs(float(iota))
    if b0 == 0.0 or iota_abs == 0.0 or r_hat <= 0.0:
        raise ValueError("B0, |iota|, and r_hat must be positive")
    r_major = abs(float(g_hat)) / b0
    if r_major == 0.0:
        raise ValueError("|g_hat| / B0 must be positive")
    eps_t = float(r_hat) / r_major
    if eps_t <= 0.0:
        raise ValueError("eps_t must be positive")

    d11_factor = 8.0 * r_major * b0 * b0 * iota_abs / math.pi
    d31_factor = (
        1.5
        * iota_abs
        * eps_t
        * b0
        / (_TRAPPED_FRACTION_FACTOR * math.sqrt(eps_t))
    )
    raw_b0 = b0 if raw_b0_over_bbar is None else abs(float(raw_b0_over_bbar))
    cross_b0_correction = raw_b0 / b0

    if (d33_spitzer is None) == (raw_fsab_b2 is None):
        raise ValueError("supply exactly one of d33_spitzer or raw_fsab_b2")
    if d33_spitzer is None:
        if nu_over_v <= 0.0 or raw_b0 == 0.0:
            raise ValueError("nu_over_v and raw B0 must be positive for the Spitzer scale")
        d33_spitzer = (
            2.0
            / (3.0 * float(nu_over_v))
            * float(raw_fsab_b2)
            / (raw_b0 * raw_b0)
        )
    if d33_spitzer == 0.0:
        raise ValueError("d33_spitzer must be nonzero")

    return {
        "D11_star": float(d11) * d11_factor,
        "D31_star": float(cross_orientation) * float(d31) * cross_b0_correction * d31_factor,
        "D13_star": float(cross_orientation) * float(d13) * cross_b0_correction * d31_factor,
        "D33_star": float(d33) / float(d33_spitzer),
        "r_major": r_major,
        "eps_t": eps_t,
        "D11_factor": d11_factor,
        "D31_factor": d31_factor,
        "cross_b0_correction": cross_b0_correction,
    }


def coefficient_relative_errors(
    candidate: Mapping[str, float], reference: Mapping[str, float]
) -> dict[str, float]:
    """Return absolute relative errors for the four Beidler coefficients."""

    errors: dict[str, float] = {}
    for key in ("D11_star", "D31_star", "D13_star", "D33_star"):
        denominator = abs(float(reference[key]))
        if denominator == 0.0:
            raise ValueError(f"reference {key} must be nonzero for a relative gate")
        errors[key] = abs(float(candidate[key]) - float(reference[key])) / denominator
    return errors



































































DEFAULT_PUBLICATION_ARTIFACTS: dict[str, str] = {
    "lhd_collisionality": "lhd_collisionality_summary.json",
    "w7x_collisionality": "w7x_collisionality_summary.json",
    "tokamak_er_sweep": "er_sweep_tokamak_reference_summary.json",
    "stellarator_er_sweep": "er_sweep_stellarator_fast_reference_summary.json",
}













































def _summarize_collisionality(records: Sequence[CollisionalityRecord]) -> dict[str, object]:
    separation = fp_pas_l11_separation(records)
    low = separation[0]
    high = separation[-1]
    return {
        "labels": collisionality_labels(records),
        "nuprime": collisionality_grid(records),
        "l11_fp_pas_separation": separation,
        "l11_low_relative_separation": float(low["relative_to_fp"]),
        "l11_high_relative_separation": float(high["relative_to_fp"]),
        "l11_high_to_low_relative_separation_ratio": float(
            high["relative_to_fp"] / max(low["relative_to_fp"], np.finfo(float).tiny)
        ),
    }


def _periodic_central_derivative(values: np.ndarray, coordinates: np.ndarray, *, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.size < 2:
        return np.zeros_like(values)
    spacing = float(coordinates[1] - coordinates[0])
    if not np.isfinite(spacing) or spacing == 0.0:
        raise ValueError("Periodic derivative coordinates must have finite nonzero spacing.")
    return (np.roll(values, -1, axis=int(axis)) - np.roll(values, 1, axis=int(axis))) / (2.0 * spacing)


def _theta_zeta_axes(shape: tuple[int, ...], *, n_theta: int, n_zeta: int) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"Expected a two-dimensional geometry field, got shape {shape}.")
    if shape == (int(n_theta), int(n_zeta)):
        return 0, 1
    if shape == (int(n_zeta), int(n_theta)):
        return 1, 0
    raise ValueError(
        f"Geometry field shape {shape} does not match theta/zeta sizes {(int(n_theta), int(n_zeta))}."
    )


def appendix_b_geometry_audit_from_h5(output_h5: Path) -> dict[str, object]:
    """Compute discrete Appendix-B geometry ingredients from a SFINCS output file.

    The returned coefficients are a normalization audit, not a final validation
    claim. They use the same checked-in geometry fields that appear in
    ``sfincsOutput.h5`` and make the Simakov-Helander/Pfirsch-Schluter comparison
    reproducible enough to identify which high-collisionality scans are still
    missing before an analytic-limit overlay is promoted.
    """

    try:
        import h5py
    except Exception as exc:  # pragma: no cover - h5py is a package dependency.
        raise RuntimeError("appendix_b_geometry_audit_from_h5 requires h5py.") from exc

    output_h5 = Path(output_h5)
    required = (
        "BHat",
        "DHat",
        "uHat",
        "BHat_sup_theta",
        "BHat_sup_zeta",
        "dBHatdtheta",
        "dBHatdzeta",
        "theta",
        "zeta",
        "GHat",
        "IHat",
        "iota",
        "FSABHat2",
    )
    with h5py.File(output_h5, "r") as h5:
        missing = [name for name in required if name not in h5]
        if missing:
            raise ValueError(f"{output_h5} is missing Appendix-B audit fields: {missing}")
        b_hat = np.asarray(h5["BHat"], dtype=np.float64)
        d_hat = np.asarray(h5["DHat"], dtype=np.float64)
        u_hat = np.asarray(h5["uHat"], dtype=np.float64)
        b_sup_theta = np.asarray(h5["BHat_sup_theta"], dtype=np.float64)
        b_sup_zeta = np.asarray(h5["BHat_sup_zeta"], dtype=np.float64)
        db_dtheta = np.asarray(h5["dBHatdtheta"], dtype=np.float64)
        db_dzeta = np.asarray(h5["dBHatdzeta"], dtype=np.float64)
        theta = np.asarray(h5["theta"], dtype=np.float64)
        zeta = np.asarray(h5["zeta"], dtype=np.float64)
        g_hat = float(np.asarray(h5["GHat"], dtype=np.float64))
        i_hat = float(np.asarray(h5["IHat"], dtype=np.float64))
        iota = float(np.asarray(h5["iota"], dtype=np.float64))
        fsab_hat2 = float(np.asarray(h5["FSABHat2"], dtype=np.float64))

    theta_axis, zeta_axis = _theta_zeta_axes(b_hat.shape, n_theta=theta.size, n_zeta=zeta.size)
    if np.any(d_hat == 0.0):
        raise ValueError(f"{output_h5} contains zero DHat entries.")
    weights = 1.0 / d_hat
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        raise ValueError(f"{output_h5} has invalid flux-surface-average weights.")

    def fsa(quantity: np.ndarray) -> float:
        return float(np.sum(weights * np.asarray(quantity, dtype=np.float64)) / weight_sum)

    def grad_parallel(quantity: np.ndarray) -> np.ndarray:
        dtheta = _periodic_central_derivative(quantity, theta, axis=theta_axis)
        dzeta = _periodic_central_derivative(quantity, zeta, axis=zeta_axis)
        return (b_sup_theta * dtheta + b_sup_zeta * dzeta) / b_hat

    gradpar_b = (b_sup_theta * db_dtheta + b_sup_zeta * db_dzeta) / b_hat
    gradpar_ln_b = gradpar_b / b_hat
    gradpar_u_b2 = grad_parallel(u_hat * b_hat * b_hat)
    gradpar_b2_fsa = fsa(gradpar_b * gradpar_b)
    if abs(gradpar_b2_fsa) <= np.finfo(float).tiny:
        raise ValueError(f"{output_h5} has a near-zero <(grad_parallel B)^2> denominator.")

    fsa_b2 = fsa(b_hat * b_hat)
    fsa_u_b2 = fsa(u_hat * b_hat * b_hat)
    fsa_u2_b2 = fsa(u_hat * u_hat * b_hat * b_hat)
    fsa_gradlnb_gradu_b2 = fsa(gradpar_ln_b * gradpar_u_b2)
    fsa_u_gradpar_b2 = fsa(u_hat * gradpar_b * gradpar_b)
    g1 = (fsa_gradlnb_gradu_b2 * fsa_gradlnb_gradu_b2) / gradpar_b2_fsa - fsa(
        (gradpar_u_b2 / b_hat) ** 2
    )
    g2 = fsa(u_hat * gradpar_ln_b * gradpar_u_b2) - fsa_gradlnb_gradu_b2 * fsa_u_gradpar_b2 / gradpar_b2_fsa
    k1 = fsa_gradlnb_gradu_b2 / (2.0 * gradpar_b2_fsa)
    k2 = 1.97213 * fsa_u_b2 / fsa_b2 - 1.03287 * 2.0 * k1 + 0.09361 * fsa_u_gradpar_b2 / gradpar_b2_fsa
    h_geom = (fsa_u_b2 * fsa_u_b2) / fsa_b2 - fsa_u2_b2
    g_plus_iota_i = g_hat + iota * i_hat
    common = 0.96 * np.sqrt(2.0) * (g_plus_iota_i**2) / (iota * iota * g_hat * g_hat)
    coefficients = {
        "L11": float(common * 0.75 * g1),
        "L12": float(common * (3.245 * g1 + 0.085 * g2)),
        "L22": float(np.sqrt(2.0) * 8.0 / 5.0 * fsa_b2 * h_geom / (iota * iota * g_hat * g_hat)),
        "L33": float(fsa_b2 * fsa_b2 / (3.0 * 0.96 * np.sqrt(2.0) * (g_plus_iota_i**2) * gradpar_b2_fsa)),
    }
    return {
        "source_output": str(output_h5),
        "grid": {
            "n_theta": int(theta.size),
            "n_zeta": int(zeta.size),
            "theta_axis": int(theta_axis),
            "zeta_axis": int(zeta_axis),
        },
        "geometry_scalars": {
            "GHat": float(g_hat),
            "IHat": float(i_hat),
            "iota": float(iota),
            "G_plus_iota_I": float(g_plus_iota_i),
            "FSABHat2_output": float(fsab_hat2),
            "FSABHat2_recomputed": float(fsa_b2),
            "FSABHat2_relative_error": float(abs(fsa_b2 - fsab_hat2) / max(abs(fsab_hat2), np.finfo(float).tiny)),
        },
        "appendix_b_discrete_quantities": {
            "G1": float(g1),
            "G2": float(g2),
            "K1": float(k1),
            "K2": float(k2),
            "H": float(h_geom),
            "gradpar_b_rms": float(np.sqrt(abs(gradpar_b2_fsa))),
            "fsa_u_b2": float(fsa_u_b2),
            "fsa_u2_b2": float(fsa_u2_b2),
        },
        "transport_matrix_coefficients_over_nuprime": coefficients,
        "notes": [
            "Coefficients follow the Appendix-B structure using checked-in normalized dkx output fields.",
            "Use these values as an audit of normalization and geometry ingredients, not as a final analytic-limit acceptance gate.",
        ],
    }


def _inverse_tail_ratio(
    records: Sequence[CollisionalityRecord],
    *,
    label: str,
    element: tuple[int, int],
    coefficient_over_nuprime: float,
) -> dict[str, float]:
    nuprime, values = transport_element_abs_series(records, label=label, element=element)
    last_nu = float(nuprime[-1])
    last_value = float(values[-1])
    predicted = abs(float(coefficient_over_nuprime)) / max(last_nu, np.finfo(float).tiny)
    return {
        "nuprime": last_nu,
        "observed_abs": last_value,
        "appendix_b_proxy_abs": float(predicted),
        "observed_to_proxy_ratio": float(last_value / max(predicted, np.finfo(float).tiny)),
    }


def _simakov_case_summary(
    records: Sequence[CollisionalityRecord],
    *,
    geometry_audit: Mapping[str, object] | None,
    n_fit: int,
    min_nuprime_for_full_limit: float,
    target_slope: float,
    slope_tolerance: float,
) -> dict[str, object]:
    trend = high_collisionality_trend_summary(records, n_fit=n_fit)
    sensitivity = high_collisionality_slope_sensitivity(records, n_fit_values=(2, 3, 4, 5))
    slopes = trend["slopes"]["Fokker-Planck"]  # type: ignore[index]
    fp_l11_l12_target_like = all(
        abs(float(slopes[name]) - float(target_slope)) <= float(slope_tolerance) for name in ("L11", "L12")
    )
    grid = collisionality_grid(records)
    max_nuprime = float(max(grid))
    scan_extends_to_required_high_nu = max_nuprime >= float(min_nuprime_for_full_limit)
    high_nu_extension = recommended_high_collisionality_nuprime_grid(
        grid,
        min_nuprime_for_full_limit=float(min_nuprime_for_full_limit),
    )
    appendix_ratios: dict[str, object] = {}
    if geometry_audit is not None:
        coeffs = geometry_audit.get("transport_matrix_coefficients_over_nuprime", {})
        if isinstance(coeffs, Mapping):
            for name in ("L11", "L12", "L22", "L33"):
                if name in coeffs:
                    appendix_ratios[name] = _inverse_tail_ratio(
                        records,
                        label="Fokker-Planck",
                        element=TRANSPORT_ELEMENTS[name],
                        coefficient_over_nuprime=float(coeffs[name]),
                    )
    return {
        "nuprime_grid": grid,
        "max_nuprime": max_nuprime,
        "recommended_high_nuprime_extension": high_nu_extension,
        "trend": trend,
        "slope_sensitivity": sensitivity,
        "appendix_b_geometry_audit": dict(geometry_audit) if geometry_audit is not None else None,
        "appendix_b_proxy_ratios_at_max_nuprime": appendix_ratios,
        "gates": {
            "scan_extends_to_required_high_nu": bool(scan_extends_to_required_high_nu),
            "fp_l11_l12_target_inverse_slope": bool(fp_l11_l12_target_like),
            "pas_l11_l12_positive": bool(trend["gates"]["pas_l11_l12_positive"]),  # type: ignore[index]
            "appendix_b_geometry_inputs_available": bool(geometry_audit is not None),
        },
        "state": "ready_for_full_overlay" if scan_extends_to_required_high_nu and fp_l11_l12_target_like else "needs_wider_high_nu_scan",
    }


def build_simakov_helander_limit_audit_summary(
    *,
    artifact_dir: Path,
    artifacts: Mapping[str, str] = DEFAULT_PUBLICATION_ARTIFACTS,
    geometry_outputs: Mapping[str, Path] | None = None,
    precomputed_geometry_audits: Mapping[str, Mapping[str, object]] | None = None,
    n_fit: int = 3,
    min_nuprime_for_full_limit: float = 50.0,
    target_slope: float = -1.0,
    slope_tolerance: float = 0.35,
) -> dict[str, object]:
    """Build the bounded audit for the Simakov-Helander high-collisionality lane."""

    artifact_dir = Path(artifact_dir)
    lhd = load_collisionality_records(artifact_dir / artifacts["lhd_collisionality"])
    w7x = load_collisionality_records(artifact_dir / artifacts["w7x_collisionality"])
    geometry_audits: dict[str, Mapping[str, object] | None] = {"lhd": None, "w7x": None}
    if precomputed_geometry_audits is not None:
        for case in ("lhd", "w7x"):
            audit = precomputed_geometry_audits.get(case)
            if audit is not None:
                geometry_audits[case] = dict(audit)
    if geometry_outputs is not None:
        for case in ("lhd", "w7x"):
            path = geometry_outputs.get(case)
            if path is not None and Path(path).exists():
                geometry_audits[case] = appendix_b_geometry_audit_from_h5(Path(path))

    cases = {
        "lhd": _simakov_case_summary(
            lhd,
            geometry_audit=geometry_audits["lhd"],
            n_fit=n_fit,
            min_nuprime_for_full_limit=min_nuprime_for_full_limit,
            target_slope=target_slope,
            slope_tolerance=slope_tolerance,
        ),
        "w7x": _simakov_case_summary(
            w7x,
            geometry_audit=geometry_audits["w7x"],
            n_fit=n_fit,
            min_nuprime_for_full_limit=min_nuprime_for_full_limit,
            target_slope=target_slope,
            slope_tolerance=slope_tolerance,
        ),
    }
    full_ready = all(bool(case["state"] == "ready_for_full_overlay") for case in cases.values())
    geometry_ready = all(bool(case["gates"]["appendix_b_geometry_inputs_available"]) for case in cases.values())  # type: ignore[index]
    literature_ready = bool(full_ready and geometry_ready)
    return {
        "metadata": {
            "schema_version": 1,
            "kind": "simakov_helander_limit_audit",
            "literature": [
                LANDREMAN_2014_URL,
                LANDREMAN_2014_OPEN_PDF,
                SIMAKOV_HELANDER_HIGH_COLLISIONALITY_URL,
            ],
            "source_artifacts": {
                "lhd_collisionality": artifacts["lhd_collisionality"],
                "w7x_collisionality": artifacts["w7x_collisionality"],
            },
            "notes": [
                "This artifact audits the normalization and high-nu sufficiency for the Appendix-B analytic-limit lane.",
                "It intentionally keeps the full reproduction gate closed until a wider nu' >> 1 scan is checked in.",
                "The current full collisionality summaries stop near nu'=10, below the default full-limit threshold.",
            ],
            "publication_figure": {
                "claim_status": (
                    "checked_in_converged_artifact" if literature_ready else "proxy_or_deferred"
                ),
                "artifact_class": (
                    "checked_in_simakov_helander_full_limit_artifact"
                    if literature_ready
                    else "checked_in_normalization_audit_deferred_full_limit"
                ),
                "checked_in_converged_artifact": bool(literature_ready),
                "ready_for_physics_validation_claim": bool(literature_ready),
                "manuscript_label": (
                    "checked-in Simakov-Helander full high-nu validation"
                    if literature_ready
                    else "normalization audit; full Simakov-Helander high-nu validation deferred"
                ),
            },
        },
        "configuration": {
            "n_fit": int(n_fit),
            "min_nuprime_for_full_limit": float(min_nuprime_for_full_limit),
            "target_fp_slope": float(target_slope),
            "slope_tolerance": float(slope_tolerance),
        },
        "cases": cases,
        "gates": {
            "appendix_b_geometry_inputs_available": bool(geometry_ready),
            "all_cases_ready_for_full_overlay": bool(full_ready),
            "checked_in_converged_artifact": bool(literature_ready),
            "ready_for_literature_claim": bool(literature_ready),
            "proxy_or_deferred_only": bool(not literature_ready),
            "full_simakov_helander_reproduction_closed": bool(not literature_ready),
        },
    }


def _summarize_er_sweep(records: Sequence[ErSweepRecord]) -> dict[str, object]:
    return {
        "models": sorted({record.model for record in records}),
        "er_values": sorted({float(record.er) for record in records}),
        "zero_field_spread": er_zero_field_spread(records),
        "nonzero_fsab_jhat_spread": er_nonzero_model_spread(records, field="fsab_jhat"),
        "nonzero_fsab_flow_spread": er_nonzero_model_spread(records, field="fsab_flow"),
    }


def build_publication_validation_summary(
    *,
    artifact_dir: Path,
    artifacts: Mapping[str, str] = DEFAULT_PUBLICATION_ARTIFACTS,
) -> dict[str, object]:
    """Build a machine-readable summary for the publication validation dashboard."""

    artifact_dir = Path(artifact_dir)
    lhd = load_collisionality_records(artifact_dir / artifacts["lhd_collisionality"])
    w7x = load_collisionality_records(artifact_dir / artifacts["w7x_collisionality"])
    tokamak = load_er_sweep_records(artifact_dir / artifacts["tokamak_er_sweep"])
    stellarator = load_er_sweep_records(artifact_dir / artifacts["stellarator_er_sweep"])
    return {
        "metadata": {
            "schema_version": 1,
            "kind": "publication_validation_dashboard",
            "literature": [LANDREMAN_2014_URL, LANDREMAN_2014_OPEN_PDF],
            "source_artifacts": dict(artifacts),
        },
        "collisionality": {
            "lhd": _summarize_collisionality(lhd),
            "w7x": _summarize_collisionality(w7x),
        },
        "trajectory_sweeps": {
            "tokamak": _summarize_er_sweep(tokamak),
            "stellarator": _summarize_er_sweep(stellarator),
        },
    }


def build_fortran_suite_benchmark_summary(
    *,
    cpu_report: Path,
    gpu_report: Path,
    min_fortran_runtime_s: float | None = None,
    enforce_public_resolution_floor: bool = True,
) -> dict[str, object]:
    """Build a CPU/GPU suite benchmark summary against the Fortran v3 reference."""

    cpu_report = Path(cpu_report)
    gpu_report = Path(gpu_report)
    raw_cpu_rows = load_suite_report(cpu_report)
    raw_gpu_rows = load_suite_report(gpu_report)
    raw_cpu_metrics = suite_case_metrics(raw_cpu_rows)
    raw_gpu_metrics = suite_case_metrics(raw_gpu_rows)
    cpu_metrics, gpu_metrics, excluded_cases = filter_suite_metrics_by_fortran_runtime(
        raw_cpu_metrics,
        raw_gpu_metrics,
        min_fortran_runtime_s=min_fortran_runtime_s,
    )
    cpu_reported_cases = {metric.case for metric in cpu_metrics}
    gpu_reported_cases = {metric.case for metric in gpu_metrics}
    cpu_rows = [row for row in raw_cpu_rows if str(row.get("case", "")) in cpu_reported_cases]
    gpu_rows = [row for row in raw_gpu_rows if str(row.get("case", "")) in gpu_reported_cases]
    resolution_floor_violations = {
        "cpu": benchmark_resolution_floor_violations(cpu_rows),
        "gpu": benchmark_resolution_floor_violations(gpu_rows),
    }
    if enforce_public_resolution_floor and (
        resolution_floor_violations["cpu"] or resolution_floor_violations["gpu"]
    ):
        raise ValueError(
            "Public benchmark summary includes below-floor or untagged rows: "
            + json.dumps(resolution_floor_violations, sort_keys=True)
        )
    payload = {
        "metadata": {
            "schema_version": FORTRAN_SUITE_BENCHMARK_SCHEMA_VERSION,
            "kind": FORTRAN_SUITE_BENCHMARK_KIND,
            "literature": [
                LANDREMAN_2014_URL,
                LANDREMAN_2014_OPEN_PDF,
                SFINCS_FORTRAN_REPO_URL,
            ],
            "source_reports": {
                "cpu": _repo_stable_path(cpu_report),
                "gpu": _repo_stable_path(gpu_report),
            },
            "source_case_counts": {
                "cpu": int(len(raw_cpu_metrics)),
                "gpu": int(len(raw_gpu_metrics)),
            },
            "reported_case_counts": {
                "cpu": int(len(cpu_metrics)),
                "gpu": int(len(gpu_metrics)),
            },
            "min_fortran_runtime_s": None if min_fortran_runtime_s is None else float(min_fortran_runtime_s),
            "excluded_low_fortran_runtime_cases": excluded_cases,
            "public_3d_benchmark_floor": dict(PUBLIC_3D_BENCHMARK_FLOOR),
            "public_tokamak_benchmark_floor": dict(PUBLIC_TOKAMAK_BENCHMARK_FLOOR),
            "resolution_floor_violations": resolution_floor_violations,
            "notes": [
                "Runtime ratios use audited wall-clock fields stored in the frozen suite reports.",
                "Process memory ratios use audited maximum-RSS fields; active JAX memory ratios use profiler dpeak_rss_mb/drss_mb deltas when available.",
                "The summary is a release gate: all audited CPU/GPU cases must remain parity_ok with no strict mismatches before filtering.",
                "README-facing performance plots filter out very short Fortran reference runs so public runtime claims are based on production-scale rows.",
                "README-facing performance plots also require final_resolution metadata meeting the public production-resolution floor.",
                "The artifacts compare dkx against the Fortran v3 reference implementation on the vendored example suite.",
            ],
        },
        "reports": {
            "cpu": suite_report_summary(cpu_rows, label="CPU"),
            "gpu": suite_report_summary(gpu_rows, label="GPU"),
        },
    }
    schema_errors = fortran_suite_benchmark_schema_errors(payload)
    if schema_errors:
        raise ValueError("Invalid Fortran-suite benchmark summary schema: " + "; ".join(schema_errors))
    return payload


def build_high_collisionality_trend_proxy_summary(
    *,
    artifact_dir: Path,
    artifacts: Mapping[str, str] = DEFAULT_PUBLICATION_ARTIFACTS,
    n_fit: int = 3,
) -> dict[str, object]:
    """Build the high-collisionality trend proxy summary from corrected artifacts."""

    artifact_dir = Path(artifact_dir)
    lhd = load_collisionality_records(artifact_dir / artifacts["lhd_collisionality"])
    w7x = load_collisionality_records(artifact_dir / artifacts["w7x_collisionality"])
    cases = {
        "lhd": high_collisionality_trend_summary(lhd, n_fit=n_fit),
        "w7x": high_collisionality_trend_summary(w7x, n_fit=n_fit),
    }
    all_pas_positive = all(
        bool(case["gates"]["pas_l11_l12_positive"])  # type: ignore[index]
        for case in cases.values()
    )
    all_fp_inverse_like = all(
        bool(case["gates"]["fp_l11_l12_inverse_like"])  # type: ignore[index]
        for case in cases.values()
    )
    return {
        "metadata": {
            "schema_version": 1,
            "kind": "high_collisionality_trend_proxy",
            "literature": [LANDREMAN_2014_URL, LANDREMAN_2014_OPEN_PDF],
            "source_artifacts": {
                "lhd_collisionality": artifacts["lhd_collisionality"],
                "w7x_collisionality": artifacts["w7x_collisionality"],
            },
            "notes": [
                "The SFINCS 2014 paper states that PAS L11/L12 scale like +nu at high collisionality.",
                "Momentum-conserving FP/model-operator L11/L12 should approach inverse-nu scaling only in the nu' >> 1 limit.",
                "The checked-in scans stop at nu'=10, so this artifact is a trend proxy, not the full Simakov-Helander analytic-limit reproduction.",
            ],
            "publication_figure": {
                "claim_status": "proxy_or_deferred",
                "artifact_class": "checked_in_high_collisionality_trend_proxy",
                "checked_in_converged_artifact": False,
                "ready_for_physics_validation_claim": False,
                "manuscript_label": "checked-in trend proxy; full analytic-limit validation deferred",
            },
        },
        "cases": cases,
        "gates": {
            "all_pas_l11_l12_positive": bool(all_pas_positive),
            "all_fp_l11_l12_inverse_like": bool(all_fp_inverse_like),
            "checked_in_converged_artifact": False,
            "ready_for_literature_claim": False,
            "full_simakov_helander_reproduction_closed": True,
            "proxy_or_deferred_only": True,
        },
    }


# Benchmark artifact release-gate policy helpers.





































































# Publication-evidence panel helpers. Keep these with artifact schemas so docs
# figures are validated by the same evidence module that loads their source data.


# ``artifacts`` remains the published address for these; the split is an
# internal reorganisation and must stay invisible to importers.
from .policy import (  # noqa: E402,F401
    ARTIFACT_CLASSES,
    ARTIFACT_CLASS_FORTRAN_SUITE_SUMMARY,
    ARTIFACT_CLASS_LEGACY,
    ARTIFACT_CLASS_NON_PAS,
    ARTIFACT_CLASS_RELEASE_BLOCKING,
    ARTIFACT_CLASS_SCHEMA_V2,
    ErrorSink,
    FORTRAN_SUITE_BACKENDS,
    FORTRAN_SUITE_MIN_RUNTIME_GATE_S,
    FORTRAN_SUITE_SUMMARY_KIND,
    OK_RESULT_STATUSES,
    TAIL_RESULT_STATUSES,
    WARM_RUNTIME_SOURCES,
    BenchmarkArtifactIndex,
    BenchmarkArtifactIndexEntry,
    BenchmarkArtifactPolicyError,
    benchmark_artifact_policy_errors,
    check_benchmark_artifact_file,
    check_benchmark_artifact_files,
    classify_benchmark_artifact_file,
    fortran_suite_benchmark_summary_errors,
    index_benchmark_artifact_files,
    validate_benchmark_artifact,
    validate_benchmark_artifact_file,
)
from .panels import (  # noqa: E402,F401
    SIMAKOV_HELANDER_PROVENANCE_FIELDS,
    W7X_AMBIPOLAR_PROVENANCE_FIELDS,
    build_simakov_helander_high_nu_panel,
    build_w7x_ambipolar_root_provenance_panel,
)


# ``artifacts`` stays the published address for the whole validation surface;
# the split is internal and must not move anyone's import.
from .lanes import (  # noqa: E402,F401
    CollisionalityLike,
    ResearchLanePolicyError,
    check_research_lane_completion_file,
    research_lane_completion_errors,
    validate_research_lane_completion,
    validate_research_lane_completion_file,
    DEFAULT_MIN_SUBSTANTIAL_DELTA_PERCENT,
    VALID_LANE_STATUSES,
    _as_number,
    _check_evidence,
    _check_nonempty_list,
    _default_repo_root,
    _lane_delta_satisfies_push_gate,
    _nonempty_string,
    _percent_field,
    _required_lane_delta,
)
from .series import (  # noqa: E402,F401
    TRANSPORT_ELEMENTS,
    CollisionalityRecord,
    ErSweepRecord,
    PhaseRecord,
    PhaseTimer,
    SuiteCaseMetric,
    autodiff_gradient_error_summary,
    benchmark_resolution_floor_violations,
    build_autodiff_sensitivity_validation_summary,
    collisionality_grid,
    collisionality_labels,
    collisionality_power_law_slope,
    er_nonzero_model_spread,
    er_zero_field_spread,
    filter_suite_metrics_by_fortran_runtime,
    fortran_suite_benchmark_schema_errors,
    fp_pas_l11_separation,
    high_collisionality_slope_sensitivity,
    high_collisionality_trend_summary,
    l11_abs_series,
    load_autodiff_sensitivity_summary,
    load_collisionality_records,
    load_er_sweep_records,
    load_suite_report,
    maxrss_mb,
    recommended_high_collisionality_nuprime_grid,
    suite_case_metrics,
    suite_report_summary,
    transport_element_abs_series,
    FORTRAN_SUITE_BENCHMARK_KIND,
    FORTRAN_SUITE_BENCHMARK_REPORT_KEYS,
    FORTRAN_SUITE_BENCHMARK_SCHEMA_VERSION,
    PAUL_2019_ADJOINT_URL,
    PUBLIC_3D_BENCHMARK_FLOOR,
    PUBLIC_TOKAMAK_BENCHMARK_FLOOR,
    SFINCS_ADJOINT_APS_URL,
    SUITE_MISMATCH_FIELDS,
    SUITE_STRICT_MISMATCH_FIELDS,
    _counts,
    _metric_row,
    _mismatch_count,
    _optional_float,
    _ratio_summary,
    _repo_stable_path,
    _row_floor,
    _row_resolution,
    _safe_ratio,
    _top_metrics,
)
