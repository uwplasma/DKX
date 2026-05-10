from __future__ import annotations

from pathlib import Path

from scripts.benchmark_pas_tz_memory_fallback import _variant_env, build_plan, main


def test_variant_env_forces_bounded_pas_tz_memory_fallback() -> None:
    env = _variant_env("zeta", block=7, overlap=2, maxiter=5, restart=11)

    assert env["SFINCS_JAX_RHSMODE1_PRECONDITIONER"] == "pas_tz"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_MAX_BYTES"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK"] == "zeta"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_BLOCK"] == "7"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_OVERLAP"] == "2"
    assert env["SFINCS_JAX_GMRES_MAXITER"] == "5"
    assert env["SFINCS_JAX_GMRES_RESTART"] == "11"


def test_dry_run_writes_reproducible_plan(tmp_path: Path) -> None:
    out = tmp_path / "pas_tz_plan.json"

    rc = main(
        [
            "--dry-run",
            "--out",
            str(out),
            "--variants",
            "hybrid",
            "zeta",
            "--timeout-s",
            "12",
            "--maxiter",
            "3",
            "--restart",
            "4",
            "--block",
            "5",
            "--overlap",
            "1",
        ]
    )

    assert rc == 0
    text = out.read_text()
    assert '"kind": "pas_tz_memory_fallback_benchmark"' in text
    assert '"variants": [' in text
    assert '"hybrid"' in text
    assert '"zeta"' in text


def test_build_plan_records_solver_limits() -> None:
    class Args:
        input = Path("case/input.namelist")
        timeout_s = 9.0
        tol = 1.0e-7
        maxiter = 6
        restart = 8
        block = 4
        overlap = 2
        variants = ["theta"]

    plan = build_plan(Args())

    assert plan["timeout_s"] == 9.0
    assert plan["tol"] == 1.0e-7
    assert plan["maxiter"] == 6
    assert plan["restart"] == 8
    assert plan["block"] == 4
    assert plan["overlap"] == 2
    assert plan["variants"] == ["theta"]
