from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.benchmark_pas_tz_memory_fallback import (
    _override_namelist_text,
    _run_child,
    _variant_env,
    _variant_provenance,
    _variant_solve_method,
    build_plan,
    main,
)


def test_variant_env_forces_bounded_pas_tz_memory_fallback() -> None:
    env = _variant_env("zeta", block=7, overlap=2, maxiter=5, restart=11)

    assert env["SFINCS_JAX_RHSMODE1_PRECONDITIONER"] == "pas_tz"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_MAX_BYTES"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK"] == "zeta"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_BLOCK"] == "7"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_OVERLAP"] == "2"
    assert env["SFINCS_JAX_GMRES_MAXITER"] == "5"
    assert env["SFINCS_JAX_GMRES_RESTART"] == "11"


def test_variant_env_supports_collision_tzfft_correction() -> None:
    env = _variant_env("collision-tzfft-correction", block=7, overlap=2, maxiter=5, restart=11)

    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK"] == "collision"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_CORRECTION"] == "tzfft"


def test_variant_env_supports_structured_tzfft_correction() -> None:
    env = _variant_env("tzfft-structured", block=7, overlap=2, maxiter=5, restart=11)

    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK"] == "tzfft"
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_LEVELS"] == "xmg,collision"


def test_variant_env_supports_lgmres_suffix_without_changing_fallback() -> None:
    env = _variant_env("tzfft-lgmres", block=7, overlap=2, maxiter=5, restart=11)

    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK"] == "tzfft"
    assert _variant_solve_method("tzfft-lgmres", "incremental") == "lgmres"
    assert _variant_solve_method("tzfft", "incremental") == "incremental"


def test_variant_provenance_records_lgmres_opt_in_source() -> None:
    provenance = _variant_provenance("tzfft-lgmres", "incremental")

    assert provenance == {
        "variant": "tzfft-lgmres",
        "base_variant": "tzfft",
        "requested_solve_method": "incremental",
        "realized_solve_method": "lgmres",
        "solve_method_source": "variant_suffix",
        "lgmres_opt_in": True,
    }


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
            "--solve-method",
            "lgmres",
            "--block",
            "5",
            "--overlap",
            "1",
            "--Ntheta",
            "31",
            "--Nzeta",
            "41",
            "--Nxi",
            "51",
            "--Nx",
            "7",
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == 2
    assert payload["kind"] == "pas_tz_memory_fallback_benchmark"
    assert payload["plan"]["variants"] == ["hybrid", "zeta"]
    assert payload["plan"]["solve_method"] == "lgmres"
    assert payload["plan"]["variant_methods"] == [
        {
            "variant": "hybrid",
            "base_variant": "hybrid",
            "requested_solve_method": "lgmres",
            "realized_solve_method": "lgmres",
            "solve_method_source": "plan_default",
            "lgmres_opt_in": False,
        },
        {
            "variant": "zeta",
            "base_variant": "zeta",
            "requested_solve_method": "lgmres",
            "realized_solve_method": "lgmres",
            "solve_method_source": "plan_default",
            "lgmres_opt_in": False,
        },
    ]
    assert payload["plan"]["input_overrides"] == {
        "Ntheta": 31,
        "Nzeta": 41,
        "Nxi": 51,
        "Nx": 7,
    }
    assert payload["results"] == []


def test_build_plan_records_solver_limits() -> None:
    class Args:
        input = Path("case/input.namelist")
        timeout_s = 9.0
        tol = 1.0e-7
        solve_method = "incremental"
        maxiter = 6
        restart = 8
        block = 4
        overlap = 2
        variants = ["theta"]
        Ntheta = 31
        Nzeta = None
        Nxi = 51
        Nx = None

    plan = build_plan(Args())

    assert plan["input_overrides"] == {"Ntheta": 31, "Nxi": 51}
    assert plan["timeout_s"] == 9.0
    assert plan["tol"] == 1.0e-7
    assert plan["solve_method"] == "incremental"
    assert plan["variant_methods"] == [
        {
            "variant": "theta",
            "base_variant": "theta",
            "requested_solve_method": "incremental",
            "realized_solve_method": "incremental",
            "solve_method_source": "plan_default",
            "lgmres_opt_in": False,
        }
    ]
    assert plan["maxiter"] == 6
    assert plan["restart"] == 8
    assert plan["block"] == 4
    assert plan["overlap"] == 2
    assert plan["variants"] == ["theta"]


def test_override_namelist_text_updates_grid_scalars_only() -> None:
    text = """&resolutionParameters
  Ntheta = 13  ! keep comment
  Nzeta = 23
  Nxi = 48
  Nx = 5
/
"""

    updated = _override_namelist_text(text, {"Ntheta": 31, "Nzeta": 41, "Nxi": 51, "Nx": 7})

    assert "  Ntheta = 31  ! keep comment" in updated
    assert "  Nzeta = 41" in updated
    assert "  Nxi = 51" in updated
    assert "  Nx = 7" in updated


def test_run_child_uses_temporary_input_overrides_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.input.namelist"
    source.write_text(
        """&resolutionParameters
  Ntheta = 13
  Nzeta = 23
  Nxi = 48
  Nx = 5
/
"""
    )
    out = tmp_path / "out.json"
    captured: dict[str, object] = {}

    def fake_run(
        cmd: list[str],
        *,
        env: dict[str, str],
        text: bool,
        capture_output: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        input_path = Path(cmd[cmd.index("--input") + 1])
        captured["cmd"] = cmd
        captured["env"] = env
        captured["input_path"] = input_path
        captured["input_text"] = input_path.read_text()
        captured["text"] = text
        captured["capture_output"] = capture_output
        captured["timeout"] = timeout
        payload = {
            "status": "ok",
            "elapsed_s": 0.25,
            "residual_norm": 0.0,
            "phase_metadata": [{"name": "solve", "status": "ok", "elapsed_s": 0.25}],
            "solver_provenance": {
                "requested_solve_method": "incremental",
                "realized_solve_method": "incremental",
            },
        }
        stdout = "__SFINCS_JAX_PAS_TZ_RESULT__=" + json.dumps(payload) + "\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = type(
        "Args",
        (),
        {
            "input": source,
            "out": out,
            "timeout_s": 9.0,
            "tol": 1.0e-7,
            "solve_method": "incremental",
            "maxiter": 6,
            "restart": 8,
            "block": 4,
            "overlap": 2,
            "variants": ["zeta"],
            "Ntheta": 31,
            "Nzeta": 41,
            "Nxi": 51,
            "Nx": 7,
        },
    )()

    row = _run_child(args, "zeta")

    child_input = captured["input_path"]
    assert isinstance(child_input, Path)
    assert child_input != source
    assert not child_input.exists()
    assert captured["input_text"] == """&resolutionParameters
  Ntheta = 31
  Nzeta = 41
  Nxi = 51
  Nx = 7
/
"""
    assert captured["cmd"][:2] == [sys.executable, str(Path("scripts/benchmark_pas_tz_memory_fallback.py").resolve())]
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK"] == "zeta"
    assert env["SFINCS_JAX_GMRES_MAXITER"] == "6"
    assert captured["text"] is True
    assert captured["capture_output"] is True
    assert captured["timeout"] == 9.0
    assert row["status"] == "ok"
    assert row["variant"] == "zeta"
    assert row["variant_provenance"] == {
        "variant": "zeta",
        "base_variant": "zeta",
        "requested_solve_method": "incremental",
        "realized_solve_method": "incremental",
        "solve_method_source": "plan_default",
        "lgmres_opt_in": False,
    }
    assert row["phase_metadata"] == [{"name": "solve", "status": "ok", "elapsed_s": 0.25}]
    assert row["solver_provenance"]["requested_solve_method"] == "incremental"
    assert row["tail_metadata"]["tail_limit_chars"] == 4000


def test_run_child_lgmres_variant_records_variant_method_without_changing_plan_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.input.namelist"
    source.write_text(
        """&resolutionParameters
  Ntheta = 13
  Nzeta = 23
  Nxi = 48
  Nx = 5
/
"""
    )
    captured: dict[str, object] = {}

    def fake_run(
        cmd: list[str],
        *,
        env: dict[str, str],
        text: bool,
        capture_output: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["env"] = env
        payload = {
            "status": "ok",
            "elapsed_s": 0.25,
            "residual_norm": 0.0,
            "solver_provenance": {
                "requested_solve_method": "lgmres",
                "realized_solve_method": "lgmres",
            },
        }
        stdout = "__SFINCS_JAX_PAS_TZ_RESULT__=" + json.dumps(payload) + "\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = type(
        "Args",
        (),
        {
            "input": source,
            "timeout_s": 9.0,
            "tol": 1.0e-7,
            "solve_method": "incremental",
            "maxiter": 6,
            "restart": 8,
            "block": 4,
            "overlap": 2,
            "variants": ["tzfft-lgmres"],
            "Ntheta": None,
            "Nzeta": None,
            "Nxi": None,
            "Nx": None,
        },
    )()

    row = _run_child(args, "tzfft-lgmres")

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--solve-method") + 1] == "lgmres"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK"] == "tzfft"
    assert row["variant_provenance"] == {
        "variant": "tzfft-lgmres",
        "base_variant": "tzfft",
        "requested_solve_method": "incremental",
        "realized_solve_method": "lgmres",
        "solve_method_source": "variant_suffix",
        "lgmres_opt_in": True,
    }
