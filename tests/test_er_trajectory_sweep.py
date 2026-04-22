from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "examples" / "publication_figures" / "generate_er_trajectory_sweep.py"
    spec = importlib.util.spec_from_file_location("generate_er_trajectory_sweep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rewrite_trajectory_input_sets_upstream_model_flags() -> None:
    mod = _load_module()
    base_text = """&physicsParameters
  Er = -1.0
  includeXDotTerm = .true.
  includeElectricFieldTermInXiDot = .true.
  useDKESExBDrift = .false.
  magneticDriftScheme = 1
/
&resolutionParameters
  Ntheta = 9
  Nzeta = 11
  Nxi = 7
  NL = 5
  Nx = 8
  solverTolerance = 1e-8
/
"""
    partial = next(m for m in mod.TRAJECTORY_MODELS if m.name == "partial")
    text = mod.rewrite_trajectory_input(base_text=base_text, er=3.5, model=partial, fast=True)
    assert "Er = 3.5" in text
    assert "includeXDotTerm = .false." in text
    assert "includeElectricFieldTermInXiDot = .false." in text
    assert "useDKESExBDrift = .false." in text
    assert "magneticDriftScheme = 0" in text
    assert "Ntheta = 5" in text
    assert "solverTolerance = 1e-4" in text


def test_collect_and_reload_sweep_summary(tmp_path: Path) -> None:
    mod = _load_module()
    out_h5 = tmp_path / "sfincsOutput.h5"
    with h5py.File(out_h5, "w") as h5:
        h5["particleFlux_vm_psiHat"] = np.asarray([1.0, 2.0])
        h5["heatFlux_vm_psiHat"] = np.asarray([3.0, 4.0])
        h5["FSABFlow"] = np.asarray([5.0, 6.0])
        h5["FSABjHat"] = np.asarray(7.0)

    model = next(m for m in mod.TRAJECTORY_MODELS if m.name == "full")
    rec = mod.collect_sweep_record(
        output_path=out_h5,
        model=model,
        er=-10.0,
        er_res=100.0,
        species_index=1,
    )
    assert rec.er_over_eres == -0.1
    assert rec.particle_flux_vm_psi_hat == 2.0
    assert rec.heat_flux_vm_psi_hat == 4.0
    assert rec.fsab_flow == 6.0
    assert rec.fsab_jhat == 7.0

    summary_path = tmp_path / "summary.json"
    mod.write_summary_json(summary_path=summary_path, records=[rec])
    payload = json.loads(summary_path.read_text())
    assert payload[0]["label"] == model.label
    reloaded = mod.load_summary_json(summary_path)
    assert reloaded[0] == rec


def test_plot_er_trajectory_sweep_from_summary(tmp_path: Path) -> None:
    mod = _load_module()
    records = [
        mod.SweepRecord(
            model="dkes",
            label="DKES trajectories",
            er=-10.0,
            er_over_eres=-0.1,
            particle_flux_vm_psi_hat=1.0,
            heat_flux_vm_psi_hat=2.0,
            fsab_flow=3.0,
            fsab_jhat=4.0,
            output_path="a.h5",
        ),
        mod.SweepRecord(
            model="dkes",
            label="DKES trajectories",
            er=10.0,
            er_over_eres=0.1,
            particle_flux_vm_psi_hat=1.5,
            heat_flux_vm_psi_hat=2.5,
            fsab_flow=3.5,
            fsab_jhat=4.5,
            output_path="b.h5",
        ),
        mod.SweepRecord(
            model="full",
            label="Full trajectories",
            er=-10.0,
            er_over_eres=-0.1,
            particle_flux_vm_psi_hat=0.5,
            heat_flux_vm_psi_hat=1.5,
            fsab_flow=2.5,
            fsab_jhat=3.5,
            output_path="c.h5",
        ),
        mod.SweepRecord(
            model="full",
            label="Full trajectories",
            er=10.0,
            er_over_eres=0.1,
            particle_flux_vm_psi_hat=0.7,
            heat_flux_vm_psi_hat=1.7,
            fsab_flow=2.7,
            fsab_jhat=3.7,
            output_path="d.h5",
        ),
    ]
    mod.plot_er_trajectory_sweep(
        records=records,
        out_dir=tmp_path,
        stem="er_sweep_test",
        title="test",
    )
    assert (tmp_path / "er_sweep_test.png").exists()
    assert (tmp_path / "er_sweep_test.pdf").exists()
