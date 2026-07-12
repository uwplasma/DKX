from __future__ import annotations

from pathlib import Path

import numpy as np

from sfincs_jax.api import write_output
from sfincs_jax.io import read_sfincs_h5


def _replace_line(text: str, old: str, new: str) -> str:
    if old not in text:
        raise AssertionError(f"did not find expected text: {old!r}")
    return text.replace(old, new, 1)


def test_phi1_output_writes_quasineutrality_option_without_adiabatic(tmp_path: Path) -> None:
    here = Path(__file__).parent
    src = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.input.namelist"
    text = src.read_text(encoding="utf-8")
    text = _replace_line(text, "  withAdiabatic = .t.", "  withAdiabatic = .f.")
    input_path = tmp_path / "phi1_no_adiabatic.input.namelist"
    input_path.write_text(text, encoding="utf-8")

    out_path = tmp_path / "phi1_no_adiabatic.sfincsOutput.h5"
    # Metadata regression only (the upstream test wrote with
    # compute_solution=False): use the canonical no-solve geometry-only write.
    write_output(input_path, out_path, geometry_only=True)

    out = read_sfincs_h5(out_path)
    assert "quasineutralityOption" in out
    assert int(np.asarray(out["quasineutralityOption"]).reshape(())) == 2
    assert int(np.asarray(out["includePhi1"]).reshape(())) == 1


def test_monoenergetic_transport_write_output_exports_delta_f_and_full_f(tmp_path: Path, monkeypatch) -> None:
    here = Path(__file__).parent
    src = here / "ref" / "monoenergetic_PAS_tiny_scheme11.input.namelist"
    text = src.read_text(encoding="utf-8")
    text = _replace_line(text, "  export_full_f = .false.", "  export_full_f = .true.")
    text = _replace_line(text, "  export_delta_f = .false.", "  export_delta_f = .true.")
    input_path = tmp_path / "monoenergetic_export.input.namelist"
    input_path.write_text(text, encoding="utf-8")

    out_path = tmp_path / "monoenergetic_export.sfincsOutput.h5"
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_LOW_MEMORY", "1")
    write_output(input_path, out_path)

    out = read_sfincs_h5(out_path)
    assert "delta_f" in out
    assert "full_f" in out
    delta_f = np.asarray(out["delta_f"])
    full_f = np.asarray(out["full_f"])
    assert delta_f.shape == full_f.shape
    assert delta_f.shape[-1] == int(np.asarray(out["NIterations"]).reshape(()))
    assert delta_f.shape[0] == int(np.asarray(out["N_export_f_x"]).reshape(()))
    assert delta_f.shape[1] == int(np.asarray(out["N_export_f_xi"]).reshape(()))
    assert delta_f.shape[2] == int(np.asarray(out["N_export_f_zeta"]).reshape(()))
    assert delta_f.shape[3] == int(np.asarray(out["N_export_f_theta"]).reshape(()))
    assert np.isfinite(delta_f).all()
    assert np.isfinite(full_f).all()
