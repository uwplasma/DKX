from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.io import _output_geom_cache_key, read_sfincs_h5, write_sfincs_jax_output_h5
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import _equilibrium_file_key, grids_from_namelist


_REPO_ROOT = Path(__file__).resolve().parents[1]
_EQUILIBRIA_DIR = _REPO_ROOT / "sfincs_jax" / "data" / "equilibria"
_W7X_WOUT = _EQUILIBRIA_DIR / "wout_w7x_standardConfig.nc"


def test_write_output_return_results(tmp_path: Path) -> None:
    input_path = Path(__file__).parent / "ref" / "output_scheme4_1species_tiny.input.namelist"
    out_path = tmp_path / "sfincsOutput.h5"

    resolved, results = write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=out_path,
        return_results=True,
    )

    assert resolved == out_path.resolve()
    assert resolved.exists()
    assert isinstance(results, dict)
    assert "Ntheta" in results
    assert int(np.asarray(results["Ntheta"]).reshape(())) > 0


def test_write_output_wout_path_override_resolves_missing_scheme5_input(tmp_path: Path) -> None:
    source_input = Path(__file__).parent / "ref" / "output_scheme5_1species_tiny.input.namelist"
    patched_input = tmp_path / "input.namelist"
    patched_input.write_text(
        source_input.read_text().replace("wout_w7x_standardConfig.nc", "missing_wout.nc"),
        encoding="utf-8",
    )
    out_path = tmp_path / "sfincsOutput.h5"

    with pytest.raises(FileNotFoundError):
        write_sfincs_jax_output_h5(
            input_namelist=patched_input,
            output_path=out_path,
        )

    write_sfincs_jax_output_h5(
        input_namelist=patched_input,
        output_path=out_path,
        wout_path=_W7X_WOUT,
    )

    data = read_sfincs_h5(out_path)
    input_text = str(data["input.namelist"])
    assert "missing_wout.nc" not in input_text
    assert str(_W7X_WOUT) in input_text


def test_output_geom_cache_key_reuses_copied_equilibrium_content(tmp_path: Path) -> None:
    here = Path(__file__).parent
    source_input = here / "ref" / "output_scheme5_1species_tiny.input.namelist"

    copied_wout = tmp_path / "copy_wout.nc"
    copied_wout.write_bytes(_W7X_WOUT.read_bytes())

    patched_input = tmp_path / "input_copy.namelist"
    patched_input.write_text(
        source_input.read_text().replace("wout_w7x_standardConfig.nc", str(copied_wout)),
        encoding="utf-8",
    )

    nml_src = read_sfincs_input(source_input)
    grids_src = grids_from_namelist(nml_src)
    nml_copy = read_sfincs_input(patched_input)
    grids_copy = grids_from_namelist(nml_copy)

    assert _equilibrium_file_key(
        nml=nml_src,
        geometry_scheme=5,
        geom_group=nml_src.group("geometryParameters"),
    ) == _equilibrium_file_key(
        nml=nml_copy,
        geometry_scheme=5,
        geom_group=nml_copy.group("geometryParameters"),
    )
    assert _output_geom_cache_key(nml=nml_src, grids=grids_src) == _output_geom_cache_key(nml=nml_copy, grids=grids_copy)


def test_output_geom_cache_key_tracks_classical_flux_inputs_not_dkes_flag(tmp_path: Path) -> None:
    here = Path(__file__).parent
    src = here / "reduced_inputs" / "geometryScheme4_2species_PAS_noEr.input.namelist"

    same_phys = tmp_path / "same_phys.namelist"
    changed_species = tmp_path / "changed_species.namelist"
    changed_irrelevant = tmp_path / "changed_irrelevant.namelist"

    text = src.read_text(encoding="utf-8")
    same_phys.write_text(text, encoding="utf-8")
    changed_species.write_text(text.replace("THats = 1.0d+0 1.0d+0", "THats = 1.0d+0 1.1d+0"), encoding="utf-8")
    changed_irrelevant.write_text(text.replace("useDKESExBDrift = .false.", "useDKESExBDrift = .true."), encoding="utf-8")

    nml_base = read_sfincs_input(same_phys)
    nml_species = read_sfincs_input(changed_species)
    nml_irrelevant = read_sfincs_input(changed_irrelevant)
    grids = grids_from_namelist(nml_base)

    key_base = _output_geom_cache_key(nml=nml_base, grids=grids)
    key_species = _output_geom_cache_key(nml=nml_species, grids=grids_from_namelist(nml_species))
    key_irrelevant = _output_geom_cache_key(nml=nml_irrelevant, grids=grids_from_namelist(nml_irrelevant))

    assert key_base != key_species
    assert key_base == key_irrelevant
