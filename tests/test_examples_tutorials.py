from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = REPO_ROOT / "examples" / "tutorials"


def _notebook(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def test_tutorial_learning_path_has_expected_files() -> None:
    expected = {
        "README.md",
        "run_quick_output_and_plot.py",
        "00_start_here.ipynb",
        "01_cli_outputs_and_plots.ipynb",
        "02_transport_and_autodiff.ipynb",
        "03_bootstrap_redl_and_optimization.ipynb",
        "04_geometry_validation_and_performance.ipynb",
    }
    assert expected <= {path.name for path in TUTORIALS.iterdir()}


def test_tutorial_notebooks_are_pedagogic_and_parseable() -> None:
    notebooks = sorted(TUTORIALS.glob("*.ipynb"))
    assert [path.name for path in notebooks] == [
        "00_start_here.ipynb",
        "01_cli_outputs_and_plots.ipynb",
        "02_transport_and_autodiff.ipynb",
        "03_bootstrap_redl_and_optimization.ipynb",
        "04_geometry_validation_and_performance.ipynb",
    ]

    for path in notebooks:
        notebook = _notebook(path)
        assert notebook["nbformat"] == 4
        cells = notebook["cells"]
        markdown_cells = [cell for cell in cells if cell["cell_type"] == "markdown"]
        code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
        text = "\n".join("".join(cell["source"]) for cell in markdown_cells)
        assert len(cells) >= 9, path.name
        assert len(markdown_cells) >= 5, path.name
        assert len(code_cells) >= 3, path.name
        assert "sfincs_jax" in text, path.name


def test_tutorial_commands_reference_existing_scripts() -> None:
    referenced_scripts = {
        "examples/tutorials/run_quick_output_and_plot.py",
        "examples/transport/transport_matrix_rhsmode2_and_rhsmode3.py",
        "examples/autodiff/autodiff_gradient_nu_n_residual.py",
        "examples/vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py",
        "examples/optimization/qa_nfp2_sfincs_jax_objectives.py",
    }
    for script in referenced_scripts:
        assert (REPO_ROOT / script).is_file(), script

    notebook_text = "\n".join(path.read_text(encoding="utf-8") for path in TUTORIALS.glob("*.ipynb"))
    for script in referenced_scripts:
        assert script in notebook_text
