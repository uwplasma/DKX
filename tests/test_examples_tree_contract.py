from __future__ import annotations

import json
import re
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
DOCS_EXAMPLES = REPO_ROOT / "docs" / "examples.rst"

ALLOWED_EXAMPLE_FOLDERS = {
    "autodiff",
    "data",
    "getting_started",
    "optimization",
    "parity",
    "performance",
    "publication_figures",
    "sfincs_examples",
    "transport",
    "tutorials",
    "upstream",
    "utils",
    "vmec_jax_finite_beta",
}

FOLDERS_REQUIRING_README = ALLOWED_EXAMPLE_FOLDERS

REQUIRED_TASK_ENTRYPOINTS = {
    "tutorials/00_start_here.ipynb",
    "tutorials/04_geometry_validation_and_performance.ipynb",
    "tutorials/run_quick_output_and_plot.py",
    "getting_started/write_sfincs_output_cli.py",
    "getting_started/write_sfincs_output_python.py",
    "transport/transport_matrix_rhsmode2_and_rhsmode3.py",
    "autodiff/implicit_diff_through_gmres_solve_scheme5.py",
    "vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py",
    "optimization/qa_nfp2_sfincs_jax_objectives.py",
    "parity/output_parity_vs_fortran_fixture.py",
    "performance/benchmark_output_formats.py",
}

CANONICAL_WORKFLOW_ENTRYPOINTS = {
    "getting_started/build_grids_and_geometry.py",
    "getting_started/apply_collisionless_operator.py",
    "getting_started/write_sfincs_output_python.py",
    "getting_started/write_sfincs_output_vmec.py",
    "getting_started/write_and_plot_multiple_formats.py",
    "transport/transport_matrix_rhsmode2_scheme11_and_scheme5.py",
    "autodiff/implicit_diff_through_gmres_solve_scheme5.py",
    "vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py",
    "optimization/QA_optimization_bootstrap_current.py",
    "parity/collisionless_operator_matvec_parity.py",
    "publication_figures/generate_fortran_suite_benchmark_summary.py",
    "performance/benchmark_transport_parallel_scaling.py",
}

APPLICATION_RECIPE_ENTRYPOINTS = {
    "tutorials/run_quick_output_and_plot.py",
    "getting_started/write_and_plot_multiple_formats.py",
    "getting_started/write_sfincs_output_tokamak.py",
    "sfincs_examples/tokamak_1species_FPCollisions_noEr/input.namelist",
    "getting_started/write_sfincs_output_vmec.py",
    "vmec_jax_finite_beta/finite_beta_vmec_to_sfincs.py",
    "transport/transport_matrix_rhsmode2_and_rhsmode3.py",
    "transport/transport_matrix_rhsmode2_scheme11_and_scheme5.py",
    "vmec_jax_finite_beta/compare_qs_paper_sfincs_jax_redl.py",
    "tutorials/03_bootstrap_redl_and_optimization.ipynb",
    "optimization/evaluate_sfincs_jax_promotion_scan.py",
    "autodiff/autodiff_gradient_nu_n_residual.py",
    "autodiff/implicit_diff_through_gmres_solve_scheme5.py",
    "autodiff/vmec_jax_to_boozer_sfincs_pipeline.py",
    "tutorials/04_geometry_validation_and_performance.ipynb",
    "optimization/qa_nfp2_sfincs_jax_objectives.py",
    "optimization/QA_optimization_bootstrap_current.py",
    "performance/benchmark_output_formats.py",
    "performance/benchmark_transport_parallel_scaling.py",
    "parity/output_parity_vs_fortran_fixture.py",
    "publication_figures/generate_fortran_suite_benchmark_summary.py",
}

APPLICATION_RECIPE_LABELS = {
    "CLI output and diagnostics panel",
    "Analytic tokamak input",
    "VMEC `wout_path` input",
    "RHSMode=2/3 transport matrix",
    "Bootstrap current vs Redl",
    "Ambipolar electric-field scan",
    "Differentiable residual or flux",
    "VMEC/Boozer/JAX handoff",
    "QA/QI optimization objective",
    "CPU/GPU timing and output I/O",
    "Frozen Fortran-v3 parity check",
}

DISALLOWED_TRACKED_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
    "outputs",
    "trace",
    "traces",
}

DISALLOWED_TRACKED_SUFFIXES = {
    ".h5",
    ".hdf5",
    ".prof",
    ".pb",
    ".gz",
    ".npy",
    ".npz",
}

MAX_TRACKED_EXAMPLE_BYTES = 2 * 1024 * 1024
README_STALE_FRAGMENTS = (
    "At the moment",
    "What works today",
    "checked docs now contain",
    "For the current support matrix",
    "For current hot-solve",
    "The current panel",
    "The current publication-grade",
    "currently pinned",
    "currently ships",
    "now supports",
    "now writes",
    "new users",
)
SCRIPT_TOKEN_RE = re.compile(r"`([^`]*?\.py)`")
TUTORIAL_NOTEBOOK_REQUIREMENTS = {
    "00_start_here.ipynb": ("drift-kinetic", "bootstrap current", "optimization"),
    "01_cli_outputs_and_plots.ipynb": ("HDF5", "NetCDF", "diagnostics"),
    "02_transport_and_autodiff.ipynb": ("RHSMode=2/3", "Autodiff", "JAX"),
    "03_bootstrap_redl_and_optimization.ipynb": ("Redl", "bootstrap", "Optimization"),
    "04_geometry_validation_and_performance.ipynb": ("VMEC", "SFINCS Fortran v3", "CPU/GPU"),
}


def _tracked_example_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "examples"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [REPO_ROOT / line for line in result.stdout.splitlines() if line]


def test_examples_top_level_folders_are_intentional() -> None:
    folders = {
        path.name
        for path in EXAMPLES_ROOT.iterdir()
        if path.is_dir() and path.name not in {".ipynb_checkpoints", "__pycache__"}
    }

    assert folders == ALLOWED_EXAMPLE_FOLDERS


def test_examples_readme_is_a_complete_user_navigation_map() -> None:
    readme = (EXAMPLES_ROOT / "README.md").read_text(encoding="utf-8")
    assert "### Learning Path" in readme
    assert "### Choose By Task" in readme
    assert "### Application Recipes" in readme
    assert "### Canonical Workflow Catalog" in readme
    assert "### Folder Map" in readme

    for folder in sorted(ALLOWED_EXAMPLE_FOLDERS):
        assert f"`{folder}/`" in readme or f"`{folder}" in readme, folder

    for folder in sorted(FOLDERS_REQUIRING_README):
        assert (EXAMPLES_ROOT / folder / "README.md").is_file(), folder

    for entrypoint in sorted(REQUIRED_TASK_ENTRYPOINTS):
        assert f"`{entrypoint}`" in readme, entrypoint
        assert (EXAMPLES_ROOT / entrypoint).is_file(), entrypoint

    for entrypoint in sorted(CANONICAL_WORKFLOW_ENTRYPOINTS):
        assert f"`{entrypoint}`" in readme, entrypoint
        assert (EXAMPLES_ROOT / entrypoint).is_file(), entrypoint

    for label in sorted(APPLICATION_RECIPE_LABELS):
        assert label in readme, label

    for entrypoint in sorted(APPLICATION_RECIPE_ENTRYPOINTS):
        assert f"`{entrypoint}`" in readme, entrypoint
        assert (EXAMPLES_ROOT / entrypoint).is_file(), entrypoint


def test_docs_examples_page_matches_application_recipe_map() -> None:
    docs = DOCS_EXAMPLES.read_text(encoding="utf-8")
    assert "Application recipe map" in docs

    for label in sorted(APPLICATION_RECIPE_LABELS):
        rst_label = label.replace("`", "``")
        assert rst_label in docs, label

    for entrypoint in sorted(APPLICATION_RECIPE_ENTRYPOINTS):
        docs_path = f"examples/{entrypoint}"
        assert f"``{docs_path}``" in docs, docs_path
        assert (EXAMPLES_ROOT / entrypoint).is_file(), entrypoint


def test_example_readmes_are_standalone_and_reference_existing_scripts() -> None:
    offenders: list[str] = []
    missing_scripts: list[str] = []

    for readme_path in sorted(EXAMPLES_ROOT.glob("*/README.md")) + [EXAMPLES_ROOT / "README.md"]:
        text = readme_path.read_text(encoding="utf-8")
        relative_readme = readme_path.relative_to(REPO_ROOT).as_posix()
        for fragment in README_STALE_FRAGMENTS:
            if fragment in text:
                offenders.append(f"{relative_readme}: {fragment!r}")

        base = readme_path.parent
        for token in SCRIPT_TOKEN_RE.findall(text):
            if " " in token or token.startswith(("/", "http")):
                continue
            script_path = (REPO_ROOT / token) if token.startswith("examples/") else (base / token)
            if not script_path.is_file():
                missing_scripts.append(f"{relative_readme}: {token}")

    assert offenders == []
    assert missing_scripts == []


def test_examples_do_not_teach_v3_driver_facade_imports() -> None:
    """Examples should teach the public API, not the compatibility shim."""

    offenders: list[str] = []
    checked_suffixes = {".md", ".py", ".ipynb"}
    for path in sorted(EXAMPLES_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in checked_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        if "v3_driver" in text or "sfincs_jax.v3_driver" in text:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []


def test_examples_do_not_track_generated_caches_or_binary_outputs() -> None:
    offenders: list[str] = []
    for path in _tracked_example_files():
        relative = path.relative_to(REPO_ROOT)
        if DISALLOWED_TRACKED_PARTS.intersection(relative.parts):
            offenders.append(relative.as_posix())
            continue
        if path.suffix in DISALLOWED_TRACKED_SUFFIXES:
            offenders.append(relative.as_posix())

    assert offenders == []


def test_examples_do_not_track_large_files() -> None:
    oversized = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _tracked_example_files()
        if path.stat().st_size > MAX_TRACKED_EXAMPLE_BYTES
    ]

    assert oversized == []


def test_tutorial_notebooks_are_pedagogic_and_output_free() -> None:
    missing_topics: list[str] = []
    structural_errors: list[str] = []
    persisted_outputs: list[str] = []

    for notebook_name, required_terms in sorted(TUTORIAL_NOTEBOOK_REQUIREMENTS.items()):
        path = EXAMPLES_ROOT / "tutorials" / notebook_name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook.get("cells", [])
        markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
        code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
        joined_markdown = "\n".join("".join(cell.get("source", [])) for cell in markdown_cells)

        if len(markdown_cells) < 5 or len(code_cells) < 3:
            structural_errors.append(f"{notebook_name}: markdown={len(markdown_cells)} code={len(code_cells)}")

        for term in required_terms:
            if term not in joined_markdown:
                missing_topics.append(f"{notebook_name}: {term}")

        for cell_index, cell in enumerate(code_cells):
            if cell.get("outputs"):
                persisted_outputs.append(f"{notebook_name}: code cell {cell_index}")
            if cell.get("execution_count") is not None:
                persisted_outputs.append(f"{notebook_name}: executed code cell {cell_index}")

    assert structural_errors == []
    assert missing_topics == []
    assert persisted_outputs == []
