from __future__ import annotations

import re
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"

ALLOWED_EXAMPLE_FOLDERS = {
    "additional_examples",
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

FOLDERS_REQUIRING_README = ALLOWED_EXAMPLE_FOLDERS - {
    "additional_examples",
    "data",
    "utils",
}

REQUIRED_TASK_ENTRYPOINTS = {
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
    "What works today",
    "checked docs now contain",
    "currently pinned",
    "currently ships",
    "now supports",
    "now writes",
    "new users",
)
SCRIPT_TOKEN_RE = re.compile(r"`([^`]*?\.py)`")


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
    assert "### Folder Map" in readme

    for folder in sorted(ALLOWED_EXAMPLE_FOLDERS):
        assert f"`{folder}/`" in readme or f"`{folder}" in readme, folder

    for folder in sorted(FOLDERS_REQUIRING_README):
        assert (EXAMPLES_ROOT / folder / "README.md").is_file(), folder

    for entrypoint in sorted(REQUIRED_TASK_ENTRYPOINTS):
        assert f"`{entrypoint}`" in readme, entrypoint
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
