"""What the examples tree has to keep true.

This used to pin the top README's nine section headings and dozens of exact
entry-point strings, which made the README impossible to simplify without
editing a test in lockstep -- and the sprawl it enforced was the thing new
users were getting lost in.  What is left checks properties rather than prose:
folders are intentional, every folder introduces itself, every script a README
names exists, nothing generated is tracked, the graded ladder stays complete.

Wording is deliberately not pinned here.  Claims about *measured numbers* are
pinned, but in the example that produced them, beside the code that re-checks.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
DOCS_EXAMPLES = REPO_ROOT / "docs" / "examples.rst"
WORKFLOW_CATALOG = EXAMPLES_ROOT / "workflow_catalog.json"

# The graded ladder, in order.  Grading is by what a folder *requires* -- 1
# needs nothing installed, 2 needs real geometry, 3 differentiates through the
# solve -- because that is objective, where "beginner/advanced" is taste.
GRADED_FOLDERS = ("1_basics", "2_equilibria", "3_gradients")

# Older topic folders, kept while their content is folded into the ladder.
LEGACY_FOLDERS = (
    "autodiff",
    "data",
    "getting_started",
    "optimization",
    "sfincs_examples",
    "transport",
    "tutorials",
    "vmex_finite_beta",
)

CASE_FOLDERS = ("cases",)

ALLOWED_EXAMPLE_FOLDERS = set(GRADED_FOLDERS) | set(LEGACY_FOLDERS) | set(CASE_FOLDERS)

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

# Phrases that date a README the moment anything around them changes.
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


def _workflow_catalog() -> dict:
    return json.loads(WORKFLOW_CATALOG.read_text(encoding="utf-8"))


def test_examples_top_level_folders_are_intentional() -> None:
    """A new folder is a decision, so it has to be made here too."""
    folders = {
        path.name
        for path in EXAMPLES_ROOT.iterdir()
        # "output" holds generated (gitignored, examples/**/output/) artifacts.
        if path.is_dir() and path.name not in {".ipynb_checkpoints", "__pycache__", "output"}
    }

    assert folders == ALLOWED_EXAMPLE_FOLDERS


def test_the_graded_ladder_is_complete_and_linked_from_the_top_readme() -> None:
    """1 -> 2 -> 3 is the path a new user is told to walk; keep it walkable."""
    readme = (EXAMPLES_ROOT / "README.md").read_text(encoding="utf-8")

    for folder in GRADED_FOLDERS:
        directory = EXAMPLES_ROOT / folder
        assert directory.is_dir(), folder
        assert (directory / "README.md").is_file(), f"{folder} does not introduce itself"
        assert list(directory.glob("*.py")), f"{folder} has no scripts"
        assert folder in readme, f"{folder} is not linked from examples/README.md"


def test_every_graded_script_is_listed_in_its_folder_readme() -> None:
    """An unlisted script is one nobody finds, which is the same as absent."""
    unlisted: list[str] = []
    for folder in GRADED_FOLDERS:
        readme = (EXAMPLES_ROOT / folder / "README.md").read_text(encoding="utf-8")
        for script in sorted((EXAMPLES_ROOT / folder).glob("*.py")):
            if script.name not in readme:
                unlisted.append(f"{folder}/{script.name}")

    assert unlisted == []


def test_every_folder_introduces_itself() -> None:
    missing = [
        folder
        for folder in sorted(ALLOWED_EXAMPLE_FOLDERS)
        if not (EXAMPLES_ROOT / folder / "README.md").is_file()
    ]
    assert missing == []


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
            if token.startswith(("examples/", "tools/")):
                script_path = REPO_ROOT / token
            else:
                script_path = base / token
            if not script_path.is_file():
                missing_scripts.append(f"{relative_readme}: {token}")

    assert offenders == []
    assert missing_scripts == []


def test_workflow_catalog_points_at_files_that_exist_and_run_unaided() -> None:
    """The catalog is the terminal route to the same map; keep it honest.

    ``requires_fortran_v3 is False`` is the load-bearing check: the catalog is
    what a new user searches, and an entry that silently needs a Fortran build
    they do not have is worse than no entry at all.
    """
    catalog = _workflow_catalog()

    assert catalog["schema_version"] == 1
    assert set(catalog["folders"]) <= ALLOWED_EXAMPLE_FOLDERS
    assert "workflow_catalog.json" in (EXAMPLES_ROOT / "README.md").read_text(encoding="utf-8")

    for folder, metadata in sorted(catalog["folders"].items()):
        assert metadata["role"], folder
        start_path = EXAMPLES_ROOT / metadata["start_here"]
        assert start_path.exists(), f"{folder}: {metadata['start_here']}"
        assert metadata["start_here"].split("/", 1)[0] == folder

    for workflow in catalog["workflows"]:
        entrypoint = workflow["entrypoint"]
        assert workflow["id"]
        assert workflow["goal"]
        assert workflow["command"].startswith("python examples/")
        assert workflow["keywords"]
        assert workflow["runtime_budget"]
        assert workflow["requires_fortran_v3"] is False, entrypoint
        assert (EXAMPLES_ROOT / entrypoint).is_file(), entrypoint


def test_docs_examples_page_names_every_folder() -> None:
    """Prose docs may say more than the README, but not less."""
    docs = DOCS_EXAMPLES.read_text(encoding="utf-8")
    missing = [
        folder
        for folder in sorted(ALLOWED_EXAMPLE_FOLDERS)
        if f"examples/{folder}" not in docs
    ]
    assert missing == []


def test_examples_do_not_teach_v3_driver_facade_imports() -> None:
    """Examples should teach the public API, not the compatibility shim."""
    offenders: list[str] = []
    checked_suffixes = {".md", ".py", ".ipynb"}
    for path in sorted(EXAMPLES_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in checked_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        if "v3_driver" in text or "dkx.v3_driver" in text:
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
        # Deleted-but-uncommitted files still appear in `git ls-files`;
        # size-check only what exists in the working tree.
        if path.exists() and path.stat().st_size > MAX_TRACKED_EXAMPLE_BYTES
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
            structural_errors.append(
                f"{notebook_name}: markdown={len(markdown_cells)} code={len(code_cells)}"
            )

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
