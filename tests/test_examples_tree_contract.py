from __future__ import annotations

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
