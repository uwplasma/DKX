from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

PUBLIC_TEXT_SUFFIXES = {".md", ".rst", ".py", ".ipynb"}
PUBLIC_TEXT_ROOTS = {
    "README.md",
    "dkx/README.md",
    "docs",
    "examples",
}
EXCLUDED_PARTS = {
    "_build",
    "_static",
    "artifacts",
    "output",
    "outputs",
    "provenance",
    "sfincs_examples",
    "upstream",
}
EXCLUDED_FILENAMES = {
    "release_notes.rst",
}

DISALLOWED_PUBLIC_PATTERNS = (
    ("On the current main branch", re.compile(r"On the current main branch")),
    (
        "not replacements for the production-resolution gates",
        re.compile(r"not replacements for the production-resolution gates"),
    ),
    (
        "The production benchmark manifest",
        re.compile(r"The production benchmark manifest"),
    ),
    ("not a public performance row", re.compile(r"not a public performance row")),
    ("current main", re.compile(r"current main")),
    ("new benchmarks", re.compile(r"new benchmarks")),
    ("At the moment", re.compile(r"At the moment")),
    ("new version", re.compile(r"new version")),
    ("previous version", re.compile(r"previous version")),
    ("previously", re.compile(r"\bpreviously\b")),
    ("now supports", re.compile(r"now supports")),
    ("now has", re.compile(r"now has")),
    ("now includes", re.compile(r"now includes")),
    ("currently", re.compile(r"\bcurrently\b")),
)


def _tracked_public_text_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", *sorted(PUBLIC_TEXT_ROOTS)],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    paths: list[Path] = []
    for line in result.stdout.splitlines():
        rel = Path(line)
        if rel.suffix not in PUBLIC_TEXT_SUFFIXES:
            continue
        if rel.name in EXCLUDED_FILENAMES:
            continue
        if EXCLUDED_PARTS.intersection(rel.parts):
            continue
        # Deleted-but-uncommitted files still appear in `git ls-files`;
        # check only what exists in the working tree.
        if not (REPO_ROOT / rel).exists():
            continue
        paths.append(REPO_ROOT / rel)
    return sorted(paths)


def test_public_docs_examples_and_readmes_are_not_branch_history() -> None:
    """Keep public prose self-contained rather than tied to branch progress logs."""

    offenders: list[str] = []
    for path in _tracked_public_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(REPO_ROOT).as_posix()
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in DISALLOWED_PUBLIC_PATTERNS:
                if pattern.search(line):
                    offenders.append(f"{rel}:{line_number}: {label}")

    assert offenders == []
