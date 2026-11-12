"""Tracked text must not contain machine paths, personal handles or AI-attribution strings."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "machine path": r"/Users/|/home/(?!runner)[a-z]|/private/tmp|/data\d+/|C:\\Users",
    "personal handle": r"rogeriojorge|rjorge",
    "AI attribution": (
        r"\bCo-Authored-By:\s*(?:Claude|ChatGPT|Copilot|Codex)\b"
        r"|\bGenerated with \[?(?:Claude|ChatGPT|Copilot|Codex)\b"
        r"|noreply@anthropic\.com"
    ),
}
HANDLE_OK = {"CITATION.cff", "LICENSE", "pyproject.toml", ".zenodo.json"}
THIRD_PARTY = ("third_party/", "vendor/", "external/")


def _tracked(root=ROOT):
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
        ).stdout.decode()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    this = Path(__file__).relative_to(ROOT).as_posix()
    return [p for p in out.split("\0") if p and p != this and not p.startswith(THIRD_PARTY)]


def test_hygiene_skips_outside_a_git_checkout(tmp_path):
    with pytest.raises(pytest.skip.Exception):
        _tracked(tmp_path)


def test_tracked_text_is_clean():
    bad = []
    for rel in _tracked():
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        text = re.sub(r"github\.com/rogeriojorge|github\.com/uwplasma", "", text)
        for name, pattern in PATTERNS.items():
            if name == "personal handle" and rel.rsplit("/", 1)[-1] in HANDLE_OK:
                continue
            for m in re.finditer(pattern, text):
                bad.append(f"{rel}: {name}: {m.group(0)}")
    assert not bad, "\n".join(bad[:20])
