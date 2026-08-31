"""Every figure the docs show can be regenerated, or is recorded as not yet.

A figure in a README is a claim.  A reader who cannot regenerate it cannot check
it, which makes it the least verifiable thing the project publishes -- and the
easiest to leave behind when the code it came from moves.

``docs/figure_provenance.json`` maps every referenced figure to the checked-in
script that produces it.  Sixteen figures predate the manifest and have no
identifiable generator; rather than delete work whose provenance might still be
recoverable, they are recorded as unresolved and held under a **ratchet**: the
count may fall and may not rise.  A new figure must therefore arrive with its
generator.
"""

from __future__ import annotations

import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "docs" / "figure_provenance.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _referenced_figures() -> set[str]:
    """Figures actually shown by the README or a docs page."""
    text = " ".join(
        path.read_text()
        for path in [*(REPO_ROOT / "docs").glob("*.rst"), REPO_ROOT / "README.md"]
    )
    return {
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "docs" / "_static" / "figures").rglob("*.png")
        if path.name in text
    }


def test_every_displayed_figure_is_in_the_manifest():
    """A figure added to the docs without provenance fails here, not in review."""
    missing = _referenced_figures() - set(_manifest()["figures"])
    assert not missing, f"figures shown but with no provenance entry: {sorted(missing)}"


def test_the_manifest_does_not_describe_figures_that_are_gone():
    """Stale entries make the manifest look more complete than it is."""
    figures = _manifest()["figures"]
    absent = {name for name in figures if not (REPO_ROOT / name).exists()}
    assert not absent, f"manifest entries with no file: {sorted(absent)}"


def test_every_named_generator_exists():
    """The whole value of the manifest is that the command it names is real."""
    broken = {
        name: entry["generator"]
        for name, entry in _manifest()["figures"].items()
        if entry.get("generator") and not (REPO_ROOT / entry["generator"]).exists()
    }
    assert not broken, f"generators named but absent: {broken}"


def test_unresolved_figures_carry_a_reason():
    """``generator: null`` without a reason is indistinguishable from an oversight."""
    for name, entry in _manifest()["figures"].items():
        if entry.get("generator") is None:
            assert entry.get("unresolved"), name


def test_the_unresolved_count_only_falls():
    """The ratchet.

    Sixteen figures predate the manifest.  Holding the count rather than
    demanding zero lets the debt be paid down without blocking unrelated work,
    while making it impossible to add to: a new figure without a generator
    raises the count and fails here.
    """
    manifest = _manifest()
    actual = sum(
        1 for entry in manifest["figures"].values() if entry.get("generator") is None
    )
    budget = int(manifest["unresolved_budget"])
    assert actual <= budget, (
        f"{actual} figures have no generator against a budget of {budget}; "
        "a new figure must arrive with the script that produces it"
    )
    assert budget <= 16, "the budget is a ratchet: lower it as provenance is found"


@pytest.mark.parametrize("key", ["schema_version", "purpose", "figures"])
def test_manifest_shape(key: str):
    assert key in _manifest()


def _evidence_figures() -> set[str]:
    """Figures cited as evidence in a tracked completion record.

    ``tools.release.release`` existence-checks these paths, so they are
    load-bearing even though no page displays them.
    """
    cited = set()
    for record in (REPO_ROOT / "docs" / "_static").glob("*.json"):
        text = record.read_text()
        for path in (REPO_ROOT / "docs" / "_static" / "figures").rglob("*.png"):
            if path.name in text:
                cited.add(str(path.relative_to(REPO_ROOT)))
    return cited


def test_no_figure_is_committed_without_a_consumer():
    """A committed figure is shown by a page or cited as evidence — or it is weight.

    Thirty-three figures were carried with neither: superseded plots, outputs of
    scripts that still regenerate them on demand, and snapshots of runs nobody
    links to.  Storing regenerable output that nothing displays costs the
    repository's size budget and tells a reader nothing.
    """
    committed = {
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "docs" / "_static" / "figures").rglob("*.png")
    }
    orphaned = committed - _referenced_figures() - _evidence_figures()
    assert not orphaned, (
        "committed but neither displayed nor cited as evidence: "
        f"{sorted(orphaned)} — display it, cite it, or delete it and let the "
        "script regenerate it"
    )
