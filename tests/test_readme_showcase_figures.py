"""The README showcase numbers, their figures and their documentation agree.

``tools/publication_figures/generate_readme_showcase.py`` draws two README
figures from measured numbers that ``docs/benchmarks/sfincs.md`` records with
their provenance.  The README quotes the same numbers in prose.  This pins all
three places together, and regenerates both figures to check they stay within
the README figure budget.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "tools" / "publication_figures" / "generate_readme_showcase.py"
README = REPO_ROOT / "README.md"
RECORD_PAGE = REPO_ROOT / "docs" / "benchmarks" / "sfincs.md"
FIGURE_BUDGET_BYTES = 150 * 1024


def _generator():
    spec = importlib.util.spec_from_file_location("generate_readme_showcase", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "tokens",
    [
        # SFINCS matrix sparsify threshold on HSX
        ("12–19%", "6.6e-11", "1.4e-11", "7e-11"),
        # SFINCS on the HSX-like gap deck
        ("22.5 min", "0.9955", "30.6 min", "2.5e-5", "20,000"),
        # exact operator assembly
        ("4,800",),
        # one factorization, many solves
        ("0.9593", "0.2822", "1.1931", "0.2779", "0.15 of"),
    ],
)
def test_every_showcase_number_is_in_the_benchmark_page(tokens: tuple[str, ...]) -> None:
    text = RECORD_PAGE.read_text(encoding="utf-8")
    for token in tokens:
        assert token in text, token


def test_the_readme_quotes_what_the_generator_plots() -> None:
    module = _generator()
    readme = README.read_text(encoding="utf-8")
    assert module.RELEASED_GAP == (0.19, 0.12)
    assert "12–19 %" in readme and "7e-11" in readme
    assert "0.9955" in readme and "2.5e-5" in readme and "open in both codes" in readme
    structured = module.REUSE_ROUTES["structured direct, 16,230 unknowns"][0]
    assert f"{structured[2]:.2f} s to {structured[3]:.2f} s" in readme
    for name in ("sfincs_reference_limits.png", "factor_reuse.png"):
        assert f"docs/_static/figures/readme/{name}" in readme


def test_the_figures_regenerate_within_the_readme_budget(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    _generator().main(["--out-dir", str(tmp_path)])
    for name in ("sfincs_reference_limits.png", "factor_reuse.png"):
        size = (tmp_path / name).stat().st_size
        assert 1024 < size <= FIGURE_BUDGET_BYTES, (name, size)
        committed = REPO_ROOT / "docs" / "_static" / "figures" / "readme" / name
        assert committed.stat().st_size <= FIGURE_BUDGET_BYTES, name
