from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_publish_workflow_tests_the_artifact_downloaded_from_pypi() -> None:
    workflow = (ROOT / ".github/workflows/publish-pypi.yml").read_text()
    smoke = workflow.split("  pypi-smoke:", 1)[1]

    assert "needs: build-and-publish" in smoke
    assert "working-directory: ${{ runner.temp }}" in smoke
    assert "--index-url https://pypi.org/simple" in smoke
    assert "--only-binary=:all:" in smoke
    assert "site-packages" in smoke
    assert "result = dkx.run(" in smoke
    assert "np.isfinite(particle_flux)" in smoke
    assert "pip install ." not in smoke


def test_publish_workflow_uses_the_single_version_source() -> None:
    workflow = (ROOT / ".github/workflows/publish-pypi.yml").read_text()

    assert 'Path("dkx/_version.py")' in workflow
    assert '["project"]["version"]' not in workflow
    assert 'Path("dkx/__init__.py")' not in workflow
