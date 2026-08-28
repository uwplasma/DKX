"""``dkx wout_*.nc`` must work from a wheel, not just from a source checkout.

The representative run built its monoenergetic base by reading
``dkx/data/representative.namelist``, falling back to a deck under
``examples/``.  Neither ships in the wheel -- verified against the published
2.2.0 artifact -- so the flagship "point dkx at an equilibrium" path worked in
a checkout and failed for every pip user.

It failed two ways.  With nothing at either path, FileNotFoundError naming a
namelist the user never asked for.  With some other namelist sitting at one of
them, a base carrying the default ``RHSMode = 1``, which surfaced three frames
deeper as "run_transport_matrix supports RHSMode 2 and 3" -- a message that
points at the solver rather than at the missing file.  That second form is
what a user reported.

The base is a module-level string now, so these tests pin that it stays one.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "dkx"


def test_the_monoenergetic_template_is_rhsmode_3() -> None:
    """RHSMode=3 is the entire point of this deck, so state it as a fact."""
    from dkx.inputs import parse_sfincs_input_text, sfincs_input_from_raw
    from dkx.representative import _MONOENERGETIC_TEMPLATE

    base = sfincs_input_from_raw(
        parse_sfincs_input_text(_MONOENERGETIC_TEMPLATE.format(equilibrium="w.nc"))
    )
    assert int(base.general.rhs_mode) == 3


def test_monoenergetic_scan_rejects_an_rhsmode_1_deck_where_the_mistake_is() -> None:
    from dkx.inputs import parse_sfincs_input_text, sfincs_input_from_raw
    from dkx.representative import _MONOENERGETIC_TEMPLATE, monoenergetic_scan

    text = _MONOENERGETIC_TEMPLATE.format(equilibrium="w.nc").replace(
        "RHSMode = 3", "RHSMode = 1"
    )
    base = sfincs_input_from_raw(parse_sfincs_input_text(text))
    with pytest.raises(ValueError, match="RHSMode=3"):
        monoenergetic_scan(base, nu_prime=(1.0,))


def test_a_missing_equilibrium_names_the_equilibrium() -> None:
    """The error should name what the user passed, not an internal template."""
    from dkx.representative import run_representative

    with pytest.raises(FileNotFoundError, match="not_a_real_wout.nc"):
        run_representative("/tmp/not_a_real_wout.nc", emit=None)


def test_the_base_builds_with_no_examples_directory_next_to_the_package(tmp_path) -> None:
    """Simulate site-packages: the package alone, no repo siblings.

    This is the layout the bug needed.  Copying the package somewhere with no
    ``examples/`` beside it is what a wheel install looks like, and building
    the base there is the step that used to raise.
    """
    shutil.copytree(PACKAGE, tmp_path / "dkx")
    assert not (tmp_path / "examples").exists()

    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from dkx.inputs import parse_sfincs_input_text, sfincs_input_from_raw\n"
        "from dkx.representative import _MONOENERGETIC_TEMPLATE\n"
        "base = sfincs_input_from_raw(parse_sfincs_input_text(\n"
        "    _MONOENERGETIC_TEMPLATE.format(equilibrium='w.nc')))\n"
        "print(int(base.general.rhs_mode))\n"
    ) % str(tmp_path)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "3"


def test_representative_does_not_reach_outside_the_package_for_its_deck() -> None:
    """A runtime read from ``examples/`` or ``data/`` is the retired pattern."""
    source = (PACKAGE / "representative.py").read_text()
    body = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("#", "#:"))
    )
    assert 'parents[1]' not in body or '"examples"' not in body
    assert '"data" / "representative.namelist"' not in body


def test_the_cli_front_door_passes_quick_through(monkeypatch, tmp_path) -> None:
    """``dkx <wout> --quick`` is what the wheel-install CI job runs.

    The representative front door is handled before argparse sees anything, so
    ``--quick`` is matched against the raw argv by hand.  That also means the
    flag has to survive the positional scan that finds the equilibrium: an
    option token must not be mistaken for the file, and the file must still be
    found when the flag comes first.
    """
    from dkx import cli, representative

    calls: list[dict] = []

    def fake_run(path, **kwargs):
        calls.append({"path": Path(path), **kwargs})
        out = tmp_path / "panels.png"
        out.write_bytes(b"")
        return out

    monkeypatch.setattr(representative, "run_representative", fake_run)
    wout = tmp_path / "wout_stub.nc"
    wout.write_bytes(b"not really netCDF")

    for argv in ([str(wout), "--quick"], ["--quick", str(wout)]):
        calls.clear()
        assert cli.main(list(argv)) == 0, argv
        assert len(calls) == 1, argv
        assert calls[0]["path"] == wout, argv
        assert calls[0]["quick"] is True, argv
        assert calls[0]["full"] is False, argv

    calls.clear()
    assert cli.main([str(wout)]) == 0
    assert calls[0]["quick"] is False, "quick must not be the default"
