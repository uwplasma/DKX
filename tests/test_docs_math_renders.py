"""The built docs contain no raw ``$...$``: math that failed to parse shows up as literal dollars.

A backtick in the info string of a backtick fence (for example ``{list-table}`` with
inline code in its title) turns the fence into plain text and swallows the rest of the page.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_built_html_has_no_raw_dollar_math(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", "-q", "-j", "auto", str(ROOT / "docs"), str(tmp_path)],
        check=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"} | {"HOME": str(tmp_path)},
    )
    bad = {}
    for page in tmp_path.rglob("*.html"):
        text = re.sub(r"<script.*?</script>|<style.*?</style>", "", page.read_text(errors="ignore"), flags=re.S)
        text = re.sub(r'<span class="math[^>]*>.*?</span>|<div class="math[^>]*>.*?</div>', "", text, flags=re.S)
        hits = re.findall(r"\$[^$<]{2,}\$", text)
        if hits:
            bad[str(page.relative_to(tmp_path))] = len(hits)
    assert not bad, f"raw $...$ in built docs: {bad}"
