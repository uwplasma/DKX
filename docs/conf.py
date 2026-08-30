from __future__ import annotations

import sys
from pathlib import Path

project = "DKX"
copyright = "2026"
author = "dkx contributors"

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    # MyST is the narrative format plan.md section 8.1 moves to. Enabling the
    # parser now means Phase F can convert pages one at a time instead of in a
    # single unreviewable commit; the .rst pages keep building unchanged.
    "myst_parser",
]

templates_path = ["_templates"]
# docs/dev/ holds internal ledgers, not user documentation. They were invisible
# to Sphinx until MyST was enabled because nothing parsed .md; they are excluded
# rather than added to a toctree, because plan.md section 8.2 keeps campaign
# diaries out of user navigation. Phase F decides whether they survive at all.
exclude_patterns: list[str] = ["dev/**"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Furo, per plan.md section 8.1. There is deliberately no fallback theme: the
# docs extra installs it, and silently building with a different theme than the
# one the pages are designed against is the kind of quiet downgrade the plan
# forbids. A missing theme should fail the build and say so.
html_theme = "furo"

# Read the Docs and some locked-down environments can block certain CDNs or inline styles.
# Pin MathJax to a widely mirrored CDN, and prefer the TeX-only bundle to avoid MathML
# fallbacks showing up as visible “math italic text” when CSS is restricted.
mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-chtml.min.js"

# Disable the assistive MathML render action (it can become visible if CSS is blocked).
mathjax3_config = {
    "options": {
        # Prefer to disable assistive MathML generation entirely. If it is generated but not hidden
        # (e.g. CSS stripped or theme quirks), it can show up as visible “math italic text” with
        # invisible operator glyphs (⁢, ⁡, …) on some hosted docs.
        "enableAssistiveMml": False,
        "renderActions": {
            # Properly disable assistive MathML output. If it is generated but not hidden (e.g. CSS stripped),
            # it can show up as “math italic text” with invisible operator glyphs (⁢, ⁡, …) on RTD pages.
            "assistiveMml": [0, "", ""],
        }
    }
}
