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
    "myst_parser",
]

# Narrative pages are MyST Markdown; math uses $...$ and $$...$$.
myst_enable_extensions = ["amsmath", "colon_fence", "dollarmath", "deflist"]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "figures/**"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "DKX"

# Furo is the only theme: the docs extra installs it, and a missing theme
# should fail the build rather than silently fall back.
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
