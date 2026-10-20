"""Copy the figures the tutorial scripts write to examples/output/ into the docs, palette-compressed.

Run every examples/tutorials/NN_*.py first (see docs/tutorials/index.md), then
``python docs/figures/publish_tutorial_figures.py``.
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIGURES = {
    "01_first_run/result.png": "first_run.png",
    "03_sfincs_namelist/fluxes.png": "sfincs_namelist.png",
    "04_vmec_geometry/result.png": "vmec.png",
    "07_species_and_impurity/species.png": "species.png",
    "06_fokker_planck_vs_pas/operators.png": "operators.png",
    "13_convergence/convergence.png": "convergence.png",
    "08_ambipolar_er/ambipolar.png": "ambipolar.png",
    "09_monoenergetic/monoenergetic.png": "monoenergetic.png",
    "10_transport_matrix/transport_matrix.png": "transport_matrix.png",
    "11_phi1/phi1.png": "phi1.png",
    "12_bootstrap_vs_redl/bootstrap.png": "bootstrap.png",
    "14_gradients/gradients.png": "gradients.png",
    "15_optimization/optimization.png": "optimization.png",
}
out = ROOT / "docs" / "_static" / "figures" / "tutorials"
out.mkdir(parents=True, exist_ok=True)
for source, name in FIGURES.items():
    image = Image.open(ROOT / "examples" / "output" / source).convert("RGB")
    image.quantize(64, dither=Image.Dither.NONE).save(out / name, optimize=True)
