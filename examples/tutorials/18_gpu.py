"""Tutorial 18 -- running on a GPU: pick the device, compare, and read the timing.

DKX runs wherever JAX does.  ``jax.devices()`` lists what this process can
use, and ``run.device`` in a case (or ``device=`` in the solver API) places
the solve: ``"auto"`` keeps JAX's default backend, ``"cpu"`` and ``"gpu"``
name a platform.  This script solves the same stellarator case on every
platform it finds, twice each: the first call pays the XLA compilation, the
second is the steady-state cost of one more case of the same shape.

Expect the GPU to lose on a case this small.  Compilation dominates short
runs, and the direct factorization behind ``method = "auto"`` is launch-bound
on a GPU; the GPU pays when many solves share one compilation (an E_r or
surface scan) or the operator is large.  Without a GPU only the CPU row
prints.  GPU memory is not preallocated by default in the CLI; in Python,
export ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` before starting when several
processes share a card.

Run:              python examples/tutorials/18_gpu.py
Runtime:          about 29 s on a loaded 36-core CPU node, compilation included; a GPU adds its own compile.
"""

# 1. Imports
import lzma
import shutil
import time
from dataclasses import replace
from pathlib import Path

import jax

import dkx

# 2. Input parameters
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
CASE_FILE = Path(__file__).with_name("04_vmec_geometry.toml")  # the VMEC case of tutorial 04
RESOLUTION = {"theta": 9, "zeta": 9, "pitch": 8, "speed": 4}  # coarse grid for a quick comparison
REPEATS = 2  # solves per platform; the first includes compilation

# 3. Run
platforms = sorted({device.platform for device in jax.devices()})
print(f"JAX {jax.__version__}, devices: {[str(device) for device in jax.devices()]}", flush=True)
case = dkx.Case.from_file(CASE_FILE)
wout = case.geometry_path
if not wout.exists():  # shipped as .nc.xz: decompress once, as in tutorial 04
    with lzma.open(wout.with_name(wout.name + ".xz")) as src, open(wout, "wb") as dst:
        shutil.copyfileobj(src, dst)
case = replace(case, resolution=replace(case.resolution, **RESOLUTION),
               run=replace(case.run, progress=False))
seconds, results = {}, {}
for platform in platforms:
    on_device = replace(case, run=replace(case.run, device=platform))
    seconds[platform] = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        results[platform] = dkx.run(on_device)
        seconds[platform].append(time.perf_counter() - start)
    print(f"{platform}: {seconds[platform][0]:.1f} s first call, {seconds[platform][-1]:.1f} s repeat", flush=True)

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
print("\n=== Summary ===")
reference = results[platforms[0]].arrays["particle_flux_m2_s"]
for platform in platforms:
    flux = results[platform].arrays["particle_flux_m2_s"]
    print(f"  {platform:4s} first {seconds[platform][0]:6.1f} s  repeat {seconds[platform][-1]:6.1f} s"
          f"  max |flux - {platforms[0]} flux| / |flux| = {abs(flux - reference).max() / abs(reference).max():.1e}")
results[platforms[0]].save(OUT / "result.nc")
print(f"  saved {OUT / 'result.nc'}")
