"""Tutorial 02 -- the same kind of run through the command line and a case file.

Everything tutorial 01 wrote as a Python dict can live in a TOML case file,
``02_cli_case.toml`` beside this script, and run without writing Python:

    dkx validate examples/tutorials/02_cli_case.toml      # schema check + case ID
    dkx run      examples/tutorials/02_cli_case.toml --out result.nc
    dkx inspect  result.nc                                 # what the file holds
    dkx plot     result.nc                                 # quick-look figure

``dkx template`` prints a fully commented case file to start your own from.
The case ID is a hash of the physics content, so two files (or a file and a
Python dict) with the same ID describe the same calculation.  This script runs
those commands for you, streaming their progress, and then reads the NetCDF
result back with ``dkx.Result.load`` to show that CLI and Python results are
the same object.  The shipped resolution is deliberately small: see the
comments in the TOML file and tutorial 13.

Run:              python examples/tutorials/02_cli_case.py
Smoke mode:       DKX_EXAMPLES_CI=1 (the case is already seconds-sized)
Expected runtime: see the table in examples/README.md.
"""

# 1. Imports
import subprocess
import sys
from pathlib import Path

import numpy as np

import dkx

# 2. Input parameters
CASE_FILE = Path(__file__).with_suffix(".toml")
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
RESULT = OUT / "result.nc"

# 3. Run -- exactly the shell commands from the docstring
OUT.mkdir(parents=True, exist_ok=True)
for command in (["validate", CASE_FILE], ["run", CASE_FILE, "--out", RESULT], ["inspect", RESULT]):
    print(f"\n$ dkx {' '.join(str(part) for part in command)}", flush=True)
    subprocess.run([sys.executable, "-m", "dkx", *map(str, command)], check=True)

# 4. Plot, save and print
result = dkx.Result.load(RESULT)
print(f"\nsaved {result.plot(OUT / 'result.png')}")
print("\n=== Summary ===")
print(f"  case ID from the file: {dkx.Case.from_file(CASE_FILE).case_id[:12]}"
      f"  stored in the result: {result.case_id[:12]}")
flux = np.asarray(result.arrays["particle_flux_m2_s"])[:, 0]
print(f"  particle flux on {flux.size} surfaces: {np.array2string(flux, precision=3)} m^-2 s^-1")
