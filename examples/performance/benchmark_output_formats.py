"""Benchmark dkx output writers and readers without running a solve.

This script uses an existing output fixture, then writes equivalent HDF5,
NetCDF4, and NPZ files. It isolates serialization cost from JAX compile/solve
cost, which is useful when choosing an output format for parameter sweeps.

Run from the repository root:

  python examples/performance/benchmark_output_formats.py --repeats 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from dkx.io import read_sfincs_h5, read_sfincs_output_file, write_sfincs_output_file


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _time_once(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5",
        type=Path,
        default=_REPO_ROOT / "tests" / "ref" / "output_scheme4_2species_quick.sfincsOutput.h5",
        help="Existing HDF5 output used as the benchmark payload.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Write/read repetitions per format.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for benchmark output files.")
    parser.add_argument("--json", type=Path, default=None, help="Optional path for machine-readable results.")
    args = parser.parse_args(argv)

    payload = read_sfincs_h5(args.input_h5)
    formats = (".h5", ".nc", ".npz")
    out_dir_cm = tempfile.TemporaryDirectory() if args.out_dir is None else None
    out_dir = Path(out_dir_cm.name) if out_dir_cm is not None else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, float | int | str]] = {}
    for suffix in formats:
        path = out_dir / f"sfincsOutput_benchmark{suffix}"
        write_times = []
        read_times = []
        for _ in range(max(1, args.repeats)):
            write_times.append(
                _time_once(lambda path=path: write_sfincs_output_file(path=path, data=payload, fortran_layout=False))
            )
            read_times.append(_time_once(lambda path=path: read_sfincs_output_file(path)))
        summary[suffix] = {
            "path": str(path),
            "datasets": len(payload),
            "bytes": path.stat().st_size,
            "write_median_s": statistics.median(write_times),
            "read_median_s": statistics.median(read_times),
        }

    for suffix, row in summary.items():
        print(
            f"{suffix:>4} datasets={row['datasets']} size={row['bytes']/1e6:.3f} MB "
            f"write_median={row['write_median_s']:.4f}s read_median={row['read_median_s']:.4f}s"
        )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if out_dir_cm is not None:
        out_dir_cm.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
