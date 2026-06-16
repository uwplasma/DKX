"""Compatibility entry point for transport parallel workers."""

from __future__ import annotations

import sys as _sys

from .problems.transport_matrix.parallel import worker as _impl

main = _impl.main

if __name__ == "__main__":
    raise SystemExit(main())

_sys.modules[__name__] = _impl
