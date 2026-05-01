from __future__ import annotations

from sfincs_jax.profiling import _resource_maxrss_to_mb


def test_resource_maxrss_to_mb_handles_darwin_bytes() -> None:
    assert _resource_maxrss_to_mb(1024.0 * 1024.0, platform="darwin") == 1.0


def test_resource_maxrss_to_mb_handles_linux_kib() -> None:
    assert _resource_maxrss_to_mb(1024.0, platform="linux") == 1.0
