from __future__ import annotations

from sfincs_jax.io import _should_precompile_v3_full_system


def test_should_precompile_v3_full_system_is_opt_in() -> None:
    assert not _should_precompile_v3_full_system(env_value="")
    assert not _should_precompile_v3_full_system(env_value="auto")
    assert not _should_precompile_v3_full_system(env_value="0")
    assert not _should_precompile_v3_full_system(env_value="off")
    assert _should_precompile_v3_full_system(env_value="1")
    assert _should_precompile_v3_full_system(env_value="true")

