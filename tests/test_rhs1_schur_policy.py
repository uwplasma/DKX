from __future__ import annotations

from sfincs_jax.rhs1_schur_policy import canonical_schur_base_kind, resolve_rhs1_schur_base_kind


def test_canonical_schur_base_kind_aliases() -> None:
    assert canonical_schur_base_kind("theta") == "theta_line"
    assert canonical_schur_base_kind("xblock_theta_zeta") == "xblock_tz"
    assert canonical_schur_base_kind("pas_l_tz") == "pas_tz"
    assert canonical_schur_base_kind("tokamak_theta") == "pas_tokamak_theta"
    assert canonical_schur_base_kind("theta_zeta") == "adi"
    assert canonical_schur_base_kind("unknown") is None


def test_resolve_schur_base_prefers_pas_tz_for_geometry4_offender(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SCHUR_BASE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_MIN", raising=False)

    assert (
        resolve_rhs1_schur_base_kind(
            base_kind_env="",
            n_theta=9,
            n_zeta=9,
            n_species=2,
            total_size=11350,
            nxi_for_x=[4, 4, 8, 12, 14],
            has_pas=True,
            has_fp=False,
            has_er_xdot=False,
            has_er_xidot=False,
            use_dkes_exb=False,
            pas_tokamak_theta_applicable=False,
            pas_tz_applicable=True,
            geom_scheme=4,
        )
        == "pas_tz"
    )


def test_resolve_schur_base_small_pas_fallback_uses_pas_schur(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "0")

    assert (
        resolve_rhs1_schur_base_kind(
            base_kind_env="",
            n_theta=5,
            n_zeta=5,
            n_species=2,
            total_size=410,
            nxi_for_x=[4, 4],
            has_pas=True,
            has_fp=False,
            has_er_xdot=False,
            has_er_xidot=False,
            use_dkes_exb=False,
            pas_tokamak_theta_applicable=False,
            pas_tz_applicable=False,
            geom_scheme=4,
        )
        == "pas_schur"
    )


def test_resolve_schur_base_dkes_uses_bounded_xblock_else_pas_ilu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_DKES_XBLOCK_TZ_MAX_BYTES", raising=False)
    kwargs = dict(
        base_kind_env="",
        n_theta=9,
        n_zeta=9,
        n_species=2,
        total_size=5000,
        nxi_for_x=[3, 3],
        has_pas=True,
        has_fp=False,
        has_er_xdot=False,
        has_er_xidot=False,
        use_dkes_exb=True,
        pas_tokamak_theta_applicable=False,
        pas_tz_applicable=True,
        geom_scheme=11,
    )
    assert resolve_rhs1_schur_base_kind(**kwargs) == "xblock_tz"

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DKES_XBLOCK_TZ_MAX_BYTES", "1")
    assert resolve_rhs1_schur_base_kind(**kwargs) == "pas_ilu"


def test_resolve_schur_base_large_pas_er_prefers_xmg(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_XMG_MIN", raising=False)

    assert (
        resolve_rhs1_schur_base_kind(
            base_kind_env="",
            n_theta=9,
            n_zeta=9,
            n_species=2,
            total_size=60000,
            nxi_for_x=[4, 4, 8, 12, 14],
            has_pas=True,
            has_fp=False,
            has_er_xdot=True,
            has_er_xidot=False,
            use_dkes_exb=False,
            pas_tokamak_theta_applicable=False,
            pas_tz_applicable=True,
            geom_scheme=4,
        )
        == "xmg"
    )
