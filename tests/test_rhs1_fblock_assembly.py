from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.operators.profile_response.kinetic import (
    assemble_partial_rhs1_fblock_operator,
    clear_structured_rhs1_fblock_csr_cache,
    select_structured_rhs1_fblock_csr_operator,
    select_structured_rhs1_fblock_operator,
)
from sfincs_jax.operators.profile_response.fblock import apply_v3_fblock_operator, fblock_operator_from_namelist
from sfincs_jax.operators.profile_response.system import apply_v3_full_system_operator, full_system_operator_from_namelist


def test_partial_fblock_assembly_matches_complete_pas_er_operator(tmp_path: Path) -> None:
    source = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny_scheme12.input.namelist"
    input_path = tmp_path / "pas_er_scheme12.input.namelist"
    input_path.write_text(source.read_text().replace("Er = 0.0d+0", "Er = 0.5d+0"))

    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=1.25)
    assembly = assemble_partial_rhs1_fblock_operator(op)

    assert assembly.is_complete
    assert assembly.unsupported_terms == ()
    assert assembly.included_terms == ("identity_shift", "collisionless", "pas", "exb_theta", "exb_zeta")
    assert assembly.operator.shape == (op.flat_size, op.flat_size)
    assert assembly.term_nnz_blocks["collisionless"] > assembly.term_nnz_blocks["pas"]

    rng = np.random.default_rng(2026060307)
    f = rng.normal(size=op.f_shape).astype(np.float64)
    expected = np.asarray(apply_v3_fblock_operator(op, jnp.asarray(f))).reshape((-1,))
    got = np.asarray(assembly.operator.matvec(jnp.asarray(f.reshape((-1,)))))

    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)


def test_partial_fblock_assembly_matches_complete_pas_full_er_operator(tmp_path: Path) -> None:
    source = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    input_path = tmp_path / "pas_full_er_scheme1.input.namelist"
    input_path.write_text(source.read_text().replace("Er = 0.0d+0", "Er = 0.5d+0"))

    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.5)
    assert op.er_xidot is not None
    assert op.er_xdot is not None
    assembly = assemble_partial_rhs1_fblock_operator(op)

    assert assembly.is_complete
    assert assembly.unsupported_terms == ()
    assert assembly.included_terms == (
        "identity_shift",
        "collisionless",
        "pas",
        "exb_theta",
        "exb_zeta",
        "er_xidot",
        "er_xdot",
    )

    rng = np.random.default_rng(2026060311)
    f = rng.normal(size=op.f_shape).astype(np.float64)
    expected = np.asarray(apply_v3_fblock_operator(op, jnp.asarray(f))).reshape((-1,))
    got = np.asarray(assembly.operator.matvec(jnp.asarray(f.reshape((-1,)))))

    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)


def test_partial_fblock_assembly_matches_complete_fp_operator() -> None:
    input_path = Path(__file__).parent / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.75)
    assert op.fp is not None

    assembly = assemble_partial_rhs1_fblock_operator(op)

    assert assembly.is_complete
    assert assembly.unsupported_terms == ()
    assert assembly.included_terms == ("identity_shift", "collisionless", "fp")

    rng = np.random.default_rng(2026060308)
    f = rng.normal(size=op.f_shape).astype(np.float64)
    expected = np.asarray(apply_v3_fblock_operator(op, jnp.asarray(f))).reshape((-1,))
    got = np.asarray(assembly.operator.matvec(jnp.asarray(f.reshape((-1,)))))

    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)


def test_partial_fblock_assembly_reports_unsupported_fp_phi1_terms() -> None:
    input_path = Path(__file__).parent / "ref" / "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.input.namelist"
    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.75)
    assert op.fp_phi1 is not None

    assembly = assemble_partial_rhs1_fblock_operator(op)

    assert not assembly.is_complete
    assert assembly.unsupported_terms == ("fp_phi1",)
    assert assembly.included_terms == ("identity_shift", "collisionless")
    with pytest.raises(NotImplementedError, match="fp_phi1"):
        assemble_partial_rhs1_fblock_operator(op, strict_complete=True)


def test_partial_fblock_assembly_matches_complete_fp_phi1_operator_with_base_phi1() -> None:
    input_path = Path(__file__).parent / "ref" / "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.input.namelist"
    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.25)
    assert op.fp_phi1 is not None
    theta_phase = np.linspace(0.0, np.pi, op.n_theta, dtype=np.float64)
    zeta_phase = np.linspace(0.0, 2.0 * np.pi, op.n_zeta, endpoint=False, dtype=np.float64)
    phi1_hat_base = 0.05 * np.sin(theta_phase)[:, None] * np.cos(zeta_phase)[None, :]

    assembly = assemble_partial_rhs1_fblock_operator(op, phi1_hat_base=phi1_hat_base)

    assert assembly.is_complete
    assert assembly.unsupported_terms == ()
    assert assembly.included_terms == ("identity_shift", "collisionless", "fp_phi1")

    rng = np.random.default_rng(2026060320)
    f = rng.normal(size=op.f_shape).astype(np.float64)
    expected = np.asarray(
        apply_v3_fblock_operator(op, jnp.asarray(f), phi1_hat_base=jnp.asarray(phi1_hat_base))
    ).reshape((-1,))
    got = np.asarray(assembly.operator.matvec(jnp.asarray(f.reshape((-1,)))))

    np.testing.assert_allclose(got, expected, rtol=2.0e-13, atol=2.0e-13)


def test_partial_fblock_assembly_matches_complete_magnetic_drift_operator() -> None:
    input_path = Path(__file__).parent / "ref" / "magdrift_1species_tiny.input.namelist"
    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.0)
    assert op.pas is not None
    assert op.magdrift_theta is not None
    assert op.magdrift_zeta is not None
    assert op.magdrift_xidot is not None

    assembly = assemble_partial_rhs1_fblock_operator(op)

    assert assembly.is_complete
    assert assembly.unsupported_terms == ()
    assert assembly.included_terms == ("collisionless", "pas", "magdrift_xidot", "magdrift_theta", "magdrift_zeta")

    rng = np.random.default_rng(2026060313)
    f = rng.normal(size=op.f_shape).astype(np.float64)
    expected = np.asarray(apply_v3_fblock_operator(op, jnp.asarray(f))).reshape((-1,))
    got = np.asarray(assembly.operator.matvec(jnp.asarray(f.reshape((-1,)))))

    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)


def test_partial_fblock_assembly_metadata_is_json_friendly(tmp_path: Path) -> None:
    source = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny_scheme12.input.namelist"
    input_path = tmp_path / "pas_er_scheme12.input.namelist"
    input_path.write_text(source.read_text().replace("Er = 0.0d+0", "Er = 0.5d+0"))

    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.0)
    metadata = assemble_partial_rhs1_fblock_operator(op).to_dict()

    assert metadata["is_complete"] is True
    assert metadata["unsupported_terms"] == ()
    assert metadata["included_terms"] == ("collisionless", "pas", "exb_theta", "exb_zeta")
    assert int(metadata["nnz_blocks"]) > 0
    assert int(metadata["data_nbytes"]) > 0


def test_structured_fblock_selector_returns_complete_linear_operator() -> None:
    input_path = Path(__file__).parent / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.5)

    selection = select_structured_rhs1_fblock_operator(op)

    assert selection.selected is True
    assert selection.reason == "complete"
    assert selection.linear_operator is not None
    assert selection.linear_operator.shape == (op.flat_size, op.flat_size)
    assert selection.to_dict()["selected"] is True

    rng = np.random.default_rng(2026060321)
    f = rng.normal(size=op.f_shape).astype(np.float64)
    expected = np.asarray(apply_v3_fblock_operator(op, jnp.asarray(f))).reshape((-1,))
    got = np.asarray(selection.matvec(jnp.asarray(f.reshape((-1,)))))

    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)


def test_structured_fblock_selector_rejects_unsupported_terms_fail_closed() -> None:
    input_path = Path(__file__).parent / "ref" / "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.input.namelist"
    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.0)

    selection = select_structured_rhs1_fblock_operator(op)

    assert selection.selected is False
    assert selection.linear_operator is None
    assert selection.reason == "unsupported_terms:fp_phi1"
    assert selection.to_dict()["operator"] is None
    with pytest.raises(RuntimeError, match="structured f-block operator was not selected"):
        selection.matvec(jnp.zeros((op.flat_size,), dtype=jnp.float64))


def test_structured_fblock_selector_accepts_frozen_phi1_operator() -> None:
    input_path = Path(__file__).parent / "ref" / "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.input.namelist"
    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.0)
    phi1_hat_base = np.full((op.n_theta, op.n_zeta), 0.025, dtype=np.float64)

    selection = select_structured_rhs1_fblock_operator(op, phi1_hat_base=phi1_hat_base)

    assert selection.selected is True
    assert selection.linear_operator is not None
    assert selection.assembly.included_terms == ("collisionless", "fp_phi1")

    rng = np.random.default_rng(2026060322)
    f = rng.normal(size=op.f_shape).astype(np.float64)
    expected = np.asarray(
        apply_v3_fblock_operator(op, jnp.asarray(f), phi1_hat_base=jnp.asarray(phi1_hat_base))
    ).reshape((-1,))
    got = np.asarray(selection.matvec(jnp.asarray(f.reshape((-1,)))))

    np.testing.assert_allclose(got, expected, rtol=2.0e-13, atol=2.0e-13)


def test_structured_fblock_selector_matches_full_system_dke_block_with_zero_sources() -> None:
    input_path = Path(__file__).parent / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    op = full_system_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.0)
    selection = select_structured_rhs1_fblock_operator(op.fblock)

    assert selection.selected is True
    assert selection.linear_operator is not None
    assert op.include_phi1 is False
    assert op.extra_size > 0

    rng = np.random.default_rng(2026060323)
    x_full = np.zeros((op.total_size,), dtype=np.float64)
    x_full[: op.f_size] = rng.normal(size=op.f_size).astype(np.float64)

    y_full = np.asarray(apply_v3_full_system_operator(op, jnp.asarray(x_full)))
    got = np.asarray(selection.matvec(jnp.asarray(x_full[: op.f_size])))

    np.testing.assert_allclose(got, y_full[: op.f_size], rtol=1.0e-13, atol=1.0e-13)


def test_structured_fblock_csr_selector_matches_block_matvec_and_reuses_cache() -> None:
    input_path = Path(__file__).parent / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.5)
    clear_structured_rhs1_fblock_csr_cache()

    first = select_structured_rhs1_fblock_csr_operator(op)
    second = select_structured_rhs1_fblock_csr_operator(op)

    assert first.selected is True
    assert first.cache_hit is False
    assert second.selected is True
    assert second.cache_hit is True
    assert second.metadata["object_cache_hit"] is True
    assert second.matrix is first.matrix
    assert int(first.metadata["nnz"]) > 0
    assert int(first.metadata["csr_nbytes_actual"]) <= int(first.metadata["csr_nbytes_estimate"])

    rng = np.random.default_rng(2026060324)
    f = rng.normal(size=op.f_shape).astype(np.float64)
    expected = np.asarray(apply_v3_fblock_operator(op, jnp.asarray(f))).reshape((-1,))
    got = first.matvec(f.reshape((-1,)))

    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)


def test_structured_fblock_csr_selector_rejects_memory_budget_fail_closed() -> None:
    input_path = Path(__file__).parent / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.5)

    selected = select_structured_rhs1_fblock_csr_operator(op, max_csr_nbytes=1)

    assert selected.selected is False
    assert selected.matrix is None
    assert selected.reason.startswith("csr_budget_exceeded:")
    with pytest.raises(RuntimeError, match="structured f-block CSR operator was not selected"):
        selected.matvec(np.zeros((op.flat_size,), dtype=np.float64))


def test_structured_fblock_csr_cache_key_tracks_physics_coefficients() -> None:
    input_path = Path(__file__).parent / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    op_a = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.5)
    op_b = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.75)
    clear_structured_rhs1_fblock_csr_cache()

    first = select_structured_rhs1_fblock_csr_operator(op_a)
    changed = select_structured_rhs1_fblock_csr_operator(op_b)

    assert first.selected is True
    assert changed.selected is True
    assert first.cache_hit is False
    assert changed.cache_hit is False
    assert changed.matrix is not first.matrix
