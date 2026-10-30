"""mfh_fenicsx without dolfinx: Mandel conversions and the batched material.

Every quantity is checked against an independent route: the conversions against
the Mandel definition written out component by component, the batched update
against Julia's single-point `point_update`, a linear law against Hooke's law in
full tensor form.
"""

import numpy as np
import pytest

import mfh_fenicsx as mf

RNG = np.random.default_rng(20261029)
J2_LAW = "J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀ = 250.0, H = 2.0e3)"


@pytest.fixture(scope="module")
def jl():
    return mf.start()


def _sym(n):
    a = RNG.standard_normal((n, 3, 3))
    return a + a.transpose(0, 2, 1)


def test_mandel_definition_and_round_trip():
    t = _sym(5)
    v = mf.mandel_from_tensor(t)
    assert v.shape == (6, 5) and v.flags.f_contiguous
    r2 = np.sqrt(2.0)
    expected = np.stack([t[:, 0, 0], t[:, 1, 1], t[:, 2, 2], r2 * t[:, 1, 2], r2 * t[:, 0, 2], r2 * t[:, 0, 1]])
    np.testing.assert_array_equal(v, expected)
    np.testing.assert_allclose(mf.tensor_from_mandel(v), t, rtol=0, atol=1e-15)
    # The Mandel inner product is the double contraction of the tensors.
    s = _sym(5)
    np.testing.assert_allclose(np.einsum("an,an->n", v, mf.mandel_from_tensor(s)), np.einsum("nij,nij->n", t, s))


def test_tensor4_from_mandel_acts_as_the_matrix():
    k = RNG.standard_normal((6, 6, 4))
    full = mf.tensor4_from_mandel(k)
    assert full.shape == (4, 3, 3, 3, 3)
    # Minor symmetries, and K : ε in full form equals the Mandel product.
    np.testing.assert_array_equal(full, full.transpose(0, 2, 1, 3, 4))
    np.testing.assert_array_equal(full, full.transpose(0, 1, 2, 4, 3))
    e = _sym(4)
    lhs = mf.mandel_from_tensor(np.einsum("nijkl,nkl->nij", full, e))
    rhs = np.einsum("abn,bn->an", k, mf.mandel_from_tensor(e))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-13, atol=1e-13)


def test_linear_law_is_hookes_law(jl):
    E, nu = 70.0e3, 0.3
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    mat = mf.Material(f"LinearElasticity(; E = {E}, ν = {nu})", 3, 2)
    e = 1.0e-3 * _sym(6)
    sig, k = mat.update(mf.mandel_from_tensor(e), 1.0)
    hooke = lam * np.einsum("nii->n", e)[:, None, None] * np.eye(3) + 2 * mu * e
    np.testing.assert_allclose(mf.tensor_from_mandel(sig), hooke, rtol=1e-12, atol=1e-9)
    iso = lam * np.einsum("ij,kl->ijkl", np.eye(3), np.eye(3)) + mu * (
        np.einsum("ik,jl->ijkl", np.eye(3), np.eye(3)) + np.einsum("il,jk->ijkl", np.eye(3), np.eye(3))
    )
    np.testing.assert_allclose(mf.tensor4_from_mandel(k), np.broadcast_to(iso, (6, 3, 3, 3, 3)), rtol=1e-12, atol=1e-8)


def test_batch_matches_point_update_and_steps(jl):
    mat = mf.Material(J2_LAW, 4, 3)
    base = np.array([1.0, -0.3, -0.3, 0.0, 0.0, 0.0])
    eps = 3.0e-3 * (base[:, None] + 0.4 * RNG.standard_normal((6, mat.n)))
    sig, k = mat.update(eps, 1.0)
    assert mat.calls == 1 and mat.julia_time > 0
    single = jl.seval(
        "(law, e) -> begin F, K, _ = point_update(law, MeanFieldHomogenization.MaterialLaws.SVector{6}(e), initial_state(law), 1.0); "
        "(collect(F), collect(K)) end"
    )
    for p in (0, 5, mat.n - 1):
        F, K = single(mat.material, eps[:, p])
        np.testing.assert_array_equal(sig[:, p], np.asarray(F))
        np.testing.assert_array_equal(k[:, :, p], np.asarray(K))
    # Rolled back, the step gives the same stresses; committed, the next step
    # starts from the plastic state, so that unloading to zero leaves a residual stress.
    first = sig.copy()
    mat.rollback()
    np.testing.assert_array_equal(mat.update(eps, 1.0)[0], first)
    mat.commit()
    residual = mat.update(np.zeros((6, mat.n)), 1.0)[0]
    assert np.max(np.abs(residual)) > 1.0
    # The committed internal variables, against the single-point update.
    names = jl.seval("law -> String.(collect(keys(law.layout)))")(mat.material)
    layout = jl.seval("(law, s) -> collect(law.layout[Symbol(s)])")
    yfirst = jl.seval(
        "(law, e) -> collect(point_update(law, MeanFieldHomogenization.MaterialLaws.SVector{6}(e), initial_state(law), 1.0)[3].y)"
    )(mat.material, eps[:, 5])
    for name in names:
        rows = np.asarray(layout(mat.material, name)) - 1
        np.testing.assert_array_equal(mat.internal(name)[:, 5], np.asarray(yfirst)[rows])


def test_refusals(jl):
    from juliacall import JuliaError

    mat = mf.Material(J2_LAW, 1, 2)
    with pytest.raises(ValueError, match="expected"):
        mat.update(np.zeros((6, 3)), 1.0)
    with pytest.raises(JuliaError, match="LocalIntegrationError"):
        e = np.zeros((6, 2))
        e[:, 1] = (0.05, -0.02, -0.02, 0.0, 0.0, 0.0)
        mat.update(e, 1.0, algorithm=jl.LocalNewton(maxiters=1), ntasks=2)


def test_start_twice_returns_the_running_julia(jl):
    assert mf.start() is jl


def test_julia_signal_handlers_are_restored(jl):
    import ctypes
    import signal

    libc = ctypes.CDLL(None)

    def handler(number):
        action = mf._SigAction()
        libc.sigaction(number, None, ctypes.byref(action))
        return action.handler

    julias = {n: handler(n) for n in mf._SIGNALS}
    assert all(julias.values())
    # What gmsh.initialize() does: every handler back to its default.
    for number in mf._SIGNALS:
        libc.signal(number, ctypes.c_void_p(0))
    assert not any(handler(n) for n in mf._SIGNALS)
    mat = mf.Material(J2_LAW, 2, 3)
    mat.update(1.0e-3 * np.ones((6, mat.n)), 1.0)
    assert {n: handler(n) for n in mf._SIGNALS} == julias
    # Collections on all the threads go through the safepoints again.
    for _ in range(3):
        mf.julia().GC.gc()
