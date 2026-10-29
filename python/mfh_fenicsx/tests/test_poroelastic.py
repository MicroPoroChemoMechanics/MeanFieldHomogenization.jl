"""The (u, p) coupling: a poroelastic material of MeanFieldHomogenization in a
mixed dolfinx problem.

Terzaghi's consolidation of a column, oedometric, loaded at once and drained at
its top, with the linear Biot law: the pressure follows the closed form of the
one-dimensional diffusion (a Fourier series) from the undrained pressure of the
first instant. The coupled tangent (the blocks σε, σp, φε, φp) is checked by
the convergence of Newton's method in one iteration on this linear problem.
"""

import numpy as np
import pytest

dolfinx = pytest.importorskip("dolfinx")
import basix.ufl  # noqa: E402
from dolfinx import fem, mesh  # noqa: E402
from mpi4py import MPI  # noqa: E402
import ufl  # noqa: E402

import mfh_fenicsx as mf  # noqa: E402
from mfh_fenicsx.fem import Newton, Poroelastic, strain  # noqa: E402

E, NU, B, M = 10.0e3, 0.25, 0.8, 5.0e3          # MPa
MOBILITY = 1.0e-3                                 # k/μ, mm²/(MPa·s)
H, Q = 10.0, 1.0                                  # mm, MPa


@pytest.fixture(scope="module")
def jl():
    return mf.start()


def _terzaghi(z, t, p0, c, terms=200):
    """Pressure of Terzaghi's column at height z (0 at the impervious bottom, H at the drained top)."""
    m = np.arange(terms)[:, None]
    k = (2 * m + 1) * np.pi / (2 * H)
    return p0 * np.sum(4 * (-1.0) ** m / ((2 * m + 1) * np.pi) * np.cos(k * z) * np.exp(-(k**2) * c * t), axis=0)


def _consolidate(steps):
    """The column consolidated in `steps` time steps: the largest gap of the
    pressure to the closed form, relative to the undrained pressure, and the
    largest number of Newton iterations of a step."""
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([1.0, H])], [1, 40])
    cell = domain.basix_cell()
    W = fem.functionspace(domain, basix.ufl.mixed_element([basix.ufl.element("Lagrange", cell, 2, shape=(2,)), basix.ufl.element("Lagrange", cell, 1)]))
    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    poro = Poroelastic(f"BiotPoroelasticity(; E = {E}, ν = {NU}, b = {B}, M = {M})", u, p, quadrature_degree=2)
    top = mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[1], H))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mesh.meshtags(domain, 1, top, np.ones_like(top)))
    load = fem.Constant(domain, np.array([0.0, -Q]))
    dt = fem.Constant(domain, 1.0)
    residual = (
        ufl.inner(poro.sigma, strain(v)) * poro.dx - ufl.dot(load, v) * ds(1)
        + (poro.phi - poro.phi_old) * q * poro.dx + dt * MOBILITY * ufl.dot(ufl.grad(p), ufl.grad(q)) * poro.dx
    )
    # Oedometric: no lateral displacement, the bottom fixed; drained at the top only.
    Vu, Vp = W.sub(0), W.sub(1)
    sides = fem.locate_dofs_topological(Vu.sub(0), 1, mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)))
    bottom = fem.locate_dofs_topological(Vu.sub(1), 1, mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[1], 0.0)))
    drained = fem.locate_dofs_topological(Vp, 1, top)
    bcs = [fem.dirichletbc(0.0, sides, Vu.sub(0)), fem.dirichletbc(0.0, bottom, Vu.sub(1)), fem.dirichletbc(0.0, drained, Vp)]
    newton = Newton(poro, residual, w, bcs, rtol=1e-11, atol=1e-11)

    K_v = E * (1 - NU) / ((1 + NU) * (1 - 2 * NU))
    S = 1 / M + B**2 / K_v
    p0, c = B * Q / (K_v * S), MOBILITY / S
    # The pressure at the nodes of the pressure space, against height.
    Vp_c, dofs = W.sub(1).collapse()
    z = Vp_c.tabulate_dof_coordinates()[:, 1]
    below = z < H - 1e-9
    times = np.concatenate([[0.0], np.geomspace(1.0e-3, 2.0, steps) * H**2 / c])
    gap, iterations = 0.0, 0
    for t0, t1 in zip(times[:-1], times[1:]):
        dt.value = t1 - t0
        poro.dt = t1 - t0
        iterations = max(iterations, newton.solve())
        poro.commit()
        if t1 > 0.01 * H**2 / c:
            gap = max(gap, np.max(np.abs(w.x.array[dofs][below] - _terzaghi(z[below], t1, p0, c))) / p0)
    return gap, iterations


def test_terzaghi_consolidation(jl):
    coarse, iterations = _consolidate(60)
    fine, _ = _consolidate(120)
    # A linear problem: one Newton iteration per step, with the coupled tangent.
    assert iterations <= 1
    # Backward Euler: the gap to the closed form, a few percent of the undrained
    # pressure, halves with the time step (first order).
    assert coarse < 5.0e-2 and fine < 0.6 * coarse
