"""mfh_fenicsx with dolfinx: the stress operator inside a Newton solve.

A homogeneous strain imposed on the whole boundary gives a homogeneous field,
whose stress is checked against Julia's single-point update (and against
Hooke's law for a linear law); a heterogeneous plastic problem checks that the
tangent is the consistent one by the quadratic convergence of Newton's method.
"""

import numpy as np
import pytest

dolfinx = pytest.importorskip("dolfinx")
from dolfinx import fem, mesh  # noqa: E402
from mpi4py import MPI  # noqa: E402
import ufl  # noqa: E402

import mfh_fenicsx as mf  # noqa: E402
from mfh_fenicsx.fem import Newton, Stress  # noqa: E402

J2 = "J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀ = 250.0, H = 2.0e3)"


@pytest.fixture(scope="module")
def jl():
    return mf.start()


def _domain(d, n):
    if d == 2:
        return mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    return mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)


def _homogeneous(material, d, E, steps, dt=1.0):
    """Impose u = t·E·x on the boundary for t in `steps`; the stress in Mandel at the end."""
    domain = _domain(d, 2)
    V = fem.functionspace(domain, ("Lagrange", 1, (d,)))
    u, v = fem.Function(V), ufl.TestFunction(V)
    stress = Stress(material, u, dt=dt)
    residual = ufl.inner(stress.sigma, mf.fem.strain(v)) * stress.dx
    domain.topology.create_connectivity(d - 1, d)
    dofs = fem.locate_dofs_topological(V, d - 1, mesh.exterior_facet_indices(domain.topology))
    g = fem.Function(V)
    newton = Newton(stress, residual, u, [fem.dirichletbc(g, dofs)])
    iterations = []
    for t in steps:
        g.interpolate(lambda x, t=t: t * (E @ x[:d]))
        iterations.append(newton.solve())
        stress.commit()
    return stress, u, iterations


def test_linear_law_three_dimensions(jl):
    E = 1.0e-3 * np.array([[1.0, 0.2, -0.1], [0.2, -0.4, 0.3], [-0.1, 0.3, 0.5]])
    stress, u, iterations = _homogeneous("LinearElasticity(; E = 70.0e3, ν = 0.3)", 3, E, [1.0])
    # The affine field is the exact solution, reached in one iteration.
    assert iterations == [1]
    x = u.function_space.tabulate_dof_coordinates()
    np.testing.assert_allclose(u.x.array.reshape(-1, 3), x @ E.T, atol=1e-15)
    lam, mu = 70.0e3 * 0.3 / (1.3 * 0.4), 70.0e3 / 2.6
    hooke = mf.mandel_from_tensor(lam * np.trace(E) * np.eye(3) + 2 * mu * E)
    np.testing.assert_allclose(stress.material.stress, np.repeat(hooke, stress.material.n, axis=1), rtol=1e-10)


def test_plastic_plane_strain_against_point_update(jl):
    E = 2.0e-3 * np.array([[1.0, 0.3], [0.3, -0.6]])
    steps = [0.5, 1.0, 1.5]
    stress, _, _ = _homogeneous(J2, 2, E, steps)
    # The same strain path at one point, through Julia's single-point update.
    path = jl.seval(
        """(law, eps, steps) -> begin
            st = initial_state(law)
            F = nothing
            for t in steps
                F, _, st = point_update(law, MeanFieldHomogenization.MaterialLaws.SVector{6}(t .* eps), st, 1.0)
            end
            collect(F)
        end"""
    )
    E3 = np.zeros((3, 3))
    E3[:2, :2] = E
    expected = np.asarray(path(jl.seval(J2), mf.mandel_from_tensor(E3)[:, 0], steps))
    np.testing.assert_allclose(stress.material.stress, np.repeat(expected[:, None], stress.material.n, axis=1), rtol=1e-9, atol=1e-6)
    # Plastic, with an out-of-plane stress from the plane-strain constraint.
    assert abs(expected[2]) > 1.0


def test_consistent_tangent_converges_quadratically(jl):
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    u = fem.Function(V)
    stress = Stress(J2, u)
    residual = ufl.inner(stress.sigma, mf.fem.strain(ufl.TestFunction(V))) * stress.dx
    bottom = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[2], 0.0))
    top_facets = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[2], 1.0))
    top = fem.locate_dofs_topological(V.sub(2), 2, top_facets)
    pull = fem.Constant(domain, 0.0)
    bcs = [fem.dirichletbc(np.zeros(3), bottom, V), fem.dirichletbc(pull, top, V.sub(2))]
    newton = Newton(stress, residual, u, bcs, rtol=1e-12, atol=1e-8)
    for value in (2.0e-3, 4.0e-3):
        pull.value = value
        newton.solve()
        stress.commit()
    h = newton.history
    # Plastic flow everywhere, the field heterogeneous (clamped bottom).
    assert len(h) >= 4
    # The order of convergence log(r₊/r)/log(r/r₋), read on the iterations above
    # the round-off floor: about 2 with the consistent tangent, 1 with any other.
    orders = [np.log(h[k + 1] / h[k]) / np.log(h[k] / h[k - 1]) for k in range(2, len(h) - 1) if h[k + 1] > 1e-10 * h[0]]
    assert len(orders) >= 2 and min(orders) > 1.5
