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
    return stress, u, iterations, newton


def test_linear_law_three_dimensions(jl):
    E = 1.0e-3 * np.array([[1.0, 0.2, -0.1], [0.2, -0.4, 0.3], [-0.1, 0.3, 0.5]])
    stress, u, iterations, newton = _homogeneous("LinearElasticity(; E = 70.0e3, ν = 0.3)", 3, E, [1.0])
    # The affine field is the exact solution, reached in one iteration.
    assert iterations == [1]
    x = u.function_space.tabulate_dof_coordinates()
    np.testing.assert_allclose(u.x.array.reshape(-1, 3), x @ E.T, atol=1e-15)
    lam, mu = 70.0e3 * 0.3 / (1.3 * 0.4), 70.0e3 / 2.6
    hooke = mf.mandel_from_tensor(lam * np.trace(E) * np.eye(3) + 2 * mu * E)
    np.testing.assert_allclose(stress.material.stress, np.repeat(hooke, stress.material.n, axis=1), rtol=1e-10)
    # The reaction on the top face, z = 1, of unit area: σ_zz, and σ_xz along x.
    domain = u.function_space.mesh
    top = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[2], 1.0))
    for component, expected in ((2, hooke[2, 0]), (0, hooke[4, 0] / np.sqrt(2.0))):
        dofs = fem.locate_dofs_topological(u.function_space.sub(component), 2, top)
        assert newton.reaction(dofs) == pytest.approx(expected, rel=1e-10)


def test_plastic_plane_strain_against_point_update(jl):
    E = 2.0e-3 * np.array([[1.0, 0.3], [0.3, -0.6]])
    steps = [0.5, 1.0, 1.5]
    stress, _, _, _ = _homogeneous(J2, 2, E, steps)
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
    iterations = []
    for value in (2.0e-3, 4.0e-3):
        pull.value = value
        iterations.append(newton.solve())
        stress.commit()
    h = newton.history
    # Plastic flow everywhere, the field heterogeneous (clamped bottom).
    assert len(h) >= 4
    # A few iterations per step: an elastic tangent does not converge in 25.
    assert max(iterations) <= 7
    # The order of convergence log(r₊/r)/log(r/r₋), read on the iterations above
    # the round-off floor. The first ones are not yet asymptotic (1.4 to 1.7
    # depending on the machine); the last one is, about 2 with the consistent
    # tangent and 1 with any other.
    orders = [np.log(h[k + 1] / h[k]) / np.log(h[k] / h[k - 1]) for k in range(2, len(h) - 1) if h[k + 1] > 1e-10 * h[0]]
    assert len(orders) >= 1 and max(orders) > 1.6


def _clamped_pull(n=2):
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    u = fem.Function(V)
    stress = Stress(J2, u)
    residual = ufl.inner(stress.sigma, mf.fem.strain(ufl.TestFunction(V))) * stress.dx
    bottom = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[2], 0.0))
    top = fem.locate_dofs_topological(V.sub(2), 2, mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[2], 1.0)))
    pull = fem.Constant(domain, 0.0)
    bcs = [fem.dirichletbc(np.zeros(3), bottom, V), fem.dirichletbc(pull, top, V.sub(2))]
    return stress, u, pull, bcs, residual


def test_quasistatic_cuts_the_steps_that_fail(jl):
    from juliacall import JuliaError

    from mfh_fenicsx.fem import quasistatic

    def prescribe(pull):
        def set_value(t):
            pull.value = 4.0e-3 * t

        return set_value

    # Four Newton iterations are not enough for the whole step: it is cut.
    stress, u, pull, bcs, residual = _clamped_pull()
    newton = Newton(stress, residual, u, bcs, rtol=1e-10, atol=1e-8, maxiter=4)
    seen = []
    accepted = quasistatic(newton, [0.0, 1.0], prescribe(pull), on_step=seen.append)
    assert len(accepted) > 2 and accepted[-1] == 1.0 and seen == accepted[1:]
    # The same steps taken directly, without a cut, give the same solution.
    stress2, u2, pull2, bcs2, residual2 = _clamped_pull()
    direct = Newton(stress2, residual2, u2, bcs2, rtol=1e-10, atol=1e-8)
    for t0, t1 in zip(accepted[:-1], accepted[1:]):
        stress2.dt = t1 - t0
        pull2.value = 4.0e-3 * t1
        direct.solve()
        stress2.commit()
    np.testing.assert_allclose(u.x.array, u2.x.array, rtol=0, atol=1e-12)
    # A point that never integrates: cut up to max_cuts, then the error.
    stress, u, pull, bcs, residual = _clamped_pull()
    stress.options = {"algorithm": jl.LocalNewton(maxiters=1)}
    with pytest.raises(JuliaError, match="LocalIntegrationError"):
        quasistatic(Newton(stress, residual, u, bcs), [0.0, 1.0], prescribe(pull), max_cuts=2)
    # Any other Julia error is raised at once.
    stress, u, pull, bcs, residual = _clamped_pull()
    stress.options = {"no_such_option": 1}
    with pytest.raises(JuliaError, match="no_such_option"):
        quasistatic(Newton(stress, residual, u, bcs), [0.0, 1.0], prescribe(pull))


def test_petsc_options_reach_this_solver_only(jl):
    from petsc4py import PETSc

    stress, u, pull, bcs, residual = _clamped_pull()
    chosen = Newton(stress, residual, u, bcs, petsc_options={"pc_factor_mat_solver_type": "petsc"})
    default = Newton(stress, residual, u, bcs)
    assert chosen.ksp.getPC().getFactorSolverType() == "petsc"
    expected = "mumps" if PETSc.Sys.hasExternalPackage("mumps") else default.ksp.getPC().getFactorSolverType()
    assert default.ksp.getPC().getFactorSolverType() == expected
    # Both solve the same step to the same displacements.
    pull.value = 2.0e-3
    chosen.solve()
    first = u.x.array.copy()
    stress.rollback()
    u.x.array[:] = 0.0
    default.solve()
    np.testing.assert_allclose(u.x.array, first, rtol=0, atol=1e-12)
