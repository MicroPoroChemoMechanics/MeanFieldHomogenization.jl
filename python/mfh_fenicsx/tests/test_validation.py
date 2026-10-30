"""mfh_fenicsx against independent references.

- The same plastic problem on the same hexahedral mesh, with the same
  quadrature, through MeanFieldHomogenization's Ferrite adapter: the nodal
  displacements agree to the tolerance of the Newton solves.
- A homogenized elastic composite against a dolfinx elasticity solve written in
  UFL with the homogenized stiffness, no Julia call in the loop.
- The thick cylinder under internal pressure in plane strain against Lamé's
  closed form.
"""

import numpy as np
import pytest

dolfinx = pytest.importorskip("dolfinx")
from dolfinx import fem, mesh  # noqa: E402
from dolfinx.fem.petsc import LinearProblem  # noqa: E402
from mpi4py import MPI  # noqa: E402
import ufl  # noqa: E402

import mfh_fenicsx as mf  # noqa: E402
from mfh_fenicsx.fem import Newton, Stress, strain  # noqa: E402

J2 = "J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀ = 250.0, H = 2.0e3)"


@pytest.fixture(scope="module")
def jl():
    return mf.start()


def _ferrite_pull(jl, n, pulls):
    """Nodal coordinates and displacements of the Ferrite solve: a unit cube of
    n³ hexahedra, clamped at z = 0, its top pulled to u_z = t for t in pulls."""
    from juliacall import JuliaError

    try:
        jl.seval("using Ferrite")
    except JuliaError:
        pytest.skip("Ferrite is not in the Julia project")
    run = jl.seval(
        """(law, n, pulls) -> begin
            grid = Ferrite.generate_grid(Ferrite.Hexahedron, (n, n, n), Ferrite.Vec(0.0, 0.0, 0.0), Ferrite.Vec(1.0, 1.0, 1.0))
            ip = Ferrite.Lagrange{Ferrite.RefHexahedron, 1}()^3
            dh = Ferrite.DofHandler(grid); Ferrite.add!(dh, :u, ip); Ferrite.close!(dh)
            ch = Ferrite.ConstraintHandler(dh)
            Ferrite.add!(ch, Ferrite.Dirichlet(:u, Ferrite.getfacetset(grid, "bottom"), (x, t) -> zeros(3), [1, 2, 3]))
            Ferrite.add!(ch, Ferrite.Dirichlet(:u, Ferrite.getfacetset(grid, "top"), (x, t) -> t, [3]))
            Ferrite.close!(ch)
            cv = Ferrite.CellValues(Ferrite.QuadratureRule{Ferrite.RefHexahedron}(2), ip)
            states = QuadratureStates(law, Ferrite.getncells(grid), Ferrite.getnquadpoints(cv))
            system!, prescribe! = material_system(dh, ch, cv, law, ThreeD(), states)
            u = zeros(Ferrite.ndofs(dh))
            quasistatic!(system!, prescribe!, u, states, vcat(0.0, pulls); rtol = 1.0e-13)
            nodal = Ferrite.evaluate_at_grid_nodes(dh, u, :u)
            ([node.x[i] for node in grid.nodes, i in 1:3], [v[i] for v in nodal, i in 1:3])
        end"""
    )
    x, u = run(jl.seval(J2), n, np.asarray(pulls, dtype=float))
    return np.asarray(x), np.asarray(u)


def test_against_the_ferrite_adapter(jl):
    n, pulls = 3, [2.0e-3, 4.0e-3]
    x_ref, u_ref = _ferrite_pull(jl, n, pulls)
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, mesh.CellType.hexahedron)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    u = fem.Function(V)
    stress = Stress(J2, u)
    residual = ufl.inner(stress.sigma, strain(ufl.TestFunction(V))) * stress.dx
    bottom = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[2], 0.0))
    top = fem.locate_dofs_topological(V.sub(2), 2, mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[2], 1.0)))
    pull = fem.Constant(domain, 0.0)
    newton = Newton(stress, residual, u, [fem.dirichletbc(np.zeros(3), bottom, V), fem.dirichletbc(pull, top, V.sub(2))], rtol=1e-13, atol=1e-11)
    for value in pulls:
        pull.value = value
        newton.solve()
        stress.commit()
    # Plastic: the yield stress is passed (ε_zz up to 4e-3, σ₀/E = 1.25e-3).
    assert np.max(np.abs(stress.material.stress)) > 250.0
    # Match the nodes by their coordinates.
    x = V.tabulate_dof_coordinates()
    order_ref = np.lexsort(np.round(x_ref, 9).T)
    order = np.lexsort(np.round(x, 9).T)
    np.testing.assert_allclose(x[order], x_ref[order_ref], atol=1e-12)
    np.testing.assert_allclose(u.x.array.reshape(-1, 3)[order], u_ref[order_ref], rtol=0, atol=1e-10 * pulls[-1])


def test_homogenized_composite_against_dolfinx_elasticity(jl):
    # Aligned spheroids, tilted: a transversely isotropic stiffness off the axes.
    jl.seval("using TensND")
    C = jl.seval(
        """let
            rve = RVE()
            add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness_young_poisson(30.0e3, 0.2)); fraction = :rest)
            add_phase!(rve, :I, Ellipsoid(3.0, 1.0, 1.0; euler_angles = (0.4, 0.9, 0.0)), Dict(:C => iso_stiffness_young_poisson(200.0e3, 0.25)); fraction = 0.25)
            homogenize(rve, MoriTanaka(), :C)
        end"""
    )
    K = np.asarray(jl.seval("C -> Array(TensND.mandel(C, TensND.CanonicalBasis{3, Float64}()))")(C))
    C4 = mf.tensor4_from_mandel(K[:, :, None])[0]
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    bottom = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[2], 0.0))
    top = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[2], 1.0))
    shear = fem.Function(V)
    shear.interpolate(lambda x: np.vstack([1.0e-3 + 0 * x[0], 0.5e-3 + 0 * x[0], -0.4e-3 + 0 * x[0]]))
    bcs = [fem.dirichletbc(np.zeros(3), bottom, V), fem.dirichletbc(shear, top)]
    # The reference: linear elasticity in UFL with the homogenized stiffness.
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})
    i, j, k, l = ufl.indices(4)
    Cu = ufl.as_tensor(C4.tolist())
    w, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.as_tensor(Cu[i, j, k, l] * strain(w)[k, l], (i, j))
    zero = fem.Constant(domain, np.zeros(3))
    problem = LinearProblem(ufl.inner(a, strain(v)) * dx, ufl.inner(zero, v) * dx, bcs=bcs, petsc_options_prefix="ref_", petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u_ref = problem.solve()
    # The MFH material: the same stiffness, at every point through Julia.
    u = fem.Function(V)
    stress = Stress(jl.LinearElasticity(C), u)
    newton = Newton(stress, ufl.inner(stress.sigma, strain(v)) * stress.dx, u, bcs, rtol=1e-14, atol=1e-14)
    newton.solve()
    np.testing.assert_allclose(u.x.array, u_ref.x.array, rtol=0, atol=1e-12 * np.max(np.abs(u_ref.x.array)))


def _lame_errors(a, b, p, E, nu, nr, nt):
    """The largest errors on u_r and σ_zz of the quarter annulus meshed nr × nt."""
    # A rectangle in (r, θ) mapped to (x, y); its boundary is polygonal.
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([a, 0.0]), np.array([b, np.pi / 2])], [nr, nt])
    r, t = domain.geometry.x[:, 0].copy(), domain.geometry.x[:, 1].copy()
    domain.geometry.x[:, 0], domain.geometry.x[:, 1] = r * np.cos(t), r * np.sin(t)
    V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    u, v = fem.Function(V), ufl.TestFunction(V)
    stress = Stress(f"LinearElasticity(; E = {E}, ν = {nu})", u, quadrature_degree=2)
    fdim = 1
    bore = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(np.hypot(x[0], x[1]), a))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mesh.meshtags(domain, fdim, bore, np.ones_like(bore)))
    n = ufl.FacetNormal(domain)
    # The pressure pushes outward on the bore, whose outer normal points inward.
    residual = ufl.inner(stress.sigma, strain(v)) * stress.dx + p * ufl.dot(n, v) * ds(1)
    sym_x = fem.locate_dofs_topological(V.sub(1), fdim, mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0)))
    sym_y = fem.locate_dofs_topological(V.sub(0), fdim, mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0)))
    bcs = [fem.dirichletbc(0.0, sym_x, V.sub(1)), fem.dirichletbc(0.0, sym_y, V.sub(0))]
    Newton(stress, residual, u, bcs, rtol=1e-12, atol=1e-12).solve()
    x = V.tabulate_dof_coordinates()[:, :2]
    rr = np.hypot(x[:, 0], x[:, 1])
    ur = np.sum(u.x.array.reshape(-1, 2) * x / rr[:, None], axis=1)
    exact = (1 + nu) * p * a**2 / (E * (b**2 - a**2)) * ((1 - 2 * nu) * rr + b**2 / rr)
    # The out-of-plane stress of plane strain, ν(σ_rr + σ_θθ) = 2ν p a²/(b² − a²).
    szz = 2 * nu * p * a**2 / (b**2 - a**2)
    return np.max(np.abs(ur - exact)) / np.max(exact), np.max(np.abs(stress.material.stress[2] - szz)) / szz


def test_lame_thick_cylinder_plane_strain(jl):
    a, b, p, E, nu = 1.0, 2.0, 50.0, 70.0e3, 0.3
    coarse = _lame_errors(a, b, p, E, nu, 8, 12)
    fine = _lame_errors(a, b, p, E, nu, 16, 24)
    # The polygonal bore limits the error to second order: four times smaller
    # when the mesh is halved, on the displacement and on σ_zz alike.
    assert fine[0] < 1e-3 and fine[1] < 1e-2
    assert coarse[0] / fine[0] > 3.5 and coarse[1] / fine[1] > 3.0
