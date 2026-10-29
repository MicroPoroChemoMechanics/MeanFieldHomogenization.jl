"""A plate with a hole made of a homogenized fiber composite (plane strain).

A square plate with a circular hole is pulled along x. It is made of a matrix
reinforced by aligned prolate spheroids, whose Mori–Tanaka stiffness
MeanFieldHomogenization computes. The same plate is solved twice:

1. with the material of MeanFieldHomogenization at every quadrature point,
   through mfh_fenicsx (one Julia call per Newton iteration);
2. with plain dolfinx, the homogenized stiffness written as a constant tensor
   in UFL.

Both give the same displacements to round-off, which checks the coupling. The
figure compares the stress concentration on the ligament above the hole with
that of the matrix alone: the fibers along x carry the load and raise it.

Run it with the Julia project of mfh_fenicsx set (see the README), from any
directory:

    python 01_plate_with_hole.py --out results/01
"""

import argparse
import json
import time
from pathlib import Path

import gmsh
import numpy as np
import ufl
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

import mfh_fenicsx as mf
from mfh_fenicsx.fem import Newton, Stress, strain

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--out", default="results/01_plate_with_hole", help="directory of the results")
parser.add_argument("--h", type=float, default=0.25, help="element size far from the hole (the plate is 10 wide)")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# Geometry and loading: a quarter of a 20 × 20 plate with a hole of radius 1,
# pulled by a traction of 100 MPa on its right edge.
L, R, traction = 10.0, 1.0, 100.0

# The size of the problem before meshing: about 4 (L/h)² triangles with the
# refinement near the hole, a few kB each with the direct solver; refuse what
# would not fit comfortably.
cells = 4 * (L / args.h) ** 2
if cells > 2.0e5:
    raise SystemExit(f"about {cells:.0f} cells: too large for this example, raise --h")

# --- Mesh (gmsh) -------------------------------------------------------------
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("plate")
plate = gmsh.model.occ.addRectangle(0, 0, 0, L, L)
hole = gmsh.model.occ.addDisk(0, 0, 0, R, R)
gmsh.model.occ.cut([(2, plate)], [(2, hole)])
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(2, [s[1] for s in gmsh.model.getEntities(2)], 1)
# Boundary curves by position: 1 left (x = 0), 2 bottom (y = 0), 3 right (x = L).
for curve in gmsh.model.getEntities(1):
    x, y, _ = gmsh.model.occ.getCenterOfMass(*curve)
    tag = 1 if np.isclose(x, 0) else 2 if np.isclose(y, 0) else 3 if np.isclose(x, L) else None
    if tag is not None:
        gmsh.model.addPhysicalGroup(1, [curve[1]], tag)
# Finer near the hole, where the stress concentrates.
field = gmsh.model.mesh.field
field.add("Distance", 1)
field.setNumbers(1, "PointsList", [p[1] for p in gmsh.model.getEntities(0) if np.isclose(np.hypot(*gmsh.model.getValue(0, p[1], [])[:2]), R)])
field.add("Threshold", 2)
for key, value in (("InField", 1), ("SizeMin", args.h / 5), ("SizeMax", args.h), ("DistMin", 0.5), ("DistMax", 4.0)):
    field.setNumber(2, key, value)
field.setAsBackgroundMesh(2)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.model.mesh.generate(2)
mesh_data = io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
gmsh.finalize()
domain, facets = mesh_data.mesh, mesh_data.facet_tags
print(f"mesh: {domain.topology.index_map(2).size_local} triangles")

V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
v = ufl.TestFunction(V)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facets)
symmetry = [
    fem.dirichletbc(0.0, fem.locate_dofs_topological(V.sub(0), 1, facets.find(1)), V.sub(0)),
    fem.dirichletbc(0.0, fem.locate_dofs_topological(V.sub(1), 1, facets.find(2)), V.sub(1)),
]
load = ufl.dot(fem.Constant(domain, np.array([traction, 0.0])), v) * ds(3)

# --- The material: Mori–Tanaka stiffness of the composite (Julia) ------------
jl = mf.start()
jl.seval("using TensND")
C_hom = jl.seval(
    """let
        Cm = iso_stiffness_young_poisson(3.0e3, 0.35)     # epoxy, MPa
        Cf = iso_stiffness_young_poisson(70.0e3, 0.22)    # glass
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => Cm); fraction = :rest)
        add_phase!(rve, :F, Ellipsoid(5.0, 1.0, 1.0), Dict(:C => Cf); fraction = 0.3)   # spheroids along x
        homogenize(rve, MoriTanaka(), :C)
    end"""
)
composite = jl.LinearElasticity(C_hom)
matrix = jl.seval("LinearElasticity(; E = 3.0e3, ν = 0.35)")


def solve_mfh(material):
    """The plate with `material` at every quadrature point; displacement, stress, time."""
    u = fem.Function(V)
    stress = Stress(material, u, quadrature_degree=2)
    residual = ufl.inner(stress.sigma, strain(v)) * stress.dx - load
    newton = Newton(stress, residual, u, symmetry)
    start = time.perf_counter()
    newton.solve()
    return u, stress, time.perf_counter() - start


u_mfh, stress_mfh, t_mfh = solve_mfh(composite)
_, stress_matrix, _ = solve_mfh(matrix)

# --- The same plate in plain dolfinx, the stiffness as a constant tensor -----
mandel = np.asarray(jl.seval("C -> Array(TensND.KM(C, TensND.CanonicalBasis{3, Float64}()))")(C_hom))
C4 = mf.tensor4_from_mandel(mandel[:, :, None])[0][:2, :2, :2, :2]
i, j, k, l = ufl.indices(4)
w = ufl.TrialFunction(V)
sigma_ref = lambda e: ufl.as_tensor(ufl.as_tensor(C4.tolist())[i, j, k, l] * e[k, l], (i, j))  # noqa: E731
dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})
start = time.perf_counter()
problem = LinearProblem(
    ufl.inner(sigma_ref(strain(w)), strain(v)) * dx, load, bcs=symmetry,
    petsc_options_prefix="plate_", petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
u_ref = problem.solve()
t_ref = time.perf_counter() - start
gap = np.max(np.abs(u_mfh.x.array - u_ref.x.array)) / np.max(np.abs(u_ref.x.array))
print(f"displacements: largest gap {gap:.1e} of the largest displacement")

# --- Stress on the ligament x = 0, at the quadrature points next to it -------
points = stress_mfh.space.element.interpolation_points
xq = fem.Expression(ufl.SpatialCoordinate(domain), points).eval(domain, np.arange(domain.topology.index_map(2).size_local, dtype=np.int32))
xq = xq.reshape(-1, 2)
near = xq[:, 0] < 0.05
order = np.argsort(xq[near, 1])
y = xq[near, 1][order]
sxx_composite = stress_mfh.material.stress[0, : len(xq)][near][order] / traction
sxx_matrix = stress_matrix.material.stress[0, : len(xq)][near][order] / traction

results = {
    "triangles": int(domain.topology.index_map(2).size_local),
    "displacement_gap": float(gap),
    "concentration_composite": float(sxx_composite[0]),
    "concentration_matrix": float(sxx_matrix[0]),
    "time_mfh_first_solve_s": t_mfh,  # includes the compilation of the first Julia call
    "time_plain_dolfinx_s": t_ref,
}
(out / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(5.5, 3.6), layout="constrained")
ax.plot(y, sxx_composite, lw=2, label="composite (Mori–Tanaka)")
ax.plot(y, sxx_matrix, lw=2, ls="--", label="matrix alone")
ax.set_xlabel("height of the quadrature points next to the ligament, y / R")
ax.set_ylabel(r"$\sigma_{xx}\,/\,\sigma_\infty$")
ax.set_xlim(R, 5 * R)
ax.legend(frameon=False)
fig.savefig(out / "ligament.svg")
print(f"written: {out / 'results.json'}, {out / 'ligament.svg'}")
