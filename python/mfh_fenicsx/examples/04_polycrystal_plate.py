"""A polycrystalline plate with a hole: the models of Kröner and Hill (plane strain).

Every quadrature point holds an aggregate of eight grains of random
orientations, each a viscoplastic crystal of twelve slip systems
(`CrystalViscoplasticity`), the grains of the tutorial on polycrystals. The
aggregate shares the strain among its grains as one of two models says:

- Kröner's: every grain is an inclusion in the elastic aggregate;
- Hill's: every grain is an inclusion in the aggregate as it flows, its
  anisotropic tangent found self-consistently at every point and every
  iteration.

A quarter of a plate with a hole is pulled. Near the hole the aggregate flows
first; the model that lets the grains deviate more from the average strain
(Hill's) flows at a lower stress, and the plate carries less load.

These materials are costly: about 50 ms per point and per update for Kröner's
model, three times more for Hill's, whose steps must also be cut more often.
The mesh is coarse, 151 linear triangles of one point each; on 4 Julia threads
the computation took 171 s with Kröner's model and 1 152 s with Hill's
(results.json keeps the times of a run). Julia shares the points among its
threads, so start it with several (PYTHON_JULIACALL_THREADS).

    python 04_polycrystal_plate.py --out results/04
"""

import argparse
import json
import time
from pathlib import Path

import gmsh
import numpy as np
import ufl
from dolfinx import fem, io
from mpi4py import MPI

import mfh_fenicsx as mf
from mfh_fenicsx.fem import Newton, Stress, quasistatic, strain

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--out", default="results/04_polycrystal_plate", help="directory of the results")
parser.add_argument("--h", type=float, default=1.5, help="element size far from the hole, 1.5 at most (the plate is 10 wide)")
parser.add_argument("--steps", type=int, default=16, help="load steps")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# A quarter of a 20 × 20 plate with a hole of radius 2; its right edge is
# pulled to a mean strain of 2e-3 in one unit of time.
L, R = 10.0, 2.0
delta_max = 2.0e-3 * L
cells = 3 * (L / args.h) ** 2
if cells > 2000:
    raise SystemExit(f"about {cells:.0f} cells: too many for materials of ~0.1 s per point update")

jl = mf.start()
jl.seval("using Random")
aggregate = jl.seval(
    """model -> let
        τ₀ = 10.0                                                    # MPa
        E, ν = 120e3, 0.3
        C₁₁, C₁₂ = E * (1 - ν) / ((1 + ν) * (1 - 2ν)), E * ν / ((1 + ν) * (1 - 2ν))
        # Grains of the isotropic elasticity that Kröner's model assumes.
        grain(frame) = CrystalViscoplasticity(; C₁₁, C₁₂, C₄₄ = (C₁₁ - C₁₂) / 2, K = 1.0, n = 5, τ₀, frame)
        frames = random_orientations(8; rng = MersenneTwister(2))
        Polycrystal(grain.(frames); model = model == "Kröner" ? KronerModel(; E, ν) : HillModel())
    end"""
)

# --- Mesh --------------------------------------------------------------------
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("plate")
plate = gmsh.model.occ.addRectangle(0, 0, 0, L, L)
hole = gmsh.model.occ.addDisk(0, 0, 0, R, R)
gmsh.model.occ.cut([(2, plate)], [(2, hole)])
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(2, [s[1] for s in gmsh.model.getEntities(2)], 1)
for curve in gmsh.model.getEntities(1):
    x, y, _ = gmsh.model.occ.getCenterOfMass(*curve)
    tag = 1 if np.isclose(x, 0) else 2 if np.isclose(y, 0) else 3 if np.isclose(x, L) else None
    if tag is not None:
        gmsh.model.addPhysicalGroup(1, [curve[1]], tag)
gmsh.option.setNumber("Mesh.MeshSizeMax", args.h)
gmsh.option.setNumber("Mesh.MeshSizeMin", args.h / 3)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
gmsh.model.mesh.generate(2)
data = io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
gmsh.finalize()
domain, facets = data.mesh, data.facet_tags
ncells = domain.topology.index_map(2).size_local
print(f"mesh: {ncells} triangles")


def pull(model):
    """The force–displacement curve of the plate made of the aggregate of `model`."""
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    u, v = fem.Function(V), ufl.TestFunction(V)
    stress = Stress(aggregate(model), u, quadrature_degree=1)
    residual = ufl.inner(stress.sigma, strain(v)) * stress.dx
    right_x = fem.locate_dofs_topological(V.sub(0), 1, facets.find(3))
    opening = fem.Constant(domain, 0.0)
    bcs = [
        fem.dirichletbc(0.0, fem.locate_dofs_topological(V.sub(0), 1, facets.find(1)), V.sub(0)),
        fem.dirichletbc(0.0, fem.locate_dofs_topological(V.sub(1), 1, facets.find(2)), V.sub(1)),
        fem.dirichletbc(opening, right_x, V.sub(0)),
    ]
    newton = Newton(stress, residual, u, bcs, rtol=1e-8, atol=1e-8)
    curve = [(0.0, 0.0)]

    def prescribe(t):
        opening.value = delta_max * t

    def record(t):
        curve.append((delta_max * t, newton.reaction(right_x)))
        np.savetxt(out / f"curve_{model}.txt", curve, header="displacement  force")

    start = time.perf_counter()
    quasistatic(newton, np.linspace(0.0, 1.0, args.steps + 1), prescribe, on_step=record)
    elapsed = time.perf_counter() - start
    print(f"{model}: {elapsed:.0f} s, of which {stress.material.julia_time:.0f} s in Julia")
    return np.array(curve), elapsed, stress.material.julia_time


runs = {model: pull(model) for model in ("Kröner", "Hill")}
summary = {
    "triangles": int(ncells),
    "threads": int(jl.Threads.nthreads()),
    **{m: {"final_force_N_per_mm": float(c[-1, 1]), "time_s": t, "julia_s": tj} for m, (c, t, tj) in runs.items()},
}
(out / "results.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(5.5, 3.6), layout="constrained")
for model, (c, _, _) in runs.items():
    ax.plot(c[:, 0], c[:, 1], lw=2, marker="o", label=f"{model}'s model")
ax.set_xlabel("displacement of the right edge (mm)")
ax.set_ylabel("force per unit thickness (N/mm)")
ax.legend(frameon=False)
fig.savefig(out / "polycrystal_plate.svg")
print(f"written: {out / 'results.json'}, {out / 'polycrystal_plate.svg'}")
