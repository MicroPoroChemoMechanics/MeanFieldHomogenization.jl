"""A double-notched plate whose microcracks grow (plane strain).

The solid is the cracked solid of the tutorial on crack growth: penny-shaped
microcracks of all orientations, of initial density 0.05, in an isotropic
matrix; their density grows when the energy released reaches Griffith's
resistance, which decreases as they grow. The growth is regularized in time by
a viscosity τ (`CrackGrowthDamage`). A plate with two semicircular notches is
pulled; a quarter is computed.

The law is local: nothing in it sets the width of the band where the cracks
grow. Only the viscosity does, through the time τ. With τ = 0.1 (the duration
of the loading is 1) the two meshes give the same curve; with τ = 0.01 the
softening is sharper and the meshes start to part (about 1 % on the peak
force). As τ decreases toward the rate-independent law, the band narrows to
the elements, the results depend more and more on the mesh, and the steps must
be cut ever finer (at τ = 0.001 the coarse mesh needed 400 steps, and the finer
one did not reach the end of the loading in 30 minutes). A mesh-objective
computation needs a regularized law (nonlocal or with gradients), which this
example does not claim.

In the band the density grows far beyond the range where a dilute density of
cracks means anything (above 100 at the end): Griffith's resistance decreases
toward zero as the cracks grow, and the law follows it.

    python 02_notched_plate_crack_growth.py --out results/02
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
parser.add_argument("--out", default="results/02_notched_plate_crack_growth", help="directory of the results")
parser.add_argument("--h", type=float, nargs="+", default=[0.5, 0.25], help="element sizes far from the notch")
parser.add_argument("--steps", type=int, default=40, help="load steps")
parser.add_argument("--tau", type=float, nargs="+", default=[0.1, 0.01], help="times of the viscous regularization")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# A quarter of the plate: 0 ≤ x ≤ W, 0 ≤ y ≤ H, the notch of radius r centered
# at (W, 0). The top is pulled to δ_max over one unit of time.
W, H, r = 5.0, 10.0, 0.5
delta_max = 3.0e-3 * H

for h in args.h:
    cells = 3 * W * H / h**2
    if cells > 1.0e5:
        raise SystemExit(f"h = {h}: about {cells:.0f} cells, too large for this example")

jl = mf.start()
jl.seval("using TensND")
cracked = jl.seval(
    """(τ) -> let
        k, μ = 30e3, 20e3                                                # MPa
        C₀ = TensISO{3}(3k, 2μ)
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
        add_phase!(rve, :F, PennyCrack(1.0), Dict(:C => C₀); density = 0.05, symmetrize = :iso)
        griffith = CrackGrowthResistance(; Gc = 0.02, a₀ = 1.0, d₀ = 0.05)  # N/mm, mm
        CrackGrowthDamage(rve; resistance = griffith, τ)
    end"""
)


def notched_quarter(h):
    """The quarter plate meshed by gmsh; its mesh and its boundary tags
    (1 left x = 0, 2 bottom ligament y = 0, 3 top y = H)."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("notched")
    plate = gmsh.model.occ.addRectangle(0, 0, 0, W, H)
    notch = gmsh.model.occ.addDisk(W, 0, 0, r, r)
    gmsh.model.occ.cut([(2, plate)], [(2, notch)])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [s[1] for s in gmsh.model.getEntities(2)], 1)
    for curve in gmsh.model.getEntities(1):
        x, y, _ = gmsh.model.occ.getCenterOfMass(*curve)
        tag = 1 if np.isclose(x, 0) else 2 if np.isclose(y, 0) else 3 if np.isclose(y, H) else None
        if tag is not None:
            gmsh.model.addPhysicalGroup(1, [curve[1]], tag)
    # Finer along the ligament, where the damage localizes.
    field = gmsh.model.mesh.field
    field.add("Box", 1)
    for key, value in (("VIn", h / 4), ("VOut", h), ("XMin", 0), ("XMax", W), ("YMin", 0), ("YMax", 1.5), ("Thickness", 2.0)):
        field.setNumber(1, key, value)
    field.setAsBackgroundMesh(1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.model.mesh.generate(2)
    data = io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return data.mesh, data.facet_tags


def pull(h, tau):
    """The force–displacement curve of the mesh of size h, and the final densities by cell."""
    material = cracked(tau)
    domain, facets = notched_quarter(h)
    V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    u, v = fem.Function(V), ufl.TestFunction(V)
    stress = Stress(material, u, quadrature_degree=2)
    residual = ufl.inner(stress.sigma, strain(v)) * stress.dx
    top_y = fem.locate_dofs_topological(V.sub(1), 1, facets.find(3))
    opening = fem.Constant(domain, 0.0)
    bcs = [
        fem.dirichletbc(0.0, fem.locate_dofs_topological(V.sub(0), 1, facets.find(1)), V.sub(0)),
        fem.dirichletbc(0.0, fem.locate_dofs_topological(V.sub(1), 1, facets.find(2)), V.sub(1)),
        fem.dirichletbc(opening, top_y, V.sub(1)),
    ]
    newton = Newton(stress, residual, u, bcs, rtol=1e-9, atol=1e-9)
    curve = [(0.0, 0.0)]

    def prescribe(t):
        opening.value = delta_max * t

    def record(t):
        # The force on the top of the quarter, per unit thickness.
        curve.append((delta_max * t, newton.reaction(top_y)))
        np.savetxt(out / f"curve_tau{tau}_h{h}.txt", curve, header="displacement  force")

    start = time.perf_counter()
    quasistatic(newton, np.linspace(0.0, 1.0, args.steps + 1), prescribe, on_step=record)
    elapsed = time.perf_counter() - start
    ncells = domain.topology.index_map(2).size_local
    density = stress.material.internal("F")[0].reshape(-1, stress.material.nqp)[:ncells].mean(axis=1)
    print(f"τ = {tau}, h = {h}: {ncells} triangles, {elapsed:.1f} s, peak force {max(f for _, f in curve):.2f} N/mm")
    return domain, np.array(curve), density, ncells, elapsed


runs = {(tau, h): pull(h, tau) for tau in args.tau for h in args.h}
summary = {
    f"tau={tau}, h={h}": {"triangles": int(n), "peak_force_N_per_mm": float(c[:, 1].max()), "final_force_N_per_mm": float(c[-1, 1]), "time_s": t}
    for (tau, h), (_, c, _, n, t) in runs.items()
}
(out / "results.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402

from matplotlib.colors import LogNorm  # noqa: E402

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), layout="constrained", width_ratios=[1.4, 1])
styles = dict(zip(sorted(args.tau, reverse=True), ("-", "--", ":", "-.")))
colors = dict(zip(sorted(args.h, reverse=True), ("C0", "C1", "C2", "C3")))
for (tau, h), (_, c, _, n, _) in runs.items():
    ax1.plot(c[:, 0], c[:, 1], lw=2, ls=styles[tau], color=colors[h], label=f"τ = {tau}, h = {h} ({n} triangles)")
ax1.set_xlabel("displacement of the top (mm)")
ax1.set_ylabel("force per unit thickness (N/mm)")
ax1.legend(frameon=False, fontsize="small")
# The densities of the finest mesh with the smallest τ, on a logarithmic scale:
# they range from the initial 0.05 to large values at the root of the notch.
shown = (min(args.tau), min(args.h))
domain, _, density, ncells, _ = runs[shown]
xy = domain.geometry.x[:, :2]
triangles = domain.geometry.dofmap[:ncells, :3]
image = ax2.tripcolor(
    mtri.Triangulation(xy[:, 0], xy[:, 1], triangles), facecolors=density, cmap="viridis",
    norm=LogNorm(vmin=0.05, vmax=density.max()), edgecolors="face", rasterized=True,
)
ax2.set_aspect("equal")
ax2.set_title(f"crack density, τ = {shown[0]}, h = {shown[1]}", fontsize="medium")
ax2.set_axis_off()
fig.colorbar(image, ax=ax2, shrink=0.8)
fig.savefig(out / "notched_plate.svg", dpi=150)  # the map of the densities as an image
print(f"written: {out / 'results.json'}, {out / 'notched_plate.svg'}")
