"""A borehole in a cracked, saturated rock whose cracks close (plane strain, u–p).

The rock holds three families of microcracks, of normals e₁, e₂ and e₃, in an
isotropic matrix, saturated by a fluid. A crack closes progressively as the
effective stress Σ′ = Σ + p 𝟏 compresses it across its plane
(`ClosingCrackPoroelasticity`): its open density is its density times
exp(−⟨−σₙ⟩/σc), σₙ the normal effective traction. The stiffness and the Biot
coefficients of the rock follow, at every quadrature point, from the
homogenization of the cracks still open.

The pressure in the borehole is lowered by 10 MPa at once and held; the outer
boundary is fixed and drained at the initial pressure. The fluid flows toward
the hole, the effective stress rises around it, and the cracks close there,
first those whose normal is radial. The same problem with cracks that never
close (σc very large) shows what the closure changes. The mobility k/μ of the
fluid is constant: this law does not let the permeability follow the apertures.
At the end the flow is steady, and the pressure is that of a radial flow,
−Δp ln(b/r)/ln(b/a), whatever the cracks: the script checks it.

    python 03_borehole_closing_cracks.py --out results/03
"""

import argparse
import json
import time
from pathlib import Path

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

import mfh_fenicsx as mf
from mfh_fenicsx.fem import Newton, Poroelastic, quasistatic, strain

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--out", default="results/03_borehole_closing_cracks", help="directory of the results")
parser.add_argument("--nr", type=int, default=24, help="elements along the radius")
parser.add_argument("--nt", type=int, default=12, help="elements along the quarter circle")
parser.add_argument("--steps", type=int, default=24, help="time steps")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

a, b = 100.0, 1000.0            # mm
drawdown = 10.0                 # MPa
mobility = 1.0e-3               # k/μ, mm²/(MPa·s)
sigma_c = 20.0                  # MPa, the closure stress of the cracks
if 2 * args.nr * args.nt > 20000:
    raise SystemExit("too many cells for this example")

jl = mf.start()
jl.seval("using TensND")
rock = jl.seval(
    """σc -> let
        C₀ = TensISO{3}(3 * 30.0e3, 2 * 20.0e3)                      # MPa
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
        add_phase!(rve, :F3, PennyCrack(1.0), Dict(:C => C₀); density = 0.1)                          # normal e₃
        add_phase!(rve, :F1, PennyCrack(1.0; euler_angles = (π / 2, 0.0)), Dict(:C => C₀); density = 0.1)       # normal e₁
        add_phase!(rve, :F2, PennyCrack(1.0; euler_angles = (π / 2, π / 2)), Dict(:C => C₀); density = 0.1)     # normal e₂
        ClosingCrackPoroelasticity(rve; closure = ExponentialClosure(σc), porosity = 0.05)
    end"""
)

# A quarter of the annulus: a rectangle in (s, θ) mapped to (x, y), the radius
# graded geometrically, r = a (b/a)^s, finer near the hole.
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([1.0, np.pi / 2])], [args.nr, args.nt])
s, t = domain.geometry.x[:, 0].copy(), domain.geometry.x[:, 1].copy()
r = a * (b / a) ** s
domain.geometry.x[:, 0], domain.geometry.x[:, 1] = r * np.cos(t), r * np.sin(t)
cell = domain.basix_cell()
W = fem.functionspace(domain, basix.ufl.mixed_element([basix.ufl.element("Lagrange", cell, 2, shape=(2,)), basix.ufl.element("Lagrange", cell, 1)]))
Vu, Vp = W.sub(0), W.sub(1)
fdim = 1
hole = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(np.hypot(x[0], x[1]), a))
outer = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(np.hypot(x[0], x[1]), b))
ds = ufl.Measure("ds", domain=domain, subdomain_data=mesh.meshtags(domain, fdim, hole, np.ones_like(hole)))
n = ufl.FacetNormal(domain)
times = np.concatenate([[0.0], np.geomspace(1.0, 1.0e5, args.steps)])   # s


def drain(name, closure):
    """The borehole of the rock whose cracks close at `closure`; histories and final fields."""
    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    poro = Poroelastic(rock(closure), u, p, quadrature_degree=2)
    p_w = fem.Constant(domain, 0.0)
    dt = fem.Constant(domain, 1.0)
    # The fluid in the hole pushes on its wall with its pressure, outward from
    # the rock (the outer normal of the rock points into the hole).
    residual = (
        ufl.inner(poro.sigma, strain(v)) * poro.dx + p_w * ufl.dot(n, v) * ds(1)
        + (poro.phi - poro.phi_old) * q * poro.dx + dt * mobility * ufl.dot(ufl.grad(p), ufl.grad(q)) * poro.dx
    )
    # Symmetry on the axes; the outer boundary fixed and drained; the pressure
    # of the hole on its wall.
    Vu_c = Vu.collapse()[0]
    fixed = fem.Function(Vu_c)
    bcs = [
        fem.dirichletbc(0.0, fem.locate_dofs_topological(Vu.sub(1), fdim, mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))), Vu.sub(1)),
        fem.dirichletbc(0.0, fem.locate_dofs_topological(Vu.sub(0), fdim, mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))), Vu.sub(0)),
        fem.dirichletbc(fixed, fem.locate_dofs_topological((Vu, Vu_c), fdim, outer), Vu),
        fem.dirichletbc(0.0, fem.locate_dofs_topological(Vp, fdim, outer), Vp),
        fem.dirichletbc(p_w, fem.locate_dofs_topological(Vp, fdim, hole), Vp),
    ]
    newton = Newton(poro, residual, w, bcs, rtol=1e-9, atol=1e-9)

    Vp_c, p_dofs = Vp.collapse()
    xp = Vp_c.tabulate_dof_coordinates()[:, :2]
    on_axis = np.isclose(xp[:, 1], 0.0)
    order = np.argsort(xp[on_axis, 0])
    radii = xp[on_axis, 0][order]
    Vu_c, u_dofs = Vu.collapse()
    xu = Vu_c.tabulate_dof_coordinates()[:, :2]
    wall = np.flatnonzero(np.isclose(xu[:, 1], 0.0) & np.isclose(xu[:, 0], a))[0]
    profiles, wall_history = {}, []

    def prescribe(t1):
        p_w.value = -drawdown
        dt.value = poro.dt

    def record(t1):
        wall_history.append((t1, w.x.array[u_dofs].reshape(-1, 2)[wall, 0]))
        profiles[t1] = w.x.array[p_dofs][on_axis][order].copy()

    start = time.perf_counter()
    quasistatic(newton, times, prescribe, on_step=record)
    elapsed = time.perf_counter() - start
    # The open fraction of the family of normal e₁ at the end, from the
    # committed effective stress Σ′ (its component 11 is the normal traction).
    sigma_eff = poro.material.internal("Σ′")
    open_e1 = np.exp(-np.maximum(-sigma_eff[0], 0.0) / closure)
    print(f"{name}: {elapsed:.0f} s, wall displacement {wall_history[-1][1]:.4e} mm")
    # Three instants of the pressure, the nearest computed to 10 s, 1000 s and the end.
    kept = {min(profiles, key=lambda t: abs(np.log(t / target))): None for target in (1.0e1, 1.0e3, times[-1])}
    return radii, {t: profiles[t] for t in kept}, np.array(wall_history), open_e1, poro, elapsed


runs = {"closing cracks": drain("closing cracks", sigma_c), "cracks always open": drain("cracks always open", 1.0e12)}
# At the end the flow is steady: the pressure of a radial flow at constant
# mobility, −Δp ln(b/r)/ln(b/a), whatever the cracks.
radii, profiles, _, _, _, _ = runs["closing cracks"]
steady = -drawdown * np.log(b / radii) / np.log(b / a)
summary = {
    name: {"wall_displacement_mm": float(hist[-1, 1]), "least_open_fraction_e1": float(open_e1.min()), "time_s": el}
    for name, (_, _, hist, open_e1, _, el) in runs.items()
}
summary["steady_pressure_gap_MPa"] = float(np.max(np.abs(profiles[max(profiles)] - steady)))
(out / "results.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.0, 3.8), layout="constrained", width_ratios=[1.2, 1.2, 1])
for (name, (radii, profiles, hist, _, _, _)), ls in zip(runs.items(), ("-", "--")):
    for (target, prof), color in zip(sorted(profiles.items()), ("C0", "C1", "C2")):
        ax1.plot(radii, prof, ls=ls, color=color, lw=2, label=f"t = {target:.3g} s" if ls == "-" else None)
    ax2.plot(hist[:, 0], -hist[:, 1], ls=ls, lw=2, color="k", label=name)
ax1.set_xscale("log")
ax1.set_xlabel("radius (mm)")
ax1.set_ylabel("pressure change (MPa)")
ax1.legend(frameon=False, fontsize="small", title="solid: closing, dashed: open", title_fontsize="small")
ax2.set_xscale("log")
ax2.set_xlabel("time (s)")
ax2.set_ylabel("inward displacement of the wall (mm)")
ax2.legend(frameon=False, fontsize="small")
_, _, _, open_e1, poro, _ = runs["closing cracks"]
ncells = domain.topology.index_map(2).size_local
fraction = open_e1.reshape(-1, poro.material.nqp)[:ncells].mean(axis=1)
xy = domain.geometry.x[:, :2]
image = ax3.tripcolor(mtri.Triangulation(xy[:, 0], xy[:, 1], domain.geometry.dofmap[:ncells, :3]), facecolors=fraction, cmap="viridis", vmin=0, vmax=1, edgecolors="face", rasterized=True)
ax3.set_xlim(0, 4 * a)
ax3.set_ylim(0, 4 * a)
ax3.set_aspect("equal")
ax3.set_title("open fraction, cracks of normal e₁", fontsize="medium")
fig.colorbar(image, ax=ax3, shrink=0.8)
fig.savefig(out / "borehole.svg", dpi=150)
print(f"written: {out / 'results.json'}, {out / 'borehole.svg'}")
