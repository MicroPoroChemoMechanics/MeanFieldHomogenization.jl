"""Creep of a cantilever made of a particle-reinforced viscoplastic composite (plane strain).

A cantilever, clamped at its left end, carries a constant load at its free
end, applied at once and held. Its matrix creeps by Norton's law; the
composite holds 20 % of stiffer particles that creep more slowly. At every
quadrature point the composite is homogenized by the quasi-elastic formulation
of Brenner, Masson, Castelnau and Zaoui (Mori–Tanaka in time,
`QuasiElasticComposite`). The deflection of the tip grows with time; the
particles slow it down, compared with the matrix alone.

    python 05_composite_beam_creep.py --out results/05
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

import mfh_fenicsx as mf
from mfh_fenicsx.fem import Newton, Stress, quasistatic, strain

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--out", default="results/05_composite_beam_creep", help="directory of the results")
parser.add_argument("--nx", type=int, default=40, help="elements along the beam (4 through its depth per 40)")
parser.add_argument("--steps", type=int, default=20, help="time steps")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# A beam 10 mm long, 1 mm deep, under 0.5 N/mm at its tip: about 30 MPa of
# bending stress at the clamp. The load is held 20 s.
length, depth, load, duration = 10.0, 1.0, 0.5, 20.0
ny = max(2, args.nx // 10)
if args.nx * ny > 5000:
    raise SystemExit(f"{2 * args.nx * ny} triangles: too many for this example")

jl = mf.start()
jl.seval("using TensND")
phases = jl.seval(
    """(;
        M = NortonViscoplasticity(; E = 70.0e3, ν = 0.3, K = 300.0, n = 3),
        I = NortonViscoplasticity(; E = 400.0e3, ν = 0.25, K = 900.0, n = 3),
    )"""
)
composite = jl.seval(
    """phases -> let
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)
        add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = 0.2)
        QuasiElasticComposite(rve, phases)
    end"""
)(phases)
materials = {"composite": composite, "matrix alone": phases.M}

domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([length, depth])], [args.nx, ny])
V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
tip_facets = mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[0], length))
ds = ufl.Measure("ds", domain=domain, subdomain_data=mesh.meshtags(domain, 1, tip_facets, np.ones_like(tip_facets)))
clamp = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
# The tip deflection is read at the node on the axis of the beam.
tip = fem.locate_dofs_geometrical(V.sub(1).collapse()[0], lambda x: np.isclose(x[0], length) & np.isclose(x[1], depth / 2))
tip_dof = V.sub(1).collapse()[1][tip[0]]
times = np.concatenate([[0.0], np.geomspace(1.0e-3, duration, args.steps)])


def creep(name):
    """The deflection of the tip over time for the material `name`."""
    u, v = fem.Function(V), ufl.TestFunction(V)
    stress = Stress(materials[name], u, quadrature_degree=2)
    traction = fem.Constant(domain, np.zeros(2))
    residual = ufl.inner(stress.sigma, strain(v)) * stress.dx - ufl.dot(traction, v) * ds(1)
    newton = Newton(stress, residual, u, [fem.dirichletbc(np.zeros(2), clamp, V)], rtol=1e-9, atol=1e-10)
    history = [(0.0, 0.0)]

    def prescribe(t):
        # The load acts from the first step on, spread over the depth of the tip.
        traction.value = (0.0, -load / depth)

    def record(t):
        history.append((t, -u.x.array[tip_dof]))
        np.savetxt(out / f"deflection_{name.replace(' ', '_')}.txt", history, header="time  deflection")

    start = time.perf_counter()
    quasistatic(newton, times, prescribe, on_step=record)
    print(f"{name}: tip deflection {history[-1][1]:.4f} mm at t = {duration} s ({time.perf_counter() - start:.0f} s)")
    return np.array(history)


runs = {name: creep(name) for name in materials}
summary = {name: {"deflection_first_step_mm": float(h[1, 1]), "deflection_end_mm": float(h[-1, 1])} for name, h in runs.items()}
# A check of the first step, before the matrix has crept: Euler–Bernoulli's
# elastic deflection F L³ / (3 E' I), with E' = E / (1 − ν²) in plane strain and
# I = h³ / 12 per unit thickness (shear adds a fraction of a percent).
summary["matrix alone"]["beam_theory_elastic_mm"] = load * length**3 / (3 * 70.0e3 / (1 - 0.3**2) * depth**3 / 12)
(out / "results.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(5.5, 3.6), layout="constrained")
for name, h in runs.items():
    ax.plot(h[1:, 0], h[1:, 1], lw=2, marker="o", ms=3, label=name)
ax.set_xscale("log")
ax.set_xlabel("time under load (s)")
ax.set_ylabel("tip deflection (mm)")
ax.legend(frameon=False)
fig.savefig(out / "beam_creep.svg")
print(f"written: {out / 'results.json'}, {out / 'beam_creep.svg'}")
