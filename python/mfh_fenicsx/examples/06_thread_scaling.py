"""How the time of a plastic computation scales with the number of Julia threads.

A cube, clamped at its base, is pulled at its top (quadratic tetrahedra), made
of one of two materials: J2 plasticity, a cheap law of a few microseconds per
point, or a composite of two viscoplastic phases homogenized at every point
(the quasi-elastic Mori–Tanaka formulation), a costlier one. The same load step
is solved with Julia
started on 1, 2, 4, … threads, each in its own process, since the number of
threads of an embedded Julia is set when it starts. Every run computes the same
solution; what changes is the time of the stress updates, which
`update_points!` shares among the threads by blocks of cells. The assembly and
the linear solves stay in dolfinx, on one process, and do not change.

The runs are timed one after the other, alone on the machine: run it with
nothing else heavy running.

    python 06_thread_scaling.py --out results/06 --threads 1 2 4 8
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--out", default="results/06_thread_scaling", help="directory of the results")
parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4], help="numbers of Julia threads")
parser.add_argument("--n", type=int, default=10, help="the cube is divided into n × n × n × 6 tetrahedra")
parser.add_argument("--materials", nargs="+", default=["j2", "composite"], choices=["j2", "composite"])
parser.add_argument("--child", type=int, default=None, help=argparse.SUPPRESS)
parser.add_argument("--material", default="j2", help=argparse.SUPPRESS)
args = parser.parse_args()
out = Path(args.out)


MATERIALS = {
    "j2": "J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀ = 250.0, H = 2.0e3)",
    "composite": """let
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)
        add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = 0.2)
        QuasiElasticComposite(rve, (M = NortonViscoplasticity(; E = 70.0e3, ν = 0.3, K = 300.0, n = 3),
                                    I = NortonViscoplasticity(; E = 400.0e3, ν = 0.25, K = 900.0, n = 3)))
    end""",
}


def child(threads):
    """One run, Julia on `threads` threads: the times of a warm load step."""
    import time

    import numpy as np
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    import mfh_fenicsx as mf
    from mfh_fenicsx.fem import Newton, Stress, strain

    jl = mf.start(threads=threads)
    jl.seval("using TensND")
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, args.n, args.n, args.n)
    V = fem.functionspace(domain, ("Lagrange", 2, (3,)))
    u = fem.Function(V)
    stress = Stress(MATERIALS[args.material], u, dt=0.1)
    residual = ufl.inner(stress.sigma, strain(ufl.TestFunction(V))) * stress.dx
    bottom = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[2], 0.0))
    top = fem.locate_dofs_topological(V.sub(2), 2, mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[2], 1.0)))
    pull = fem.Constant(domain, 0.0)
    bcs = [fem.dirichletbc(np.zeros(3), bottom, V), fem.dirichletbc(pull, top, V.sub(2))]
    newton = Newton(stress, residual, u, bcs, rtol=1e-10, atol=1e-8)
    # A first step compiles; the second, into the nonlinear range, is timed.
    timings = []
    for value in (1.0e-3, 3.0e-3):
        pull.value = value
        julia_before, calls_before = stress.material.julia_time, stress.material.calls
        start = time.perf_counter()
        iterations = newton.solve()
        total = time.perf_counter() - start
        stress.commit()
        timings.append(
            {
                "iterations": iterations,
                "total_s": total,
                "julia_s": stress.material.julia_time - julia_before,
                "julia_calls": stress.material.calls - calls_before,
            }
        )
    result = {
        "material": args.material,
        "threads": int(jl.Threads.nthreads()),
        "points": stress.material.n,
        "step": timings[1],
        "checksum": float(np.linalg.norm(u.x.array)),
    }
    print(json.dumps(result))


if args.child is not None:
    child(args.child)
    sys.exit()

# The parent: one process per material and number of threads, one after the other.
out.mkdir(parents=True, exist_ok=True)
runs = []
for material in args.materials:
    for threads in args.threads:
        # One thread for OpenBLAS and for the OpenMP of the direct solver: their
        # threads would compete with Julia's, and an OpenMP factorization sums in
        # an order that changes from run to run, which the check below would see.
        env = dict(os.environ, PYTHON_JULIACALL_THREADS=str(threads), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        done = subprocess.run(
            [sys.executable, __file__, "--child", str(threads), "--material", material, "--n", str(args.n)],
            env=env, capture_output=True, text=True, check=True,
        )
        runs.append(json.loads(done.stdout.strip().splitlines()[-1]))
        step = runs[-1]["step"]
        print(f"{material}, {threads} threads: step {step['total_s']:.2f} s, of which {step['julia_s']:.2f} s in Julia ({step['julia_calls']} calls)")
        (out / "results.json").write_text(json.dumps(runs, indent=2))

# Every run of a material must have computed the same solution.
for material in args.materials:
    checksums = {r["checksum"] for r in runs if r["material"] == material}
    if max(checksums) - min(checksums) > 1e-12 * max(checksums):
        raise SystemExit(f"{material}: the runs differ: {checksums}")
summary = {"machine": f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs", "runs": runs}
(out / "results.json").write_text(json.dumps(summary, indent=2))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, len(args.materials), figsize=(4.2 * len(args.materials), 4.0), layout="constrained", squeeze=False)
for ax, material in zip(axes[0], args.materials):
    mine = [r for r in runs if r["material"] == material]
    labels = [str(r["threads"]) for r in mine]
    julia = [r["step"]["julia_s"] for r in mine]
    rest = [r["step"]["total_s"] - r["step"]["julia_s"] for r in mine]
    ax.bar(labels, julia, label="stress updates (Julia)")
    ax.bar(labels, rest, bottom=julia, label="assembly and solves (dolfinx)")
    ax.set_title(f"{material}, {mine[0]['points']} points", fontsize="medium")
    ax.set_xlabel("Julia threads")
axes[0][0].set_ylabel("time of the load step (s)")
fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside lower center", ncols=2, frameon=False)
fig.savefig(out / "thread_scaling.svg")
print(f"written: {out / 'results.json'}, {out / 'thread_scaling.svg'}")
