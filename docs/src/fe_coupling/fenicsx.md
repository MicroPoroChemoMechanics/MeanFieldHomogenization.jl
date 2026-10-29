# [FEniCSx backend](@id fe-fenicsx)

!!! info "Before this page"
    [Materials](@ref fe-materials), for what a material computes at a
    quadrature point, and [Ferrite backend](@ref fe-backends), for the pieces
    of the coupling that belong to the package whatever the finite-element code.

[FEniCSx](https://fenicsproject.org) writes a problem in UFL and assembles it
in C++; the Python package `mfh_fenicsx`, in `python/mfh_fenicsx` of the
repository, lets the stress of such a problem be a material of this package:
a law of `MaterialLaws`, a nonlinear composite, a polycrystal. Julia runs
inside the Python process, through
[juliacall](https://github.com/JuliaPy/PythonCall.jl).

## How it works

The stress is an *external operator* of
[dolfinx-external-operator](https://github.com/a-latyshev/dolfinx-external-operator)
[latyshevEtAl2025](@cite): a UFL object that stands for ``\boldsymbol{\sigma}(\boldsymbol{\varepsilon}(\underline{u}))``
in the forms, and whose values and derivative are supplied, at every quadrature
point, by a Python function. The residual is written with it as with any stress,

```math
\int_\Omega \boldsymbol{\sigma}(\boldsymbol{\varepsilon}(\underline{u})) : \boldsymbol{\varepsilon}(\underline{v})\,\mathrm{d}\Omega
- \int_{\partial\Omega} \underline{t}\cdot\underline{v}\,\mathrm{d}S = 0 ,
```

and its Jacobian is the derivative of the operator, the consistent tangent
``\mathbb{K} = \partial\boldsymbol{\sigma}/\partial\boldsymbol{\varepsilon}``.

At each Newton iteration the strains of **all** the quadrature points are
evaluated in one array and handed to Julia in **one** call,
[`update_points!`](@ref), which computes every stress and every tangent from
the states of the last converged step, on as many threads as Julia has, and
writes them into arrays that NumPy reads without a copy. The tangent the
Jacobian asks for is the one of that call: no point is integrated twice. The
states stay in Julia, in a [`QuadratureStates`](@ref), and a step is
committed or rolled back as with Ferrite.

| Python (`mfh_fenicsx`) | Julia |
|:--|:--|
| `mf.start()` | starts Julia, loads the package; installs nothing |
| `Material(material, ncells, nqp)` | the material and its [`QuadratureStates`](@ref) |
| `Stress(material, u)` | the external operator ``\boldsymbol{\sigma}(\boldsymbol{\varepsilon}(\underline{u}))`` and its quadrature space |
| `Newton(stress, residual, u, bcs)` | Newton's method on one step; MUMPS when PETSc has it |
| `quasistatic(newton, times, prescribe)` | the steps, a failing one halved, as [`quasistatic!`](@ref) |
| `stress.commit()`, `stress.rollback()` | [`commit!`](@ref), [`rollback!`](@ref) |
| `newton.reaction(dofs)` | the force of a Dirichlet condition |
| `stress.material.internal("p")` | the committed values of an internal variable |

## Installation

Two environments meet: a Julia project that holds `MeanFieldHomogenization`
and `PythonCall`, and a Python environment that holds dolfinx (from
conda-forge) and `mfh_fenicsx`:

```bash
pip install -e "python/mfh_fenicsx[dolfinx,test]"
export PYTHON_JULIAPKG_EXE=$(which julia)              # the Julia to embed
export PYTHON_JULIAPKG_PROJECT=/path/to/julia/project  # holds MeanFieldHomogenization
export PYTHON_JULIACALL_THREADS=4                       # Julia's threads
```

`mf.start()` sets `PYTHON_JULIAPKG_OFFLINE=yes`: juliacall then uses that
project as it is and downloads nothing.

## A load step

The code below is the core of every example: a displacement field, the
material at its quadrature points, the residual, and the steps.

```python
import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

import mfh_fenicsx as mf
from mfh_fenicsx.fem import Newton, Stress, quasistatic, strain

mf.start()
domain = mesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
u = fem.Function(V)

# The material, written in Julia; any material of the package goes there.
stress = Stress("J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀ = 250.0, H = 2.0e3)", u)
residual = ufl.inner(stress.sigma, strain(ufl.TestFunction(V))) * stress.dx

# Clamped at the bottom, the top pulled along z.
bottom = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[2], 0.0))
top_facets = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[2], 1.0))
top = fem.locate_dofs_topological(V.sub(2), 2, top_facets)
pull = fem.Constant(domain, 0.0)
bcs = [fem.dirichletbc(np.zeros(3), bottom, V), fem.dirichletbc(pull, top, V.sub(2))]

newton = Newton(stress, residual, u, bcs)
quasistatic(newton, np.linspace(0.0, 1.0, 11), lambda t: setattr(pull, "value", 4.0e-3 * t))
print(newton.reaction(top))   # the force on the top
```

| code | what it does |
|:--|:--|
| `Stress(material, u)` | builds the quadrature space, the states, and the operator `stress.sigma` |
| `stress.dx` | the measure whose quadrature matches the space of the stress; use it in the forms |
| `Newton(...)` | compiles the forms once; `solve()` runs one step |
| `quasistatic(...)` | a step that fails (Newton, or a point that does not integrate) is rolled back and halved |

## Conventions

- **Components.** The stress and the strain cross the boundary in
  Kelvin–Mandel components, ordered ``(11, 22, 33, \sqrt2\,23, \sqrt2\,13, \sqrt2\,12)``,
  the package's convention; `mandel_from_tensor`, `tensor_from_mandel` and
  `tensor4_from_mandel` convert from and to full tensors.
- **Points.** The points are ordered cell by cell, as a quadrature space of
  dolfinx orders them; the ghost cells of a process are included.
- **Two dimensions** mean plane strain: the material is integrated in three
  dimensions with ``\varepsilon_{13} = \varepsilon_{23} = \varepsilon_{33} = 0``,
  and the out-of-plane stress ``\sigma_{33}`` is kept in `stress.material.stress[2]`.
  Axisymmetry and the coupled ``(\underline{u}, p)`` problem of the
  [fractured rock](@ref fe-fractured-rock) are not covered yet: [`update_points!`](@ref)
  refuses a material that also needs a pressure gradient.
- **Processes.** The tests and the examples run on one MPI process; Julia's
  threads do the parallel work of the materials.

!!! warning "gmsh and Julia's signals"
    Julia stops its threads at safepoints through the signal SIGSEGV.
    `gmsh.initialize()` resets SIGSEGV, SIGBUS, SIGILL, SIGFPE (and SIGINT) to
    their defaults, and the next garbage collection on several threads then
    kills the process without a message. `mfh_fenicsx` saves Julia's handlers
    when it starts and puts them back before every call into Julia and at exit.
    After such a library, call `mfh_fenicsx.restore_signals()` before calling
    Julia yourself.

## What is checked

The tests of `python/mfh_fenicsx/tests` compare the coupling with references
that do not go through it:

| test | reference | agreement |
|:--|:--|:--|
| a homogeneous strain on a cube, 3-D and plane strain | Hooke's law; Julia's single-point update along a plastic path | to round-off |
| a plastic cube, clamped and pulled, the same hexahedral mesh and quadrature | the package's [Ferrite adapter](@ref fe-backends) | nodal displacements to ``10^{-10}`` of the pull |
| a cube of a Mori–Tanaka composite of tilted spheroids | a dolfinx solve written in UFL with the homogenized stiffness | ``10^{-12}`` |
| the thick cylinder under pressure, plane strain | Lamé's closed form | second-order convergence |
| a heterogeneous plastic problem | the order of convergence of Newton's method | about 2; an elastic tangent does not converge |

## Examples

Each example is a script of `python/mfh_fenicsx/examples`, which writes its
results and its figure in the directory `--out`;
`examples/run_all.sh` runs them all and copies the figures and the results
(`*.json`) here. The times below were measured on 8 logical CPUs (AMD
EPYC-Milan), Julia on 4 threads unless stated.

### A plate with a hole in a fiber composite

`01_plate_with_hole.py`. A plate with a hole, in plane strain, of an epoxy
reinforced by 30 % of glass spheroids aligned with the load, whose
Mori–Tanaka stiffness the package computes. The material at every point gives
the displacements of a dolfinx solve with the homogenized stiffness written in
UFL, to better than ``10^{-13}``; the stress concentration at the hole rises from
3.04 for the matrix alone to 3.65.

![stress along the ligament above the hole](../assets/fe_coupling/fenicsx/ligament.svg)

### A notched plate whose microcracks grow

`02_notched_plate_crack_growth.py`. Penny-shaped microcracks of all
orientations grow at Griffith's resistance, which decreases as they grow; the
growth is regularized by a viscosity of time ``\tau``
([`CrackGrowthDamage`](@ref)). With ``\tau = 0.1`` (the loading lasts 1) two
meshes give the same curve; with ``\tau = 0.01`` the cracks grow in a band
along the ligament and the meshes start to part, by about 1 % on the peak. The
law is local: as ``\tau`` decreases the band follows the elements, which no
mesh refinement cures.

![force–displacement curves and crack density](../assets/fe_coupling/fenicsx/notched_plate.svg)

### A polycrystalline plate: Kröner and Hill

`04_polycrystal_plate.py`. Every point is an aggregate of eight viscoplastic
crystals of random orientations ([`Polycrystal`](@ref)), whose grains share the
strain as [`KronerModel`](@ref) or [`HillModel`](@ref) says. Hill's aggregate,
constrained by its own flowing tangent, carries less load than Kröner's,
constrained by the elastic aggregate. These materials are costly: on 151
triangles, 171 s with Kröner's model and 1,152 s with Hill's.

![force–displacement of the polycrystalline plate](../assets/fe_coupling/fenicsx/polycrystal_plate.svg)

### Creep of a composite beam

`05_composite_beam_creep.py`. A cantilever of a matrix that creeps by
Norton's law, reinforced by 20 % of particles, under a load held for 20 s; the
composite is homogenized at every point by [`QuasiElasticComposite`](@ref)
[brenner2002](@cite). The first deflection of the matrix alone is the elastic
one of beam theory (0.0261 mm against 0.0260 mm); after 20 s the particles have
cut the deflection from 0.232 mm to 0.152 mm.

![tip deflection over time](../assets/fe_coupling/fenicsx/beam_creep.svg)

### Julia's threads

`06_thread_scaling.py`. One load step of a cube of 24,000 points, with Julia
on 1, 2, 4 and 8 threads, each run in its own process. For J2 plasticity the
stress updates are 1 to 5 % of the step, which the assembly and the linear
solves fill. For the composite of two viscoplastic phases they are 96 % of it,
and 4 threads take 29.4 s where one takes 92.5 s (3.2 times faster), 8 threads
26.8 s; every run computes the same solution.

![time of a load step against the number of Julia threads](../assets/fe_coupling/fenicsx/thread_scaling.svg)
