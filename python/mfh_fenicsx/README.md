# mfh_fenicsx

The mechanical materials of
[MeanFieldHomogenization.jl](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl)
at the quadrature points of a [FEniCSx](https://fenicsproject.org) (dolfinx 0.10)
problem: a constitutive law of `MaterialLaws`, a nonlinear composite, a polycrystal.

The stress is a `FEMExternalOperator` of
[dolfinx-external-operator](https://github.com/a-latyshev/dolfinx-external-operator)
([Latyshev et al., 2025](https://doi.org/10.46298/jtcam.14449)): the residual is
written in UFL with it, and its Jacobian is the consistent tangent. At each Newton
iteration one Julia call, `update_points!`, integrates every quadrature point, on as
many threads as Julia was started with.

## Installation

The Julia side is a Julia project that holds `MeanFieldHomogenization` and
`PythonCall`; nothing is installed at run time. The Python side needs dolfinx
(conda-forge), then

```bash
pip install -e "python/mfh_fenicsx[dolfinx,test]"
```

and the environment that tells juliacall which Julia and which project to use:

```bash
export PYTHON_JULIAPKG_EXE=$(which julia)
export PYTHON_JULIAPKG_PROJECT=/path/to/the/julia/project
export PYTHON_JULIACALL_THREADS=4
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # reproducible solves, cores left to Julia
```

## A load step

```python
import ufl
from dolfinx import fem
import mfh_fenicsx as mf
from mfh_fenicsx.fem import Newton, Stress, strain

mf.start()
u = fem.Function(V)
stress = Stress("J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀ = 250.0, H = 2.0e3)", u)
residual = ufl.inner(stress.sigma, strain(ufl.TestFunction(V))) * stress.dx
newton = Newton(stress, residual, u, bcs)
for load in loads:
    ...                      # set the boundary values of the step
    newton.solve()
    stress.commit()          # stress.rollback() to cut the step instead
```

`quasistatic(newton, times, prescribe)` runs the steps between `times` and halves a
step whose Newton iterations fail or where a point fails to integrate;
`newton.reaction(dofs)` gives the force a Dirichlet condition exerts;
`stress.material.internal("p")` the committed values of an internal variable.

Tensors cross the boundary in Kelvin–Mandel components, ordered
(11, 22, 33, √2·23, √2·13, √2·12); `mandel_from_tensor`, `tensor_from_mandel` and
`tensor4_from_mandel` convert from and to full tensors.

## Signals: gmsh and other libraries

Julia handles the signals SIGSEGV (its threads stop at safepoints through it),
SIGBUS, SIGILL, SIGFPE and SIGINT. `gmsh.initialize()` resets them to their
defaults, after which the next garbage collection on several threads kills the
process. `mfh_fenicsx` saves Julia's handlers when it starts and puts them back
before each of its calls into Julia and at exit; call `mfh_fenicsx.restore_signals()`
before calling Julia yourself after such a library.

## Examples

`examples/` holds scripts that run as they are, each writing its results and its
figure in a directory given by `--out`:

1. `01_plate_with_hole.py`: a plate with a hole made of a Mori–Tanaka fiber
   composite, against plain dolfinx with the same stiffness;
2. `02_notched_plate_crack_growth.py`: a double-notched plate whose microcracks
   grow (`CrackGrowthDamage`), on two meshes;
4. `04_polycrystal_plate.py`: a plate with a hole of polycrystals of eight grains,
   the models of Kröner and Hill (costly: minutes to a quarter of an hour);
5. `05_composite_beam_creep.py`: the creep of a cantilever of a homogenized
   viscoplastic composite, against the matrix alone;
6. `06_thread_scaling.py`: the time of a plastic load step with 1, 2, 4, …
   Julia threads.

## Tests

```bash
python -m pytest python/mfh_fenicsx/tests
```

`tests/test_material.py` needs Julia only; `tests/test_fem.py` and
`tests/test_validation.py` need dolfinx too, and the validations need `TensND` and
`Ferrite` in the Julia project: the same plastic problem solved through the package's
Ferrite adapter, a homogenized composite against a plain dolfinx elasticity solve, and
Lamé's thick cylinder in plane strain.
