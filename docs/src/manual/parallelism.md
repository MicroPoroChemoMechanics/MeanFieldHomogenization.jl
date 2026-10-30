# [Parallelism](@id man-parallelism)

The package computes in parallel where the work splits into independent
pieces of a useful size: the quadrature points of a finite-element mesh. A
single call of [`homogenize`](@ref) or [`point_response`](@ref) runs on one
thread.

## The quadrature points of a mesh

[`update_points!`](@ref) computes the stresses and the tangents of a
mechanical material at every quadrature point of a mesh in one call, the call
a finite-element code makes once per Newton iteration (it is the call the
[FEniCSx backend](@ref fe-fenicsx) makes). The points are shared among tasks by
blocks of whole cells; each point reads only its own state and writes only its
own results, so that the result does not depend on the number of tasks:

````@example parallelism
using MeanFieldHomogenization

law = J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀ = 250.0, H = 2.0e3)
ncells, nqp = 50, 4
ε = 3.0e-3 .* repeat([1.0, -0.3, -0.3, 0.0, 0.0, 0.0], 1, ncells * nqp) .* (1 .+ 0.1 .* sin.(1:ncells * nqp))'

results = map((1, 4)) do ntasks
    states = QuadratureStates(law, ncells, nqp)
    σ, 𝕂 = zeros(6, ncells * nqp), zeros(6, 6, ncells * nqp)
    update_points!(σ, 𝕂, ε, states, 1.0, law; ntasks)
    (σ, 𝕂)
end
results[1] == results[2]
````

| code | what it does |
|:--|:--|
| `QuadratureStates(law, ncells, nqp)` | the states of the points, the converged ones and the new ones |
| `update_points!(σ, 𝕂, ε, states, Δt, law; ntasks)` | every point, by `ntasks` tasks (the number of Julia threads by default) |

The tasks run on Julia's threads, which are set when Julia starts:
`julia --threads 4`, the variable `JULIA_NUM_THREADS`, or
`PYTHON_JULIACALL_THREADS` for a Julia embedded in Python. Keep
`OPENBLAS_NUM_THREADS=1` alongside, so that the linear algebra of each point
does not start threads of its own on top of Julia's.

## What it gains

The gain is that of the share of the materials in the computation. For J2
plasticity a point costs a few microseconds: 4 threads compute 10,000 points
3.9 times faster than one, but in a finite-element step the assembly and the
linear solves take most of the time. For a homogenized material a point costs
about a millisecond, and the points are almost all of the step: a load step of
24,000 points of a composite of two viscoplastic phases took 92.5 s on one
thread and 29.4 s on four (the [scaling example](@ref fe-fenicsx) of the FEniCSx
backend, which states the machine).

## Shared state

The Hill tensors read back by [`with_hill_cache`](@ref) live in a
[`HillCache`](@ref) guarded by a lock: one cache serves several threads, each
tensor computed once. Everything else a computation needs is local to its
task.
