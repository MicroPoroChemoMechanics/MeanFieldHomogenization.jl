# [Installation](@id man-installation)

`MeanFieldHomogenization` is registered in Julia's General registry.

```julia
julia> import Pkg; Pkg.add("MeanFieldHomogenization")
```

or, from the Pkg REPL mode (`]`):

```julia
pkg> add MeanFieldHomogenization
```

No additional registry is required: its dependencies (`TensND.jl`,
`OrdinaryDiffEq.jl`, `Elliptic.jl`, `QuadGK.jl`, `Polynomials.jl`,
`PolynomialRoots.jl`, `Tensors.jl`, …) come from General as well. Type-generic
elliptic integrals are bundled internally as the
[`MeanFieldHomogenization.Elliptic`](@ref MeanFieldHomogenization.Elliptic)
submodule.

## Optional package extensions

Nothing below is needed for the core: each extension loads by itself when you
`import` its trigger packages, and the feature it unlocks is the only thing that
becomes unavailable without it.

| Load | Unlocks |
| :--- | :--- |
| `DECUHR`, `Integrals` | the adaptive-cubature backend `method = :decuhr` for arbitrary anisotropy; the bundled `method = :nestedquadgk` covers the same cases |
| `NonlinearSolve` | any SciML algorithm for the self-consistent fixed points ([Iterative solvers](@ref man-schemes)) |
| `SymPy` | symbolic closed forms of the elliptic integrals |
| `Ferrite`, `FerriteGmsh`, `Gmsh` | the reference finite-element backend for [FE inclusions](@ref man-fe-inclusions) |
| `Gridap`, `GridapGmsh` | the second FE backend for the same morphologies |
| `Ferrite` (alone) | the material interface of the [finite-element coupling](@ref fe-coupling) — a microstructure as a Gauss-point law |
| `Lux`, `Optimisers`, `Zygote` | *training* a [neural surrogate](@ref man-neural-inclusions); evaluating a shipped model needs none of them |

For development from a clone of the repository, instantiate the project
before first use:

```shell
cd /path/to/MeanFieldHomogenization.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run the test suite with:

```shell
julia --project=. -e 'using Pkg; Pkg.test()'
```
