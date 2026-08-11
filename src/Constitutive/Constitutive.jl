"""
    MeanFieldHomogenization.Constitutive

**MeanFieldHomogenization as a constitutive law at each Gauss point** of a
structural finite-element computation — the role an MFront behavior or an
Abaqus UMAT plays.

!!! note "Not to be confused with `MeanFieldHomogenization.FiniteElements`"
    [`FiniteElements`](@ref MeanFieldHomogenization.FiniteElements) points the
    other way: it uses finite elements *inside* the package, to solve the
    Eshelby problem of a single inclusion whose response no closed form covers.
    Here the finite-element code is the **caller**, and a whole microstructure
    plays the part of one material point.

# The contract

A material bundles a microstructure, a scheme, and whatever internal state the
model needs. An FE driver builds it once, allocates one state per quadrature
point, and calls [`material_response`](@ref) in its element loop:

```julia
mat    = HomogenizedElastic(rve, MoriTanaka())
states = [[initial_state(mat) for _ in 1:nqp] for _ in 1:ncells]
cache  = MaterialCache()

r  = material_response(mat, ε, states[cell][qp], Δt; cache = cache)
σ  = to_tensors(stress(r))       # -> Tensors.SymmetricTensor{2,3}, global frame
ℂ  = to_tensors(tangent(r))      # -> Tensors.SymmetricTensor{4,3}, global frame
states[cell][qp] = state(r)
```

The contract is **multi-gradient / multi-flux** — the shape MGIS uses for
MFront's generic behaviors — because the poroelastic case takes a strain *and*
a pore pressure and returns a stress *and* a variation of fluid content. A
purely mechanical law only ever sees the one-gradient specialization.

# Contents

| | |
|---|---|
| [`AbstractMFHMaterial`](@ref), [`AbstractMaterialState`](@ref) | the contract |
| [`material_response`](@ref), [`initial_state`](@ref) | the two required methods |
| [`MaterialResponse`](@ref), [`stress`](@ref), [`tangent`](@ref), [`state`](@ref) | what a response carries |
| [`HomogenizedElastic`](@ref) | linear law from any scheme — the control case |
| [`MaterialCache`](@ref), [`cache_stats`](@ref) | memoization on the microstructural configuration |
| [`to_tensors`](@ref), [`from_tensors`](@ref) | the `Tensors.jl` bridge |
| [`check_material_interface`](@ref) | conformance checker |

# Two rules that are easy to break

- **Frames.** A `Tensors.jl` tensor has no basis; its components are global. A
  TensND tensor returns its components in its *own* basis, and a homogenized
  property whose RVE carries tilted inclusions comes back rotated. Always cross
  the boundary through [`to_tensors`](@ref) / [`from_tensors`](@ref), never
  through `get_array`.
- **State is immutable.** [`material_response`](@ref) returns a new state
  instead of mutating its argument, so a rejected Newton iteration is recovered
  by keeping the old one.
"""
module Constitutive

using LinearAlgebra
using TensND
using Tensors

import ..Core
const MFH_Core = Core

import ..Schemes
import ..Poromechanics

include("tensors_bridge.jl")
include("material.jl")
include("cache.jl")
include("elastic.jl")
include("check.jl")

# ── The contract ─────────────────────────────────────────────────────────────
export AbstractMFHMaterial, AbstractMaterialState, NoState
export MaterialResponse, stress, tangent, state
export initial_state, material_response
export gradient_names, flux_names, tangent_blocks, transport_property
export check_material_interface

# ── Materials ────────────────────────────────────────────────────────────────
export HomogenizedElastic, stiffness

# ── Memoization ──────────────────────────────────────────────────────────────
export MaterialCache, cached!, cache_stats, reset_cache!

# ── Tensors.jl bridge ────────────────────────────────────────────────────────
export to_tensors, from_tensors
export voigt_stress, voigt_strain, stress_from_voigt, strain_from_voigt

end # module
