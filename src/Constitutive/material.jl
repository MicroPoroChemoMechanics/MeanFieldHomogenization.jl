# =============================================================================
#  material.jl — the Gauss-point constitutive contract.
#
#  This is the seam that lets a microstructure act as a *material law* inside a
#  structural finite-element computation, the way an MFront behavior or an
#  Abaqus UMAT does. It is deliberately backend-agnostic: nothing here knows
#  about Ferrite, Gridap, dolfinx or Abaqus. An FE code only ever needs to
#  build the material once, allocate one state per quadrature point, and call
#  `material_response` in its element loop.
#
#  THE SHAPE OF THE CONTRACT.  A purely mechanical law maps one gradient to one
#  flux, but the fractured-reservoir model of Barthélémy & Daniel (ARMA 2011)
#  does not fit that: it takes a strain *and* a pore pressure, and returns a
#  stress *and* a variation of fluid content, with four tangent blocks between
#  them. Rather than bolt a second API on later, the contract is
#  multi-gradient / multi-flux from the start — the same choice MGIS makes for
#  MFront's generic behaviors — with a one-gradient specialization that keeps
#  the common case terse.
# =============================================================================

"""
    AbstractMFHMaterial

Supertype of every Gauss-point constitutive law built on a homogenization
model. A concrete material bundles the microstructure (an
[`RVE`](@ref MeanFieldHomogenization.Schemes.RVE) or any
[`AbstractHomogenizationCell`](@ref MeanFieldHomogenization.Core.AbstractHomogenizationCell)),
the scheme used to upscale it, and whatever is needed to evaluate it
repeatedly and cheaply.

# The contract

A material must implement:

- [`initial_state`](@ref) — the internal state at the start of the computation;
- [`material_response`](@ref) — the incremental response.

and may implement:

- [`gradient_names`](@ref), [`flux_names`](@ref), [`tangent_blocks`](@ref) —
  self-description, which [`check_material_interface`](@ref) uses and which a
  generic FE driver can introspect instead of hard-coding field names.

Materials with a single gradient and a single flux (every purely mechanical
law) need only the three-argument form of [`material_response`](@ref); the
multi-field machinery then costs them nothing.

See also [`HomogenizedElastic`](@ref).
"""
abstract type AbstractMFHMaterial end

"""
    AbstractMaterialState

Supertype of the internal state carried by a Gauss-point material between
steps: crack aperture ratios, open/closed flags, fracture conductivities, and
so on.

A state is **immutable by convention**. [`material_response`](@ref) returns a
new one rather than mutating its argument, which is what makes a rejected
Newton iteration trivially recoverable — the caller simply keeps the old state
— and what keeps the law usable from a multithreaded element loop. The FE
driver holds two arrays, `states` and `states_old`, and swaps them once a step
has converged.

[`NoState`](@ref) is the state of a law that has none.
"""
abstract type AbstractMaterialState end

"""
    NoState() <: AbstractMaterialState

The internal state of a law that carries none — a linear elastic material, for
instance. Costs nothing to store per quadrature point.
"""
struct NoState <: AbstractMaterialState end

"""
    MaterialResponse(fluxes, tangents, state)

What [`material_response`](@ref) returns.

- `fluxes` — a `NamedTuple` of the thermodynamic forces, e.g. `(; σ)` for a
  mechanical law, `(; σ, φ)` for a poroelastic one (`φ` being the variation of
  fluid content).
- `tangents` — a `NamedTuple` of the tangent blocks, keyed by *flux then
  gradient*: `(; σε = ℂ, σp = -𝐁, φε = 𝐁, φp = 1/M)`. Keys concatenate the flux
  and gradient names so the block structure stays readable at the call site.
- `state` — the updated [`AbstractMaterialState`](@ref).

Access the common mechanical case with [`stress`](@ref) and
[`tangent`](@ref) rather than by digging into the fields.
"""
struct MaterialResponse{F <: NamedTuple, T <: NamedTuple, S <: AbstractMaterialState}
    fluxes::F
    tangents::T
    state::S
end

"""
    stress(r::MaterialResponse)

The `σ` flux of a response — the stress of a mechanical or poroelastic law.
"""
stress(r::MaterialResponse) = r.fluxes.σ

"""
    tangent(r::MaterialResponse)

The `σε` tangent block — the consistent tangent stiffness ``\\partial\\sigma /
\\partial\\varepsilon`` an FE code assembles into its Jacobian.
"""
tangent(r::MaterialResponse) = r.tangents.σε

"""
    state(r::MaterialResponse)

The updated internal state carried by a response.
"""
state(r::MaterialResponse) = r.state

# ── The generics every material extends ──────────────────────────────────────

"""
    initial_state(m::AbstractMFHMaterial) -> AbstractMaterialState

The internal state a fresh quadrature point starts from. Called once per
quadrature point when the FE driver allocates its state arrays.
"""
function initial_state end

"""
    material_response(m, gradients, state_old, Δt; cache = nothing) -> MaterialResponse
    material_response(m, ε, state_old, Δt; cache = nothing) -> MaterialResponse

Integrate the constitutive law over one step and return the fluxes, the tangent
blocks and the new state.

`gradients` is a `NamedTuple` — `(; ε)` for a mechanical law, `(; ε, p)` for a
poroelastic one. The second form accepts a bare strain tensor and wraps it, so
a mechanical law reads

```julia
r = material_response(mat, ε, st, Δt)
σ, ℂ, st_new = stress(r), tangent(r), state(r)
```

`state_old` is **not** mutated: the response carries a new state. That is what
makes an FE Newton iteration that has to be retried safe, and what allows a
threaded element loop.

`Δt` is the time increment; rate-independent laws ignore it. `cache` is an
optional [`MaterialCache`](@ref) shared across quadrature points — for a
homogenization-backed law it is the difference between a scheme solve per
Gauss point and one per distinct microstructural state.

!!! note "Tensors are TensND objects in the global frame"
    `ε` is expected in the canonical (global) frame, which is where an FE code
    computes it. Use [`from_tensors`](@ref) on the way in and
    [`to_tensors`](@ref) on the way out; do not hand raw component arrays
    across the boundary.
"""
function material_response end

material_response(
    m::AbstractMFHMaterial, ε::TensND.AbstractTens{2, 3},
    state_old::AbstractMaterialState, Δt::Real; kw...
) = material_response(m, (; ε = ε), state_old, Δt; kw...)

# `Δt` defaults to zero for rate-independent laws, so a caller that has no
# notion of time does not have to invent one.
material_response(
    m::AbstractMFHMaterial, gradients, state_old::AbstractMaterialState; kw...
) = material_response(m, gradients, state_old, 0.0; kw...)

"""
    gradient_names(m::AbstractMFHMaterial) -> Tuple{Vararg{Symbol}}

The gradients the law consumes, in order. Defaults to `(:ε,)`.
"""
gradient_names(::AbstractMFHMaterial) = (:ε,)

"""
    flux_names(m::AbstractMFHMaterial) -> Tuple{Vararg{Symbol}}

The thermodynamic forces the law produces, in order. Defaults to `(:σ,)`.
"""
flux_names(::AbstractMFHMaterial) = (:σ,)

"""
    tangent_blocks(m::AbstractMFHMaterial) -> Tuple{Vararg{Symbol}}

The tangent blocks the law supplies, keyed *flux then gradient*. Defaults to
`(:σε,)`.

A generic FE driver can read this to know which blocks to assemble instead of
hard-coding them, which is what makes the same driver serve a mechanical and a
poroelastic material.
"""
tangent_blocks(::AbstractMFHMaterial) = (:σε,)

"""
    transport_property(m::AbstractMFHMaterial, state) -> Union{Nothing,AbstractTens{2,3}}

The 2nd-order transport property (permeability, conductivity, diffusivity)
implied by the current internal state, or `nothing` for a material that carries
none.

This is *not* a flux: it is a coefficient the surrounding balance equation
needs — Darcy's law, in the fractured-reservoir case, where the permeability
follows the fracture apertures and so changes with the state. Keeping it off
the flux/tangent structure reflects that it belongs to a different balance
equation than the one the material closes.
"""
transport_property(::AbstractMFHMaterial, ::AbstractMaterialState) = nothing
