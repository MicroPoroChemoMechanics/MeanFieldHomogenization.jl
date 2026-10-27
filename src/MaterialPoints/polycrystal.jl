# =============================================================================
#  polycrystal.jl — an aggregate of many grains, each with its own law and
#  orientation, under a uniform strain (Taylor).
#
#  The composites of uniform_fields.jl solve all the phases in one system, which
#  suits a few phases; an aggregate has hundreds of grains. Here every grain
#  keeps its own state and is integrated on its own, the grains being
#  independent under a uniform strain.
# =============================================================================

import Random

"""
    PolycrystalModel

How the grains of a [`Polycrystal`](@ref) share the macroscopic fields:
[`TaylorModel`](@ref).
"""
abstract type PolycrystalModel end

"""
    TaylorModel()

Every grain undergoes the macroscopic strain [bishopHill1951](@cite): the
[`UniformStrain`](@ref) composite of the grains.
"""
struct TaylorModel <: PolycrystalModel end

"""
    Polycrystal(grains, weights = equal; model = TaylorModel())

An aggregate of `grains`, a vector of laws of `MaterialPoints` of one type (the
same law with different orientations, typically a [`CrystalViscoplasticity`](@ref)
per grain), with the volume `weights` (equal by default), sharing the macroscopic
fields as `model` says. It runs in [`point_response`](@ref) and in the
finite-element coupling like a law; its state holds the state of every grain.
See [the theory page](@ref mp-theory-polycrystals).
"""
struct Polycrystal{L <: AbstractPointLaw, W <: Real, M <: PolycrystalModel} <: AbstractMFHMaterial
    grains::Vector{L}
    weights::Vector{W}
    model::M
end

function Polycrystal(grains::AbstractVector{<:AbstractPointLaw}, weights::AbstractVector{<:Real} = fill(1 / length(grains), length(grains)); model::PolycrystalModel = TaylorModel())
    isempty(grains) && throw(ArgumentError("a polycrystal needs at least one grain"))
    length(weights) == length(grains) || throw(ArgumentError("$(length(grains)) grains and $(length(weights)) weights"))
    TensND.is_hard_numeric(eltype(weights)) || throw(ArgumentError("the weights must be numbers that can be compared; got $(eltype(weights))"))
    all(>(0), weights) && isapprox(sum(weights), 1; rtol = 1.0e-12) || throw(ArgumentError("the weights must be positive and add up to one"))
    _sizes(first(grains).gradients) == (ε = 6,) && _sizes(first(grains).fluxes) == (σ = 6,) ||
        throw(ArgumentError("a polycrystal takes grains of the strain and the stress"))
    return Polycrystal(collect(grains), collect(weights), model)
end

"""
    PolycrystalState

The state of a [`Polycrystal`](@ref): the state of every grain (`grains`), and
the macroscopic strain and stress it was reached at.
"""
struct PolycrystalState{S <: PointState, T} <: AbstractMaterialState
    grains::Vector{S}
    ε::SVector{6, T}
    σ::SVector{6, T}
end

Constitutive.initial_state(m::Polycrystal) = PolycrystalState(map(initial_state, m.grains), zero(SVector{6, Float64}), zero(SVector{6, Float64}))
Constitutive.gradient_names(::Polycrystal) = (:ε,)
Constitutive.flux_names(::Polycrystal) = (:σ,)

"""
    grain_states(st::PolycrystalState) -> Vector

The states of the grains, in the order of the grains of the polycrystal.
"""
grain_states(st::PolycrystalState) = st.grains

# One step on Kelvin–Mandel components: the driver and the element loops call
# this, as for a law.
function _mandel_update(m::Polycrystal, E, st::PolycrystalState, Δt; algorithm, external, cache = nothing)
    Σ, 𝕂, grains = _aggregate_step(m.model, m, SVector{6}(E), st, Δt, algorithm, external)
    return Σ, 𝕂, PolycrystalState(grains, SVector{6}(E), Σ)
end

# The material of the Gauss-point contract, for the callers that use it.
function Constitutive.material_response(m::Polycrystal, gradients::NamedTuple, st::PolycrystalState, Δt::Real; cache = nothing)
    Σ, 𝕂, new = _mandel_update(m, _mandel(gradients.ε), st, Δt; algorithm = LocalNewton(), external = (;))
    return MaterialResponse((σ = _tensor(Σ),), (σε = _tensor(𝕂),), new)
end

_grain_update(law, ε, st, Δt, alg, ext) = _point_update(_with_external(law, ext), ε, st, Δt, alg)

# Every grain at the macroscopic strain: independent grains, averaged.
function _aggregate_step(::TaylorModel, m, E, st, Δt, alg, ext)
    steps = map((law, s) -> _grain_update(law, E, s, Δt, alg, ext), m.grains, st.grains)
    Σ = sum(w * first(r) for (w, r) in zip(m.weights, steps))
    𝕂 = sum(w * r[2] for (w, r) in zip(m.weights, steps))
    return Σ, 𝕂, map(last, steps)
end

"""
    random_orientations(n; rng = Random.default_rng()) -> Vector

`n` orientations drawn uniformly over the rotations (a random texture), as
TensND bases: Euler angles ``(\\theta, \\phi, \\psi)`` in the convention of TensND,
with ``\\cos\\theta`` uniform on ``[-1, 1]`` and ``\\phi``, ``\\psi`` uniform on
``[0, 2\\pi)``, which is the uniform measure on the rotations.
"""
function random_orientations(n::Integer; rng::Random.AbstractRNG = Random.default_rng())
    return [TensND.RotatedBasis(acos(2rand(rng) - 1), 2π * rand(rng), 2π * rand(rng)) for _ in 1:n]
end
