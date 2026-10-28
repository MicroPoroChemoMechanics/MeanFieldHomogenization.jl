# =============================================================================
#  polycrystal.jl — an aggregate of many grains, each with its own law and
#  orientation, under a uniform strain (Taylor) or as inclusions in an elastic
#  or elastoplastic matrix (Kröner, Berveiller–Zaoui).
#
#  The composites of uniform_fields.jl solve all the phases in one system, which
#  suits a few phases; an aggregate has hundreds of grains. Here every grain
#  keeps its own state and is integrated on its own: independent under a
#  uniform strain, coupled only through the average plastic strain under an
#  interaction law, which an outer Newton iteration on its six components
#  solves.
# =============================================================================

import Random

"""
    PolycrystalModel

How the grains of a [`Polycrystal`](@ref) share the macroscopic fields:
[`TaylorModel`](@ref), [`KronerModel`](@ref), [`BerveillerZaouiModel`](@ref).
"""
abstract type PolycrystalModel end

"""
    TaylorModel()

Every grain undergoes the macroscopic strain [bishopHill1951](@cite): the
[`UniformStrain`](@ref) composite of the grains.
"""
struct TaylorModel <: PolycrystalModel end

"""
    KronerModel(; E, ν)

Every grain is a spherical inclusion, of plastic strain ``\\boldsymbol{\\varepsilon}^p_g``,
in a matrix of plastic strain ``\\boldsymbol{E}^p``, the average of the grains',
grains and matrix sharing the isotropic elasticity (`E`, `ν`)
[kroner1961](@cite). Eshelby's solution gives the interaction law
``\\boldsymbol{\\sigma}_g = \\boldsymbol{\\Sigma} + 2\\mu(1-\\beta)(\\boldsymbol{E}^p - \\boldsymbol{\\varepsilon}^p_g)``
for deviatoric plastic strains, ``\\beta = 2(4-5\\nu)/(15(1-\\nu))``. See
[the theory page](@ref mp-theory-polycrystals).
"""
struct KronerModel{T <: Real} <: PolycrystalModel
    k::T
    μ::T
end

"""
    BerveillerZaouiModel(; E, ν)

[`KronerModel`](@ref) with a matrix that flows plastically: its stiffness in
the interaction is the secant one of an isotropic elastoplastic medium,
``\\mu^* = \\mu/(1 + 2\\mu h)`` with ``\\boldsymbol{E}^p = h\\,\\boldsymbol{S}``
(``\\boldsymbol{S}`` the deviator of ``\\boldsymbol{\\Sigma}``)
[berveillerZaoui1978](@cite), which tempers the interaction by the plastic
accommodation factor ``\\alpha \\in (0, 1]``:
``\\boldsymbol{\\sigma}_g = \\boldsymbol{\\Sigma} + 2\\alpha\\mu(1-\\beta)(\\boldsymbol{E}^p - \\boldsymbol{\\varepsilon}^p_g)``.
Out of proportional loading, ``h`` is read as the ratio of the norms of
``\\boldsymbol{E}^p`` and ``\\boldsymbol{S}``. See
[the theory page](@ref mp-theory-polycrystals).
"""
struct BerveillerZaouiModel{T <: Real} <: PolycrystalModel
    k::T
    μ::T
end

for M in (:KronerModel, :BerveillerZaouiModel)
    @eval function $M(; E, ν)
        # The interaction is solved iteratively: numbers that can be compared only.
        TensND.is_hard_numeric(promote_type(typeof(E), typeof(ν))) ||
            throw(ArgumentError($(string(M)) * ": numeric moduli only (the grains are solved iteratively); got $(typeof(E)), $(typeof(ν))"))
        E > 0 || throw(DomainError(E, $(string(M)) * ": the Young modulus must be positive"))
        -1 < ν < 1 / 2 || throw(DomainError(ν, $(string(M)) * ": the Poisson ratio must lie in (-1, 1/2)"))
        return $M(promote(E / (3 * (1 - 2ν)), E / (2 * (1 + ν)))...)
    end
end

const _InclusionModel = Union{KronerModel, BerveillerZaouiModel}

"""
    Polycrystal(grains, weights = equal; model = TaylorModel())

An aggregate of `grains`, a vector of laws of `MaterialLaws` of one type (the
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

# ── Grains as inclusions in a matrix (Kröner, Berveiller–Zaoui) ───────────────
#
# With the elasticity ℂ shared by grains and matrix, the plastic strain of a
# grain is what its stress leaves of its strain, εᵖ = ε − ℂ⁻¹σ, whatever its
# law. A spherical grain in an isotropic matrix of moduli (k, μ*) strains as
# ε = E + 𝒞:(εᵖ − Eᵖ), with 𝒞 = c_J 𝕁 + c_K 𝕂: Eshelby's tensor when μ* = μ
# (Kröner), its secant counterpart otherwise (Berveiller–Zaoui). The grains are
# coupled only through Eᵖ = ⟨εᵖ⟩, the unknown of an outer Newton iteration.

# The shear modulus of the matrix in the interaction.
_matrix_shear(model::KronerModel, ℂ, E, Eᵖ) = model.μ
function _matrix_shear(model::BerveillerZaouiModel, ℂ, E, Eᵖ)
    # Without plastic strain the matrix is elastic (α = 1): a value test, the
    # polycrystal being numerical only.
    iszero(TensND.primal.(Eᵖ)) && return oftype(model.μ * first(Eᵖ), model.μ)
    S = deviator(ℂ * (E - Eᵖ))
    nS, nEᵖ = sqrt(sum(abs2, S)), sqrt(sum(abs2, Eᵖ))
    # μ* = μ/(1 + 2μh), h = ‖Eᵖ‖/‖S‖, written to stay finite when S vanishes.
    return model.μ * nS / (nS + 2 * model.μ * nEᵖ)
end

# (c_J, c_K): the volumetric and deviatoric parts of 𝒞 for a sphere in the
# matrix of moduli (k, μ*).
function _interaction(model::_InclusionModel, ℂ, E, Eᵖ)
    k, μ, μs = model.k, model.μ, _matrix_shear(model, ℂ, E, Eᵖ)
    βs = 6 * (k + 2μs) / (5 * (3k + 4μs))
    return SVector(3k / (3k + 4μs), μ * βs / (μ * βs + μs * (1 - βs)))
end

# One grain in the matrix: its strain solves ε = b + 𝒞:(ε − 𝕊:σ(ε)), b = E − 𝒞:Eᵖ,
# its own law giving σ and its tangent 𝕂 at each iterate.
function _grain_in_matrix(law, 𝒞, b, 𝕊, ε₀, prev, Δt, alg, ext, ::Type{T}) where {T}
    I6 = one(SMatrix{6, 6, T})
    return _root_with_payload(ε₀, alg, law.name, T) do ε
        σ, 𝕂, s = _grain_update(law, ε, prev, Δt, alg, ext)
        Jg = I6 - 𝒞 * (I6 - 𝕊 * 𝕂)
        (ε - 𝒞 * (ε - 𝕊 * σ) - b, Jg, (σ, 𝕂, s, Jg))
    end
end

function _aggregate_step(model::_InclusionModel, m, E₀, st, Δt, alg, ext)
    T = float(promote_type(eltype(E₀), typeof(model.k), eltype(st.σ), eltype(m.weights), _numeric_type(first(m.grains).p)))
    ℂ = SMatrix{6, 6, T}(_mandel(TensND.TensISO{3}(3model.k, 2model.μ)))
    𝕊 = SMatrix{6, 6, T}(_mandel(TensND.TensISO{3}(1 / (3model.k), 1 / (2model.μ))))
    𝕁 = SMatrix{6, 6, T}(_mandel(TensND.TensISO{3}(1, 0)))
    𝕂ᵈ = SMatrix{6, 6, T}(_mandel(TensND.TensISO{3}(0, 1)))
    I6 = one(SMatrix{6, 6, T})
    E = SVector{6, T}(E₀)
    ΔE = E - st.ε
    # The strain of each grain at the last outer iterate: the start of the next.
    guesses = [SVector{6, T}(s.g) + ΔE for s in st.grains]
    c(Ep) = _interaction(model, ℂ, E, Ep)
    outer(Eᵖ) = begin
        cJ, cK = c(Eᵖ)
        ∂c = ForwardDiff.jacobian(c, Eᵖ)
        𝒞 = cJ * 𝕁 + cK * 𝕂ᵈ
        b = E - 𝒞 * Eᵖ
        grains = map(m.grains, st.grains, eachindex(guesses)) do law, prev, i
            ε, out = _grain_in_matrix(law, 𝒞, b, 𝕊, guesses[i], prev, Δt, alg, ext, T)
            guesses[i] = ε
            (ε, out...)
        end
        F = sum(w * (ε - 𝕊 * σ) for (w, (ε, σ)) in zip(m.weights, grains)) - Eᵖ
        # dεᵖ_g/dEᵖ = (𝕀 − 𝕊𝕂_g) J_g⁻¹ (−𝒞 + 𝕁r_g ⊗ ∂c_J + 𝕂r_g ⊗ ∂c_K), r_g = ε_g − 𝕊σ_g − Eᵖ.
        JF = sum(
            m.weights[i] * (I6 - 𝕊 * 𝕂) * (Jg \ (-𝒞 + (𝕁 * (ε - 𝕊 * σ - Eᵖ)) * transpose(∂c[1, :]) + (𝕂ᵈ * (ε - 𝕊 * σ - Eᵖ)) * transpose(∂c[2, :])))
                for (i, (ε, σ, 𝕂, s, Jg)) in enumerate(grains)
        ) - I6
        (F, JF, grains)
    end
    Eᵖ₀ = SVector{6, T}(st.ε - 𝕊 * st.σ)
    Eᵖ, grains = _root_with_payload(outer, Eᵖ₀, alg, :Polycrystal, T)
    # The tangent: Σ = ℂ:(E − Eᵖ) at the root, Eᵖ following E through the
    # implicit-function theorem on F(Eᵖ, E) = ⟨εᵖ⟩ − Eᵖ.
    ∂c∂E = ForwardDiff.jacobian(e -> _interaction(model, ℂ, e, Eᵖ), E)
    cJ, cK = c(Eᵖ)
    𝒞 = cJ * 𝕁 + cK * 𝕂ᵈ
    ∂c∂Eᵖ = ForwardDiff.jacobian(c, Eᵖ)
    FE, FEᵖ = zero(I6), -I6
    for (w, (ε, σ, 𝕂, s, Jg)) in zip(m.weights, grains)
        P = (I6 - 𝕊 * 𝕂) / Jg
        r = ε - 𝕊 * σ - Eᵖ
        FE += w * P * (I6 + (𝕁 * r) * transpose(∂c∂E[1, :]) + (𝕂ᵈ * r) * transpose(∂c∂E[2, :]))
        FEᵖ += w * P * (-𝒞 + (𝕁 * r) * transpose(∂c∂Eᵖ[1, :]) + (𝕂ᵈ * r) * transpose(∂c∂Eᵖ[2, :]))
    end
    Σ = sum(w * σ for (w, (ε, σ)) in zip(m.weights, grains))
    return Σ, ℂ * (I6 + FEᵖ \ FE), [g[4] for g in grains]
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
