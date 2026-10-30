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

import ..Schemes

"""
    PolycrystalModel

How the grains of a [`Polycrystal`](@ref) share the macroscopic fields:
[`TaylorModel`](@ref), [`KronerModel`](@ref), [`BerveillerZaouiModel`](@ref),
[`HillModel`](@ref).
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
[the theory page](@ref ml-theory-polycrystals).
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
[the theory page](@ref ml-theory-polycrystals).
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
    HillModel(; reltol = 1e-6)

The self-consistent model of [hill1965](@citet) for elastoplastic aggregates:
each spherical grain lies in the aggregate itself, whose tangent stiffness
``\\mathbb{L}`` constrains it, ``\\dot{\\boldsymbol{\\sigma}}_g - \\dot{\\boldsymbol{\\Sigma}} = \\mathbb{L}^*:(\\dot{\\boldsymbol{E}} - \\dot{\\boldsymbol{\\varepsilon}}_g)``
with ``\\mathbb{L}^* = \\mathbb{P}(\\mathbb{L})^{-1} - \\mathbb{L}``, and
``\\mathbb{L} = \\langle\\mathbb{L}_g\\,\\mathbb{A}_g\\rangle`` is found self-consistently from
the tangents ``\\mathbb{L}_g`` of the grains, anisotropic as they make it. The
interaction is written on the increments of each step with the tangents at its
end. The Hill tensor ``\\mathbb{P}`` is integrated to the relative tolerance
`reltol`, which bounds the accuracy of the result. Where [`KronerModel`](@ref)
constrains every grain by the elastic aggregate, this one constrains it by the
aggregate as it flows. See [the theory page](@ref ml-theory-polycrystals).
"""
struct HillModel <: PolycrystalModel
    reltol::Float64
    HillModel(reltol::Real) = new(_quadrature_reltol(:HillModel, reltol))
end
HillModel(; reltol::Real = 1.0e-6) = HillModel(reltol)

# The tolerance of a quadrature inside a model.
_quadrature_reltol(name, reltol) = 0 < reltol < 1 ? Float64(reltol) : throw(DomainError(reltol, "$name: reltol must lie in (0, 1)"))

# The Hill tensor of a sphere in the medium of stiffness L (Kelvin–Mandel, in the
# canonical basis), to the relative tolerance reltol: by the package's entry for a
# compressible medium, by the given Green `kernel` otherwise.
_sphere_hill(L::StaticMatrix{6, 6, T}, reltol) where {T} = _canonical_mandel(
    Elasticity.hill_tensor(Elasticity.Ellipsoid(1.0), TensND.from_mandel(L); reltol, abstol = _hill_abstol(L, reltol)), T
)
_sphere_hill(L::StaticMatrix{6, 6, T}, reltol, kernel::F) where {T, F} = _canonical_mandel(
    Elasticity._hill_3d_aniso_nestedquadgk(Elasticity.Ellipsoid(1.0), TensND.from_mandel(L); kernel, reltol, abstol = _hill_abstol(L, reltol)), T
)
_hill_abstol(L, reltol) = reltol / maximum(abs ∘ TensND.primal, L)
# The Hill tensor reaches here with its dimension left to run time by the
# quadrature: the element type of the medium fixes the result.
_canonical_mandel(P, ::Type{T}) where {T} = SMatrix{6, 6, T, 36}(_mandel(TensND.canonical(P)))::SMatrix{6, 6, T, 36}

"""
    Polycrystal(grains, weights = equal; model = TaylorModel())

An aggregate of `grains`, a vector of laws of `MaterialLaws` of one type (the
same law with different orientations, typically a [`CrystalViscoplasticity`](@ref)
per grain), with the volume `weights` (equal by default), sharing the macroscopic
fields as `model` says. It runs in [`point_response`](@ref) and in the
finite-element coupling like a law; its state holds the state of every grain.
See [the theory page](@ref ml-theory-polycrystals).
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

The state of a [`Polycrystal`](@ref): the state of every grain (`grains`), the
macroscopic strain and stress it was reached at, and the tangent of the step
that reached it (`tangent`, zero in the initial state).
"""
struct PolycrystalState{S <: PointState, T} <: AbstractMaterialState
    grains::Vector{S}
    ε::SVector{6, T}
    σ::SVector{6, T}
    tangent::SMatrix{6, 6, T, 36}
end

function PolycrystalState(grains::Vector{<:PointState}, ε::StaticVector{6}, σ::StaticVector{6}, tangent::StaticMatrix{6, 6})
    T = promote_type(eltype(ε), eltype(σ), eltype(tangent))
    return PolycrystalState(grains, SVector{6, T}(ε), SVector{6, T}(σ), SMatrix{6, 6, T}(tangent))
end

# Without a tangent, as before it was kept: none known yet.
PolycrystalState(grains::Vector{<:PointState}, ε::StaticVector{6}, σ::StaticVector{6}) =
    PolycrystalState(grains, ε, σ, zero(SMatrix{6, 6, promote_type(eltype(ε), eltype(σ))}))

Constitutive.initial_state(m::Polycrystal) =
    PolycrystalState(map(initial_state, m.grains), zero(SVector{6, Float64}), zero(SVector{6, Float64}))
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
    return Σ, 𝕂, PolycrystalState(grains, SVector{6}(E), Σ, 𝕂)
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

# One grain under a linear interaction: its strain solves G ε + H σ(ε) = b, its
# own law giving σ and its tangent 𝕂 at each iterate, the Jacobian G + H 𝕂. In
# the matrix of Kröner and Berveiller–Zaoui, ε = b + 𝒞:(ε − 𝕊:σ(ε)) is
# G = 𝕀 − 𝒞, H = 𝒞𝕊; in Hill's medium, σ + 𝕃*:ε = b is G = 𝕃*, H = 𝕀.
function _grain_interacting(law, G, H, b, ε₀, prev, Δt, alg, ext, ::Type{T}; known = nothing) where {T}
    residual(ε) = begin
        σ, 𝕂, s = _known_or_update(known, law, ε, prev, Δt, alg, ext)
        (G * ε + H * σ - b, (σ, 𝕂, s, G + H * 𝕂))
    end
    return _root_with_payload(residual, (ε, out) -> out[4], ε₀, alg, law.name, T)
end

# The root of a grain in an outer iterate is where its solve starts in the next
# one, whose first evaluation would integrate the grain again at the same
# strain: half of the grain updates of an aggregate step. `known`, the strain of
# the last root and the update there, is read back instead when the strain is
# the same to the bit (`===`, partials of duals included): the same numbers,
# integrated once.
_known_or_update(::Nothing, law, ε, prev, Δt, alg, ext) = _grain_update(law, ε, prev, Δt, alg, ext)
_known_or_update(known::Tuple{E, U}, law, ε::E, prev, Δt, alg, ext) where {E, U} =
    known[1] === ε ? known[2] : _grain_update(law, ε, prev, Δt, alg, ext)
_known_or_update(::Tuple, law, ε, prev, Δt, alg, ext) = _grain_update(law, ε, prev, Δt, alg, ext)

# One slot per grain for its last root and the update there, typed by the
# update's inferred type, so that reading a slot stays a small union.
function _known_slots(m, st, ::Type{T}, Δt, alg, ext) where {T}
    U = Base.promote_op(_grain_update, eltype(m.grains), SVector{6, T}, eltype(st.grains), typeof(Δt), typeof(alg), typeof(ext))
    return Vector{Union{Nothing, Tuple{SVector{6, T}, U === Union{} ? Any : U}}}(nothing, length(m.grains))
end

# Σ_g w_g (𝕀 − 𝕊𝕂_g) J_g⁻¹ (A + 𝕁r_g ⊗ ∂c_J + 𝕂r_g ⊗ ∂c_K), r_g = ε_g − 𝕊σ_g − Eᵖ:
# the derivative of ⟨εᵖ⟩ along a change that moves b by A and the interaction
# coefficients by ∂c, shared by the outer iteration (A = −𝒞, the derivatives in
# Eᵖ) and the consistent tangent (A = 𝕀, the derivatives in E).
_plastic_strain_sensitivity(grains, weights, 𝕊, 𝕁, 𝕂ᵈ, I6, Eᵖ, A, ∂c) = sum(
    w * (I6 - 𝕊 * 𝕂) * (Jg \ (A + (𝕁 * (ε - 𝕊 * σ - Eᵖ)) * transpose(∂c[1, :]) + (𝕂ᵈ * (ε - 𝕊 * σ - Eᵖ)) * transpose(∂c[2, :])))
        for (w, (ε, σ, 𝕂, s, Jg)) in zip(weights, grains)
)

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
    known = _known_slots(m, st, T, Δt, alg, ext)
    c(Ep) = _interaction(model, ℂ, E, Ep)
    outer(Eᵖ) = begin
        cJ, cK = c(Eᵖ)
        𝒞 = cJ * 𝕁 + cK * 𝕂ᵈ
        b = E - 𝒞 * Eᵖ
        grains = map(m.grains, st.grains, eachindex(guesses)) do law, prev, i
            ε, out = _grain_interacting(law, I6 - 𝒞, 𝒞 * 𝕊, b, guesses[i], prev, Δt, alg, ext, T; known = known[i])
            guesses[i] = ε
            known[i] = (ε, (out[1], out[2], out[3]))
            (ε, out...)
        end
        (sum(w * (ε - 𝕊 * σ) for (w, (ε, σ)) in zip(m.weights, grains)) - Eᵖ, grains)
    end
    # dF/dEᵖ, F = ⟨εᵖ⟩ − Eᵖ.
    outer_jacobian(Eᵖ, grains) = begin
        cJ, cK = c(Eᵖ)
        _plastic_strain_sensitivity(grains, m.weights, 𝕊, 𝕁, 𝕂ᵈ, I6, Eᵖ, -(cJ * 𝕁 + cK * 𝕂ᵈ), ForwardDiff.jacobian(c, Eᵖ)) - I6
    end
    Eᵖ₀ = SVector{6, T}(st.ε - 𝕊 * st.σ)
    Eᵖ, grains = _root_with_payload(outer, outer_jacobian, Eᵖ₀, alg, :Polycrystal, T)
    # The tangent: Σ = ℂ:(E − Eᵖ) at the root, Eᵖ following E through the
    # implicit-function theorem on F(Eᵖ, E) = ⟨εᵖ⟩ − Eᵖ.
    FE = _plastic_strain_sensitivity(grains, m.weights, 𝕊, 𝕁, 𝕂ᵈ, I6, Eᵖ, I6, ForwardDiff.jacobian(e -> _interaction(model, ℂ, e, Eᵖ), E))
    FEᵖ = outer_jacobian(Eᵖ, grains)
    Σ = sum(w * σ for (w, (ε, σ)) in zip(m.weights, grains))
    return _aggregate_result(Σ, ℂ * (I6 + FEᵖ \ FE), [g[4] for g in grains], st, T)
end

# The result of a step in the element type T, which inference loses through the
# nested solves of the grains.
_state_type(::Type{PointState{N, M, S}}, ::Type{T}) where {N, M, S, T} = PointState{N, M, T}
_aggregate_result(Σ, 𝕂, grains, st, ::Type{T}) where {T} =
    (SVector{6, T}(Σ)::SVector{6, T}, SMatrix{6, 6, T, 36}(𝕂)::SMatrix{6, 6, T, 36}, grains::Vector{_state_type(eltype(st.grains), T)})

# ── Hill's self-consistent model ─────────────────────────────────────────────
#
# Equation (2) of Hill (1965), σ̇_g − Σ̇ = 𝕃*:(Ė − ε̇_g), written on the increments
# of the step, about the means of the previous state:
#   σ_g + 𝕃*:ε_g = Σ + 𝕃*:E + (σ_g,n − ⟨σ_n⟩) + 𝕃*:(ε_g,n − ⟨ε_n⟩),
# with the constraint tensor 𝕃* = ℙ(𝕃)⁻¹ − 𝕃 of a sphere in the tangent 𝕃 of
# the aggregate at the end of the step (equations 1 and 7). For a given 𝕃, Σ
# solves ⟨ε_g⟩ = E by Newton, the Jacobian ⟨(𝕃* + 𝕂_g)⁻¹⟩, 𝕂_g the consistent
# tangent of the grain on the step; averaged, the interaction then gives
# ⟨σ_g⟩ = Σ exactly. 𝕃 is the fixed point of the self-consistency in the form
# of equation 18, (𝕃* + 𝕃)⁻¹ = ⟨(𝕃* + 𝕂_g)⁻¹⟩, equivalent to 𝕃 = ⟨𝕂_g 𝔸_g⟩
# (equation 13), its successive substitutions contracting by about 0.05 per
# iteration where those of equation 13 contract by 0.35 (measured), iterated
# with Anderson's acceleration. 𝕃 is kept symmetric, as Hill's moduli are: the
# consistent tangent of a grain need not be, by the time discretization only.
function _aggregate_step(model::HillModel, m, E₀, st, Δt, alg, ext)
    T = float(promote_type(eltype(E₀), eltype(st.σ), eltype(m.weights), _numeric_type(first(m.grains).p)))
    I6 = one(SMatrix{6, 6, T})
    E = SVector{6, T}(E₀)
    σₙ = [SVector{6, T}(_fluxes(_with_external(law, ext), s.g, s.y)) for (law, s) in zip(m.grains, st.grains)]
    εₙ = [SVector{6, T}(s.g) for s in st.grains]
    Σₙ, Eₙ = sum(w * σ for (w, σ) in zip(m.weights, σₙ)), sum(w * ε for (w, ε) in zip(m.weights, εₙ))
    # The start: the tangent of the previous step, or, from the initial state,
    # the mean of the tangents of the grains held at their initial strains over
    # the step, which every grain can follow (a value test: the polycrystal is
    # numerical only).
    𝕃₀ = iszero(TensND.primal.(st.tangent)) ?
        sum(w * _grain_update(law, ε, s, Δt, alg, ext)[2] for (w, law, ε, s) in zip(m.weights, m.grains, εₙ, st.grains)) :
        SMatrix{6, 6, T}(st.tangent)
    # The stress of the last iterate: the start of the next solve on Σ.
    Σ₀ = Ref(Σₙ + 𝕃₀ * (E - Eₙ))
    # The strain of each grain at the last iterate: the start of the next.
    guesses = [ε + E - Eₙ for ε in εₙ]
    known = _known_slots(m, st, T, Δt, alg, ext)
    # 𝕃 converged to the accuracy of the Hill tensor: below it the change is the
    # noise of the quadrature.
    tol = 10 * model.reltol
    map𝕃(l) = begin
        𝕃 = _sym_matrix(l, Val(6))
        𝕃s = inv(_sphere_hill(𝕃, model.reltol)) - 𝕃
        residual(Σ) = begin
            grains = map(m.grains, st.grains, eachindex(guesses)) do law, prev, i
                b = Σ + 𝕃s * E + (σₙ[i] - Σₙ) + 𝕃s * (εₙ[i] - Eₙ)
                ε, out = _grain_interacting(law, 𝕃s, I6, b, guesses[i], prev, Δt, alg, ext, T; known = known[i])
                guesses[i] = ε
                known[i] = (ε, (out[1], out[2], out[3]))
                (ε, out...)
            end
            (sum(w * first(g) for (w, g) in zip(m.weights, grains)) - E, grains)
        end
        # The grains, started from their last strains, resolve them to alg.abstol,
        # that is Σ to 𝕃*·abstol: Σ is solved to ten times it.
        Σalg = LocalNewton(; abstol = 10 * alg.abstol * maximum(abs ∘ TensND.primal, 𝕃s), reltol = alg.reltol, maxiters = alg.maxiters)
        Σ, grains = _root_with_payload(residual, (Σ, grains) -> sum(w * inv(g[5]) for (w, g) in zip(m.weights, grains)), Σ₀[], Σalg, :Polycrystal, T)
        Σ₀[] = Σ
        # Equation 18: (𝕃* + 𝕃)⁻¹ = ⟨(𝕃* + 𝕂_g)⁻¹⟩, the mean already formed as
        # the Jacobian of the solve on Σ.
        (_sym_components(inv(sum(w * inv(g[5]) for (w, g) in zip(m.weights, grains))) - 𝕃s), grains)
    end
    converged(f, l) = maximum(abs ∘ TensND.primal, f) ≤ tol * maximum(abs ∘ TensND.primal, l) &&
        Schemes._sc_partials_converged(f, l; abstol = zero(tol), reltol = tol)
    l, solved = _anderson(map𝕃, _sym_components(𝕃₀), alg.maxiters, converged, :Polycrystal)
    # The tangent of the step at the constraint reached, ⟨(𝕃* + 𝕂_g)⁻¹⟩⁻¹ − 𝕃*,
    # is 𝕃 itself at the fixed point.
    return _aggregate_result(sum(w * g[2] for (w, g) in zip(m.weights, solved)), _sym_matrix(l, Val(6)), [g[4] for g in solved], st, T)
end
