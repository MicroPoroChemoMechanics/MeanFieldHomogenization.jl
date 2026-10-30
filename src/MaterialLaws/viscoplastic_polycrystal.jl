# =============================================================================
#  viscoplastic_polycrystal.jl — the steady flow of an aggregate of
#  rigid–viscoplastic crystals, under the Taylor model or the tangent
#  self-consistent model of Molinari, Canova and Ahzi (1987).
#
#  A grain of power-law slip is homogeneous of degree n in its stress: its
#  tangent viscosity is n times smaller than its secant one. The tangent model
#  lets each grain interact with a homogeneous matrix through the tangent
#  viscosity of the aggregate, taken isotropic as in the paper's computations:
#  for a sphere in an incompressible isotropic matrix the interaction law is
#  s_g − S = −3η_t (d_g − D), closed by S = ⟨s_g⟩.
# =============================================================================

import ..Elasticity

"""
    PowerLawCrystal(; γ̇₀, τ₀, n, lattice = :fcc, frame = CanonicalBasis)

A rigid–viscoplastic single crystal: on each slip system ``s`` of `lattice`
(see [`slip_systems`](@ref)), oriented by `frame`, the slip rate is a power of
the resolved shear stress,

```math
\\dot\\gamma_s = \\dot\\gamma_0\\left|\\frac{\\tau_s}{\\tau_0}\\right|^{n}\\operatorname{sign}\\tau_s,
\\qquad
\\boldsymbol{d} = \\sum_s \\dot\\gamma_s\\,\\boldsymbol{P}_s ,
```

``\\boldsymbol{P}_s`` the Schmid tensors, ``\\tau_s = \\boldsymbol{\\sigma}:\\boldsymbol{P}_s``.
The strain rate is deviatoric and homogeneous of degree ``n`` in the stress.
The grain of a [`ViscoplasticPolycrystal`](@ref).
"""
struct PowerLawCrystal{T <: Real, N, M <: SMatrix{6, N}}
    P::M
    γ̇₀::T
    τ₀::T
    n::T
end

function PowerLawCrystal(; γ̇₀, τ₀, n, lattice::Symbol = :fcc, frame::TensND.OrthonormalBasis{3} = TensND.CanonicalBasis{3, Float64}())
    # The grains are solved iteratively: numbers that can be compared only.
    TensND.is_hard_numeric(promote_type(typeof(γ̇₀), typeof(τ₀), typeof(n))) ||
        throw(ArgumentError("PowerLawCrystal: numeric parameters only (the grains are solved iteratively); got $(typeof(γ̇₀)), $(typeof(τ₀)), $(typeof(n))"))
    γ̇₀ > 0 || throw(DomainError(γ̇₀, "PowerLawCrystal: the reference slip rate must be positive"))
    τ₀ > 0 || throw(DomainError(τ₀, "PowerLawCrystal: the reference shear stress must be positive"))
    n ≥ 1 || throw(DomainError(n, "PowerLawCrystal: the exponent n must be at least 1"))
    systems = slip_systems(lattice)
    P = SMatrix{6, length(systems)}(_schmid_matrix(systems, SMatrix{3, 3}(TensND.vecbasis(frame))))
    return PowerLawCrystal(P, promote(γ̇₀, τ₀, n)...)
end

"""
    crystal_strain_rate(c::PowerLawCrystal, σ) -> SVector{6}

The strain rate of the crystal under the stress `σ` (Kelvin–Mandel components).
"""
crystal_strain_rate(c::PowerLawCrystal, σ) = c.P * _slip_rates(c, transpose(c.P) * σ)

_slip_rates(c::PowerLawCrystal, τ) = c.γ̇₀ .* abs.(τ ./ c.τ₀) .^ c.n .* sign.(τ)

"""
    TangentModel()

The tangent self-consistent model of [molinari1987](@citet), with the isotropic
matrix of their computations: each spherical grain interacts with a matrix of
the tangent viscosity of the aggregate, ``\\eta_t = \\eta_s/n``, ``\\eta_s`` its
secant viscosity. See [the theory page](@ref ml-theory-polycrystals).
"""
struct TangentModel <: PolycrystalModel end

"""
    ViscoplasticPolycrystal(grains, weights = equal)

An aggregate of [`PowerLawCrystal`](@ref) grains of one exponent, with the volume
`weights`, in steady viscoplastic flow: [`flow_stress`](@ref) gives its stress
under a strain rate.
"""
struct ViscoplasticPolycrystal{C <: PowerLawCrystal, W <: Real}
    grains::Vector{C}
    weights::Vector{W}
    # The checks sit in the inner constructor: an outer one would be shadowed
    # by the generated constructor for vectors of concrete types.
    function ViscoplasticPolycrystal{C, W}(grains::Vector{C}, weights::Vector{W}) where {C <: PowerLawCrystal, W <: Real}
        isempty(grains) && throw(ArgumentError("a polycrystal needs at least one grain"))
        length(weights) == length(grains) || throw(ArgumentError("$(length(grains)) grains and $(length(weights)) weights"))
        TensND.is_hard_numeric(W) || throw(ArgumentError("the weights must be numbers that can be compared; got $W"))
        all(>(0), weights) && isapprox(sum(weights), 1; rtol = 1.0e-12) || throw(ArgumentError("the weights must be positive and add up to one"))
        all(g -> g.n == first(grains).n, grains) || throw(ArgumentError("the grains must share one exponent n: the tangent of the aggregate is then n times smaller than its secant"))
        return new{C, W}(grains, weights)
    end
end

ViscoplasticPolycrystal(grains::AbstractVector{<:PowerLawCrystal}, weights::AbstractVector{<:Real} = fill(1 / max(length(grains), 1), length(grains))) =
    ViscoplasticPolycrystal{eltype(grains), eltype(weights)}(collect(grains), collect(weights))

"""
    flow_stress(p::ViscoplasticPolycrystal, D; model = TangentModel(), algorithm = LocalNewton())
        -> (σ = Σ, grains = σ_g)

The stress deviator ``\\boldsymbol{\\Sigma}`` of the aggregate under the deviatoric
strain rate `D` (Kelvin–Mandel components), and the stress of every grain, by
the [`TaylorModel`](@ref) (every grain at the strain rate `D`) or the
[`TangentModel`](@ref). Iterative: numbers that can be compared only.
"""
function flow_stress(p::ViscoplasticPolycrystal, D::AbstractVector; model::PolycrystalModel = TangentModel(), algorithm::LocalNewton = LocalNewton())
    TensND.is_hard_numeric(eltype(D)) || throw(ArgumentError("flow_stress solves the grains iteratively: numeric strain rates only; got $(eltype(D))"))
    D6 = SVector{6}(D)
    abs(trace_mandel(TensND.primal.(D6))) <= 1.0e-12 * maximum(abs ∘ TensND.primal, D6) ||
        throw(ArgumentError("flow_stress: the strain rate of rigid–viscoplastic grains is deviatoric; got the trace $(trace_mandel(D6))"))
    # At rest every grain is: no iteration (nor any equivalent stress to divide by).
    iszero(TensND.primal.(D6)) && return (σ = zero(D6) .* zero(_flow_type(p, D6)), grains = [zero(D6) .* zero(_flow_type(p, D6)) for _ in p.grains])
    return _flow_stress(model, p, D6, algorithm)
end

const _𝕁 = SMatrix{6, 6, Float64}(_mandel(TensND.TensISO{3}(1, 0)))
const _𝕂 = SMatrix{6, 6, Float64}(_mandel(TensND.TensISO{3}(0, 1)))

# The stress of a grain under the load b and the interaction compliance A (a
# symmetric operator on the deviators, positive or zero): the minimum of the
# strictly convex φ*(σ) + ½ σ:A:σ + ½ σ:𝕁:σ − σ:b, where its gradient
# d(σ) + Aσ + 𝕁σ − b vanishes. Newton's method for convex
# minimization with Armijo's backtracking line search (Boyd and Vandenberghe,
# Convex Optimization, 2004, §9.5): for large n the systems that barely slip
# make the Hessian nearly singular, which a trust-region root finder does not
# survive while the descent on φ does. Converged on the residual (a strain
# rate) relative to the load; then the steps that attach exact derivatives.
function _grain_flow(c::PowerLawCrystal, b, A, σ₀, alg::LocalNewton, ::Type{T}) where {T}
    cv, bv, av = _primal_crystal(c), TensND.primal.(b), TensND.primal.(A)
    σ = SVector{6, Float64}(TensND.primal.(σ₀))
    for it in 1:alg.maxiters
        r = _grain_gradient(cv, σ, bv, av)
        H = ForwardDiff.jacobian(s -> _grain_gradient(cv, s, bv, av), σ)
        # The tolerance relative to the load, plus the residual a stress known to
        # rounding can reach, ‖H‖·eps·|σ|: for large n the slip rates are so steep
        # that this floor approaches the tolerance, and a machine whose rounding
        # differs would otherwise iterate on noise.
        if maximum(abs, r) <= alg.reltol * maximum(abs, bv) + 8 * eps(Float64) * opnorm(H, Inf) * maximum(abs, σ)
            σT = SVector{6, T}(σ)
            for _ in 1:_derivative_steps(T)
                σT = σT - ForwardDiff.jacobian(s -> _grain_gradient(c, s, b, A), σT) \ _grain_gradient(c, σT, b, A)
            end
            return σT
        end
        δ = H \ r
        all(isfinite, δ) || throw(LocalIntegrationError(:PowerLawCrystal, it, NaN, "singular or non-finite Hessian"))
        # Armijo's sufficient decrease of φ; near the minimum, where that
        # decrease falls below rounding, a step that lowers the residual.
        φ₀, slope, λ = _grain_objective(cv, σ, bv, av), transpose(r) * δ, 1.0
        while !(
                _grain_objective(cv, σ - λ * δ, bv, av) <= φ₀ - 1.0e-4 * λ * slope ||
                    maximum(abs, _grain_gradient(cv, σ - λ * δ, bv, av)) < maximum(abs, r)
            ) && λ > 1.0e-12
            λ /= 2
        end
        σ = σ - λ * δ
    end
    throw(LocalIntegrationError(:PowerLawCrystal, alg.maxiters, NaN, "no convergence"))
end

_grain_gradient(c, σ, b, A) = crystal_strain_rate(c, σ) + A * σ + _𝕁 * σ - b
_grain_objective(c, σ, b, A) =
    sum(c.γ̇₀ * c.τ₀ / (c.n + 1) .* abs.((transpose(c.P) * σ) ./ c.τ₀) .^ (c.n + 1)) +
    transpose(σ) * (A * σ) / 2 + transpose(σ) * (_𝕁 * σ) / 2 - transpose(σ) * b

_primal_crystal(c::PowerLawCrystal) = PowerLawCrystal(TensND.primal.(c.P), TensND.primal(c.γ̇₀), TensND.primal(c.τ₀), TensND.primal(c.n))

_flow_type(p, D) = float(promote_type(eltype(D), eltype(p.weights), typeof(first(p.grains).γ̇₀), eltype(first(p.grains).P)))

# Every grain at the strain rate D: the Taylor model.
function _flow_stress(::TaylorModel, p, D, alg)
    T = _flow_type(p, D)
    grains = [_grain_flow(c, D, zero(SMatrix{6, 6, T}), _start_along(c, D), alg, T) for c in p.grains]
    return (σ = sum(w * σ for (w, σ) in zip(p.weights, grains)), grains = grains)
end

# A start for the iteration on a grain: the direction of its stress if it were
# linear (n = 1), σ̂ = (γ̇₀/τ₀ P Pᵀ + 𝕁)⁻¹ D, scaled so that the dissipation
# σ:d(σ) equals σ:D; the strain rate being homogeneous of degree n,
# d(s σ̂) = sⁿ d(σ̂), which gives s in closed form.
function _start_along(c::PowerLawCrystal, D)
    cv, Dv = _primal_crystal(c), TensND.primal.(D)
    σ̂ = (cv.γ̇₀ / cv.τ₀ * cv.P * transpose(cv.P) + _𝕁) \ Dv
    return σ̂ * ((transpose(σ̂) * Dv) / (transpose(σ̂) * crystal_strain_rate(cv, σ̂)))^(1 / cv.n)
end

# The tangent self-consistent model: the grain stresses s_g solve
# d_g(s_g) + a 𝕂 s_g = D + a S with a = 1/(3η_t) = n D_eq / Σ_eq, and S = ⟨s_g⟩:
# a fixed point on the six components of S, solved by Newton's method.
function _flow_stress(::TangentModel, p, D, alg)
    T = _flow_type(p, D)
    n = first(p.grains).n
    taylor = _flow_stress(TaylorModel(), _primal_polycrystal(p), TensND.primal.(D), alg)
    guesses = copy(taylor.grains)
    residual(p, D) = S -> begin
        a = n * equivalent_strain(D) / von_mises(S)
        b = D + a * (_𝕂 * S)
        grains = map(eachindex(p.grains)) do i
            # The type from p and D, not a captured type (a closure field of
            # type DataType, which inference cannot see through).
            σ = _grain_flow(p.grains[i], b, a * _𝕂, guesses[i], alg, promote_type(_flow_type(p, D), eltype(S)))
            guesses[i] = TensND.primal.(σ)
            σ
        end
        sum(w * σ for (w, σ) in zip(p.weights, grains)) - S, grains
    end
    S, grains = _root_with_payload(
        residual(p, D), (S, _) -> ForwardDiff.jacobian(s -> first(residual(p, D)(s)), S), SVector{6, T}(taylor.σ), _outer_algorithm(alg), :ViscoplasticPolycrystal, T;
        values = _values_pair(residual, p, D)
    )
    # The Newton iteration runs through closures whose results inference does
    # not follow: their types, known here, are asserted.
    return (σ = S::SVector{6, T}, grains = grains::Vector{SVector{6, T}})
end

# The tolerance of a fixed point over the grains, three orders above that of
# the grains themselves (an inexact Newton iteration): each grain is solved to
# its tolerance from the stress it last reached, so the residual of the fixed
# point carries that much noise, on which a tighter iteration would never stop.
_outer_algorithm(alg::LocalNewton; floor = zero(alg.reltol)) =
    LocalNewton(; abstol = alg.abstol, reltol = max(1000 * alg.reltol, floor), maxiters = alg.maxiters)

# A polycrystal and a strain rate reduced to their values, and the residual of a
# model on them with its Jacobian: what the iteration on the values works with.
_primal_polycrystal(p::ViscoplasticPolycrystal) = ViscoplasticPolycrystal(map(_primal_crystal, p.grains), TensND.primal.(p.weights))
function _values_pair(residual, p, D)
    Fv = residual(_primal_polycrystal(p), TensND.primal.(D))
    return (Fv, (x, _) -> ForwardDiff.jacobian(y -> first(Fv(y)), x))
end

"""
    AnisotropicTangentModel(; reltol = 1e-10)

The tangent self-consistent model with the anisotropic matrix of
[lebensohnTome1993](@citet): each spherical grain interacts with the aggregate
itself, linearized at its tangent, whose secant compliance ``\\mathbb{M}`` on the
deviators is found self-consistently, ``\\mathbb{M} = \\langle\\mathbb{M}_g\\,\\mathbb{B}_g\\rangle``.
The interaction goes through the Hill tensor of the incompressible anisotropic
matrix, integrated to the relative tolerance `reltol`, which bounds the accuracy
of the result: the fixed point is converged to ten times it. Where
[`TangentModel`](@ref) takes the matrix isotropic, as [molinari1987](@citet)
computed it, this one keeps its anisotropy. See
[the theory page](@ref ml-theory-polycrystals).
"""
struct AnisotropicTangentModel <: PolycrystalModel
    reltol::Float64
    AnisotropicTangentModel(reltol::Real) = new(_quadrature_reltol(:AnisotropicTangentModel, reltol))
end
AnisotropicTangentModel(; reltol::Real = 1.0e-10) = AnisotropicTangentModel(reltol)

# An orthonormal basis of the deviators in Kelvin–Mandel components, and the
# symmetric operators on the deviators by their 15 components in it.
const _DEV_BASIS = SMatrix{6, 5, Float64}(
    1 / sqrt(2), -1 / sqrt(2), 0, 0, 0, 0,
    1 / sqrt(6), 1 / sqrt(6), -2 / sqrt(6), 0, 0, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 1
)
_dev_operator(m::SVector{15}) = _DEV_BASIS * _sym_matrix(m, Val(5)) * transpose(_DEV_BASIS)
_dev_components(M) = _sym_components(transpose(_DEV_BASIS) * M * _DEV_BASIS)

# The secant compliance of a grain at its stress: d = Σₛ (γ̇₀/τ₀)|τₛ/τ₀|ⁿ⁻¹ τₛ Pₛ.
_grain_secant(c::PowerLawCrystal, σ) =
    c.P * Diagonal(c.γ̇₀ / c.τ₀ .* abs.((transpose(c.P) * σ) ./ c.τ₀) .^ (c.n - 1)) * transpose(c.P)

# The self-consistent fixed point on the secant compliance M of the aggregate,
# its 15 components on the deviators, started from the isotropic matrix of
# the tangent model. At the root ⟨M_g B_g⟩ = M, so that ⟨d_g⟩ = D and ⟨σ_g⟩ = Σ.
function _flow_stress(model::AnisotropicTangentModel, p, D, alg)
    T = _flow_type(p, D)
    n = first(p.grains).n
    iso = _flow_stress(TangentModel(), _primal_polycrystal(p), TensND.primal.(D), alg)
    guesses = copy(iso.grains)
    η = von_mises(iso.σ) / (3 * equivalent_strain(TensND.primal.(D)))
    residual(p, D) = m -> begin
        Tm = promote_type(_flow_type(p, D), eltype(m))
        M = _dev_operator(m)
        L = TensND.deviatoric_pinv(M)
        # The interaction compliance through the tangent matrix L/n, whose Hill
        # tensor is n P(L): M̃ = ((nP)⁺ − L/n)⁺ = n (P⁺ − L)⁺.
        M̃ = n * TensND.deviatoric_pinv(TensND.deviatoric_pinv(_sphere_hill(L, model.reltol, Elasticity._sym3_inv_acoustic_incompressible)) - L)
        Σ = L * D
        b = D + M̃ * Σ
        grains = map(eachindex(p.grains)) do i
            σ = _grain_flow(p.grains[i], b, M̃, guesses[i], alg, Tm)
            guesses[i] = TensND.primal.(σ)
            σ
        end
        B = sum(
            w * _grain_secant(c, σ) * TensND.deviatoric_pinv(_grain_secant(c, σ) + M̃) * (M + M̃)
                for (w, c, σ) in zip(p.weights, p.grains, grains)
        )
        _dev_components(B) - m, (Σ, grains)
    end
    m₀ = _dev_components(SMatrix{6, 6, T}(_𝕂 / (2η)))
    # Converged to the accuracy of the Hill tensor too: below it the residual is
    # the noise of the quadrature.
    outer = _outer_algorithm(alg; floor = 10 * model.reltol)
    _, (Σ, grains) = _root_with_payload(
        residual(p, D), (m, _) -> ForwardDiff.jacobian(x -> first(residual(p, D)(x)), m), m₀, outer, :ViscoplasticPolycrystal, T;
        values = _values_pair(residual, p, D)
    )
    return (σ = SVector{6, T}(Σ)::SVector{6, T}, grains = grains::Vector{SVector{6, T}})
end
