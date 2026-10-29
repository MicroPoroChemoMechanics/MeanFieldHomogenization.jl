# =============================================================================
#  quasi_elastic.jl — the "quasi-elastic" affine formulation of Brenner,
#  Masson, Castelnau and Zaoui (2002) for a composite of two nonlinear
#  viscoelastic phases, matrix and inclusions.
#
#  At the time ζ, each phase is linearized about its current stress, ε̇ =
#  s:σ̇ + m:σ + ε̇⁰ with m = ∂g/∂σ(σ(ζ)); its creep function s + m t, transformed
#  and brought back by the direct inversion p = h(t)/t, becomes s + (t/h) m: at
#  t = ζ the phase is an elastic solid of that compliance with the eigenstrain
#  ε⁰(ζ) = ∫₀^ζ (g(σ) − m:σ) dθ of its actual history. The composite of the two
#  is thermoelastic in time; no Laplace–Carson inversion is left.
# =============================================================================

"""
    QuasiElasticComposite(rve, laws; relaxation_time = nothing, isotropic = true, options...)

The composite of a matrix and inclusions of nonlinear viscoelastic `laws` (a
`NamedTuple` by phase name, the matrix first, laws of Maxwell type without
internal variable, such as [`NortonViscoplasticity`](@ref)) by the
"quasi-elastic" affine formulation of [brenner2002](@citet). At the time
``\\zeta`` each phase is linearized about its current stress,
``\\mathbb{m} = \\partial\\boldsymbol{g}/\\partial\\boldsymbol{\\sigma}``, and the direct
inversion of the Laplace–Carson transform at ``p = h(\\zeta)/\\zeta`` makes it an
elastic solid with an eigenstrain,

```math
\\boldsymbol{\\varepsilon}_r(\\zeta) = \\Big(\\mathbb{s}_r + \\frac{\\zeta}{h(\\zeta)}\\,\\mathbb{m}_r\\Big):\\boldsymbol{\\sigma}_r(\\zeta)
  + \\int_0^\\zeta \\big(\\boldsymbol{g}_r(\\boldsymbol{\\sigma}_r) - \\mathbb{m}_r:\\boldsymbol{\\sigma}_r\\big)\\,\\mathrm{d}\\theta,
```

whose composite the Mori–Tanaka scheme homogenizes, as in
[`AffineEVPComposite`](@ref) but in time. ``h = 1`` (the default) is the
choice of the paper for creep, exact for a single phase in creep; with a
`relaxation_time` ``\\bar\\tau``, ``h(t) = t\\,e^{-t/\\bar\\tau}/(\\bar\\tau(1 - e^{-t/\\bar\\tau}))``,
its choice for relaxation and for a constant strain rate, ``\\bar\\tau`` an average
relaxation time of the phases. See [the theory page](@ref ml-theory-quasi-elastic).
"""
struct QuasiElasticComposite{L <: Tuple, F, S, H, O <: NamedTuple} <: AbstractMFHMaterial
    phases::L
    fractions::NTuple{2, F}
    shape::S
    relaxation_time::H
    isotropic::Bool
    options::O
end

function QuasiElasticComposite(rve::Schemes.RVE, laws::NamedTuple; relaxation_time::Union{Nothing, Real} = nothing, isotropic::Bool = true, options...)
    length(laws) == 2 || throw(ArgumentError("QuasiElasticComposite: a matrix and inclusions, two laws; got $(length(laws))"))
    Set(keys(laws)) == Set(Schemes.phase_names(rve)) ||
        throw(ArgumentError("one law per phase of the cell: the cell has $(Schemes.phase_names(rve)), the laws $(keys(laws))"))
    for (φ, law) in pairs(laws)
        _maxwell_type(law) || throw(ArgumentError("phase :$φ: the quasi-elastic formulation needs a nonlinear Maxwell law without internal variable (NortonViscoplasticity)"))
    end
    relaxation_time === nothing || relaxation_time > 0 ||
        throw(DomainError(relaxation_time, "QuasiElasticComposite: the relaxation time must be positive"))
    matrix, inclusion = keys(laws)
    fractions = (Schemes.volume_fraction(rve, matrix), Schemes.volume_fraction(rve, inclusion))
    return QuasiElasticComposite(Tuple(values(laws)), fractions, rve.phases[inclusion].geometry, relaxation_time, isotropic, NamedTuple(options))
end

# Laws of Maxwell type, s:σ̇ + g(σ), without internal variable of their own.
_maxwell_type(law) = false
_maxwell_type(::ResidualLaw{<:Any, typeof(_norton_residual)}) = true

# h(t) of the direct inversion p = h(t)/t (eqs. 12 and 32): 1, or the function
# that makes it exact for the relaxation of a phase of relaxation time τ̄, which
# tends to 1 at t = 0.
_inversion_factor(::Nothing, t) = one(t)
_inversion_factor(τ̄, t) = iszero(t) ? one(t / τ̄) : (t / τ̄) * exp(-t / τ̄) / -expm1(-t / τ̄)

"""
    QuasiElasticState

The state of a [`QuasiElasticComposite`](@ref): the macroscopic strain, stress
and time, and for each phase its average strain and stress, its inelastic
strain ``\\int_0^t \\boldsymbol{g}(\\boldsymbol{\\sigma})\\,\\mathrm{d}\\theta`` and the
integral of its stress ``\\int_0^t \\boldsymbol{\\sigma}\\,\\mathrm{d}\\theta``.
"""
struct QuasiElasticState{T} <: AbstractMaterialState
    ε::SVector{6, T}
    σ::SVector{6, T}
    t::T
    εr::NTuple{2, SVector{6, T}}
    σr::NTuple{2, SVector{6, T}}
    εin::NTuple{2, SVector{6, T}}
    Iσ::NTuple{2, SVector{6, T}}
end

function Constitutive.initial_state(::QuasiElasticComposite)
    z = zero(SVector{6, Float64})
    return QuasiElasticState(z, z, 0.0, (z, z), (z, z), (z, z), (z, z))
end
Constitutive.gradient_names(::QuasiElasticComposite) = (:ε,)
Constitutive.flux_names(::QuasiElasticComposite) = (:σ,)

function Constitutive.material_response(c::QuasiElasticComposite, gradients::NamedTuple, st::QuasiElasticState, Δt::Real; cache = nothing)
    Σ, 𝕂, new = _mandel_update(c, _mandel(gradients.ε), st, Δt; algorithm = LocalNewton(), external = (;))
    return MaterialResponse((σ = _tensor(Σ),), (σε = _tensor(𝕂),), new)
end

# The phase at the time ζ, at the stress σ: its quasi-elastic compliance and
# eigenstrain, the integrals of the history taken by the backward Euler scheme
# over the step, as the inelastic strain of the law is.
function _quasi_elastic_phase(law, σ, εin, Iσ, Δt, ζh)
    g = _norton_rate(σ, law.p)
    m = ForwardDiff.jacobian(s -> _norton_rate(s, law.p), σ)
    return inv(law.p.ℂ) + ζh * m, (εin + Δt * g) - m * (Iσ + Δt * σ)
end

# The stress of a phase at the strain ε: the root of q(σ):σ + ε⁰(σ) = ε, by the
# local Newton iteration of the laws. At long times q ≈ (ζ/h) m outgrows the
# elastic compliance in shear while the volume stays elastic: the stress is
# resolved to the condition number of q only, by which the tolerance of the
# iteration is relaxed.
function _quasi_elastic_stress(law, ε, σ₀, εin, Iσ, Δt, ζh, alg, ::Type{T}) where {T}
    residual(σ) = (qe = _quasi_elastic_phase(law, σ, εin, Iσ, Δt, ζh); first(qe) * σ + last(qe) - ε)
    q₀ = first(_quasi_elastic_phase(law, TensND.primal.(σ₀), TensND.primal.(εin), TensND.primal.(Iσ), TensND.primal(Δt), TensND.primal(ζh)))
    conditioned = LocalNewton(; abstol = alg.abstol, reltol = alg.reltol * cond(TensND.primal.(q₀)), maxiters = alg.maxiters)
    return _root(residual, TensND.primal.(σ₀), conditioned, law.name, T)
end

function _mandel_update(c::QuasiElasticComposite, E₀, st::QuasiElasticState, Δt; algorithm, external, cache = nothing)
    Δt ≥ 0 || throw(ArgumentError("QuasiElasticComposite: a step of negative duration, Δt = $Δt"))
    T = float(promote_type(eltype(E₀), typeof(Δt), eltype(st.σ), _numeric_type(c.phases[1].p), _numeric_type(c.phases[2].p), typeof(first(c.fractions))))
    laws = map(law -> _with_external(law, external), c.phases)
    E = SVector{6, T}(E₀)
    ζ = st.t + Δt
    ζh = ζ / _inversion_factor(c.relaxation_time, ζ)
    f₀, f₁ = c.fractions
    # The stresses of the phases at the strains of a trial x, then the strain of
    # the inclusions the comparison composite gives.
    stresses(x, E) = map(
        (law, ε, σ, εin, Iσ) -> _quasi_elastic_stress(law, ε, σ, εin, Iσ, Δt, ζh, algorithm, promote_type(T, eltype(x), eltype(E))),
        laws, ((E - f₁ * x) / f₀, x), st.σr, st.εin, st.Iσ
    )
    G(x, E) = begin
        σx = stresses(x, E)
        qe = map((law, s, εin, Iσ) -> _quasi_elastic_phase(law, s, εin, Iσ, Δt, ζh), laws, σx, st.εin, st.Iσ)
        A, a = _carson_concentration(c, first(qe[1]), first(qe[2]), last(qe[1]), last(qe[2]))
        (A * E + a, σx)
    end
    # The fixed point by accelerated substitutions, as for AffineEVPComposite.
    converged = _fixed_point_test(algorithm)
    x₀ = SVector{6, T}(st.εr[2] + (E - st.ε))
    x, σ = _anderson(z -> G(z, E), x₀, algorithm.maxiters, converged, :QuasiElasticComposite)
    # The tangent: x follows E by the implicit-function theorem on x = G(x, E),
    # and each phase stress its strain, dσ/dε the inverse of the derivative of
    # its quasi-elastic relation.
    J = _directional_jacobian(z -> first(G(z[SOneTo(6)], z[SVector{6}(7:12)])), vcat(x, E))
    dx = (one(J[SOneTo(6), SOneTo(6)]) - J[SOneTo(6), SOneTo(6)]) \ J[SOneTo(6), SVector{6}(7:12)]
    𝕂₀, 𝕂₁ = map(
        (law, s, εin, Iσ) -> inv(ForwardDiff.jacobian(z -> (qe = _quasi_elastic_phase(law, z, εin, Iσ, Δt, ζh); first(qe) * z + last(qe)), s)),
        laws, σ, st.εin, st.Iσ
    )
    Σ = f₀ * σ[1] + f₁ * σ[2]
    𝕂 = 𝕂₀ * (one(𝕂₀) - f₁ * dx) + f₁ * 𝕂₁ * dx
    εr = ((E - f₁ * x) / f₀, x)
    εin = map((law, s, e) -> e + Δt * _norton_rate(s, law.p), laws, σ, st.εin)
    Iσ = map((s, I) -> I + Δt * s, σ, st.Iσ)
    new = QuasiElasticState(E, Σ, T(ζ), map(SVector{6, T}, εr), map(SVector{6, T}, σ), map(SVector{6, T}, εin), map(SVector{6, T}, Iσ))
    return SVector{6, T}(Σ)::SVector{6, T}, SMatrix{6, 6, T, 36}(𝕂)::SMatrix{6, 6, T, 36}, new::QuasiElasticState{T}
end
