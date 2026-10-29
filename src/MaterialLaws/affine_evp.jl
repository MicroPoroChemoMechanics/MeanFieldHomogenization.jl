# =============================================================================
#  affine_evp.jl — the affine formulation of Pierard and Doghri (2006) for a
#  composite of two elasto-viscoplastic phases, matrix and inclusions.
#
#  At the end τ of a step, the rates of each phase, ε̇ = S:σ̇ + ε̇in(σ, p) and
#  ṗ = ṗ(σ, p), are linearized about the current stress and accumulated strain.
#  The linearized phase is a linear viscoelastic solid with an eigenstrain that
#  makes it follow its actual history up to τ; in the Laplace–Carson domain it
#  is a thermoelastic solid, ε* = S*:σ* + J, and the Mori–Tanaka scheme gives
#  the strain of the inclusions, A*:E* + a*. Both are brought back to time by
#  Schapery's collocation on fixed relaxation times, and the average strain of
#  the inclusions at τ must be the one the phases were integrated at: a fixed
#  point on it.
#
#  The history enters through integrals ∫₀^τ f(t) e^{−st} dt at the collocation
#  points, of the stress and the inelastic strain rate of each phase, and through
#  the exponential memories of the macroscopic strain rate, all updated step by
#  step: the state keeps a fixed number of them, whatever the length of the
#  history.
# =============================================================================

"""
    AffineEVPComposite(rve, laws; relaxation_times, isotropic = true, options...)

The composite of a matrix and inclusions of elasto-viscoplastic `laws` (a
`NamedTuple` by phase name, the matrix first, laws in rate form such as
[`PerzynaViscoplasticity`](@ref) or [`NortonViscoplasticity`](@ref)) by the
affine formulation of [pierardDoghri2006](@citet). At the end of each step, the
rates of each phase are linearized about its current stress and accumulated
strain, which makes it a linear viscoelastic solid with an eigenstrain that
follows its actual history; in the Laplace–Carson domain the composite of the
two is thermoelastic, and the Mori–Tanaka scheme gives the average strain of
the inclusions,

```math
\\bar{\\boldsymbol{\\varepsilon}}_1^{*}(s) = \\mathbb{A}^{*}(s):\\boldsymbol{E}^{*}(s) + \\boldsymbol{a}^{*}(s),
```

brought back to time by the collocation of the paper (its appendix B) on the
`relaxation_times`, which
should span the time scales of the phases and of the loading. The average
strain of the inclusions at the end of the step must be the one the phases
were integrated at: a fixed point, solved by Newton's method. The macroscopic
stress is the average of the stresses of the phases. With `isotropic = true`,
as in the paper, the Eshelby tensor of the inclusions is computed in the
isotropic part of the modulus of the matrix. `rve` holds the two phases, the
shape of the inclusions and the fractions; `options` go to the Hill tensor.
See [the theory page](@ref ml-theory-affine-evp).
"""
struct AffineEVPComposite{L <: Tuple, F, S, Θ, G, O <: NamedTuple} <: AbstractMFHMaterial
    phases::L
    fractions::NTuple{2, F}
    shape::S
    θ::Θ
    points::Θ
    collocation::G
    isotropic::Bool
    options::O
end

function AffineEVPComposite(rve::Schemes.RVE, laws::NamedTuple; relaxation_times::AbstractVector{<:Real}, isotropic::Bool = true, options...)
    length(laws) == 2 || throw(ArgumentError("AffineEVPComposite: a matrix and inclusions, two laws; got $(length(laws))"))
    Set(keys(laws)) == Set(Schemes.phase_names(rve)) ||
        throw(ArgumentError("one law per phase of the cell: the cell has $(Schemes.phase_names(rve)), the laws $(keys(laws))"))
    for (φ, law) in pairs(laws)
        _rate_form(law) || throw(ArgumentError("phase :$φ: the affine formulation needs a law in rate form (PerzynaViscoplasticity, NortonViscoplasticity)"))
    end
    θ = sort(collect(float.(relaxation_times)))
    length(θ) ≥ 2 && first(θ) > 0 ||
        throw(ArgumentError("AffineEVPComposite: at least two positive relaxation times; got $relaxation_times"))
    matrix, inclusion = keys(laws)
    fractions = (Schemes.volume_fraction(rve, matrix), Schemes.volume_fraction(rve, inclusion))
    # Schapery's collocation at s_l = 1/θ_l: f*(s_l) − A − B/s_l = Σ_k b_k/(1 + s_l θ_k),
    # whose matrix is kept inverted, to solve for tensors of any size.
    collocation = inv([1 / (1 + θk / θl) for θl in θ, θk in θ])
    # The history is carried at the collocation points, and at s₀ and 2s₀, far
    # below them, for the limit at s = 0.
    s₀ = 1.0e-3 / last(θ)
    points = vcat(1 ./ θ, s₀, 2s₀)
    shape = rve.phases[inclusion].geometry
    return AffineEVPComposite(Tuple(values(laws)), fractions, shape, θ, points, collocation, isotropic, NamedTuple(options))
end

# ── Phase laws in rate form ─────────────────────────────────────────────────

# The inelastic strain rate and the rate of the accumulated strain of a phase
# at its stress σ and state y: `nothing` for a law without accumulated strain.
_rate_form(law) = false
_rate_form(::ResidualLaw{<:Any, typeof(_perzyna_residual)}) = true
_rate_form(::ResidualLaw{<:Any, typeof(_norton_residual)}) = true
_inelastic_rates(law::ResidualLaw{<:Any, typeof(_perzyna_residual)}, σ, y) = _perzyna_rates(σ, y[7], law.p)
_inelastic_rates(law::ResidualLaw{<:Any, typeof(_norton_residual)}, σ, y) = (_norton_rate(σ, law.p), nothing)

# The linearization of the rates of a phase at (σ, p): ε̇in, ṗ and the four
# derivatives m = ∂ε̇in/∂σ, n = ∂ε̇in/∂p, l = ∂ṗ/∂σ, q = ∂ṗ/∂p (eqs. 4–9). A law
# without accumulated strain has n = 0 and l = 0; q is then set to −1, a value
# that never enters, n being zero, but keeps s − q away from zero.
function _linearized_rates(law, σ, y)
    ε̇, ṗ = _inelastic_rates(law, σ, y)
    return _linearized_rates(law, σ, y, ε̇, ṗ)
end
function _linearized_rates(law, σ, y, ε̇, ::Nothing)
    T = eltype(ε̇)
    m = ForwardDiff.jacobian(s -> first(_inelastic_rates(law, s, y)), σ)
    return (; ε̇, ṗ = zero(T), m, n = zero(SVector{6, T}), l = zero(SVector{6, T}), q = -one(T))
end
function _linearized_rates(law, σ, y, ε̇, ṗ)
    ∂σ = ForwardDiff.jacobian(s -> (r = _inelastic_rates(law, s, y); vcat(r[1], r[2])), σ)
    ∂p = ForwardDiff.derivative(κ -> (r = _inelastic_rates(law, σ, vcat(y[SOneTo(6)], SVector(κ))); vcat(r[1], r[2])), y[7])
    return (; ε̇, ṗ, m = ∂σ[SOneTo(6), :], n = ∂p[SOneTo(6)], l = ∂σ[7, :], q = ∂p[7])
end

# ── The phase in the Laplace–Carson domain ──────────────────────────────────
#
# Linearized at τ for every time, the phase creeps as
#   ε̇ = S:σ̇ + m:σ + n ∫₀ᵗ e^{q(t−u)} l:σ(u) du + ε̇⁰,
# so that S*(s) = S + m/s + n⊗l/(s(s − q)) (eq. 47). The eigenstrain ε⁰ is what
# makes it follow the actual inelastic rate up to τ and the linearized one
# after; from a rest state its transform is
#   J(s) = I_in(s) − m:I_σ(s) + n (l:I_σ(s))/(q − s)
#          + (ε̇in − m:σ)(τ) e^{−sτ}/s + n (ṗ − l:σ)(τ) e^{−sτ}/(s(s − q)),
# I_f(s) = ∫₀^τ f(t) e^{−st} dt: the terms of the past through e^{q(t−u)} cancel
# between before and after τ. Equation 48 of the paper is s J(s), the transform
# of the rate of the eigenstrain.

_carson_compliance(S, lin, s) = S + lin.m / s + (lin.n * transpose(lin.l)) / (s * (s - lin.q))

function _carson_eigenstrain(lin, σ, Iσ, Iin, τ, s)
    e = exp(-s * τ)
    return Iin - lin.m * Iσ + lin.n * ((transpose(lin.l) * Iσ) / (lin.q - s)) +
        (lin.ε̇ - lin.m * σ) * (e / s) + lin.n * ((lin.ṗ - transpose(lin.l) * σ) * e / (s * (s - lin.q)))
end

# ∫ over a step of length Δ starting at t₀ of a function linear from α to β,
# times e^{−st}: Δ e^{−s t₀} [α φ₁(a) + (β − α) φ₂(a)], a = sΔ, with
# φ₁(a) = (1 − e^{−a})/a and φ₂(a) = (1 − (1 + a)e^{−a})/a², by their series
# where the closed forms cancel.
_φ₁(a) = a < 1.0e-4 ? 1 - a / 2 + a^2 / 6 : -expm1(-a) / a
_φ₂(a) = a < 1.0e-3 ? 1 / 2 - a / 3 + a^2 / 8 - a^3 / 30 : (-expm1(-a) - a * exp(-a)) / a^2
_step_integral(α, β, t₀, Δ, s) = (Δ * exp(-s * t₀)) * (α * _φ₁(s * Δ) + (β - α) * _φ₂(s * Δ))

# ── The Mori–Tanaka composite in the Laplace–Carson domain ──────────────────
#
# The inclusion strain of eq. 20, ε₁ = B:(ε₀ + d), B = [𝕀 + ℙ:(C₁ − C₀)]⁻¹,
# with ℙ = 𝔼:C₀⁻¹, 𝔼 the Eshelby tensor of the inclusion in the matrix (or in
# its isotropic part), and d = ℙ:(C₁:J₁ − C₀:J₀) for the eigenstrains: the
# polarization of the inclusion relative to the matrix. The mixture rule
# f₀ε₀ + f₁ε₁ = E gives A₁ = B:(f₁B + f₀𝕀)⁻¹ and a₁ = f₀ A₁:d, written without
# the inverse of C₁ − C₀ of eq. 19, so that identical phases are allowed.
# The quasi-elastic composite uses it in time, with its compliances and
# eigenstrains (`c` gives the fractions, the shape of the inclusions, whether
# the matrix is taken isotropic, and the options of the Hill tensor).
function _carson_concentration(c, S₀, S₁, J₀, J₁)
    C₀, C₁ = inv(S₀), inv(S₁)
    𝔼 = _eshelby(c, C₀)
    ℙ = 𝔼 / C₀
    I6 = one(C₀)
    f₀, f₁ = c.fractions
    B = inv(I6 + ℙ * (C₁ - C₀))
    A₁ = B / (f₁ * B + f₀ * I6)
    return A₁, f₀ * A₁ * (ℙ * (C₁ * J₁ - C₀ * J₀))
end

_eshelby(c, C₀) = c.isotropic ? _eshelby_iso(c, C₀) : _eshelby_full(c, C₀)
function _eshelby_iso(c, C₀)
    k, μ = TensND.bulk_shear(TensND.isotropify(TensND.from_mandel((C₀ + transpose(C₀)) / 2)))
    Ciso = TensND.TensISO{3}(3k, 2μ)
    return _hill_mandel(c, Ciso) * SMatrix{6, 6}(_mandel(Ciso))
end
_eshelby_full(c, C₀) = _hill_mandel(c, TensND.from_mandel((C₀ + transpose(C₀)) / 2)) * C₀
_hill_mandel(c, C) = SMatrix{6, 6}(_mandel(TensND.change_tens_canon(Elasticity.hill_tensor(c.shape, C; c.options...))))

# ── State and step ──────────────────────────────────────────────────────────

"""
    AffineEVPState

The state of an [`AffineEVPComposite`](@ref): the states of the two phases (their
strains are their average strains), the macroscopic strain and stress, the
time, and the memories of the history: the integrals ``\\int_0^t f\\,e^{-s_l u}\\,\\mathrm{d}u``
of the stress and of the inelastic strain rate of each phase at the collocation
points ``s_l = 1/\\theta_l``, and those ``\\int_0^t e^{-(t-u)/\\theta_l}\\,\\dot{\\boldsymbol{E}}\\,\\mathrm{d}u``
of the macroscopic strain rate.
"""
struct AffineEVPState{S₀ <: PointState, S₁ <: PointState, T} <: AbstractMaterialState
    phases::Tuple{S₀, S₁}
    ε::SVector{6, T}
    σ::SVector{6, T}
    t::T
    Iσ::NTuple{2, Vector{SVector{6, T}}}
    Iin::NTuple{2, Vector{SVector{6, T}}}
    H::Vector{SVector{6, T}}
end

function Constitutive.initial_state(c::AffineEVPComposite)
    zeros6(points) = [zero(SVector{6, Float64}) for _ in points]
    return AffineEVPState(
        map(initial_state, c.phases), zero(SVector{6, Float64}), zero(SVector{6, Float64}), 0.0,
        (zeros6(c.points), zeros6(c.points)), (zeros6(c.points), zeros6(c.points)), zeros6(c.θ)
    )
end
Constitutive.gradient_names(::AffineEVPComposite) = (:ε,)
Constitutive.flux_names(::AffineEVPComposite) = (:σ,)

function Constitutive.material_response(c::AffineEVPComposite, gradients::NamedTuple, st::AffineEVPState, Δt::Real; cache = nothing)
    Σ, 𝕂, new = _mandel_update(c, _mandel(gradients.ε), st, Δt; algorithm = LocalNewton(), external = (;))
    return MaterialResponse((σ = _tensor(Σ),), (σε = _tensor(𝕂),), new)
end

# A time function f(t) = A + B t + Σ_k b_k (1 − e^{−t/θ_k}) from its transform at
# the collocation points (eqs. 49–53), given A = f*(∞) and B = lim s f*(s) at 0;
# returns the coefficients b_k. Several functions at once: `values[l]` holds
# them at s_l.
function _collocation_coefficients(c::AffineEVPComposite, values, A, B)
    rhs = [values[l] - A - B * c.θ[l] for l in eachindex(c.θ)]
    return [sum(c.collocation[k, l] * rhs[l] for l in eachindex(rhs)) for k in eachindex(c.θ)]
end

function _mandel_update(c::AffineEVPComposite, E₀, st::AffineEVPState, Δt; algorithm, external, cache = nothing)
    Δt ≥ 0 || throw(ArgumentError("AffineEVPComposite: a step of negative duration, Δt = $Δt"))
    T = float(promote_type(eltype(E₀), typeof(Δt), eltype(st.σ), _numeric_type(c.phases[1].p), _numeric_type(c.phases[2].p), typeof(first(c.fractions))))
    laws = map(law -> _with_external(law, external), c.phases)
    E = SVector{6, T}(E₀)
    τ = st.t + Δt
    f₀, f₁ = c.fractions
    S = map(law -> inv(law.p.ℂ), laws)
    # The elastic concentration, A*(∞).
    A∞ = first(_carson_concentration(c, S[1], S[2], zero(E), zero(E)))
    # A step of no duration is a jump: elastic, the inclusions taking A*(∞) of
    # it (a value test on the time, which is numerical).
    iszero(Δt) && return _jump(c, laws, A∞, E, st, algorithm, T)
    σₙ = map((law, s) -> _fluxes(law, s.g, s.y), laws, st.phases)
    # The phases at the strains of a trial x: (stress, tangent, state).
    phases(x, E) = map((law, ε, prev) -> _point_update(law, ε, prev, Δt, algorithm), laws, ((E - f₁ * x) / f₀, x), st.phases)
    # The average strain of the inclusions that the comparison composite gives,
    # from the stresses and states the phases reached, and what it computed on
    # the way. It solves nothing, so that its derivatives nest no solve.
    concentration(σ₀, y₀, σ₁, y₁, E) = _evp_concentration(c, laws, S, A∞, st, σₙ, Δt, (σ₀, σ₁), (y₀, y₁), E)
    G(x, E) = (r = phases(x, E); (concentration(r[1][1], r[1][3].y, r[2][1], r[2][3].y, E)..., r))
    # The fixed point x = G(x) by successive substitutions, accelerated, as the
    # paper iterates (its Fig. 1), on the values and on their derivatives.
    converged = _fixed_point_test(algorithm)
    x₀ = SVector{6, T}(st.phases[2].g + (E - st.ε))
    x, (Iσ₊, Iin₊, H₊, solved) = _anderson(z -> (r = G(z, E); (r[1], Base.tail(r))), x₀, algorithm.maxiters, converged, :AffineEVPComposite)
    # The tangent: x follows E by the implicit-function theorem on x = G(x, E).
    # G depends on x and E through the stresses and states of the phases, whose
    # derivatives in their strains are their consistent tangents 𝕂 and the
    # sensitivities of their states, and through E directly; the concentration
    # is differentiated in those, never through the solves of the phases.
    σ₀, 𝕂₀, s₀ = solved[1]
    σ₁, 𝕂₁, s₁ = solved[2]
    N₀, N₁ = length(s₀.y), length(s₁.y)
    i(a, n) = SVector{n}(ntuple(k -> a + k, Val(n)))
    z = vcat(σ₀, s₀.y, σ₁, s₁.y, E)
    # Column by column: a derivative in all 32 directions at once, nested with
    # the one of the linearized rates, through 6×6 matrices at every
    # collocation point, takes more compilation memory than a machine may have.
    Ĝ(w) = first(concentration(w[i(0, 6)], w[i(6, N₀)], w[i(6 + N₀, 6)], w[i(12 + N₀, N₁)], w[i(12 + N₀ + N₁, 6)]))
    J = _directional_jacobian(Ĝ, z)
    cols(a, n) = J[:, i(a, n)]
    Y₀ = _phase_state_sensitivity(laws[1], s₀, st.phases[1], Δt)
    Y₁ = _phase_state_sensitivity(laws[2], s₁, st.phases[2], Δt)
    d₀ = cols(0, 6) * 𝕂₀ + cols(6, N₀) * Y₀
    d₁ = cols(6 + N₀, 6) * 𝕂₁ + cols(12 + N₀, N₁) * Y₁
    Gx = d₁ - (f₁ / f₀) * d₀
    GE = d₀ / f₀ + cols(12 + N₀ + N₁, 6)
    dx = (one(Gx) - Gx) \ GE
    Σ = f₀ * σ₀ + f₁ * σ₁
    𝕂 = 𝕂₀ * (one(𝕂₀) - f₁ * dx) + f₁ * 𝕂₁ * dx
    return _evp_result(Σ, 𝕂, AffineEVPState((s₀, s₁), E, Σ, T(τ), Iσ₊, Iin₊, H₊), st, T)
end

# A jump of the macroscopic strain at the time of the state: the phases respond
# elastically, the inclusions taking A*(∞) of it, and the memories of the
# macroscopic strain rate take it whole.
function _jump(c, laws, A∞, E, st, algorithm, ::Type{T}) where {T}
    f₀, f₁ = c.fractions
    ΔE = E - st.ε
    x = SVector{6, T}(st.phases[2].g) + A∞ * ΔE
    steps = map((law, ε, prev) -> _point_update(law, ε, prev, zero(T), algorithm), laws, ((E - f₁ * x) / f₀, x), st.phases)
    Σ = f₀ * steps[1][1] + f₁ * steps[2][1]
    𝕂 = steps[1][2] * (one(A∞) - f₁ * A∞) + f₁ * steps[2][2] * A∞
    retype(I) = [SVector{6, T}(v) for v in I]
    return _evp_result(Σ, 𝕂, AffineEVPState((steps[1][3], steps[2][3]), E, Σ, T(st.t), map(retype, st.Iσ), map(retype, st.Iin), [SVector{6, T}(h) + ΔE for h in st.H]), st, T)
end

# The average strain of the inclusions at the end τ of the step, from the
# stresses σ and states y the phases reached and the macroscopic strain E, and
# the updated memories: the history integrals at the collocation points, those
# of the macroscopic strain rate.
function _evp_concentration(c, laws, S, A∞, st, σₙ, Δt, σ, y, E)
    τ = st.t + Δt
    lins = map(_linearized_rates, laws, σ, y)
    Iσ = map((I, σn, σr) -> [I[l] + _step_integral(σn, σr, st.t, Δt, s) for (l, s) in enumerate(c.points)], st.Iσ, σₙ, σ)
    Iin = map((I, lin) -> [I[l] + _step_integral(lin.ε̇, lin.ε̇, st.t, Δt, s) for (l, s) in enumerate(c.points)], st.Iin, lins)
    # The memories of the macroscopic strain rate at τ, the rate constant over
    # the step.
    Ė = (E - st.ε) / Δt
    H = [exp(-Δt / θ) * h + Ė * (-θ * expm1(-Δt / θ)) for (h, θ) in zip(st.H, c.θ)]
    carson = map(enumerate(c.points)) do (l, s)
        _carson_concentration(
            c, _carson_compliance(S[1], lins[1], s), _carson_compliance(S[2], lins[2], s),
            _carson_eigenstrain(lins[1], σ[1], Iσ[1][l], Iin[1][l], τ, s),
            _carson_eigenstrain(lins[2], σ[2], Iσ[2][l], Iin[2][l], τ, s)
        )
    end
    M = length(c.θ)
    # B of the eigenstrain term, lim s a*(s) at s = 0, by Richardson's
    # extrapolation from s₀ and 2s₀.
    Ba = 2 * c.points[M + 1] * last(carson[M + 1]) - c.points[M + 2] * last(carson[M + 2])
    bA = _collocation_coefficients(c, [first(carson[l]) for l in 1:M], A∞, zero(A∞))
    ba = _collocation_coefficients(c, [last(carson[l]) for l in 1:M], zero(E), Ba)
    ε₁ = A∞ * E + sum(b * (E - h) for (b, h) in zip(bA, H)) + Ba * τ + sum(b * (-expm1(-τ / θ)) for (b, θ) in zip(ba, c.θ))
    return ε₁, Iσ, Iin, H
end

# dy/dε of a phase over the step: zero when the step leaves its state unchanged.
# The laws in rate form accepted (Perzyna with m ≥ 1, Norton with n ≥ 1) have
# finite derivatives everywhere, so that the sensitivity always exists.
function _phase_state_sensitivity(law, s, prev, Δt)
    N, T = length(s.y), eltype(s.y)
    _frozen(law, s.g, prev, Δt) && return zero(SMatrix{N, 6, T})
    return SMatrix{N, 6, T}(_state_sensitivity(law, s.y, s.g, prev, Δt))
end

# The k-th vector of the canonical basis of the space of z.
_unit(z::SVector{N}, k) where {N} = SVector{N}(ntuple(j -> j == k, Val(N)))

# The Jacobian of f at z, one direction at a time.
_directional_jacobian(f::F, z::SVector{N}) where {F, N} =
    hcat(ntuple(k -> ForwardDiff.derivative(t -> f(z + t * _unit(z, k)), zero(eltype(z))), Val(N))...)

# The result of a step in the element type T, which inference loses through the
# solves of the phases (as for the polycrystals).
_evp_result(Σ, 𝕂, new, st, ::Type{T}) where {T} = (
    SVector{6, T}(Σ)::SVector{6, T}, SMatrix{6, 6, T, 36}(𝕂)::SMatrix{6, 6, T, 36},
    new::AffineEVPState{_state_type(typeof(st.phases[1]), T), _state_type(typeof(st.phases[2]), T), T},
)

# The test of a fixed point x = G(x) iterated over phases solved to the
# tolerance of `alg`: the change r judged on the values and on their
# derivatives, against ten times the relative tolerance of the phases, whose
# rounding it stays above, and their absolute tolerance. A looser test leaves
# the stress less accurate than a stress-driven step asks.
function _fixed_point_test(alg::LocalNewton)
    tol = 10 * alg.reltol
    return (r, x) -> maximum(abs ∘ TensND.primal, r) ≤ alg.abstol + tol * maximum(abs ∘ TensND.primal, x) &&
        Schemes._sc_partials_converged(r, x; abstol = alg.abstol, reltol = tol)
end
