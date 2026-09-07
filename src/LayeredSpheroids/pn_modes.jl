# =============================================================================
#  pn_modes.jl — one Papkovich–Neuber harmonic mode, and the fields it makes.
#
#  The three elementary problems of Duan et al. (2005) differ only in WHICH
#  potentials are active, at which order `m`, and what the remote field is. The
#  operators are the same for all three, because Papkovich–Neuber is:
#
#      2μ u = ∇Φ - 4(1-ν) φ⃗,          Φ = φ₀ + x⃗·φ⃗,   φ⃗ = (φ₁, φ₂, φ₃)
#      σ    = -2ν (div φ⃗) 𝟙 + ∇∇Φ - 4(1-ν) sym(∇φ⃗)
#
#  both verified symbolically for four arbitrary harmonic potentials (see
#  `test/LayeredSpheroids/test_pn_symbolic.jl`). So there is one evaluator, and
#  the cases are data.
#
#  WHY A JET AND NOT AUTOMATIC DIFFERENTIATION. A mode is a product of three
#  univariate functions, `Pₙᵐ(p) · Rₙᵐ(q) · T(mφ)`, so every derivative of it is
#  a product of tabulated pieces — nothing to differentiate. What does need
#  differentiating is `Φ`, whose `x⃗·φ⃗` term multiplies a mode by a coordinate.
#  A second-order jet carries value, gradient and Hessian through that product
#  by Leibniz, exactly, in whatever element type is in play, and without
#  nesting `ForwardDiff` inside a solve that may itself be differentiated.
# =============================================================================

"""
    Jet2{T}

Value, gradient and Hessian of a scalar function of `(φ, p, q)`, carried
together so that products differentiate themselves.

`d = (∂φ, ∂p, ∂q)` and `h = (∂φφ, ∂pp, ∂qq, ∂φp, ∂φq, ∂pq)`.
"""
struct Jet2{T}
    v::T
    d::NTuple{3, T}
    h::NTuple{6, T}
end

Jet2{T}(v) where {T} = Jet2{T}(T(v), ntuple(_ -> zero(T), 3), ntuple(_ -> zero(T), 6))
Base.eltype(::Jet2{T}) where {T} = T
Base.zero(::Type{Jet2{T}}) where {T} = Jet2{T}(zero(T))

Base.:+(a::Jet2{T}, b::Jet2{T}) where {T} =
    Jet2{T}(a.v + b.v, a.d .+ b.d, a.h .+ b.h)
Base.:-(a::Jet2{T}) where {T} = Jet2{T}(-a.v, .-a.d, .-a.h)
Base.:-(a::Jet2{T}, b::Jet2{T}) where {T} = a + (-b)
Base.:*(s::Number, a::Jet2{T}) where {T} = Jet2{T}(s * a.v, s .* a.d, s .* a.h)
Base.:*(a::Jet2, s::Number) = s * a

Base.convert(::Type{Jet2{T}}, a::Jet2{T}) where {T} = a
Base.convert(::Type{Jet2{T}}, a::Jet2) where {T} =
    Jet2{T}(T(a.v), T.(a.d), T.(a.h))
Base.promote_rule(::Type{Jet2{A}}, ::Type{Jet2{B}}) where {A, B} = Jet2{promote_type(A, B)}

# Leibniz. The Hessian slots are ordered (φφ, pp, qq, φp, φq, pq), so the
# mixed ones pair the two distinct first derivatives.
const _MIX = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))

function Base.:*(a::Jet2, b::Jet2)
    T = promote_type(eltype(a), eltype(b))
    return convert(Jet2{T}, a) * convert(Jet2{T}, b)
end

function Base.:*(a::Jet2{T}, b::Jet2{T}) where {T}
    v = a.v * b.v
    d = ntuple(i -> a.d[i] * b.v + a.v * b.d[i], 3)
    h = ntuple(
        k -> let (i, j) = _MIX[k]
            a.h[k] * b.v + a.v * b.h[k] + a.d[i] * b.d[j] + a.d[j] * b.d[i]
        end, 6
    )
    return Jet2{T}(v, d, h)
end

"""
    _coord_jets(ϕ, p, q, c) -> (x, y, z)

The Cartesian coordinates as jets in `(φ, p, q)`:
`x = c p̄ q̄ cos φ`, `y = c p̄ q̄ sin φ`, `z = c p q`, with `p̄ = √(1-p²)`,
`q̄ = √(q²-1)`. These are what multiply `φ₁, φ₂, φ₃` inside `Φ`.
"""
function _coord_jets(ϕ, p, q, c, ::Type{T}) where {T}
    z0 = zero(T)
    pb, qb = sqrt(one(T) - T(p)^2), sqrt(T(q)^2 - one(T))
    # ρ = c p̄ q̄ and its derivatives; p̄' = -p/p̄, q̄' = q/q̄
    ρ = c * pb * qb
    ρp = c * (-p / pb) * qb
    ρq = c * pb * (q / qb)
    ρpp = c * qb * (-(one(T)) / pb - p^2 / pb^3)
    ρqq = c * pb * (one(T) / qb - q^2 / qb^3)
    ρpq = c * (-p / pb) * (q / qb)
    cϕ, sϕ = cos(T(ϕ)), sin(T(ϕ))
    # x = ρ cos φ
    xj = Jet2{T}(
        ρ * cϕ, (-ρ * sϕ, ρp * cϕ, ρq * cϕ),
        (-ρ * cϕ, ρpp * cϕ, ρqq * cϕ, -ρp * sϕ, -ρq * sϕ, ρpq * cϕ)
    )
    yj = Jet2{T}(
        ρ * sϕ, (ρ * cϕ, ρp * sϕ, ρq * sϕ),
        (-ρ * sϕ, ρpp * sϕ, ρqq * sϕ, ρp * cϕ, ρq * cϕ, ρpq * sϕ)
    )
    zj = Jet2{T}(
        c * p * q, (z0, c * q, c * p),
        (z0, z0, z0, z0, z0, c)
    )
    return xj, yj, zj
end

"""
    PNMode(potential, regularity, n, m, trig)

One harmonic mode of one Papkovich–Neuber potential.

- `potential ∈ 0:3` — which of `φ₀, φ₁, φ₂, φ₃` it belongs to.
- `regularity ∈ (:regular, :irregular)` — the `q`-dependence, in Barthélémy &
  Bignonnet's own words (2020, §2): `Pₙᵐ(p)Pₙᵐ(q)` has a finite limit as
  `q → 1` and is a **regular** harmonic, `Pₙᵐ(p)Qₙᵐ(q)` blows up there and is
  an **irregular** one. A core admits only regular harmonics, the matrix only
  irregular ones plus the remote field.
- `n`, `m` — degree and order.
- `trig ∈ (:cos, :sin)` — the azimuthal factor `cos mφ` or `sin mφ`.

## Amplitude naming

Barthélémy & Bignonnet write the series of layer `ℓ` as

    Σ Pₙᵐ(p) { [aᵐ_{ℓ,n} Pₙᵐ(q) + bᵐ_{ℓ,n} Qₙᵐ(q)] cos mφ
             + [cᵐ_{ℓ,n} Pₙᵐ(q) + dᵐ_{ℓ,n} Qₙᵐ(q)] sin mφ }

so `a, b, c, d` mean *regular-cos, irregular-cos, regular-sin, irregular-sin* —
see [`bb_letter`](@ref). Conduction has one field; elasticity has four
potentials, so the letters carry a potential index as well,
`a^{i,m}_{ℓ,n} … d^{i,m}_{ℓ,n}`.

!!! warning "A letter encodes `(regularity, trig)`, never a potential"
    An earlier version of this module labeled case I's four families
    `:A, :B, :C, :D` for `(φ₀, reg), (φ₀, irr), (φ₃, reg), (φ₃, irr)`. That
    collides with the convention above, where `c` and `d` are the *sine*
    families. The pair is carried explicitly here for that reason.
"""
struct PNMode
    potential::Int
    regularity::Symbol
    n::Int
    m::Int
    trig::Symbol
end

"""
    bb_letter(mode) -> Symbol

Barthélémy & Bignonnet's amplitude letter for a mode: `:a` regular-cos, `:b`
irregular-cos, `:c` regular-sin, `:d` irregular-sin. The potential index rides
alongside as a superscript and is not part of the letter.
"""
function bb_letter(mode::PNMode)
    reg = mode.regularity === :regular
    cs = mode.trig === :cos
    reg && cs && return :a
    !reg && cs && return :b
    reg && return :c
    return :d
end

"""
    _branch_kinds(m, regularity) -> (p_branch, q_branch)

Legendre table symbols for the two branches at order `m`. The `p` branch is
always of the first kind (`|p| ≤ 1`); the `q` branch is `Pₙᵐ` for a regular
harmonic, `Qₙᵐ` for an irregular one.
"""
@inline function _branch_kinds(m::Int, regularity::Symbol)
    reg = regularity === :regular
    m == 0 && return (:P0, reg ? :P0 : :Q0)
    m == 1 && return (:P1p, reg ? :P1 : :Q1)
    m == 2 && return (:P2p, reg ? :P2 : :Q2)
    throw(ArgumentError("PNMode: order m = $m is not tabulated (0, 1, 2 only)"))
end

"""
    _assoc_second(y, dy, x, n, m) -> y''

Second derivative from the associated Legendre equation,

    (1-x²) y'' - 2x y' + [n(n+1) - m²/(1-x²)] y = 0,

which `Pₙᵐ` and `Qₙᵐ` both satisfy on either branch. Nothing tabulates it.
"""
@inline function _assoc_second(y, dy, x, n::Int, m::Int)
    om = 1 - x^2
    return (2 * x * dy - (n * (n + 1) - m^2 / om) * y) / om
end

"""
    _mode_jet(mode, ϕ, p, q, ::Type{T}) -> Jet2{T}

The mode's own potential as a jet: `Pₙᵐ(p) Rₙᵐ(q) T(mφ)` with all nine
derivatives, each a product of tabulated univariate pieces.
"""
function _mode_jet(mode::PNMode, ϕ, p, q, ::Type{T}) where {T}
    kp, kq = _branch_kinds(mode.m, mode.regularity)
    n, m = mode.n, mode.m
    Pv, Pd = legendre_degrees(kp, T(p), n:n)
    Rv, Rd = legendre_degrees(kq, T(q), n:n)
    P, dP = Pv[1], Pd[1]
    R, dR = Rv[1], Rd[1]
    ddP = _assoc_second(P, dP, T(p), n, m)
    ddR = _assoc_second(R, dR, T(q), n, m)
    mϕ = m * T(ϕ)
    Tt, dTt = mode.trig === :cos ? (cos(mϕ), -m * sin(mϕ)) : (sin(mϕ), m * cos(mϕ))
    ddTt = -m^2 * Tt
    return Jet2{T}(
        P * R * Tt,
        (P * R * dTt, dP * R * Tt, P * dR * Tt),
        (P * R * ddTt, ddP * R * Tt, P * ddR * Tt, dP * R * dTt, P * dR * dTt, dP * dR * Tt)
    )
end

"""
    _chart_grad(j, p, q, c) -> (gφ, gp, gq)

Physical gradient components of a jet in the orthonormal chart frame:
`(∇F)_φ = F_φ/(c p̄ q̄)`, `(∇F)_p = p̄ F_p/(c w)`, `(∇F)_q = q̄ F_q/(c w)`,
with `w = √(q²-p²)`.
"""
@inline function _chart_grad(j::Jet2{T}, p, q, c) where {T}
    pb, qb, w = sqrt(one(T) - T(p)^2), sqrt(T(q)^2 - one(T)), sqrt(T(q)^2 - T(p)^2)
    return (j.d[1] / (c * pb * qb), pb * j.d[2] / (c * w), qb * j.d[3] / (c * w))
end

"""
    _chart_hess(j, p, q, c) -> NTuple{6}

Physical Hessian components in the chart frame, ordered
`(φφ, pp, qq, φp, φq, pq)`. Extracted from the chart rather than quoted — the
expression is linear in the derivatives, so each coefficient is unambiguous,
and the six are checked in `test_pn_symbolic.jl`. With `p̄² = 1-p²`,
`q̄² = q²-1`, `w² = q²-p²` and `S = q F_q - p F_p`:

    c² H_φφ = F_φφ/(p̄²q̄²) + S/w²
    c² H_pp = p̄² F_pp/w²   + q̄² S/w⁴
    c² H_qq = q̄² F_qq/w²   + p̄² S/w⁴
    c² H_φp = [F_φp + p F_φ/p̄²] / (w q̄)
    c² H_φq = [F_φq - q F_φ/q̄²] / (p̄ w)
    c² H_pq = p̄ q̄ [F_pq/w² + (p F_q - q F_p)/w⁴]

Their trace is the spheroidal Laplacian, which is how they were first checked.
"""
@inline function _chart_hess(j::Jet2{T}, p, q, c) where {T}
    pb2, qb2, w2 = one(T) - T(p)^2, T(q)^2 - one(T), T(q)^2 - T(p)^2
    pb, qb, w = sqrt(pb2), sqrt(qb2), sqrt(w2)
    Fϕ, Fp, Fq = j.d
    Fϕϕ, Fpp, Fqq, Fϕp, Fϕq, Fpq = j.h
    S = q * Fq - p * Fp
    ic2 = one(T) / c^2
    return (
        ic2 * (Fϕϕ / (pb2 * qb2) + S / w2),
        ic2 * (pb2 * Fpp / w2 + qb2 * S / w2^2),
        ic2 * (qb2 * Fqq / w2 + pb2 * S / w2^2),
        ic2 * (Fϕp + p * Fϕ / pb2) / (w * qb),
        ic2 * (Fϕq - q * Fϕ / qb2) / (pb * w),
        ic2 * pb * qb * (Fpq / w2 + (p * Fq - q * Fp) / w2^2),
    )
end

"""
    _cartesian_in_chart(ϕ, p, q, ::Type{T}) -> NTuple{3,NTuple{3,T}}

`ê_i · e_a` for `i = 1,2,3` (Cartesian) and `a = φ, p, q` (chart) — the chart's
own normalized basis, read column-wise. Needed because `φ⃗` is a vector of
CARTESIAN components while everything else lives in the chart frame.
"""
@inline function _cartesian_in_chart(ϕ, p, q, ::Type{T}) where {T}
    pb, qb, w = sqrt(one(T) - T(p)^2), sqrt(T(q)^2 - one(T)), sqrt(T(q)^2 - T(p)^2)
    cϕ, sϕ = cos(T(ϕ)), sin(T(ϕ))
    return (
        (-sϕ, -p * qb * cϕ / w, q * pb * cϕ / w),
        (cϕ, -p * qb * sϕ / w, q * pb * sϕ / w),
        (zero(T), q * pb / w, p * qb / w),
    )
end

"""
    mode_fields(mode, ϕ, p, q, c, μ, ν, ::Type{T}) -> (u, t)

Displacement and traction of one unit-amplitude mode, in the chart frame:
`u = (u_φ, u_p, u_q)` and `t = σ·e_q = (σ_φq, σ_pq, σ_qq)`.

This is the single evaluator all three elementary problems go through. What
distinguishes them is which modes are in the list, not how a mode is turned
into fields.
"""
function mode_fields(mode::PNMode, ϕ, p, q, c, μ, ν, ::Type{T}) where {T}
    jφ = _mode_jet(mode, ϕ, p, q, T)
    i = mode.potential
    # Φ = φ₀ + x φ₁ + y φ₂ + z φ₃ — only this mode's own term survives.
    Φj = if i == 0
        jφ
    else
        _coord_jets(ϕ, p, q, c, T)[i] * jφ
    end
    gΦ = _chart_grad(Φj, p, q, c)
    H = _chart_hess(Φj, p, q, c)

    # The `·q` column of the Hessian, in the slot order (φφ, pp, qq, φp, φq, pq):
    # (H_φq, H_pq, H_qq) = (H[5], H[6], H[3]).
    Hq = (H[5], H[6], H[3])

    if i == 0
        # `φ⃗ = 0`: this mode lives in `φ₀` alone, so `σ = ∇∇Φ`.
        return ntuple(a -> gΦ[a] / (2 * μ), 3), Hq
    end

    e = _cartesian_in_chart(ϕ, p, q, T)[i]  # ê_i · e_a
    gφ = _chart_grad(jφ, p, q, c)           # ∇φ_i in the chart frame
    fourν = 4 * (1 - ν)
    # `φ⃗ = φ_i ê_i`, so its chart components are `φ_i (ê_i·e_a)`.
    u = ntuple(a -> (gΦ[a] - fourν * jφ.v * e[a]) / (2 * μ), 3)
    # `div φ⃗ = ∇φ_i · ê_i`, and `sym(∇φ⃗)_{aq} = ½[(∇φ_i)_a e_q + (∇φ_i)_q e_a]`.
    divφ = gφ[1] * e[1] + gφ[2] * e[2] + gφ[3] * e[3]
    δ_q = (zero(T), zero(T), one(T))
    t = ntuple(
        a -> -2 * ν * divφ * δ_q[a] + Hq[a] -
            fourν * (gφ[a] * e[3] + gφ[3] * e[a]) / 2, 3
    )
    return u, t
end
