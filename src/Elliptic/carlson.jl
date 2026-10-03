# ═════════════════════════════════════════════════════════════════════════════
#  Carlson symmetric elliptic integrals  R_F(x, y, z)  and  R_D(x, y, z)
#
#  Type-generic iterative duplication, following
#      B. C. Carlson, "Numerical computation of real or complex elliptic
#      integrals", Numerical Algorithms 10 (1995) 13-26,
#  with the same Taylor expansions (eq. 2.15 and eq. 2.29) as the
#  public-domain SLATEC routines DRF / DRD.
# ═════════════════════════════════════════════════════════════════════════════

# ─── Tolerance ───────────────────────────────────────────────────────────────
#
# The Taylor remainder is O(Δ⁶) where Δ = max(|ΔX|, |ΔY|, |ΔZ|) and
# ΔI = (μ − I) / μ.  To align truncation error with the working
# precision we iterate until Δ < eps(T)^{1/6}.  For Float64 this gives
# ≈ 2.4·10⁻³ (consistent with the SLATEC default); for BigFloat the
# threshold tightens automatically with the working precision.

@inline _carlson_tol(::Type{T}) where {T <: AbstractFloat} = eps(T)^(1 / 6)
@inline _carlson_tol(::Type{T}) where {T <: Real} = eps(Float64)^(1 / 6)
@inline _carlson_tol(::Type{T}) where {T <: Number} = zero(Float64)


"""
    ell_RF(x, y, z) -> T

Carlson's symmetric elliptic integral of the first kind,

``R_F(x, y, z) = \\tfrac{1}{2}\\int_0^{\\infty}
  \\bigl[(t+x)(t+y)(t+z)\\bigr]^{-1/2}\\,\\mathrm{d}t``.

Type-generic: the duplication recursion uses only arithmetic and
square roots, so it extends unchanged to `BigFloat`,
`ForwardDiff.Dual`, `Symbolics.Num`, `SymPy.Sym`, and any other
`Number` subtype.
"""
function ell_RF(x::Tx, y::Ty, z::Tz) where {Tx <: Number, Ty <: Number, Tz <: Number}
    T = promote_type(Tx, Ty, Tz)
    xk, yk, zk = T(x), T(y), T(z)
    tol = _carlson_tol(T)
    for _ in 1:50
        sx = sqrt(xk); sy = sqrt(yk); sz = sqrt(zk)
        # Carlson duplication step (Carlson 1995 eq. 2.3):
        #   (xₖ₊₁, yₖ₊₁, zₖ₊₁) = ((xₖ + λ)/4, (yₖ + λ)/4, (zₖ + λ)/4)
        λ = sx * (sy + sz) + sy * sz
        xk = (xk + λ) / 4
        yk = (yk + λ) / 4
        zk = (zk + λ) / 4
        if T <: Real
            μ = (xk + yk + zk) / 3
            spread = max(abs((μ - xk) / μ), abs((μ - yk) / μ), abs((μ - zk) / μ))
            spread < tol && break
        end
    end
    μ = (xk + yk + zk) / 3
    ΔX = (μ - xk) / μ
    ΔY = (μ - yk) / μ
    ΔZ = (μ - zk) / μ
    # Elementary-symmetric invariants of (ΔX, ΔY, ΔZ); at convergence
    # ΔX + ΔY + ΔZ → 0 (Carlson 1995 eq. 2.14).
    E₂ = ΔX * ΔY + ΔY * ΔZ + ΔZ * ΔX
    E₃ = ΔX * ΔY * ΔZ
    # Fifth-order Taylor polynomial (Carlson 1995 eq. 2.15).
    poly = one(T) - E₂ / 10 + E₃ / 14 + (E₂ * E₂) / 24 - (3 * E₂ * E₃) / 44
    return poly / sqrt(μ)
end


"""
    ell_RD(x, y, z) -> T

Carlson's symmetric elliptic integral of the second kind, degenerate in
`z`:

``R_D(x, y, z) = \\tfrac{3}{2}\\int_0^{\\infty}
  \\bigl[(t+z)\\sqrt{(t+x)(t+y)(t+z)}\\bigr]^{-1}\\,\\mathrm{d}t``.

Same type-generic recursion as [`ell_RF`](@ref) with the additional
``(1, 1, 3)``-weighted mean and a running sum that accounts for the
degenerate ``(t + z)`` factor (Carlson 1995 §2).
"""
function ell_RD(x::Tx, y::Ty, z::Tz) where {Tx <: Number, Ty <: Number, Tz <: Number}
    T = promote_type(Tx, Ty, Tz)
    xk, yk, zk = T(x), T(y), T(z)
    tol = _carlson_tol(T)
    # Running sum  Σₖ 4^{−k} / (√zₖ · (zₖ + λₖ))  that accumulates the
    # extra 1/(t + z) factor which distinguishes R_D from R_F
    # (Carlson 1995 eq. 2.28).
    tail = zero(T)
    weight = one(T)
    for _ in 1:50
        sx = sqrt(xk); sy = sqrt(yk); sz = sqrt(zk)
        λ = sx * (sy + sz) + sy * sz
        tail += weight / (sz * (zk + λ))
        weight /= 4
        xk = (xk + λ) / 4
        yk = (yk + λ) / 4
        zk = (zk + λ) / 4
        if T <: Real
            μ = (xk + yk + 3 * zk) / 5
            spread = max(abs((μ - xk) / μ), abs((μ - yk) / μ), abs((μ - zk) / μ))
            spread < tol && break
        end
    end
    μ = (xk + yk + 3 * zk) / 5
    ΔX = (μ - xk) / μ
    ΔY = (μ - yk) / μ
    ΔZ = (μ - zk) / μ
    # Weighted elementary-symmetric invariants with (1, 1, 3) weighting
    # (Carlson 1995 eq. 2.29; at convergence ΔX + ΔY + 3·ΔZ → 0).
    ΔZ² = ΔZ * ΔZ
    PXY = ΔX * ΔY
    E₂ = PXY - 6 * ΔZ²
    E₃ = (3 * PXY - 8 * ΔZ²) * ΔZ
    E₄ = 3 * (PXY - ΔZ²) * ΔZ²
    E₅ = PXY * ΔZ² * ΔZ
    # Fifth-order Taylor polynomial (Carlson 1995 eq. 2.29).
    poly = one(T) -
        (3 * E₂) / 14 +
        E₃ / 6 +
        (9 * E₂ * E₂) / 88 -
        (3 * E₄) / 22 -
        (9 * E₂ * E₃) / 52 +
        (3 * E₅) / 26
    return 3 * tail + weight * poly / (μ * sqrt(μ))
end


# ─── Incomplete Legendre integrals via Carlson ───────────────────────────────
#
# F(φ, m) = sin(φ) · R_F(cos²φ, 1 − m sin²φ, 1)
# E(φ, m) = F(φ, m) − (m/3) · sin³(φ) · R_D(cos²φ, 1 − m sin²φ, 1)
# (DLMF 19.25.5; Carlson 1995 eq. 4.6).

function _ell_F_inc(φ::Tφ, m::Tm) where {Tφ <: Number, Tm <: Number}
    T = promote_type(Tφ, Tm)
    s = sin(T(φ))
    c² = cos(T(φ))^2
    return s * ell_RF(c², one(T) - T(m) * s^2, one(T))
end

function _ell_E_inc(φ::Tφ, m::Tm) where {Tφ <: Number, Tm <: Number}
    T = promote_type(Tφ, Tm)
    s = sin(T(φ))
    c² = cos(T(φ))^2
    sm = T(m) * s^2
    y = one(T) - sm
    return s * ell_RF(c², y, one(T)) -
        (T(m) / 3) * s^3 * ell_RD(c², y, one(T))
end
