# =============================================================================
#  spheroid_integrals.jl
#
#  The one-dimensional integrals behind the closed forms of a spheroid:
#  Newton potentials (isotropic matrix, conduction) and the Hill tensor in a
#  coaxial transversely isotropic matrix. Evaluated so that they stay exact to
#  rounding near the sphere, near coincident acoustic roots and for flat
#  shapes, and differentiable everywhere.
# =============================================================================

#
#   I_n(a)      = ∫₋₁¹ zⁿ dz / D(a, z),               D(a, z) = a + (1 − a) z²
#   J_n(a₁, a₂) = ∫₋₁¹ zⁿ dz / [D(a₁, z) D(a₂, z)]
#
# for n = 0, 2, 4. They carry the shape of a spheroid: with ω its axial over its
# transverse semi-axis, a = ω² gives the Newton potentials of the spheroid
# (`newton_potential_3d`), and a = ω²γ², with γ² a root of the acoustic
# polynomial of a transversely isotropic matrix, the Hill tensor of a coaxial
# spheroid in that matrix (`_hill_ti_walpole`). There, J_n enters through
# symmetric functions of (a₁, a₂) only: the mean m = (a₁ + a₂)/2 and the
# squared half-difference d² = ((a₁ − a₂)/2)², both rational in the moduli.
# J_n is evaluated from them in three regimes, so that no square root is taken
# where the roots coincide (an isotropic matrix stored as TI, where the
# derivative of √d² is infinite) and no closed form is used where it cancels:
#
# 1. near the sphere, |aᵢ − 1| ≤ 1/4: with D = 1 + k (1 − z²), k = a − 1,
#        J_n = Σⱼ (−1)ʲ hⱼ(k₁, k₂) β_{n,j},   β_{n,j} = ∫₋₁¹ zⁿ (1 − z²)ʲ dz,
#    hⱼ = Σ_{p+q=j} k₁ᵖ k₂^q, which only needs k₁ + k₂ and k₁k₂;
# 2. near coincident roots, |d| ≤ |m|/20: with D̄ = D(m, z),
#        1 / (D₁D₂) = Σⱼ d^{2j} (1 − z²)^{2j} / D̄^{2j+2},   truncated at j = 4,
#    the integrals of zⁿ/D̄ᵖ following from I_n(m) by the reduction formulas;
# 3. elsewhere, the closed forms in `acosh` of the two roots by partial
#    fractions, J₂ and J₄ by the recurrence on z² = (D₁ − a₁)/(1 − a₁) for a
#    flat spheroid, where the partial fractions of J₄ cancel as 1/a.
#
# The split |d| = |m|/20 balances, in derivatives, the O(d⁸) truncation of
# regime 2 against the cancellation of regime 3, which grows as 1/d³: both
# stay near 1e-10.

const _SPHEROID_SERIES_RADIUS = 0.25
const _SPHEROID_SERIES_TERMS = 32          # (j + 1)·4⁻ʲ < 1e-18 beyond
const _SPHEROID_COINCIDENT_SPLIT = 0.05
const _SPHEROID_COINCIDENT_ORDER = 4       # terms in d⁰, d², …, d⁸

# Σⱼ (−1)ʲ hⱼ β_{n,j}, hⱼ the complete homogeneous polynomials of the two roots
# of x² − e₁x + e₂ (hⱼ = e₁hⱼ₋₁ − e₂hⱼ₋₂); e₂ = 0 gives I_n.
@inline function _spheroid_series(e1, e2, n)
    β = 2 * one(real(e1)) / (n + 1)
    h_prev, h = zero(e1), one(e1)
    s = β * h
    for j in 1:_SPHEROID_SERIES_TERMS
        β = β * (2j) / (2j + n + 1)
        h, h_prev = e1 * h - e2 * h_prev, h
        s += isodd(j) ? -(h * β) : h * β
    end
    return s
end

# R_C(a, 1) = acosh(√a)/√(a − 1), the elementary function behind I_n: real for
# a real positive a on either side of 1 (as acos(√a)/√(1 − a) below it), and the
# analytic continuation for a complex one.
@inline _spheroid_rc(a::Real) = a < 1 ? acos(sqrt(a)) / sqrt(1 - a) : acosh(sqrt(a)) / sqrt(a - 1)
@inline _spheroid_rc(a) = acosh(sqrt(a)) / sqrt(a - 1)

@inline function _spheroid_I0(a)
    k = a - 1
    abs(k) ≤ _SPHEROID_SERIES_RADIUS && return _spheroid_series(k, zero(k), 0)
    return 2 * _spheroid_rc(a) / sqrt(a)
end

@inline function _spheroid_I2(a)
    k = a - 1
    abs(k) ≤ _SPHEROID_SERIES_RADIUS && return _spheroid_series(k, zero(k), 2)
    return 2 * (sqrt(a) * _spheroid_rc(a) - 1) / k
end

@inline function _spheroid_I4(a)
    k = a - 1
    abs(k) ≤ _SPHEROID_SERIES_RADIUS && return _spheroid_series(k, zero(k), 4)
    return 2 * (a * sqrt(a) * _spheroid_rc(a) - 1 - 4 * k / 3) / k^2
end

# Regime 2. With (1 − z²) = (D̄ − 1)/(m − 1) and c = d²/(m − 1)², the expansion
# to order d^{2N} is Σ_p w_p U_{n,p}, U_{n,p} = ∫ zⁿ/D̄ᵖ, with weights
#     w_p = (−1)ᵖ Σ_{j ≤ N} cʲ binomial(2j, 2j + 2 − p),   p = 2, …, 2N + 2.
# U_{0,p+1} = (1 + (p − ½) U_{0,p}) / (p m) integrates d/dz [z/D̄ᵖ], and
# U_{n+2,p} = (U_{n,p−1} − m U_{n,p}) / (1 − m) follows from z² = (D̄ − m)/(1 − m).
@inline function _spheroid_coincident_weight(c, p)
    w = zero(c)
    for j in max(0, cld(p - 2, 2)):_SPHEROID_COINCIDENT_ORDER
        w += c^j * binomial(2j, 2j + 2 - p)
    end
    return isodd(p) ? -w : w
end

@inline function _spheroid_J024_coincident(m, d2)
    β = 1 - m
    c = d2 / β^2
    U0 = _spheroid_I0(m)                                  # p = 1
    U2 = (2 - m * U0) / β
    U4 = (2 * one(m) / 3 - m * U2) / β
    J0 = J2 = J4 = zero(U0)
    for p in 1:(2 * _SPHEROID_COINCIDENT_ORDER + 1)
        U0_next = (1 + (p - 1 / 2) * U0) / (p * m)
        U2_next = (U0 - m * U0_next) / β
        U4_next = (U2 - m * U2_next) / β
        w = _spheroid_coincident_weight(c, p + 1)
        J0 += w * U0_next
        J2 += w * U2_next
        J4 += w * U4_next
        U0, U2, U4 = U0_next, U2_next, U4_next
    end
    return (J0, J2, J4)
end

"""
    _spheroid_J024(m, d²) -> (J₀, J₂, J₄)

The integrals ``J_n(a_1, a_2)``, ``n = 0, 2, 4``, from the mean
``m = (a_1 + a_2)/2`` and the squared half-difference ``d^2 = ((a_1 - a_2)/2)^2``
of the squared scaled roots, in the three regimes described in
`src/Core/spheroid_integrals.jl`. Generic in the number type, real or complex,
`ForwardDiff.Dual` included: no square root of ``d^2`` is taken near
``d^2 = 0``, so derivatives stay exact where the two roots coincide.
"""
@inline function _spheroid_J024(m, d2)
    k̄ = m - 1
    if abs(k̄) + sqrt(abs(d2)) ≤ _SPHEROID_SERIES_RADIUS
        e1 = 2k̄
        e2 = k̄^2 - d2
        return (_spheroid_series(e1, e2, 0), _spheroid_series(e1, e2, 2), _spheroid_series(e1, e2, 4))
    elseif abs(d2) ≤ _SPHEROID_COINCIDENT_SPLIT^2 * abs(m)^2
        return _spheroid_J024_coincident(m, d2)
    end
    d = sqrt(d2)
    a1 = m + d
    a2 = m - d
    J0 = ((1 - a1) * _spheroid_I0(a1) - (1 - a2) * _spheroid_I0(a2)) / (a2 - a1)
    if max(abs(a1), abs(a2)) < 1
        ai, b, aj = abs(1 - a1) ≥ abs(1 - a2) ? (a1, 1 - a1, a2) : (a2, 1 - a2, a1)
        J2 = (_spheroid_I0(aj) - ai * J0) / b
        return (J0, J2, (_spheroid_I2(aj) - ai * J2) / b)
    end
    J2 = ((1 - a1) * _spheroid_I2(a1) - (1 - a2) * _spheroid_I2(a2)) / (a2 - a1)
    J4 = ((1 - a1) * _spheroid_I4(a1) - (1 - a2) * _spheroid_I4(a2)) / (a2 - a1)
    return (J0, J2, J4)
end


"""
    _newton_spheroid(L, R) -> (I_L, I_R, I_LR, I_LL, I_RR)

Newton potentials of a spheroid of axial semi-axis `L` and transverse semi-axis
`R`: ``I_L`` along the axis, ``I_R`` across it, and the second-order ones
``I_{LR}``, ``I_{LL}``, ``I_{RR}`` (the latter equal to the mixed transverse
one, as `newton_potential_3d` returns them). With ``m = L^2/R^2``,

```math
I_L = 2\\pi\\,I_2(m), \\qquad
I_{LR} = \\frac{2\\pi}{R^2}\\int_{-1}^{1}\\frac{z^4\\,\\mathrm{d}z}{D(m,z)^2},
```

then ``I_R = (4\\pi - I_L)/2``, ``I_{LL} = 4\\pi/(3L^2) - 2I_{LR}/3`` and
``I_{RR} = \\pi/R^2 - I_{LR}/4``. The divided difference
``(I_R - I_L)/(L^2 - R^2)`` that defines ``I_{LR}`` cancels near the sphere; the
integral does not.
"""
@inline function _newton_spheroid(L, R)
    L2, R2 = L * L, R * R
    m = L2 / R2
    k = m - 1
    if abs(k) ≤ _SPHEROID_SERIES_RADIUS
        I2 = _spheroid_series(k, zero(k), 2)
        U42 = _spheroid_series(2k, k^2, 4)       # coincident roots: hⱼ = (j + 1)kʲ
    else
        rc = _spheroid_rc(m)
        I0 = 2 * rc / sqrt(m)
        I2 = 2 * (sqrt(m) * rc - 1) / k
        U02 = (1 + I0 / 2) / m
        U22 = (I0 - m * U02) / (1 - m)
        U42 = (I2 - m * U22) / (1 - m)
    end
    π_ = oftype(float(real(m)), π)
    I_L = 2 * π_ * I2
    I_LR = 2 * π_ * U42 / R2
    return (I_L, (4 * π_ - I_L) / 2, I_LR, 4 * π_ / (3 * L2) - 2 * I_LR / 3, π_ / R2 - I_LR / 4)
end
