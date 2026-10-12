# =============================================================================
#  spheroid_integrals.jl
#
#  The one-dimensional integrals behind the closed forms of a spheroid, and of
#  an ellipsoid close to one: Newton potentials (isotropic matrix, conduction)
#  and the Hill tensor in a coaxial transversely isotropic matrix. Evaluated so
#  that they stay exact to rounding near the sphere, near coincident acoustic
#  roots, for flat shapes and for needles, and differentiable everywhere.
# =============================================================================

# With D(a, z) = a + (1 − a) z² and w = 1 − z², over z ∈ [−1, 1]:
#
#   I_n(a)          = ∫ zⁿ / D(a)                 (one root)
#   J_{n,q}(a₁, a₂) = ∫ zⁿ w^q / [D(a₁) D(a₂)]    (two roots)
#   M_{n,k,p}(m)    = ∫ zⁿ wᵏ / D(m)ᵖ             (moments of one root)
#
# For a spheroid of axial over transverse semi-axis ω, a = ω² gives the Newton
# potentials (`newton_potential_3d`); a = ω²γ², with γ² a root of the acoustic
# polynomial of a transversely isotropic matrix, the Hill tensor of a coaxial
# spheroid in that matrix (`_hill_ti_walpole`). There, J enters through
# symmetric functions of (a₁, a₂) only: the mean m = (a₁ + a₂)/2 and the squared
# half-difference d² = ((a₁ − a₂)/2)², both rational in the moduli. J is
# evaluated from them in three regimes, so that no square root is taken where
# the roots coincide (an isotropic matrix stored as TI, where the derivative of
# √d² is infinite) and no closed form is used where it cancels:
#
# 1. near the sphere, |aᵢ − 1| ≤ 1/4: with D = 1 + k w, k = a − 1,
#        J_{n,q} = Σⱼ (−1)ʲ hⱼ(k₁, k₂) β_{n,q+j},   β_{n,j} = ∫ zⁿ wʲ,
#    hⱼ = Σ_{p+q=j} k₁ᵖ k₂^q, which only needs k₁ + k₂ and k₁k₂;
# 2. near coincident roots, |d| ≤ |m|/20: with D̄ = D(m),
#        1 / (D₁D₂) = Σⱼ d^{2j} w^{2j} / D̄^{2j+2},   truncated at j = 4,
#    so that J_{n,q} = Σⱼ d^{2j} M_{n,2j+q,2j+2}(m);
# 3. elsewhere, the closed forms of the two roots by partial fractions,
#        J_{n,q} = [(1 − a₁) M(a₁) − (1 − a₂) M(a₂)] / (a₂ − a₁),   M = ∫ zⁿ w^q / D,
#    arranged so that they do not cancel: for a flat spheroid (|aᵢ| < 1), J₂ and
#    J₄ by the recurrence on z² = (D₁ − a₁)/(1 − a₁); for a needle, the parts of
#    w/D = (1 − z²/D)/a that are rational in a summed exactly, leaving partial
#    fractions of I₄/a and I₄/a² that cancel as 1/d only.
#
# In derivatives, the split |d| = |m|/20 balances the O(d⁸) truncation of
# regime 2 against the cancellation of regime 3, which grows as 1/d³: both stay
# near 1e-10.
#
# The moments M_{n,k,p} are, near the sphere, the power series
#     M_{n,k,p} = Σⱼ binomial(−p, j) kʲ β_{n,k+j},
# and elsewhere, with wᵏ = (D − 1)ᵏ/kᵏ, combinations of U_{n,p} = M_{n,0,p}, which
# follow from I₀ by U_{0,p+1} = (1 + (p − ½) U_{0,p}) / (p m), the integral of
# d/dz [z/Dᵖ], and U_{n+2,p} = (U_{n,p−1} − m U_{n,p}) / (1 − m), from
# z² = (D − m)/(1 − m).

const _SPHEROID_SERIES_RADIUS = 0.25
const _SPHEROID_SERIES_TERMS = 32          # (j + 1)·4⁻ʲ < 1e-18 beyond
const _SPHEROID_MOMENT_TERMS = 64          # the same for powers p ≤ 13
const _SPHEROID_COINCIDENT_SPLIT = 0.05
const _SPHEROID_COINCIDENT_ORDER = 4       # terms in d⁰, d², …, d⁸

# Σⱼ (−1)ʲ hⱼ β_{n,q+j}, hⱼ the complete homogeneous polynomials of the two
# roots of x² − e₁x + e₂ (hⱼ = e₁hⱼ₋₁ − e₂hⱼ₋₂); e₂ = 0 gives one root.
@inline function _spheroid_series(e1, e2, n, q = 0)
    β = 2 * one(real(e1)) / (n + 1)
    for i in 1:q
        β = β * (2i) / (2i + n + 1)
    end
    h_prev, h = zero(e1), one(e1)
    s = β * h
    for j in 1:_SPHEROID_SERIES_TERMS
        β = β * (2 * (q + j)) / (2 * (q + j) + n + 1)
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

# ∫ w / D(a) = I₀ − I₂, or (2 − I₂)/a, the same by w = (D − z²)/a, for a ≥ 1:
# there I₀ and I₂ both vanish as log(a)/a while the second form does not cancel.
@inline function _spheroid_Iw(a)
    k = a - 1
    abs(k) ≤ _SPHEROID_SERIES_RADIUS && return _spheroid_series(k, zero(k), 0, 1)
    return abs(a) < 1 ? _spheroid_I0(a) - _spheroid_I2(a) : (2 - _spheroid_I2(a)) / a
end

# ── Moments M_{n,k,p} of one root ──────────────────────────────────────────

function _spheroid_moment_series(k̄, n, k, p)
    F = typeof(float(_primal(real(k̄))))
    β = 2 * one(F) / (n + 1)
    for i in 1:k
        β = β * (2i) / (2i + n + 1)
    end
    c = one(F)
    kj = one(k̄)
    s = β * kj
    for j in 1:_SPHEROID_MOMENT_TERMS
        β = β * (2 * (k + j)) / (2 * (k + j) + n + 1)
        c = -c * (p + j - 1) / j
        kj *= k̄
        s += c * kj * β
    end
    return s
end

# U_{n,p} for n = 0, 2, 4 and p = 0 … P, away from the sphere.
function _spheroid_U_table(m, P)
    T = typeof(m / one(m))
    U0 = Vector{T}(undef, P + 1)
    U2 = Vector{T}(undef, P + 1)
    U4 = Vector{T}(undef, P + 1)
    U0[1], U2[1], U4[1] = 2 * one(T), 2 * one(T) / 3, 2 * one(T) / 5
    U0[2] = _spheroid_I0(m)
    for p in 1:(P - 1)
        U0[p + 2] = (1 + (p - 1 / 2) * U0[p + 1]) / (p * m)
    end
    β = 1 - m
    for p in 1:P
        U2[p + 1] = (U0[p] - m * U0[p + 1]) / β
        U4[p + 1] = (U2[p] - m * U2[p + 1]) / β
    end
    return (U0, U2, U4)
end

# The moments of a root m: the series near the sphere (`U === nothing`), the
# table `U = _spheroid_U_table(m, P)` elsewhere.
_spheroid_moment_table(m, P) = abs(m - 1) ≤ _SPHEROID_SERIES_RADIUS ? nothing : _spheroid_U_table(m, P)

function _spheroid_moment(m, U, n, k, p)
    k̄ = m - 1
    U === nothing && return _spheroid_moment_series(k̄, n, k, p)
    Un = U[n ÷ 2 + 1]
    s = zero(k̄)
    for l in 0:k
        s += (isodd(k - l) ? -1 : 1) * binomial(k, l) * Un[p - l + 1]
    end
    return s / k̄^k
end

# ── Two roots: the integrals of the transversely isotropic closed form ──────

"""
    _spheroid_JL(m, d²) -> (J₄, L₂₁, L₀₂)

The integrals ``J_{4,0}``, ``J_{2,1}`` and ``J_{0,2}`` of two roots
``a_{1,2} = m \\pm d``,

```math
\\int_{-1}^{1}\\frac{z^4\\,\\mathrm{d}z}{D_1D_2},\\qquad
\\int_{-1}^{1}\\frac{z^2(1-z^2)\\,\\mathrm{d}z}{D_1D_2},\\qquad
\\int_{-1}^{1}\\frac{(1-z^2)^2\\,\\mathrm{d}z}{D_1D_2},
```

from their mean ``m`` and squared half-difference ``d^2``, in the three regimes
described in `src/Core/spheroid_integrals.jl`. The Walpole coefficients of the
Hill tensor are written with these three rather than with ``J_0, J_2, J_4``,
whose combinations cancel as ``\\omega^2`` for a needle. Generic in the number
type, real or complex, `ForwardDiff.Dual` included: no square root of ``d^2`` is
taken near ``d^2 = 0``, so derivatives stay exact where the two roots coincide.
"""
function _spheroid_JL(m, d2)
    k̄ = m - 1
    if abs(k̄) + sqrt(abs(d2)) ≤ _SPHEROID_SERIES_RADIUS
        e1 = 2k̄
        e2 = k̄^2 - d2
        return (_spheroid_series(e1, e2, 4), _spheroid_series(e1, e2, 2, 1), _spheroid_series(e1, e2, 0, 2))
    elseif abs(d2) ≤ _SPHEROID_COINCIDENT_SPLIT^2 * abs(m)^2
        N = _SPHEROID_COINCIDENT_ORDER
        U = _spheroid_moment_table(m, 2N + 2)
        J4 = L21 = L02 = zero(k̄)
        dj = one(d2)
        for j in 0:N
            J4 += dj * _spheroid_moment(m, U, 4, 2j, 2j + 2)
            L21 += dj * _spheroid_moment(m, U, 2, 2j + 1, 2j + 2)
            L02 += dj * _spheroid_moment(m, U, 0, 2j + 2, 2j + 2)
            dj *= d2
        end
        return (J4, L21, L02)
    end
    d = sqrt(d2)
    a1 = m + d
    a2 = m - d
    pf(f) = ((1 - a1) * f(a1) - (1 - a2) * f(a2)) / (a2 - a1)
    if max(abs(a1), abs(a2)) < 1                 # flat
        J0 = pf(_spheroid_I0)
        ai, b, aj = abs(1 - a1) ≥ abs(1 - a2) ? (a1, 1 - a1, a2) : (a2, 1 - a2, a1)
        J2 = (_spheroid_I0(aj) - ai * J0) / b
        J4 = (_spheroid_I2(aj) - ai * J2) / b
        return (J4, J2 - J4, J0 - 2 * J2 + J4)
    end
    p = a1 * a2                                  # needle side
    L21 = 2 / (3 * p) - pf(a -> _spheroid_I4(a) / a)
    L02 = 2 / p - 2 * (a1 + a2) / (3 * p^2) + pf(a -> _spheroid_I4(a) / a^2)
    return (pf(_spheroid_I4), L21, L02)
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

# ── Triaxial ellipsoid close to a spheroid ──────────────────────────────────
#
# With L the semi-axis apart and B₁ ≈ B₂ the two close ones, ℓ = L², bᵢ = Bᵢ²,
# the substitution ℓ + u = ℓ/z² turns every Newton potential into
#
#     2π L B₁ B₂ L^(2−2e_L) B₁^(−2e₁) B₂^(−2e₂) ∫₋₁¹ zⁿ D₁^(−e₁) D₂^(−e₂) dz,
#
# with Dᵢ = D(mᵢ, z), mᵢ = ℓ/bᵢ, n = 2(e_L + e₁ + e₂) − 3, and the exponents
# e_L, e₁, e₂ ∈ {½, 3/2, 5/2} of (ℓ + u), (b₁ + u), (b₂ + u) in the integrand.
# With m̄ = (m₁ + m₂)/2 and d = (m₁ − m₂)/2, Dᵢ = D̄ ± d (1 − z²), and
#
#     D₁^(−e₁) D₂^(−e₂) = Σₖ cₖ dᵏ (1 − z²)ᵏ / D̄^(e₁+e₂+k),
#
# cₖ the coefficients of (1 + y)^(−e₁) (1 − y)^(−e₂). The series converges as
# (|d|/m̄)ᵏ = (|b₁ − b₂|/(b₁ + b₂))ᵏ; its terms are the moments M_{n,k,p}(m̄). Exact
# for B₁ = B₂ — derivatives included, which is what the elliptic integrals and
# their divided differences (I_b − I_a)/(a² − b²) cannot give near a spheroid.

const _NEAR_SPHEROID_SPLIT = 0.02    # |b₁ − b₂|/(b₁ + b₂) below which it is used
const _NEAR_SPHEROID_ORDER = 10      # terms in d⁰ … d¹⁰: 0.02¹¹ < 1e-18

# Coefficients of (1 + y)^(−a/2) (1 − y)^(−b/2), for the odd integers a, b.
function _near_spheroid_coefficients(::Type{F}, a, b, K) where {F}
    p = Vector{F}(undef, K + 1)
    q = Vector{F}(undef, K + 1)
    p[1] = q[1] = one(F)
    for i in 1:K
        p[i + 1] = -p[i] * (a / 2 + i - 1) / i
        q[i + 1] = q[i] * (b / 2 + i - 1) / i
    end
    return [sum(p[i + 1] * q[k - i + 1] for i in 0:k) for k in 0:K]
end

"""
    _newton_near_spheroid(L, B₁, B₂) -> (I_L, I₁, I₂, I_LL, I₁₁, I₂₂, I_L1, I_L2, I₁₂)

Newton potentials of the ellipsoid of semi-axes `L`, `B₁`, `B₂` with `B₁ ≈ B₂`,
by the expansion around the spheroid described above. Raw values, normalized
as those of [`newton_potential_3d`](@ref).
"""
function _newton_near_spheroid(L, B₁, B₂)
    ℓ, b₁, b₂ = L * L, B₁ * B₁, B₂ * B₂
    m₁, m₂ = ℓ / b₁, ℓ / b₂
    m̄ = (m₁ + m₂) / 2
    d = (m₁ - m₂) / 2
    K = _NEAR_SPHEROID_ORDER
    F = typeof(float(_primal(m̄)))
    U = _spheroid_moment_table(m̄, K + 3)
    π_ = oftype(float(_primal(m̄)), π)
    function G(aL, a₁, a₂)
        c = _near_spheroid_coefficients(F, a₁, a₂, K)
        n = aL + a₁ + a₂ - 3
        R = (a₁ + a₂) ÷ 2
        s = zero(m̄)
        dk = one(m̄)
        for k in 0:K
            s += c[k + 1] * dk * _spheroid_moment(m̄, U, n, k, R + k)
            dk *= d
        end
        return 2 * π_ * L * B₁ * B₂ * L^(2 - aL) * B₁^(-a₁) * B₂^(-a₂) * s
    end
    return (
        G(3, 1, 1), G(1, 3, 1), G(1, 1, 3),
        G(5, 1, 1), G(1, 5, 1), G(1, 1, 5), G(3, 3, 1), G(3, 1, 3), G(1, 3, 3),
    )
end
