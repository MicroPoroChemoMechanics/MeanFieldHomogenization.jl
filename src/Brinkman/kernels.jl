# =============================================================================
#  kernels.jl — scalar solutions of the Brinkman inhomogeneity problems.
#
#  Each problem is solved twice, once per formulation, because each must stay
#  finite at its own end of the range:
#
#  * resistance form: the variable is α₀ = R √r₀ ∈ [0, ∞), finite for a
#    clear-fluid reference (r₀ = 0);
#  * permeability form: the variable is s = √k₀ / R = 1/α₀ ∈ [0, ∞), finite for
#    an impervious reference (k₀ = 0).
#
#  The two are the same rational function of α₀ written in α₀ and in 1/α₀, so
#  neither divides by zero nor overflows at its own end. The inclusion enters
#  through α₁ = R/√k₁ (porous inclusion). Derivation: `theory/brinkman.md` and
#  `scripts/71_brinkman_permeability.jl` (TensND curvilinear operators, SymPy).
# =============================================================================

# ── Modified Bessel ratios ───────────────────────────────────────────────────
#
#  K₀/K₁ and I₁/I₀ come from the exponentially scaled functions, which do not
#  under- or overflow. Beyond x = 10⁸ the AMOS routines refuse the argument
#  (they lose all accuracy past ~10⁹), and the ratios are their asymptotic series,
#  whose truncation error there is below x⁻⁶ = 10⁻⁴⁸: the two coincide to the
#  last bit at the switch, and the series also gives the exact limit 1 at x = ∞,
#  which the permeability form reaches for an impervious reference.
const _BESSEL_ASYMPTOTIC = 1.0e8

_k0k1(x) = _k0k1(x, Val(is_hard_numeric(typeof(x))))
_k0k1(x, ::Val{true}) = x > _BESSEL_ASYMPTOTIC ?
    evalpoly(inv(x), (1, -1 // 2, 3 // 8, -3 // 8, 63 // 128, -27 // 32)) :
    besselkx(0, x) / besselkx(1, x)
_k0k1(x, ::Val{false}) = besselk(0, x) / besselk(1, x)

_i1i0(x) = _i1i0(x, Val(is_hard_numeric(typeof(x))))
_i1i0(x, ::Val{true}) = x > _BESSEL_ASYMPTOTIC ?
    evalpoly(inv(x), (1, -1 // 2, -1 // 8, -1 // 8, -25 // 128, -13 // 32)) :
    besselix(1, x) / besselix(0, x)
_i1i0(x, ::Val{false}) = besseli(1, x) / besseli(0, x)

# ── Well-conditioned functions of α₁ ────────────────────────────────────────
#
#  The porous closed forms subtract nearly equal quantities as α₁ = b → 0. They
#  are written with
#
#      τ = tanh(b)/b,  g = (b − tanh b)/b³,  h = (b³ − 3b + 3 tanh b)/b⁵,
#      w = q/b,        p = (b − 2q)/b³,      q = I₁(b)/I₀(b),
#
#  which tend to 1, 1/3, 2/5, 1/2 and 1/8. For b ≤ 1 they are evaluated from the
#  Taylor series of tanh and of I₁/I₀, whose coefficients follow from the Riccati
#  equations y' = 1 − y² and q' = 1 − q/b − q²; beyond, the direct expressions
#  lose less than a few ulps (the cancellation is at most a factor 10 there).
#  The series converge up to π/2 and 2.40 respectively, so the terms kept make
#  their truncation error negligible at b = 1.

# Odd Taylor coefficients c₁, c₃, … of the solution of y' = 1 − y² − κ y/b with
# y(0) = 0: κ = 0 is tanh, κ = 1 is I₁/I₀. Computed exactly, stored as Float64
# constants: they multiply the variable, whose type (dual included) the result
# takes; only an extended-precision input would see them rounded.
function _riccati_odd_coefficients(κ, n)
    c = zeros(Rational{BigInt}, 2n)
    c[1] = 1 // (1 + κ)
    for k in 3:2:(2n - 1)
        c[k] = -sum(c[i] * c[k - 1 - i] for i in 1:(k - 2)) // (k + κ)
    end
    return Float64.(c[1:2:end])
end
const _TANH_ODD = _riccati_odd_coefficients(0, 40)
const _I1I0_ODD = _riccati_odd_coefficients(1, 24)
# Series in b² of τ, g, h (tanh) and of w, p (I₁/I₀), as tuples for `evalpoly`.
const _SERIES_τ = Tuple(_TANH_ODD)
const _SERIES_g = Tuple(-_TANH_ODD[2:end])
const _SERIES_h = Tuple(3 .* _TANH_ODD[3:end])
const _SERIES_w = Tuple(_I1I0_ODD)
const _SERIES_p = Tuple(-2 .* _I1I0_ODD[2:end])

# (τ, g, h, b²g) of tanh.
_tanh_parts(b) = _tanh_parts(b, Val(is_hard_numeric(typeof(b))))
function _tanh_parts(b, ::Val{true})
    if b ≤ 1
        x = b^2
        g = evalpoly(x, _SERIES_g)
        return (evalpoly(x, _SERIES_τ), g, evalpoly(x, _SERIES_h), x * g)
    end
    τ = tanh(b) / b
    return (τ, (1 - τ) / b^2, (1 - 3 / b^2 + 3τ / b^2) / b^2, 1 - τ)
end
function _tanh_parts(b, ::Val{false})
    τ = tanh(b) / b
    return (τ, (1 - τ) / b^2, (b^3 - 3b + 3 * tanh(b)) / b^5, 1 - τ)
end

# (w, p, b q) of I₁/I₀.
_bessel_parts(b) = _bessel_parts(b, Val(is_hard_numeric(typeof(b))))
function _bessel_parts(b, ::Val{true})
    if b ≤ 1
        x = b^2
        w = evalpoly(x, _SERIES_w)
        return (w, evalpoly(x, _SERIES_p), x * w)
    end
    q = _i1i0(b)
    return (q / b, (1 - 2q / b) / b^2, b * q)
end
function _bessel_parts(b, ::Val{false})
    q = _i1i0(b)
    return (q / b, (b - 2q) / b^3, b * q)
end

# ── Sphere ───────────────────────────────────────────────────────────────────

# Average velocity in a Stokes sphere per unit remote velocity.
_sphere_fluid_r(a) = 3 * evalpoly(a, (15, 15, 6, 1)) / evalpoly(a, (45, 45, 6, 1))
# Same, times k₀/R² (permeability form, s = 1/α₀).
_sphere_fluid_k(s) = 3 * s^2 * evalpoly(s, (1, 6, 15, 15)) / evalpoly(s, (1, 6, 45, 45))

# Drag on an impervious sphere, R² r₁A (Brinkman's drag), and its value times
# k₀/R² = s², the force concentration of the permeability form.
_sphere_solid_r(a) = 3 * evalpoly(a, (3, 3, 1)) / 2
_sphere_solid_k(s) = 3 * evalpoly(s, (1, 3, 3)) / 2

# Average velocity in a porous sphere, α₁ = b. The closed form subtracts nearly
# equal terms as b → 0 (it reads 0/0 at b = 0), so it is written with the
# well-conditioned functions g and h of `_tanh_parts`.
function _sphere_porous_r(a, b)
    _, g, h, bg = _tanh_parts(b)
    num = a^2 * (a + 1) * h + (3 + 3a - a^3) * g
    return 3 * num / (num + 2 * (a + 1) - 2a * bg)
end
# Same quantity, numerator and denominator multiplied by s³.
function _sphere_porous_k(s, b)
    _, g, h, bg = _tanh_parts(b)
    num = (1 + s) * h + (3s^3 + 3s^2 - 1) * g
    return 3 * num / (num + 2s^2 * (1 + s) - 2s^2 * bg)
end

# ── Cylinder: (axial, transverse) pairs ─────────────────────────────────────

function _cylinder_fluid_r(a)
    ψ = a * _k0k1(a)
    return (1 + a^2 / 8 + ψ / 2, 2 * (8 + a^2 + 4ψ) / (16 + a^2 + 4ψ))
end
# Times k₀/R².
function _cylinder_fluid_k(s)
    ψ = s * _k0k1(inv(s))
    return (s^2 + one(s) / 8 + ψ / 2, 2 * s^2 * (8 * s^2 + 1 + 4ψ) / (16 * s^2 + 1 + 4ψ))
end

# Axial drag R² r₁Aₙ on an impervious cylinder; the transverse one is twice it.
_cylinder_solid_r(a) = 2a / _k0k1(a) + a^2
_cylinder_solid_k(s) = 2s / _k0k1(inv(s)) + 1

# Porous cylinder, written with w = q/b and p = (b − 2q)/b³ (q = I₁/I₀), which
# stay well conditioned as b → 0 where the closed form reads 0/0.
function _cylinder_porous_r(a, b)
    ρ = _k0k1(a)
    w, p, bq = _bessel_parts(b)
    common = 2w + a * ρ * w + a^2 * p
    An = a * common / (bq * ρ + a)
    At = 2a * common / (a^3 * p + 2a * w + a + ρ * (a^2 * w + bq))
    return (An, At)
end
# Times k₀/R².
function _cylinder_porous_k(s, b)
    ρ = _k0k1(inv(s))
    w, p, bq = _bessel_parts(b)
    common = 2w * s^2 + ρ * w * s + p
    kAn = common / (bq * ρ * s + 1)
    kAt = 2 * s^2 * common / (p + 2w * s^2 + s^2 + ρ * s * (w + bq * s^2))
    return (kAn, kAt)
end

# ── Layer: in-plane component (the normal one is 1) ─────────────────────────

_layer_fluid_r(a) = evalpoly(a, (1, 1, 1 // 3))
_layer_fluid_k(s) = evalpoly(s, (1 // 3, 1, 1))

function _layer_porous_r(a, b)
    τ, g, _, _ = _tanh_parts(b)
    return a * ((1 + a) * τ + a^2 * g) / (b^2 * τ + a)
end
function _layer_porous_k(s, b)
    τ, g, _, _ = _tanh_parts(b)
    return ((s^2 + s) * τ + g) / (b^2 * τ * s + 1)
end

# ── Coated sphere: impervious core R, Stokes shell up to R/η ────────────────
#
#  A = 3(1-η)² N/D and R² ⟨r·A⟩ = 18 η³ M/D, with N, M, D cubic in α₀; the
#  coefficients are polynomials in η, listed from the constant term up.
_coated_N(η) = (
    60η^3 + 30η^4, 60η^2 + 30η^3,
    24η + 3η^2 - 18η^3 - 9η^4, 4 - η - 6η^2 - η^3 + 4η^4,
)
_coated_M(η) = (45η^3, 45η^2, 21η - 5η^3 - η^6, 6 - 5η^2 - η^5)
_coated_D(η) = (
    180η^3, 180η^2 - 180η^3,
    24η - 45η^2 + 30η^4 - 9η^6, 4 - 9η + 10η^3 - 9η^5 + 4η^6,
)

# (A, R²⟨r·A⟩) in α₀, and the same multiplied out by s³ for the permeability form.
function _coated_r(a, η)
    D = evalpoly(a, _coated_D(η))
    return (3 * (1 - η)^2 * evalpoly(a, _coated_N(η)) / D, 18η^3 * evalpoly(a, _coated_M(η)) / D)
end
function _coated_k(s, η)
    D = evalpoly(s, reverse(_coated_D(η)))
    return (3 * (1 - η)^2 * evalpoly(s, reverse(_coated_N(η))) / D, 18η^3 * evalpoly(s, reverse(_coated_M(η))) / D)
end
