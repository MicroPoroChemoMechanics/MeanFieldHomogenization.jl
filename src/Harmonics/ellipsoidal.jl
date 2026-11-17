# =============================================================================
#  ellipsoidal.jl — Lamé functions and ellipsoidal harmonics (dassios2012).
#
#  An ellipsoid of semi-axes a₁ > a₂ > a₃ along x₁, x₂, x₃ carries the
#  ellipsoidal coordinates (ρ, μ, ν), h₃² ≤ μ² ≤ h₂² ≤ ρ², 0 ≤ ν² ≤ h₃², with
#  h₁² = a₂² − a₃², h₂² = a₁² − a₃², h₃² = a₁² − a₂², defined by
#
#      x₁²/t + x₂²/(t − h₃²) + x₃²/(t − h₂²) = 1 ,   t ∈ {ρ², μ², ν²} ,
#
#  (Dassios, Eqs. (1.26)–(1.34)); the surface of the ellipsoid is ρ = a₁. A
#  harmonic function separates as E(ρ) E(μ) E(ν), E being a Lamé function of
#  degree n: one of the 2n + 1 solutions of
#
#      (x² − h₃²)(x² − h₂²) E″ + x (2x² − h₃² − h₂²) E′ + [(h₃² + h₂²) p − n(n+1) x²] E = 0
#
#  that are of the form ψ(x) P(x), P a polynomial of every other power and ψ one
#  of 1, √|x² − h₃²|, √|x² − h₂²| and their product: the classes K, L, M, N.
#  Writing P = Σ_k c_k x^{d−2k} turns the equation into a three-term recurrence
#  on the c_k whose truncation is a tridiagonal eigenproblem for p (Dassios,
#  Eqs. (3.22), (3.48), (3.81)); its r eigenvalues give the r functions of the
#  class, with c₀ = 1.
#
#  The interior harmonics 𝔼_n^m(x) = E(ρ) E(μ) E(ν) are polynomials of degree n
#  in x; the exterior ones are 𝔽_n^m(x) = F(ρ) E(μ) E(ν), with the Lamé
#  function of the second kind
#
#      F(ρ) = (2n + 1) E(ρ) I(ρ) ,   I(ρ) = ∫_ρ^∞ dt / [E(t)² √(t² − h₃²) √(t² − h₂²)] ,
#
#  which decays as ρ^{−n−1}. On ρ = const the surface harmonics S = E(μ) E(ν)
#  are orthogonal for the weight
#
#      dΩ = (μ² − ν²) dμ dν / [√(μ² − h₃²) √(h₂² − μ²) √(h₃² − ν²) √(h₂² − ν²)] .
#
#  The substitutions μ² = h₃² + (h₂² − h₃²) sin²α (α ∈ [0, π]) and ν = h₃ sin β
#  (β ∈ [0, 2π)) cover the eight octants with the sign conventions of Dassios
#  (§ 1.2) and remove the singular factors: dΩ = (μ² − ν²)/(μ √(h₂² − ν²)) dα dβ.
#
#  The geometry enters only through numbers compared with one another (the
#  roots of a cubic, the eigenvalues of the recurrence): it is taken in
#  `Float64`, while the fields built on it may carry duals.
# =============================================================================

"""
    LameFunction

Lamé function ``E(x) = \\psi(x) \\sum_k c_k x^{d - 2k}`` of degree `n`, with
``\\psi`` the product of ``\\sqrt{|x^2 - h_3^2|}`` (if `a == 1`) and
``\\sqrt{|x^2 - h_2^2|}`` (if `b == 1`), and `p` its separation constant.
"""
struct LameFunction
    n::Int
    a::Int
    b::Int
    p::Float64
    coefficients::Vector{Float64}
end

"""
    EllipsoidalGeometry(a₁, a₂, a₃)

The ellipsoidal coordinates of the ellipsoid of semi-axes ``a_1 > a_2 > a_3``
along ``x_1, x_2, x_3``: the focal constants ``h_1, h_2, h_3`` and, filled up
to the degree asked by [`lame_functions`](@ref), the Lamé functions.
"""
struct EllipsoidalGeometry
    a::NTuple{3, Float64}
    h1::Float64
    h2::Float64
    h3::Float64
end

function EllipsoidalGeometry(a₁::Real, a₂::Real, a₃::Real)
    a₁ > a₂ > a₃ > 0 || throw(
        ArgumentError("EllipsoidalGeometry: the semi-axes must satisfy a₁ > a₂ > a₃ > 0, got ($a₁, $a₂, $a₃)")
    )
    a = Float64.((a₁, a₂, a₃))
    return EllipsoidalGeometry(a, sqrt(a[2]^2 - a[3]^2), sqrt(a[1]^2 - a[3]^2), sqrt(a[1]^2 - a[2]^2))
end

"""
    lame_functions(g::EllipsoidalGeometry, n) -> Vector{LameFunction}

The ``2n + 1`` Lamé functions of degree `n`, by class (K, L, M, N) and, within
a class, by increasing separation constant.
"""
function lame_functions(g::EllipsoidalGeometry, n::Integer)
    n ≥ 0 || throw(ArgumentError("lame_functions: the degree must be nonnegative, got $n"))
    h₂², h₃² = g.h2^2, g.h3^2
    α, β = h₃² + h₂², h₃² * h₂²
    out = LameFunction[]
    for (a, b) in ((0, 0), (1, 0), (0, 1), (1, 1))
        d = n - a - b
        d ≥ 0 || continue
        r = d ÷ 2 + 1                   # coefficients of x^d, x^{d−2}, …, x^{d mod 2}
        # recurrence 2(k+1)(2n−2k−1) c_{k+1} = [α (p − s_k²) − e_k] c_k + β f_k c_{k−1},
        # that is α p c = A c with A tridiagonal, c_r = 0 closing it
        A = zeros(r, r)
        for k in 0:(r - 1)
            s, e, f = _lame_recurrence(n, a, b, k, h₂², h₃²)
            A[k + 1, k + 1] = α * s^2 + e
            k + 1 < r && (A[k + 1, k + 2] = 2 * (k + 1) * (2n - 2k - 1))
            k > 0 && (A[k + 1, k] = -β * f)
        end
        F = eigen(A ./ α)
        order = sortperm(real.(F.values))
        for j in order
            c = real.(F.vectors[:, j])
            push!(out, LameFunction(n, a, b, real(F.values[j]), c ./ c[1]))
        end
    end
    length(out) == 2n + 1 || throw(ErrorException("lame_functions: $(length(out)) functions at the degree $n"))
    return out
end

# The terms of the recurrence of the class (a, b) at the step k: the square
# s² of the shifted degree, the extra diagonal term e (classes L, M) and the
# factor f of c_{k−1} (Dassios, Eqs. (3.22), (3.48), (3.81)).
function _lame_recurrence(n, a, b, k, h₂², h₃²)
    if a == b == 0
        return (n - 2k, 0.0, (n - 2k + 2) * (n - 2k + 1))
    elseif a + b == 1
        h² = a == 1 ? h₂² : h₃²          # class L carries h₂² in its diagonal, M h₃²
        return (n - 2k - 1, (2n - 4k - 1) * h², (n - 2k + 1) * (n - 2k))
    end
    return (n - 2k - 1, 0.0, (n - 2k) * (n - 2k - 1))
end

# The polynomial part and its derivative at x.
function _lame_polynomial(E::LameFunction, x)
    d = E.n - E.a - E.b
    P = zero(x)
    dP = zero(x)
    for (k, c) in enumerate(E.coefficients)
        j = d - 2 * (k - 1)
        P += c * x^j
        j > 0 && (dP += c * j * x^(j - 1))
    end
    return P, dP
end

"""
    lame_value(E, x, s₃, s₂, σ₃, σ₂) -> (value, derivative)

The Lamé function `E` and its derivative at `x`, the square roots of its class
being given as ``s_3 = \\sqrt{|x^2 - h_3^2|}`` and ``s_2 = \\sqrt{|x^2 - h_2^2|}`` with the
branch (sign) of the coordinate, and their derivatives following from
``s\\,s' = \\sigma x``, ``\\sigma`` (`σ₃`, `σ₂`) being the sign of ``x^2 - h^2``.
"""
function lame_value(E::LameFunction, x, s₃, s₂, σ₃::Int, σ₂::Int)
    P, dP = _lame_polynomial(E, x)
    ψ, dψ = one(x), zero(x)
    if E.a == 1
        ψ, dψ = ψ * s₃, dψ * s₃ + ψ * σ₃ * x / s₃
    end
    if E.b == 1
        ψ, dψ = ψ * s₂, dψ * s₂ + ψ * σ₂ * x / s₂
    end
    return ψ * P, dψ * P + ψ * dP
end

"""
    ellipsoidal_coordinates(g, x) -> NamedTuple

The ellipsoidal coordinates ``(\\rho, \\mu, \\nu)`` of the point `x` and the square
roots that enter the Lamé functions, signed by the octant of `x` (Dassios,
§ 1.2): ``\\nu`` carries the sign of ``x_1``, ``\\sqrt{h_3^2 - \\nu^2}`` that of ``x_2``
and ``\\sqrt{h_2^2 - \\mu^2}`` that of ``x_3``. The squares are the roots of the
cubic of the three confocal families.
"""
function ellipsoidal_coordinates(g::EllipsoidalGeometry, x)
    h₂², h₃² = g.h2^2, g.h3^2
    x₁², x₂², x₃² = x[1]^2, x[2]^2, x[3]^2
    # −t³ + c₂ t² − c₁ t + c₀ = 0
    c₂ = h₃² + h₂² + x₁² + x₂² + x₃²
    c₁ = h₃² * h₂² + x₁² * (h₃² + h₂²) + x₂² * h₂² + x₃² * h₃²
    c₀ = x₁² * h₃² * h₂²
    # roots in Float64, then two Newton steps in the type of x: the second order
    # of the implicit-function theorem, so that duals (nested) carry the
    # derivatives of the coordinates exactly
    t₀ = sort(real.(eigvals(primal.([c₂ -c₁ c₀; 1 0 0; 0 1 0]))))
    t = map(t₀) do τ
        τ = oftype(c₂, τ)
        for _ in 1:2
            τ -= (-τ^3 + c₂ * τ^2 - c₁ * τ + c₀) / (-3τ^2 + 2c₂ * τ - c₁)
        end
        τ
    end
    ν² = clamp(t[1], zero(h₃²), h₃²)
    μ² = clamp(t[2], h₃², h₂²)
    ρ² = max(t[3], h₂²)
    sgn(v) = v < 0 ? -1.0 : 1.0
    ρ, μ = sqrt(ρ²), sqrt(μ²)
    ν = sgn(x[1]) * sqrt(ν²)
    return (;
        ρ, μ, ν,
        ρ3 = sqrt(ρ² - h₃²), ρ2 = sqrt(ρ² - h₂²),
        μ3 = sqrt(μ² - h₃²), μ2 = sgn(x[3]) * sqrt(h₂² - μ²),
        ν3 = sgn(x[2]) * sqrt(h₃² - ν²), ν2 = sqrt(h₂² - ν²),
    )
end

# The Lamé function at the three coordinates, with the square roots and the
# signs of x² − h² of each.
_lame_at_ρ(E, c) = lame_value(E, c.ρ, c.ρ3, c.ρ2, 1, 1)
_lame_at_μ(E, c) = lame_value(E, c.μ, c.μ3, c.μ2, 1, -1)
_lame_at_ν(E, c) = lame_value(E, c.ν, c.ν3, c.ν2, -1, -1)

"""
    lame_second_kind(g, E, ρ, nodes) -> (F, dF)

The Lamé function of the second kind ``F(\\rho) = (2n+1)\\,E(\\rho)\\,I(\\rho)`` and its
derivative, with ``I(\\rho) = \\int_\\rho^\\infty \\mathrm{d}t / [E(t)^2 \\sqrt{t^2-h_3^2}\\sqrt{t^2-h_2^2}]``
computed as ``\\rho \\int_0^1 \\mathrm{d}s / [E(\\rho/s)^2 \\sqrt{\\rho^2-h_3^2 s^2}\\sqrt{\\rho^2-h_2^2 s^2}]``.
The factor ``\\sqrt{\\rho^2-h_2^2 s^2}`` falls to ``\\kappa = \\sqrt{\\rho^2-h_2^2}`` at
``s = 1``, which is ``a_3`` on the surface: for an elongated ellipsoid the
integrand varies there over a length ``(\\kappa/h_2)^2``. The interval is
split at ``s = 1/2``; on ``[1/2, 1]`` the variable ``\\tau``, with
``1 - s^2 = (\\kappa/h_2)^2 \\sinh^2\\tau``, turns ``\\sqrt{\\rho^2-h_2^2 s^2}`` into
``\\kappa\\cosh\\tau`` and leaves a smooth integrand. Both parts are integrated
by Gauss–Legendre on `nodes` (nodes and weights on ``[0, 1]``, ``n + 24``
points by default; a caller evaluating many functions passes them once).
"""
function lame_second_kind(g::EllipsoidalGeometry, E::LameFunction, ρ, nodes = gauss(E.n + 24, 0.0, 1.0))
    h₂, h₃ = g.h2, g.h3
    xs, ws = nodes
    κ = sqrt(ρ^2 - h₂^2)
    τ₁ = asinh(sqrt(oftype(κ, 3)) * h₂ / (2κ))     # s = 1/2
    Eρs(s) = first(lame_value(E, ρ / s, sqrt((ρ / s)^2 - h₃^2), sqrt((ρ / s)^2 - h₂^2), 1, 1))
    I = zero(κ)
    for (x, w) in zip(xs, ws)
        s = x / 2                                             # on [0, 1/2]
        I += (w / 2) / (Eρs(s)^2 * sqrt(ρ^2 - h₃^2 * s^2) * sqrt(ρ^2 - h₂^2 * s^2))
        τ = τ₁ * x                                            # on [1/2, 1]
        sh = sinh(τ)
        s = sqrt(1 - (κ * sh / h₂)^2)
        I += τ₁ * w * κ * sh / (h₂^2 * s * Eρs(s)^2 * sqrt(ρ^2 - h₃^2 * s^2))
    end
    I *= ρ
    s₃ = sqrt(ρ^2 - h₃^2)
    Eρ, dEρ = lame_value(E, ρ, s₃, κ, 1, 1)
    dI = -1 / (Eρ^2 * s₃ * κ)
    return (2E.n + 1) * Eρ * I, (2E.n + 1) * (dEρ * I + Eρ * dI)
end
