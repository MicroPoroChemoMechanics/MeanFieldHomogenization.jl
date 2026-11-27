# =============================================================================
#  field.jl — the stress in the matrix around a free hole of the plane.
#
#  Isotropic matrix, any map. With t = 1/ζ the potentials are rational in t:
#  Φ′ = A(t), ω′ = D(t), polynomials, and the free contour gives the second
#  potential as Ψ = −conj(Φ)(1/ζ) − ω̄(1/ζ) Φ′/ω′, a function that holds the
#  powers ζ, …, ζᴺ. These cancel but for Γ′ζ; evaluated as written they would
#  be subtracted from one another, with a loss of |ζ|ᴺ far from the hole. The
#  cancellation is carried out on the coefficients instead: with
#  ω̄(1/ζ) = E(t)/tᴺ and C the first N terms of the series of E A/D,
#
#       Ψ(ζ) = Γ′ ζ − Γ t − Q(t)/D(t) ,   Q = (E A − D C)/tᴺ ,
#
#  Q a polynomial, and every evaluation is at |t| ≤ 1.
#
#  Anisotropic matrix, the ellipse. The stress is the sum over the two roots
#  μₖ of Bₖ Tₖ (μₖ², 1, −μₖ), with Tₖ = −2αₖ/((zₖ + Rₖ) Rₖ) the derivative of
#  1/ζₖ and Rₖ² = zₖ² − 4αₖβₖ (`anisotropic.jl`). The Bₖ hold a division by
#  μ₁ − μ₂. Two evaluations are written:
#   - with the two roots, as the formulas read, ill-conditioned when the roots
#     come close;
#   - on the pairs (mean, divided difference) of each quantity over the two
#     roots, whose algebra is that of a + a′ε with ε² = (μ₁ − μ₂)²/4: no
#     division by μ₁ − μ₂ is left, and the isotropic matrix is an ordinary
#     case. The pair of Rₖ needs R₁ + R₂, by which it divides.
#  Both are exact; the one whose divisor is the larger, relatively, is used.
#
#  The point is given by its coordinates, and the variable ζ found from them:
#  a comparison of moduli, numbers only.
# =============================================================================

"""
    matrix_stress(hole::ConformalHole, Σ, x)
    matrix_stress(hole::ConformalHole, Σ, x, C₀)

Stress at the point `x` of the matrix around a free hole of the plane, under
the remote stress `Σ`, as a tensor of order 2 of the plane. `x = (x₁, x₂)` is
in the axes and at the scale of the map ``z = \\zeta + \\sum_k m_k\\,\\zeta^{-k}``,
``z = x_1 + \\mathrm{i}\\,x_2``; a collection of points gives a collection of
tensors, the potentials being computed once.

Without a stiffness the matrix is isotropic and the hole has any map: the
moduli do not enter. With the stiffness `C₀` of an anisotropic matrix of the
plane the hole is an ellipse
([The stress in the matrix](@ref th-matrix-stress)). On the contour the
tensor has the one component that [`contour_stress`](@ref) returns; far from
the hole it tends to `Σ`.

A point inside the hole throws a `DomainError`. Numbers and dual numbers; the
point of the circle that a point of the matrix comes from is found by a
comparison, which symbols do not take.

# Examples
The circular hole pulled along ``\\underline{e}_2``: ahead of the hole, at the
distance ``r`` from its center, ``\\sigma_{22} = 1 + 1/(2r^2) + 3/(2r^4)``.
```jldoctest
julia> using TensND

julia> σ = matrix_stress(ConformalHole(()), Tens([0.0 0.0; 0.0 1.0]), [2.0, 0.0]);

julia> canonical_components(σ)[2, 2] ≈ 1 + 1 / 8 + 3 / 32
true
```
"""
matrix_stress(h::ConformalHole, Σ::TensND.AbstractTens{2, 2}, x) = _at_points(_isotropic_field(h, _remote_stress(Σ)), x)
matrix_stress(h::ConformalHole, Σ::TensND.AbstractTens{2, 2}, x, ::TensND.TensISO{4, 2}) = matrix_stress(h, Σ, x)
function matrix_stress(h::ConformalHole, Σ::TensND.AbstractTens{2, 2}, x, C₀::TensND.AbstractTens{4, 2})
    _is_ellipse(h) || throw(
        ArgumentError(
            "the stress in an anisotropic matrix is known around the ellipse, whose map has one term; " *
                "this map has $(length(h.coefficients))"
        )
    )
    return _at_points(_AnisotropicField(first(h.coefficients), _characteristic_pair(C₀)..., _remote_stress(Σ)), x)
end

_at_points(field, x::AbstractVector{<:Number}) = _stress_tensor(field(x[1], x[2]))
_at_points(field, x::Tuple{Number, Number}) = _stress_tensor(field(x[1], x[2]))
_at_points(field, xs) = map(x -> _at_points(field, x), xs)

_stress_tensor((σ₁₁, σ₂₂, σ₁₂)) = TensND.from_mandel(SVector{3}(σ₁₁, σ₂₂, sqrt(2 * one(σ₁₂)) * σ₁₂))

@noinline _throw_inside_hole(x₁, x₂) = throw(
    DomainError((TensND.primal(x₁), TensND.primal(x₂)), "the point is inside the hole, where there is no stress")
)

# The point of the circle is found by comparing moduli.
_numbers_only(::Type{T}) where {T} = is_hard_numeric(T) || throw(
    ArgumentError(
        "matrix_stress takes numbers: the point of the unit circle that a point of the matrix comes from is " *
            "found by a comparison, which a $T does not take; contour_stress accepts symbols"
    )
)

# ── Isotropic matrix ─────────────────────────────────────────────────────────

# Coefficients, in increasing powers of t = 1/ζ, of A = Φ′, D = ω′ and Q, each
# with its derivative, with the constants of the two potentials and the map.
struct _IsotropicField{T}
    m::Vector{Complex{T}}
    Γ::T
    Γ′::Complex{T}
    A::NTuple{2, Vector{Complex{T}}}
    D::NTuple{2, Vector{Complex{T}}}
    Q::NTuple{2, Vector{Complex{T}}}
end

# c₀ + c₁t + … of 1 − Σ k cₖ tᵏ⁺¹ scaled by `c₀`: the derivative of c₀ ζ + Σ cₖ ζ⁻ᵏ.
_derivative_in_t(c₀, c::Vector{Complex{T}}) where {T} =
    [k == 0 ? Complex{T}(c₀) : (k == 1 ? zero(Complex{T}) : -(k - 1) * c[k - 1]) for k in 0:(length(c) + 1)]

# Product of two polynomials given by their coefficients.
_polynomial_product(a::Vector{S}, b::Vector{S}) where {S} =
    [sum(a[i] * b[k + 2 - i] for i in max(1, k + 2 - length(b)):min(length(a), k + 1)) for k in 0:(length(a) + length(b) - 2)]

_polynomial_derivative(c::Vector{S}) where {S} = [k * c[k + 1] for k in 1:(length(c) - 1)]

function _isotropic_field(h::ConformalHole, Σ)
    m, Γ, Γ′, b = _loaded_potential(h, Σ)
    N = length(m)
    A, D = _derivative_in_t(Γ, b), _derivative_in_t(one(Γ), m)
    # ω̄(1/ζ) tᴺ = tᴺ⁺¹ + Σ conj(mₖ) tᴺ⁻ᵏ
    E = [k == N + 1 ? one(eltype(m)) : (k < N ? conj(m[N - k]) : zero(eltype(m))) for k in 0:(N + 1)]
    EA = _polynomial_product(E, A)
    # The first N terms of the series of E A/D: each reads the previous ones (D₀ = 1).
    C = zeros(eltype(m), N)
    for j in 0:(N - 1)
        C[j + 1] = EA[j + 1] - sum(D[i + 1] * C[j + 1 - i] for i in 1:j; init = zero(eltype(m)))
    end
    Q = (EA - [_polynomial_product(D, C); zeros(eltype(m), length(EA) - length(D) - N + 1)])[(N + 1):end]
    return _IsotropicField(m, Γ, Γ′, map(c -> (c, _polynomial_derivative(c)), (A, D, Q))...)
end

# The point ζ, |ζ| ≥ 1, whose image by the map is z: the root of largest
# modulus of ζᴺ⁺¹ − z ζᴺ + Σ mₖ ζᴺ⁻ᵏ, found on the values, then brought to the
# precision of its type, derivatives included, by Newton steps on ω(ζ) = z.
# The eigenvalues of the companion matrix are the cost of a point, 0.4 ms for
# a map of 31 terms: no cheaper start is known to reach the right root from
# every point of the matrix.
function _exterior_preimage(m::Vector{Complex{T}}, z::Complex) where {T}
    N = length(m)
    # the ellipse has its root in closed form (the size of the map is a value)
    N == 1 && return (z + _exterior_root(z, m[1])) / 2
    c = [k == 0 ? -ComplexF64(TensND.primal(real(z)), TensND.primal(imag(z))) : ComplexF64(TensND.primal(real(m[k])), TensND.primal(imag(m[k]))) for k in 0:N]
    companion = [(j == i - 1 ? one(ComplexF64) : zero(ComplexF64)) - (j == N + 1 ? c[N + 2 - i] : zero(ComplexF64)) for i in 1:(N + 1), j in 1:(N + 1)]
    ζ₀ = argmax(abs, eigvals(companion))
    ζ = Complex{promote_type(T, typeof(real(z)))}(ζ₀)
    for _ in 1:4
        ζ = _preimage_step(m, ζ, z)
    end
    return ζ
end

# One Newton step on ω(ζ) = z; apart, so that the sums do not capture the
# variable the iteration reassigns.
_preimage_step(m, ζ, z) =
    ζ - (ζ + sum(m[k] * ζ^(-k) for k in eachindex(m)) - z) / (1 - sum(k * m[k] * ζ^(-k - 1) for k in eachindex(m)))

function (f::_IsotropicField{T})(x₁, x₂) where {T}
    _numbers_only(promote_type(T, typeof(x₁), typeof(x₂)))
    z = complex(x₁, x₂)
    ζ = _exterior_preimage(f.m, z)
    abs(ζ) ≥ 1 - 1.0e-10 || _throw_inside_hole(x₁, x₂)
    t = inv(ζ)
    (A, A′), (D, D′), (Q, Q′) = map(c -> (evalpoly(t, c[1]), evalpoly(t, c[2])), (f.A, f.D, f.Q))
    # derivatives in z: d/dζ = −t² d/dt, then a division by ω′ = D
    φ′ = A / D
    φ″ = -t^2 * (A′ * D - A * D′) / D^3
    ψ′ = (f.Γ′ + t^2 * (f.Γ + (Q′ * D - Q * D′) / D^2)) / D
    trace, deviator = 4 * real(φ′), 2 * (conj(z) * φ″ + ψ′)       # σ₁₁ + σ₂₂ and σ₂₂ − σ₁₁ + 2i σ₁₂
    return ((trace - real(deviator)) / 2, (trace + real(deviator)) / 2, imag(deviator) / 2)
end

# ── Anisotropic matrix: the ellipse ──────────────────────────────────────────

struct _AnisotropicField{M <: Complex, P <: Complex, S <: Tuple}
    m::M
    s::P
    p::P
    Σ::S
end

# R with R² = z² − 4αβ on the branch of the point outside the ellipse, for
# which ζ = (z + R)/(2α) is the root of larger modulus of α ζ² − z ζ + β.
_exterior_root(z, αβ) = (R = sqrt(z^2 - 4 * αβ); abs(z + R) ≥ abs(z - R) ? R : -R)

# The square root of `w` on the branch whose value is the closer to `near`.
_sqrt_near(w, near) = (r = sqrt(w); abs(TensND.primal(real(r)) + im * TensND.primal(imag(r)) - near) ≤ abs(TensND.primal(real(r)) + im * TensND.primal(imag(r)) + near) ? r : -r)

function (f::_AnisotropicField)(x₁, x₂)
    _numbers_only(promote_type(typeof(real(f.s)), typeof(real(f.m)), typeof(f.Σ[1]), typeof(x₁), typeof(x₂)))
    ξ, η = _contour_vectors(f.m)
    # inside the hole: the variable of the map of the ellipse itself
    z = complex(x₁, x₂)
    abs(z + _exterior_root(z, f.m)) ≥ 2 * (1 - 1.0e-10) || _throw_inside_hole(x₁, x₂)
    c₁, c₂ = _boundary_constants(f.Σ, ξ, η)
    s, p = f.s, f.p
    δ = sqrt(s^2 - 4 * p)                                   # μ₁ − μ₂
    μ₁, μ₂ = (s + δ) / 2, (s - δ) / 2
    R₁, R₂ = map((μ₁, μ₂)) do μ
        _exterior_root(x₁ + μ * x₂, (conj(ξ) + μ * conj(η)) * (ξ + μ * η) / 4)
    end
    # Relative size of the two divisors, on the values.
    separated = abs(_value(δ)) * (abs(_value(R₁)) + abs(_value(R₂))) > abs(_value(R₁ + R₂)) * (abs(_value(μ₁)) + abs(_value(μ₂)))
    S₀, S₁, S₂ = separated ? _field_by_roots(x₁, x₂, ξ, η, c₁, c₂, (μ₁, μ₂), (R₁, R₂)) :
        _field_by_pairs(x₁, x₂, ξ, η, c₁, c₂, s, p, _value(R₁), _value(R₂))
    return (f.Σ[1] + 2 * real(S₂), f.Σ[2] + 2 * real(S₀), f.Σ[3] - 2 * real(S₁))
end

_value(w::Complex) = complex(TensND.primal(real(w)), TensND.primal(imag(w)))

# Sₙ = Σₖ μₖⁿ Bₖ Tₖ, n = 0, 1, 2, with the two roots.
function _field_by_roots(x₁, x₂, ξ, η, c₁, c₂, (μ₁, μ₂), (R₁, R₂))
    B₁, B₂ = (c₂ - μ₂ * c₁) / (μ₁ - μ₂), (μ₁ * c₁ - c₂) / (μ₁ - μ₂)
    T₁, T₂ = map((μ₁, μ₂), (R₁, R₂)) do μ, R
        -(conj(ξ) + μ * conj(η)) / ((x₁ + μ * x₂ + R) * R)
    end
    return ntuple(n -> μ₁^(n - 1) * B₁ * T₁ + μ₂^(n - 1) * B₂ * T₂, Val(3))
end

# A quantity on the two roots as its mean `a` and its divided difference `d`:
# its values are a ± d (μ₁ − μ₂)/2, and the products follow ε² = Δ/4.
struct _RootPair{S}
    a::S
    d::S
end
_pair_product(x::_RootPair, y::_RootPair, Δ₄) = _RootPair(x.a * y.a + Δ₄ * x.d * y.d, x.a * y.d + x.d * y.a)
_pair_inverse(x::_RootPair, Δ₄) = (n = x.a^2 - Δ₄ * x.d^2; _RootPair(x.a / n, -x.d / n))

# The same sums on the pairs: with Dₙ the divided difference of μⁿ T,
# Sₙ = c₂ Dₙ − c₁ p Dₙ₋₁. `R₁`, `R₂` are values, which fix the branches.
function _field_by_pairs(x₁, x₂, ξ, η, c₁, c₂, s, p, R₁::Complex, R₂::Complex)
    Δ₄ = (s^2 - 4 * p) / 4
    S = typeof(Δ₄ * x₁)
    μ = _RootPair{S}(s / 2, one(S))
    z = _RootPair{S}(x₁ + s * x₂ / 2, x₂)
    α = _RootPair{S}((conj(ξ) + s * conj(η) / 2) / 2, conj(η) / 2)
    β = _RootPair{S}((ξ + s * η / 2) / 2, η / 2)
    zz, αβ = _pair_product(z, z, Δ₄), _pair_product(α, β, Δ₄)
    q = _RootPair{S}(zz.a - 4 * αβ.a, zz.d - 4 * αβ.d)       # R² on the two roots
    Π = _sqrt_near(q.a^2 - Δ₄ * q.d^2, R₁ * R₂)              # R₁ R₂
    ΣR = _sqrt_near(2 * q.a + 2 * Π, R₁ + R₂)                # R₁ + R₂
    R = _RootPair{S}(ΣR / 2, q.d / ΣR)
    zR = _RootPair{S}(z.a + R.a, z.d + R.d)
    T = _pair_product(_RootPair{S}(-2 * α.a, -2 * α.d), _pair_inverse(_pair_product(zR, R, Δ₄), Δ₄), Δ₄)
    μT = _pair_product(μ, T, Δ₄)
    D₋₁, D₀, D₁, D₂ = _pair_product(_pair_inverse(μ, Δ₄), T, Δ₄).d, T.d, μT.d, _pair_product(μ, μT, Δ₄).d
    return (c₂ * D₀ - c₁ * p * D₋₁, c₂ * D₁ - c₁ * p * D₀, c₂ * D₂ - c₁ * p * D₁)
end
