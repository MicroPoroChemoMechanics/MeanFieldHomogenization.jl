# =============================================================================
#  solid_harmonics.jl — regular and decaying solid harmonics.
#
#  With (r, θ, φ) the spherical coordinates of x about the axis e₃ and
#  χ_n^m = (−1)^m P_n^m(cos θ) e^{imφ} the surface harmonic of degree n and
#  order 0 ≤ m ≤ n (P_n^m the associated Legendre function of the first kind as
#  in Barthélémy & Bignonnet (2020) and in LayeredSpheroids/legendre.jl, for
#  which P_1^1(p) = −√(1−p²); so χ_1^1 = sin θ e^{iφ}), the solid harmonics are
#  normalized as in kushch2013, Eq. (2.9):
#
#       y_n^m(x) = r^n χ_n^m / (n + m)! ,     Y_n^m(x) = (n − m)! χ_n^m / r^{n+1} .
#
#  This normalization is the one in which the translation of a decaying
#  harmonic to another center has coefficients that are decaying harmonics
#  themselves, with no further factor:
#
#       Y_n′^m′(r + R) = Σ_{n,m} (−1)^{n+m} Y_{n′+n}^{m′−m}(R) y_n^m(r) ,  ‖r‖ < ‖R‖ .
#
#  Both families are evaluated by recurrences on the Cartesian components,
#  with ζ = x₁ + i x₂:
#
#       y_m^m     = y_{m−1}^{m−1} ζ / (2m) ,
#       y_{n+1}^m = [(2n+1) x₃ y_n^m − r² y_{n−1}^m] / [(n−m+1)(n+m+1)] ,
#       Y_m^m     = Y_{m−1}^{m−1} (2m−1) ζ / r² ,
#       Y_{n+1}^m = [(2n+1) x₃ Y_n^m − (n−m)(n+m) Y_{n−1}^m] / r² ,
#
#  which involve neither an angle nor a factorial, and hold for any number type
#  the coordinates carry. Negative orders follow from
#  y_n^{−m} = (−1)^m conj(y_n^m), and the same for Y.
# =============================================================================

"""
    HarmonicTable{C}

Values of a family of solid harmonics at one point, for the degrees
``0 \\le n \\le n_{\\max}`` and every order: `h[n, m]`. Only the orders
``m \\ge 0`` are stored, the others following from
``h_n^{-m} = (-1)^m\\,\\overline{h_n^m}``; an order beyond the degree, which the
translation formulas ask for, reads as zero.
"""
struct HarmonicTable{C <: Complex}
    nmax::Int
    values::Vector{C}
end

@inline _packed(n::Integer, m::Integer) = (n * (n + 1)) >> 1 + m + 1

function Base.getindex(h::HarmonicTable{C}, n::Integer, m::Integer) where {C}
    0 ≤ n ≤ h.nmax || throw(BoundsError(h, (n, m)))
    abs(m) > n && return zero(C)
    v = h.values[_packed(n, abs(m))]
    return m ≥ 0 ? v : (isodd(m) ? -conj(v) : conj(v))
end

# Fill the sectoral (n = m) terms, then each order upward in the degree.
function _harmonic_table(x::AbstractVector, nmax::Integer, sectoral::F, step::G) where {F, G}
    length(x) == 3 ||
        throw(ArgumentError("solid harmonics are defined in 3D; got a $(length(x))-component point"))
    nmax ≥ 0 || throw(ArgumentError("solid harmonics: the highest degree must be non-negative, got $nmax"))
    T = float(eltype(x))
    ζ = complex(T(x[1]), T(x[2]))
    z = T(x[3])
    r² = abs2(ζ) + z * z
    v = Vector{Complex{T}}(undef, _packed(nmax, nmax))
    v[1] = sectoral(0, zero(Complex{T}), ζ, r²)
    for m in 0:nmax
        m > 0 && (v[_packed(m, m)] = sectoral(m, v[_packed(m - 1, m - 1)], ζ, r²))
        prev = zero(Complex{T})
        for n in m:(nmax - 1)
            cur = v[_packed(n, m)]
            v[_packed(n + 1, m)] = step(n, m, cur, prev, z, r²)
            prev = cur
        end
    end
    return HarmonicTable(Int(nmax), v)
end

"""
    regular_harmonics(x, nmax) -> HarmonicTable

Regular solid harmonics ``y_n^m(\\underline{x}) = (-1)^m\\,r^n\\,P_n^m(\\cos\\theta)\\,
\\mathrm{e}^{\\mathrm{i}m\\varphi}/(n+m)!`` at the point `x`, up to the degree
`nmax`. They are homogeneous polynomials of the coordinates, hence defined at
the origin too.
"""
regular_harmonics(x::AbstractVector, nmax::Integer) = _harmonic_table(
    x, nmax,
    (m, prev, ζ, r²) -> m == 0 ? one(ζ) : prev * ζ / (2m),
    (n, m, cur, prev, z, r²) -> ((2n + 1) * z * cur - r² * prev) / ((n - m + 1) * (n + m + 1))
)

"""
    irregular_harmonics(x, nmax) -> HarmonicTable

Decaying solid harmonics ``Y_n^m(\\underline{x}) = (-1)^m\\,(n-m)!\\,P_n^m(\\cos\\theta)\\,
\\mathrm{e}^{\\mathrm{i}m\\varphi}/r^{n+1}`` at the point `x`, up to the degree
`nmax`. The point must differ from the origin, where they are singular.
"""
function irregular_harmonics(x::AbstractVector, nmax::Integer)
    all(iszero, x) && throw(DomainError(x, "the decaying solid harmonics are singular at the origin"))
    return _harmonic_table(
        x, nmax,
        (m, prev, ζ, r²) -> m == 0 ? complex(inv(sqrt(r²))) : prev * (2m - 1) * ζ / r²,
        (n, m, cur, prev, z, r²) -> ((2n + 1) * z * cur - (n - m) * (n + m) * prev) / r²
    )
end

"""
    harmonic_norm(n, m) -> Float64

``\\sqrt{(n-m)!\\,(n+m)!}``, the factor by which the coefficients of an
expansion in solid harmonics are scaled so that the linear system they solve
keeps entries of order one at every degree [kushch2013; Eq. (2.52)](@cite).
"""
harmonic_norm(n::Integer, m::Integer) = exp((logfactorial(n - m) + logfactorial(n + m)) / 2)
