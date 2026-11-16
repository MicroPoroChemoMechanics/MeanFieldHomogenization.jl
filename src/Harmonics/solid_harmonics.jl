# =============================================================================
#  solid_harmonics.jl — regular and decaying solid harmonics.
#
#  With (r, θ, φ) the spherical coordinates of x about the axis e₃ and
#  χ_t^s = P_t^s(cos θ) e^{isφ} the surface harmonic of degree t and order s
#  (P_t^s the associated Legendre function WITHOUT the Condon-Shortley phase,
#  so that P_1^1 = sin θ), the solid harmonics are normalized as in
#  kushch2013, Eq. (2.9):
#
#       y_t^s(x) = r^t χ_t^s / (t + s)! ,     Y_t^s(x) = (t − s)! χ_t^s / r^{t+1} .
#
#  This normalization is the one in which the translation of a decaying
#  harmonic to another center has coefficients that are decaying harmonics
#  themselves, with no further factor:
#
#       Y_k^l(r + R) = Σ_{t,s} (−1)^{t+s} Y_{k+t}^{l−s}(R) y_t^s(r) ,  ‖r‖ < ‖R‖ .
#
#  Both families are evaluated by recurrences on the Cartesian components,
#  with ζ = x₁ + i x₂:
#
#       y_s^s     = y_{s−1}^{s−1} ζ / (2s) ,
#       y_{t+1}^s = [(2t+1) x₃ y_t^s − r² y_{t−1}^s] / [(t−s+1)(t+s+1)] ,
#       Y_s^s     = Y_{s−1}^{s−1} (2s−1) ζ / r² ,
#       Y_{t+1}^s = [(2t+1) x₃ Y_t^s − (t−s)(t+s) Y_{t−1}^s] / r² ,
#
#  which involve neither an angle nor a factorial, and hold for any number type
#  the coordinates carry. Negative orders follow from
#  y_t^{−s} = (−1)^s conj(y_t^s), and the same for Y.
# =============================================================================

"""
    HarmonicTable{C}

Values of a family of solid harmonics at one point, for the degrees
``0 \\le t \\le t_{\\max}`` and every order: `h[t, s]`. Only the orders
``s \\ge 0`` are stored, the others following from
``h_t^{-s} = (-1)^s\\,\\overline{h_t^s}``; an order beyond the degree, which the
translation formulas ask for, reads as zero.
"""
struct HarmonicTable{C <: Complex}
    tmax::Int
    values::Vector{C}
end

@inline _packed(t::Integer, s::Integer) = (t * (t + 1)) >> 1 + s + 1

function Base.getindex(h::HarmonicTable{C}, t::Integer, s::Integer) where {C}
    0 ≤ t ≤ h.tmax || throw(BoundsError(h, (t, s)))
    abs(s) > t && return zero(C)
    v = h.values[_packed(t, abs(s))]
    return s ≥ 0 ? v : (isodd(s) ? -conj(v) : conj(v))
end

# Fill the sectoral (t = s) terms, then each order upward in the degree.
function _harmonic_table(x::AbstractVector, tmax::Integer, sectoral::F, step::G) where {F, G}
    length(x) == 3 ||
        throw(ArgumentError("solid harmonics are defined in 3D; got a $(length(x))-component point"))
    tmax ≥ 0 || throw(ArgumentError("solid harmonics: the highest degree must be non-negative, got $tmax"))
    T = float(eltype(x))
    ζ = complex(T(x[1]), T(x[2]))
    z = T(x[3])
    r² = abs2(ζ) + z * z
    v = Vector{Complex{T}}(undef, _packed(tmax, tmax))
    v[1] = sectoral(0, zero(Complex{T}), ζ, r²)
    for s in 0:tmax
        s > 0 && (v[_packed(s, s)] = sectoral(s, v[_packed(s - 1, s - 1)], ζ, r²))
        prev = zero(Complex{T})
        for t in s:(tmax - 1)
            cur = v[_packed(t, s)]
            v[_packed(t + 1, s)] = step(t, s, cur, prev, z, r²)
            prev = cur
        end
    end
    return HarmonicTable(Int(tmax), v)
end

"""
    regular_harmonics(x, tmax) -> HarmonicTable

Regular solid harmonics ``y_t^s(\\underline{x}) = r^t\\,P_t^s(\\cos\\theta)\\,
\\mathrm{e}^{\\mathrm{i}s\\varphi}/(t+s)!`` at the point `x`, up to the degree
`tmax`. They are homogeneous polynomials of the coordinates, hence defined at
the origin too.
"""
regular_harmonics(x::AbstractVector, tmax::Integer) = _harmonic_table(
    x, tmax,
    (s, prev, ζ, r²) -> s == 0 ? one(ζ) : prev * ζ / (2s),
    (t, s, cur, prev, z, r²) -> ((2t + 1) * z * cur - r² * prev) / ((t - s + 1) * (t + s + 1))
)

"""
    irregular_harmonics(x, tmax) -> HarmonicTable

Decaying solid harmonics ``Y_t^s(\\underline{x}) = (t-s)!\\,P_t^s(\\cos\\theta)\\,
\\mathrm{e}^{\\mathrm{i}s\\varphi}/r^{t+1}`` at the point `x`, up to the degree
`tmax`. The point must differ from the origin, where they are singular.
"""
function irregular_harmonics(x::AbstractVector, tmax::Integer)
    all(iszero, x) && throw(DomainError(x, "the decaying solid harmonics are singular at the origin"))
    return _harmonic_table(
        x, tmax,
        (s, prev, ζ, r²) -> s == 0 ? complex(inv(sqrt(r²))) : prev * (2s - 1) * ζ / r²,
        (t, s, cur, prev, z, r²) -> ((2t + 1) * z * cur - (t - s) * (t + s) * prev) / r²
    )
end

"""
    harmonic_norm(t, s) -> Float64

``\\sqrt{(t-s)!\\,(t+s)!}``, the factor by which the coefficients of an
expansion in solid harmonics are scaled so that the linear system they solve
keeps entries of order one at every degree [kushch2013; Eq. (2.52)](@cite).
"""
harmonic_norm(t::Integer, s::Integer) = exp((logfactorial(t - s) + logfactorial(t + s)) / 2)
