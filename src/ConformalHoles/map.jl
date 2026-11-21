# ── The conformal map and its series ─────────────────────────────────────────
#
# ω(ζ) = ζ + Σₖ mₖ ζ⁻ᵏ, 1 ≤ k ≤ N, maps the outside of the unit disk onto the
# outside of the hole. The size R of the hole is left out: the tensors per unit
# area of the hole do not depend on it.

# The map is one-to-one outside the disk when Σ k|mₖ| ≤ 1: for two points of
# that domain, ω(ζ₁) - ω(ζ₂) = (ζ₁ - ζ₂)(1 - Σ mₖ sₖ), where sₖ is a sum of k
# terms of modulus at most 1. Symbols do not compare: the condition is then left
# to the caller.
_check_map(m::Vector{Complex{T}}) where {T} = _check_map(Val(is_hard_numeric(T)), m)
_check_map(::Val{false}, m) = nothing
function _check_map(::Val{true}, m::Vector{Complex{T}}) where {T}
    s = sum(k * abs(mₖ) for (k, mₖ) in pairs(m))
    s ≤ 1 + 4 * eps(float(TensND.primal(one(T)))) || throw(
        DomainError(
            TensND.primal(s),
            "the coefficients mₖ of a ConformalHole must satisfy Σ k|mₖ| ≤ 1, which makes the map " *
                "one-to-one; got Σ k|mₖ| = $(TensND.primal(s))."
        )
    )
    _hole_area(m) > 0 || throw(
        DomainError(
            TensND.primal(m[1]),
            "the map with |m₁| = 1 is that of a slit, which has no area: use a crack (RibbonCrack) instead."
        )
    )
    return nothing
end

# Area of the hole divided by π R²: ½ Im ∮ z̄ dz over the image of the circle.
_hole_area(m::Vector{<:Complex}) = 1 - sum(k * abs2(mₖ) for (k, mₖ) in pairs(m))

# Coefficients of 1/ω′(ζ) = Σₗ gₗ ζ⁻ˡ, 0 ≤ l ≤ L, with ω′ = 1 - Σ k mₖ ζ⁻ᵏ⁻¹:
# g₀ = 1 and gₗ = Σₖ k mₖ g_{l-k-1}. Each term reads the previous ones.
function _inverse_derivative_series(m::Vector{Complex{T}}, L::Integer) where {T}
    g = zeros(Complex{T}, L + 1)
    g[1] = one(T)
    for l in 1:L
        g[l + 1] = sum(k * m[k] * g[l - k] for k in 1:min(length(m), l - 1); init = zero(Complex{T}))
    end
    return g
end

"""
    polygon_map(n, terms) -> Vector{Pair{Int, Float64}}

First `terms` terms of the map of the outside of the unit disk onto the outside
of a regular polygon of `n ≥ 3` sides, as pairs `degree => coefficient` for
[`ConformalHole`](@ref).

The derivative of the Schwarz-Christoffel map of the polygon is
``\\omega'(\\zeta) = (1 - \\zeta^{-n})^{2/n}``; its binomial series, integrated
term by term, gives

```math
m_{jn-1} = \\frac{(-1)^{j}}{1 - jn}\\binom{2/n}{j},\\qquad j = 1, 2, \\dots
```

The truncated map is a polygon with rounded corners, the sharper the more terms
are kept; one vertex lies on the first axis.

```jldoctest
julia> polygon_map(4, 2)
2-element Vector{Pair{Int64, Float64}}:
 3 => 0.16666666666666666
 7 => 0.017857142857142856
```
"""
function polygon_map(n::Integer, terms::Integer)
    n ≥ 3 || throw(ArgumentError("a polygon has three sides at least; got n = $n"))
    terms ≥ 1 || throw(ArgumentError("polygon_map needs one term at least; got terms = $terms"))
    binomials = accumulate((c, j) -> c * (2 / n - j + 1) / j, 1:terms; init = 1.0)
    return [j * n - 1 => (-1)^j * binomials[j] / (1 - j * n) for j in 1:terms]
end
