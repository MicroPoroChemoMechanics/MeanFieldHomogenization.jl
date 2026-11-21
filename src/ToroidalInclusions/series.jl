# ── Series of toroidal harmonics around a torus ──────────────────────────────
#
# Outside the torus `α = α₀` a field is the sum of a remote part, written on the
# functions of the second kind `Q`, and of a perturbation that vanishes at
# infinity, written on those of the first kind `P`:
#
#     √(cosh α - cos β) Σₙ [sₙ Q_{n-1/2}^m(cosh α) + bₙ P_{n-1/2}^m(cosh α)] {sin nβ, cos nβ},
#
# with a common factor `cos mϕ`. The source amplitudes `sₙ` are known; a
# condition on the torus gives the `bₙ`.

# The amplitudes decay as `e^{-2nα₀}`: the number of terms that brings them
# below the rounding of the number type, with a margin.
_series_length(α₀) = ceil(Int, precision(typeof(TensND.primal(α₀))) * log(2) / (2 * TensND.primal(α₀))) + 8

# First index of a series, and weight of the term `n-1` in the relation of
# index `n`: a series of sines starts at 1; in a series of cosines the product
# `2 cos β cos nβ` sends the term 0 twice onto the index 1.
_first_index(::Val{:sine}) = 1
_first_index(::Val{:cosine}) = 0
_lower_weight(::Val{:sine}, n::Integer) = n == 1 ? 0 : 1
_lower_weight(::Val{:cosine}, n::Integer) = n == 0 ? 0 : n == 1 ? 2 : 1

"""
    _neumann_amplitudes(parity, m, s, α₀) -> Vector

Amplitudes `bₙ` of the perturbation that makes the normal derivative of the
series vanish on the torus `α = α₀`, for the source amplitudes `s[n+1] = sₙ`,
`0 ≤ n ≤ N+1`, of a series of sines (`Val(:sine)`, `n ≥ 1`) or of cosines
(`Val(:cosine)`, `n ≥ 0`) of order `m`. The result holds `bₙ` from the first
index of the series to `N`.

With `fₙ = sₙ Qₙ + bₙ Pₙ`, the derivative in `α` of `√δ Σ fₙ {sin, cos} nβ`
vanishes when

    sinh α₀ fₙ + 2 cosh α₀ fₙ′ - wₙ f′ₙ₋₁ - f′ₙ₊₁ = 0,

`wₙ` being the weight of `_lower_weight`. The system is tridiagonal; it is
written in the unknowns `xₙ = bₙ Pₙ`, which keeps its coefficients of the order
of `n` while `Pₙ` grows as `e^{nα₀}`, and solved by elimination with pivoting.
`b_{N+1} = 0` closes it.
"""
function _neumann_amplitudes(parity::Val, m::Integer, s::AbstractVector, α₀)
    N = length(s) - 2
    sh, ch = sinh(α₀), cosh(α₀)
    P, dP = toroidal_P(N + 1, m, α₀), toroidal_dP(N + 1, m, α₀)
    Q, dQ = toroidal_Q(N + 1, m, α₀), toroidal_dQ(N + 1, m, α₀)
    ℓ = dP ./ P
    n₀ = _first_index(parity)
    diagonal = [sh + 2 * ch * ℓ[n + 1] for n in n₀:N]
    lower = [-_lower_weight(parity, n) * ℓ[n] for n in (n₀ + 1):N]
    upper = [-ℓ[n + 2] for n in n₀:(N - 1)]
    # The term `n-1` of the source has the weight 0 at the first index, where
    # `max(n, 1)` only keeps the index in the vectors.
    rhs = [
        -(
            s[n + 1] * (sh * Q[n + 1] + 2 * ch * dQ[n + 1]) -
                _lower_weight(parity, n) * s[max(n, 1)] * dQ[max(n, 1)] - s[n + 2] * dQ[n + 2]
        ) for n in n₀:N
    ]
    x = Tridiagonal(lower, diagonal, upper) \ rhs
    return x ./ view(P, (n₀ + 1):(N + 1))
end
