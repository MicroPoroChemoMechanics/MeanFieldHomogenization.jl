# =============================================================================
#  coupling.jl — interface coupling matrices I, J, K, L (Barthélémy &
#  Bignonnet, IJES 2020, eq:Iij/Jij/Kij/Lij).
#
#  These matrices couple the spheroidal-harmonic degrees at an imperfect
#  (LC/HC) interface — see `conductivity.jl`. For a layer at confocal
#  parameter `q`, with odd degrees `i = 2r-1`, `j = 2s-1` (r,s = 1..𝒩):
#
#      I_{rs}(q) = ∫₋₁¹  Pᵢ(x) Pⱼ(x)              / √(q²-x²)   dx
#      J_{rs}(q) = ∫₋₁¹  Pᵢ¹(x) Pⱼ¹(x)             / √(q²-x²)   dx
#      K_{rs}(q) = ∫₋₁¹  Pᵢ¹′(x) Pⱼ¹′(x) (1-x²)    / √(q²-x²)   dx
#      L_{rs}(q) = ∫₋₁¹  Pᵢ′(x) Pⱼ′(x) √(q²-x²)                dx
#
#  DEFAULT METHOD — direct Gauss quadrature (`method = :quadrature`):
#  the integrands only ever involve Legendre *values* Pᵢ(x), Pᵢ¹(x) and
#  their *derivatives* at real `x ∈ [-1,1]` (`|x| ≤ 1`, never the
#  monomial expansion of Pᵢ), obtained by the stable recurrence of
#  `legendre.jl`. For prolate `q` (real, `q > 1`) the weight
#  `1/√(q²-x²)` is smooth and bounded on `[-1,1]`; for oblate `q = iτ`
#  the integrand is complex-analytic on a strip around `[-1,1]` (no
#  real singularity). Either way `QuadGK.quadgk` converges at
#  essentially machine precision in `Float64`, and — crucially — NEVER
#  forms the ill-conditioned monomial-coefficient sums
#  `Iᵢⱼ = Σₖ γ₂ₖ^{i,j} Wₖ(q)` of the original mpmath implementation
#  (see the module docstring and the theory page for the precision
#  argument, eq:maxlogcij).
#
#  OPTIONAL METHOD — `method = :series`: a faithful BigFloat port of the
#  original monomial-coefficient summation (`γ, η, δ` tables + the
#  hypergeometric `Wₖ(q)`, eq:cij/eq:eij/eq:dij/eq:Wk2), kept as an
#  independent validation oracle (cross-checked against `:quadrature`
#  in the test suite) and to reproduce the paper's convergence figure.
#  Precision follows the paper's rule `dps ≳ 0.8(2𝒩-1)` decimal digits.
# =============================================================================

using QuadGK: quadgk

# ── Default path: direct quadrature of the closed-form integrals ───────────

function _coupling_I(q, Nseries::Int)
    f(x) = begin
        P, _ = legendre_odd(:P0, x, Nseries)
        (P * P') ./ sqrt(q^2 - x^2)
    end
    M, _ = quadgk(f, -1.0, 1.0)
    return M
end

function _coupling_J(q, Nseries::Int)
    f(x) = begin
        P1, _ = legendre_odd(:P1p, x, Nseries)
        (P1 * P1') ./ sqrt(q^2 - x^2)
    end
    M, _ = quadgk(f, -1.0, 1.0)
    return M
end

function _coupling_K(q, Nseries::Int)
    f(x) = begin
        _, dP1 = legendre_odd(:P1p, x, Nseries)
        ((1 - x^2) / sqrt(q^2 - x^2)) .* (dP1 * dP1')
    end
    M, _ = quadgk(f, -1.0, 1.0)
    return M
end

function _coupling_L(q, Nseries::Int)
    f(x) = begin
        _, dP0 = legendre_odd(:P0, x, Nseries)
        sqrt(q^2 - x^2) .* (dP0 * dP0')
    end
    M, _ = quadgk(f, -1.0, 1.0)
    return M
end

"""
    coupling_matrices(q, Nseries; method = :quadrature) -> (I, J, K, L)

The four ``\\mathcal{N}\\times\\mathcal{N}`` interface coupling matrices (``\\mathcal{N}`` = `Nseries`) at confocal
parameter ``q`` (real for prolate, ``q = \\mathrm{i}\\tau`` for oblate), restricted to
the odd degrees ``1, 3, \\dots, 2\\mathcal{N}-1`` (index ``r`` ``\\leftrightarrow`` degree ``2r-1``).

`method = :quadrature` (default) integrates the paper's closed-form
definitions directly (stable in `Float64`); `method = :series` uses the
BigFloat monomial-coefficient summation of the original implementation
(see the module docstring).
"""
function coupling_matrices(q, Nseries::Int; method::Symbol = :quadrature)
    if method === :quadrature
        return _coupling_I(q, Nseries), _coupling_J(q, Nseries),
            _coupling_K(q, Nseries), _coupling_L(q, Nseries)
    elseif method === :series
        return _coupling_matrices_series(q, Nseries)
    else
        throw(ArgumentError("coupling_matrices: unknown method $method"))
    end
end

# =============================================================================
#  Optional path: BigFloat monomial-coefficient series (validation oracle)
# =============================================================================
#
#  Faithful implementation of the paper's monomial-coefficient series,
#  using its recursive
#  algorithms for the polynomial coefficients (eq:cij/eq:eij/eq:dij) and
#  the hypergeometric closed form for `Wₖ(q)` (eq:Wk2), summed directly
#  as a series (converges since `|1/q²| < 1` for `|q| > 1`).
#
#  γ^{i,j}, η^{i,j}, δ^{i,j} are coefficient VECTORS of the monomial
#  expansions of `Pᵢ Pⱼ`, `Pᵢ′ Pⱼ`, `Pᵢ′ Pⱼ′` (index `l ↔` power `xˡ`,
#  1-indexed as `[l+1]`); `_gam[i+1,j+1]` stores `γ^{i,j}` for `i,j = 0,…,Nmax-1`.

# The valid-index bound is the EXPLICIT degree formula `i+j-d`,
# NOT the actual array length: `d`
# is a fixed constant per table (`γ`: 0, `η`: 1, `δ`: 2), reflecting that
# `γ^{i,j}`, `η^{i,j}`, `δ^{i,j}` are the coefficients of `Pᵢ Pⱼ`,
# `Pᵢ′ Pⱼ`, `Pᵢ′ Pⱼ′` — polynomials of degree `i+j`, `i+j-1`, `i+j-2`
# respectively — evaluated at *possibly shifted* arguments (`i-1`, or
# `j,i` swapped) for which the bound must be recomputed from (i,j), not
# read off whatever array happens to be stored.
@inline function _tab_get(tab, i, j, l, d::Int)
    (l < 0 || l > i + j - d) && return nothing
    return tab[i + 1, j + 1][l + 1]
end
@inline _tab_getz(tab, i, j, l, ::Type{Tb}, d::Int) where {Tb} =
    something(_tab_get(tab, i, j, l, d), zero(Tb))

@inline _γget(γ, i, j, l, ::Type{Tb}) where {Tb} = _tab_getz(γ, i, j, l, Tb, 0)
@inline _ηget(η, i, j, l, ::Type{Tb}) where {Tb} = _tab_getz(η, i, j, l, Tb, 1)
@inline _δget(δ, i, j, l, ::Type{Tb}) where {Tb} = _tab_getz(δ, i, j, l, Tb, 2)

function _gamma_table(Nmax::Int, ::Type{Tb}) where {Tb}
    γ = Matrix{Vector{Tb}}(undef, Nmax, Nmax)
    γ[1, 1] = Tb[1]
    γ[2, 1] = Tb[0, 1]
    γ[1, 2] = Tb[0, 1]
    γ[2, 2] = Tb[0, 0, 1]
    for j in 0:(Nmax - 1)
        imin = j < 3 ? 1 : j - 1
        for i in imin:(Nmax - 2)
            co = Tb[]
            for l in 0:(i + j + 1)
                push!(
                    co,
                    (
                        (2i + 1) * _γget(γ, i, j, l - 1, Tb) -
                            i * _γget(γ, i - 1, j, l, Tb)
                    ) / (i + 1)
                )
            end
            γ[i + 2, j + 1] = co
            i + 1 != j && (γ[j + 1, i + 2] = co)
        end
    end
    return γ
end

function _eta_table(Nmax::Int, γ, ::Type{Tb}) where {Tb}
    η = Matrix{Vector{Tb}}(undef, Nmax, Nmax)
    η[1, 1] = Tb[0]
    η[1, 2] = Tb[0]
    η[2, 1] = Tb[1]
    η[2, 2] = Tb[0, 1]
    for j in 0:(Nmax - 1)
        imin = j < 3 ? 1 : j - 1
        for i in imin:(Nmax - 2)
            eo = Tb[]
            for l in 0:(i + j)
                push!(
                    eo,
                    (
                        (2i + 1) * _γget(γ, i, j, l, Tb) +
                            (2i + 1) * _ηget(η, i, j, l - 1, Tb) -
                            i * _ηget(η, i - 1, j, l, Tb)
                    ) / (i + 1)
                )
            end
            η[i + 2, j + 1] = eo
            if i + 1 != j
                eo2 = Tb[]
                for l in 0:(i + j)
                    push!(
                        eo2,
                        (
                            (2i + 1) * _ηget(η, j, i, l - 1, Tb) -
                                i * _ηget(η, j, i - 1, l, Tb)
                        ) / (i + 1)
                    )
                end
                η[j + 1, i + 2] = eo2
            end
        end
    end
    return η
end

function _delta_table(Nmax::Int, η, ::Type{Tb}) where {Tb}
    δ = Matrix{Vector{Tb}}(undef, Nmax, Nmax)
    δ[1, 1] = Tb[0]
    δ[2, 1] = Tb[0]
    δ[1, 2] = Tb[0]
    δ[2, 2] = Tb[1]
    for j in 0:(Nmax - 1)
        imin = j < 3 ? 1 : j - 1
        for i in imin:(Nmax - 2)
            d = Tb[]
            for l in 0:(i + j - 1)
                push!(
                    d,
                    (
                        (2i + 1) * _ηget(η, j, i, l, Tb) +
                            (2i + 1) * _δget(δ, i, j, l - 1, Tb) -
                            i * _δget(δ, i - 1, j, l, Tb)
                    ) / (i + 1)
                )
            end
            δ[i + 2, j + 1] = d
            i + 1 != j && (δ[j + 1, i + 2] = d)
        end
    end
    return δ
end

"""
    _Wk_series(q, kmax, ::Type{Tb}) -> Vector{Tb}

``W_k(q) = \\int_{-1}^{1} x^{2k}/\\sqrt{q^2-x^2}\\,\\mathrm{d}x``, ``k = 0, \\dots, k_{\\max}``, via the hypergeometric
series (eq:Wk2): ``W_k(q) = \\frac{2}{q(1+2k)}\\,{}_2F_1\\bigl(\\tfrac{1}{2}, \\tfrac{1}{2}+k; \\tfrac{3}{2}+k; 1/q^2\\bigr)``,
summed directly in the (Big)Float type `Tb` (converges since
``|1/q^2| < 1`` for ``|q| > 1``).
"""
function _Wk_series(q::Tq, kmax::Int, ::Type{Tb}) where {Tq, Tb}
    z = one(Tq) / q^2
    out = Vector{Tq}(undef, kmax + 1)
    for k in 0:kmax
        a, b, c = Tb(1) / 2, Tb(1) / 2 + k, Tb(3) / 2 + k
        term = one(Tq)
        s = one(Tq)
        n = 0
        while true
            term *= (a + n) * (b + n) / ((c + n) * (n + 1)) * z
            n += 1
            s += term
            abs(term) < eps(Tb) * abs(s) && break
            n > 10_000 && error("_Wk_series: failed to converge")
        end
        out[k + 1] = 2 * s / (q * (1 + 2k))
    end
    return out
end

"""
    _coupling_matrices_series(q, Nseries) -> (I, J, K, L)

BigFloat monomial-coefficient computation of the coupling matrices, on
the odd-degree, i.e. "chess-filtered", submatrix. Precision is set from
`Nseries` following the paper's rule ``\\mathrm{dps} \\gtrsim 0.8\\,(2\\mathcal{N}-1)``, ``\\mathcal{N}`` = `Nseries`.
"""
function _coupling_matrices_series(q, Nseries::Int)
    Nmax = 2 * Nseries
    digits = max(round(Int, 0.8 * (2 * Nseries - 1)), 16)
    bits = ceil(Int, digits * log2(10)) + 16
    Tb = BigFloat
    Tq = q isa Complex ? Complex{Tb} : Tb
    return setprecision(() -> _coupling_matrices_series_impl(Tq(q), Nseries, Nmax, Tb), Tb, bits)
end

function _coupling_matrices_series_impl(q::Tq, Nseries::Int, Nmax::Int, ::Type{Tb}) where {Tq, Tb}
    γ = _gamma_table(Nmax, Tb)
    η = _eta_table(Nmax, γ, Tb)
    δ = _delta_table(Nmax, η, Tb)
    W = _Wk_series(q, Nmax, Tb)

    I = zeros(Tq, Nseries, Nseries)
    J = zeros(Tq, Nseries, Nseries)
    K = zeros(Tq, Nseries, Nseries)
    L = zeros(Tq, Nseries, Nseries)
    for r in 1:Nseries, s in r:Nseries
        i = 2r - 1
        j = 2s - 1
        # Iᵢⱼ = Σ_{k=0}^{(i+j)/2} γ_{2k} Wₖ
        kmax_ij = div(i + j, 2)
        Iv = sum(_γget(γ, i, j, 2k, Tb) * W[k + 1] for k in 0:kmax_ij)
        # Jᵢⱼ = Σ_{k=0}^{(i+j)/2-1} δ_{2k} (Wₖ - Wₖ₊₁)
        kmax_j = kmax_ij - 1
        Jv = kmax_j < 0 ? zero(Tq) :
            sum(_δget(δ, i, j, 2k, Tb) * (W[k + 1] - W[k + 2]) for k in 0:kmax_j)
        # Kᵢⱼ (eq:Kijsum)
        Kv = i * (i + 1) * j * (j + 1) * _γget(γ, i, j, 0, Tb) * W[1]
        for k in 1:kmax_ij
            Kv += (
                _δget(δ, i, j, 2k - 2, Tb) -
                    j * (j + 1) * _ηget(η, i, j, 2k - 1, Tb) -
                    i * (i + 1) * _ηget(η, j, i, 2k - 1, Tb) +
                    i * (i + 1) * j * (j + 1) * _γget(γ, i, j, 2k, Tb)
            ) * W[k + 1]
        end
        # Lᵢⱼ = Σ δ_{2k} (q² Wₖ - Wₖ₊₁)
        Lv = kmax_j < 0 ? zero(Tq) :
            sum(_δget(δ, i, j, 2k, Tb) * (q^2 * W[k + 1] - W[k + 2]) for k in 0:kmax_j)

        I[r, s] = Iv; I[s, r] = Iv
        J[r, s] = Jv; J[s, r] = Jv
        K[r, s] = Kv; K[s, r] = Kv
        L[r, s] = Lv; L[s, r] = Lv
    end
    return I, J, K, L
end
