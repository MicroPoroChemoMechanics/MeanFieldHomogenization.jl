# =============================================================================
#  green_residue.jl
#
#  Factored Masson / Cauchy residue algorithm used by both `Elasticity`
#  (Hill polarization tensor — Masson 2008) and `Cracks` (line-integrated
#  Green kernel ``\\hat{\\mathbf Q}^{\\star}_{nn}`` — Cauchy residue).
#
#  Strategy
#  ========
#  Both algorithms reduce their integrand to the same rational-function
#  pattern :
#
#      f(z) = N(z) / Q(z)
#
#  where
#
#    * `Q(z) = det K(z)` — degree-6 polynomial in `z`
#    * `K(z)` — acoustic tensor evaluated on a caller-supplied parametric
#               wave-vector `ζ(z) = α₀ + α₁ z` (linear polynomial)
#
#  The common building blocks are factored in [`_build_poly_system`](@ref):
#  the coefficient matrices `α₀, α₁, α₂` such that
#  `K(z) = α₀ + α₁ z + α₂ z²`, the polynomial adjugate `adj(K)(z)`, the
#  determinant `Q(z)` and its roots in the upper half plane; and the sum of the
#  residues of a rational function, times a logarithm or not, over groups of
#  poles ([`_cluster_weights`](@ref)).
#
#  The algorithm-specific numerators, and the poles they are summed over, are
#  set by the downstream wrappers — see
#  `Elasticity/hill_3d_aniso_residue.jl` and
#  `Cracks/green_residue.jl`.
# =============================================================================

"""
    _build_acoustic_coeffs(C, α₀ζ, α₁ζ) -> (A₀, A₁, A₂)

Build the three 3×3 coefficient matrices of
``\\boldsymbol{K}(z) = \\boldsymbol{A}_0 + \\boldsymbol{A}_1\\,z + \\boldsymbol{A}_2\\,z^2``
given the stiffness `C` (Float64 `3×3×3×3` array) and the two linear
coefficients `α₀ζ`, `α₁ζ` of the wave-vector parametrization
``\\underline{\\zeta}(z) = \\underline{\\zeta}_0 + \\underline{\\zeta}_1\\,z``.

Each `Aₖ` is a 3×3 matrix of real scalars ; they are assembled below
into `ComplexF64` polynomial entries before root finding.
"""
function _build_acoustic_coeffs(
        C::AbstractArray{<:Real, 4},
        α₀ζ::AbstractVector{<:Real},
        α₁ζ::AbstractVector{<:Real}
    )
    A₀ = zeros(Float64, 3, 3)
    A₁ = zeros(Float64, 3, 3)
    A₂ = zeros(Float64, 3, 3)
    @inbounds for j in 1:3, i in 1:3
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for k in 1:3, l in 1:3
            cc = Float64(C[i, k, j, l])
            s0 += cc * α₀ζ[k] * α₀ζ[l]
            s1 += cc * (α₀ζ[k] * α₁ζ[l] + α₁ζ[k] * α₀ζ[l])
            s2 += cc * α₁ζ[k] * α₁ζ[l]
        end
        A₀[i, j] = s0; A₁[i, j] = s1; A₂[i, j] = s2
    end
    return A₀, A₁, A₂
end

"""
    _residues_in_floats(::AbstractInclusion{T}, C₀::AbstractTens) -> nothing

The residue algorithm locates the poles as the eigenvalues of a companion matrix,
which LAPACK computes in `Float64`. A geometry or a stiffness of another element
type (a dual number, a symbol, a complex modulus) is refused here with the way
out, rather than met by a `MethodError` deep in the kernel: the cubature
(`method = :nestedquadgk`, which `:auto` selects) is generic and differentiates.
"""
_residues_in_floats(::AbstractInclusion{T}, C₀::TensND.AbstractTens) where {T} =
    _residues_in_floats(promote_type(T, eltype(C₀)))
_residues_in_floats(::Type{<:Union{AbstractFloat, Integer, Rational}}) = nothing
_residues_in_floats(::Type{T}) where {T} = throw(
    ArgumentError(
        "`method = :residues` evaluates the residues in Float64 (the poles are the " *
            "eigenvalues of a companion matrix); with $T entries use the cubature, " *
            "`method = :nestedquadgk`, which `:auto` selects and which differentiates"
    )
)

"""
    _build_poly_system(C, α₀ζ, α₁ζ) -> (adj, Q, roots_uhp)

Build the polynomial acoustic tensor
``\\boldsymbol{K}(z) = \\boldsymbol{A}_0 + \\boldsymbol{A}_1\\,z + \\boldsymbol{A}_2\\,z^2``, its
polynomial adjugate ``\\mathrm{adj}(\\boldsymbol{K})(z)`` (`adj`, a matrix of
ascending coefficient vectors) and the determinant
``Q(z) = \\det\\boldsymbol{K}(z)``, then compute the three roots of ``Q`` in
the upper half plane, listed with their multiplicity. A reference whose
acoustic tensor is singular along the integration, so that ``Q`` loses its
degree or gains a real root, is refused with an `ArgumentError`.

`Q` is returned as a `Polynomial` with real coefficients.
"""
function _build_poly_system(
        C::AbstractArray{<:Real, 4},
        α₀ζ::AbstractVector{<:Real},
        α₁ζ::AbstractVector{<:Real}
    )
    _bump!(RESIDUE_SOLVES)
    A₀, A₁, A₂ = _build_acoustic_coeffs(C, α₀ζ, α₁ζ)
    # K(z) = A₀ + A₁ z + A₂ z², entry by entry as coefficient vectors; its
    # adjugate, of degree 4, and its determinant Q, of degree 6, by products of
    # fixed size.
    K = [SVector(A₀[i, j], A₁[i, j], A₂[i, j]) for i in 1:3, j in 1:3]
    other(m) = m == 1 ? (2, 3) : m == 2 ? (1, 3) : (1, 2)
    function cofactor(i, j)       # adj(K)ᵢⱼ, the cofactor of Kⱼᵢ
        (r₁, r₂), (c₁, c₂) = other(j), other(i)
        minor = _conv(K[r₁, c₁], K[r₂, c₂]) - _conv(K[r₁, c₂], K[r₂, c₁])
        return isodd(i + j) ? -minor : minor
    end
    adj = [cofactor(i, j) for i in 1:3, j in 1:3]
    q = sum(_conv(K[1, i], adj[i, 1]) for i in 1:3)

    # The leading coefficient of `Q` is `det K(α₁)`, positive for a reference
    # that is strongly elliptic. Where it vanishes the Green operator, and with
    # it the Hill tensor, is not defined: a reference without shear stiffness,
    # say. Dropping it to carry on with a polynomial of lower degree, as was done,
    # only turned an undefined integral into a wrong number.
    abs(q[7]) >= 1.0e-14 * maximum(abs, q) || throw(
        ArgumentError(
            "the acoustic tensor of the reference stiffness is singular along a " *
                "direction of the integration: the reference is not strongly elliptic " *
                "(no shear stiffness, for instance), and its Green operator is not defined"
        )
    )

    # The roots are the eigenvalues of the companion matrix of `Q`, whose
    # coefficients are real: the QR algorithm is backward stable and returns the
    # conjugate pairs exactly. A root of multiplicity k comes out split into k
    # roots ε^(1/k) apart, around a centroid that is accurate to rounding; the
    # residue sums take such groups as a whole, so no root is polished on its own,
    # which would move the centroid. (The iterative root finder used up to 0.17.2
    # returned five roots out of six far from any root of a polynomial with two
    # triple roots, that of an isotropic reference around a sphere.)
    companion = [j == 6 ? -q[i] / q[7] : Float64(i == j + 1) for i in 1:6, j in 1:6]
    roots_all = ComplexF64.(eigvals!(companion))

    # `Q` has real coefficients and, for a strongly elliptic reference, no real
    # root: three roots in each half plane, counted with their multiplicity.
    roots_uhp = ComplexF64[zr for zr in roots_all if imag(zr) > 1.0e-12]
    length(roots_uhp) == 3 || throw(
        ArgumentError(
            "the acoustic polynomial of the reference stiffness has a real root: the " *
                "reference is not strongly elliptic, and its Green operator is not defined"
        )
    )

    return (adj = adj, Q = Polynomial(q, :z), roots_uhp = roots_uhp)
end

# Ascending coefficients of the product of two polynomials given by theirs.
_conv(a::SVector{N}, b::SVector{M}) where {N, M} = SVector{N + M - 1}(
    ntuple(p -> sum(a[k] * b[p - k + 1] for k in max(1, p - M + 1):min(p, N)), Val(N + M - 1))
)

# ─── Residue sums over clusters of poles
#
# Both integrands are rational, N/D, times log w for the Hill tensor, and the
# integral is a sum of residues over the poles of the upper half plane. Poles
# that coincide, or nearly, cannot be taken one at a time: a root of
# multiplicity k comes out of the root finder split into k roots ε^(1/k) apart,
# and the residue at each, which grows as the inverse of their distance, is then
# known to little better than ε^(1/k). Nor can they be declared coincident below
# some distance and moved together: that error is of the order of the distance.
#
# A group of close poles is summed as a whole instead: the sum of its residues
# is the contour integral around the group, which depends on its members only
# through symmetric functions and is well conditioned. With c the center of the
# group, δⱼ = wⱼ - c the offsets of its k members and R = D / Πⱼ(w - wⱼ),
#
#     Σⱼ Res_{wⱼ} f N / D = Σ_{l ≥ 0} g_{l+k-1} h_l(δ),
#
# where gₙ are the Taylor coefficients of f N / R at c and h_l the complete
# homogeneous symmetric polynomial of degree l in the δⱼ, the coefficient of s^l
# in Πⱼ 1 / (1 - δⱼ s): 1/D expands in powers of 1/(w - c) outside the group and
# f N / R in powers of (w - c) inside the next pole. The series converges as
# (radius of the group / distance to the next singularity)^l. A lone pole leaves
# g₀ = f N / D′, and k coincident ones g_{k-1}, the residue at a pole of order k.

"""
    _taylor_at(a, c, n) -> Vector

The first `n` Taylor coefficients at `c` of the polynomial of ascending
coefficients `a`, zero beyond its degree, by `n` passes of synthetic division,
in the type of `c`. With `|a|` and `|c|` it bounds, coefficient by coefficient,
the moduli of the terms the same divisions add up.
"""
function _taylor_at(a::AbstractVector, c::Number, n::Int)
    S = promote_type(eltype(a), typeof(c))
    n == 1 && return [evalpoly(c, S.(a))]
    d = length(a) - 1
    b = [S.(a); zeros(S, max(n - d - 1, 0))]
    # A recurrence: each pass finalizes one coefficient.
    @inbounds for j in 0:(min(n, d) - 1), m in (d - 1):-1:j
        b[m + 1] += c * b[m + 2]
    end
    return b[1:n]
end

"""
    _series_product(a, b, n) -> Vector

The first `n` coefficients of the product of two power series given by their
first coefficients.
"""
_series_product(a::AbstractVector, b::AbstractVector, n::Int) = [
    sum(
        a[q + 1] * b[p - q + 1] for q in max(0, p - length(b) + 1):min(p, length(a) - 1);
        init = zero(promote_type(eltype(a), eltype(b)))
    ) for p in 0:(n - 1)
]

# The powers 0, …, n of a power series, each truncated to its first M
# coefficients: a recurrence on the product.
function _series_powers(b::AbstractVector, n::Int, M::Int)
    p = [[one(eltype(b)); zeros(eltype(b), M - 1)]]
    for _ in 1:n
        push!(p, _series_product(p[end], b, M))
    end
    return p
end

# Distance from `c` to the nearest pole outside `members`, and to the branch
# point of the logarithm at 0 when it multiplies the integrand.
function _outer_distance(poles, members, c, logfactor::Bool)
    ρ = minimum((abs(poles[j] - c) for j in eachindex(poles) if j ∉ members); init = Inf)
    return logfactor ? min(ρ, abs(c)) : ρ
end

_cluster_center(poles, members) = sum(view(poles, members)) / length(members)
_cluster_radius(poles, members, c) = maximum(p -> abs(p - c), view(poles, members))

function _cluster_ratio(poles, members, logfactor::Bool)
    c = _cluster_center(poles, members)
    δ = _cluster_radius(poles, members, c)
    return δ == 0 ? 0.0 : δ / _outer_distance(poles, members, c, logfactor)
end

"""
    _pole_clusters(poles, nsum; logfactor, τ = 0.1) -> Vector{Vector{Int}}

Group the first `nsum` entries of `poles`, the roots of a denominator listed with
their multiplicity, into clusters to be summed as a whole. The candidates are
the nodes of the single-linkage tree of these poles, each merge of two groups at
the distance of their closest members; a node qualifies when its radius is below
`τ` times its distance to every other pole (and to 0 when a logarithm multiplies
the integrand), and each pole goes to the largest qualifying node that holds it.
A tree is needed: four roots spread around a double pole, as a double root of
`Q` at `i` gives, form a tight group of six that no pair, and no group of three,
qualifies to start. The poles after `nsum`, those of the other half plane, are
never summed but bound the clusters.
"""
function _pole_clusters(poles::AbstractVector{ComplexF64}, nsum::Int; logfactor::Bool, τ::Float64 = 0.1)
    # Single linkage (Kruskal): pairs in order of distance, each joining two
    # groups making a node.
    pairs = sort!([(abs(poles[i] - poles[j]), i, j) for j in 2:nsum for i in 1:(j - 1)])
    group = collect(1:nsum)
    members = [[j] for j in 1:nsum]
    nodes = copy(members)
    for (_, i, j) in pairs
        a, b = group[i], group[j]
        a == b && continue
        members[a] = vcat(members[a], members[b])
        group[members[b]] .= a
        push!(nodes, members[a])
        length(members[a]) == nsum && break
    end
    taken = falses(nsum)
    clusters = Vector{Int}[]
    for node in Iterators.reverse(nodes)
        any(view(taken, node)) && continue
        if length(node) == 1 || _cluster_ratio(poles, node, logfactor) < τ
            push!(clusters, node)
            taken[node] .= true
        end
    end
    return clusters
end

# Σ_l h_l g_{l+k-1} regrouped by the Taylor coefficient of the numerator it
# multiplies: H_i = Σ_l g_{l+k-1-i} h_l, zero from i = k + L on.
_regroup_weights(g::AbstractVector, h::AbstractVector, k::Int) = [
    sum(g[l + k - i] * h[l + 1] for l in max(0, i - k + 1):(length(h) - 1)) for i in 0:(length(h) + k - 2)
]

# The two modes of the recurrences below. `Val(false)` computes the weights;
# `Val(true)` runs the same recurrences on the moduli of their operands, every
# difference turned into a sum, which bounds the modulus of each term they add
# up: the rounding error of the first is at most a small multiple of the unit
# roundoff times the second (a running error bound).
_modulus(::Val{false}, x) = x
_modulus(::Val{true}, x) = abs(x)
_minus(::Val{false}, a, b) = a - b
_minus(::Val{true}, a, b) = a + b

# Terms of the series of a cluster: enough for the remainder, which decays as the
# ratio `r` of its radius to its distance to the next singularity, to fall below
# the unit roundoff of `T`. A cluster qualifies only with `r < τ`, so that this
# stays below about `precision(T) / log2(1/τ)` terms.
# Rounding error of a residue sum, per unit roundoff and per unit of the sum of
# the moduli of its terms, the bound `_cluster_weights` gives with `Val(true)`.
# Against the same sums in 256-bit arithmetic, on clusters from a lone pole to an
# eightfold one, the error stayed below that sum times the unit roundoff, by a
# factor 2.8 at least; four leaves a margin on top.
const _RESIDUE_ROUNDING = 4

_series_terms(r::Real, ::Type{T}) where {T} =
    iszero(r) ? 0 : clamp(ceil(Int, log(eps(T) / 16) / log(r)), 1, cld(precision(T), 3) + 8)

"""
    _cluster_weights(poles, members, lead; logfactor, bound = Val(false)) -> (c, H_one, H_log)

Center `c` of the cluster `poles[members]` of the roots `poles` of a
denominator ``D`` of leading coefficient `lead`, and the weights such that, for
every polynomial ``N`` with Taylor coefficients ``n_i`` at `c`,
```math
\\sum_{j \\in \\mathrm{members}} \\operatorname{Res}_{w_j} \\frac{f\\,N}{D} = \\sum_i n_i\\,H_i,
```
`H_one` for ``f = 1`` and, when `logfactor`, `H_log` for ``f = \\log``
(principal branch; empty otherwise). Both have the `k + L` entries the sum
reaches, `k` the size of the cluster and `L` the number of terms of the series
beyond the first: one for a lone pole, whose weight is ``1/D'``.

The arithmetic is that of the element type of `poles`, `Complex{Float64}` or an
extended precision. With `bound = Val(true)` the same recurrences run on moduli
(see `_modulus`) and return, entry by entry, an upper bound of the sum of the
moduli of the terms each weight adds up.
"""
function _cluster_weights(
        poles::AbstractVector{<:Complex}, members::AbstractVector{Int},
        lead::Complex; logfactor::Bool, bound::Val = Val(false)
    )
    T = real(eltype(poles))
    k = length(members)
    c = _cluster_center(poles, members)
    δ = view(poles, members) .- c
    # Terms of the series beyond l decay as (max|δ| / ρ)^l: enough of them for
    # the remainder to fall below rounding.
    r = maximum(abs, δ) == 0 ? zero(T) : maximum(abs, δ) / _outer_distance(poles, members, c, logfactor)
    L = _series_terms(r, T)
    m = L + k
    S = typeof(_modulus(bound, zero(Complex{T})))
    # h_l(δ), l = 0, …, L: the product of the geometric series 1 / (1 - δⱼ s).
    # Multiplying by one of them is the recurrence h_l += δⱼ h_{l-1}.
    h = [one(S); zeros(S, L)]
    @inbounds for δⱼ in δ, l in 1:L
        h[l + 1] += _modulus(bound, δⱼ) * h[l]
    end
    # Taylor coefficients at c of 1 / R, R = lead · Π_{j ∉ members} (w - wⱼ), up
    # to the order L + k - 1 the sum reaches. Dividing by (c - wⱼ) + t is the
    # recurrence g_p = (g_p - g_{p-1}) / (c - wⱼ).
    g = [_modulus(bound, inv(lead)); zeros(S, m - 1)]
    @inbounds for j in eachindex(poles)
        j in members && continue
        ie = _modulus(bound, inv(c - poles[j]))
        g[1] *= ie
        for p in 2:m
            g[p] = _minus(bound, g[p], g[p - 1]) * ie
        end
    end
    H_one = _regroup_weights(g, h, k)
    logfactor || return c, H_one, S[]
    # times log(c + t) = log c - Σ_{p ≥ 1} (-t / c)^p / p
    ℓ = [_modulus(bound, log(c)); [_modulus(bound, -(-inv(c))^p / p) for p in 1:(m - 1)]]
    return c, H_one, _regroup_weights(_series_product(ℓ, g, m), h, k)
end
