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
    _build_poly_system(C, α₀ζ, α₁ζ) -> (K_poly, adj_poly, Q, roots_uhp)

Build the polynomial acoustic tensor
``\\boldsymbol{K}(z) = \\boldsymbol{A}_0 + \\boldsymbol{A}_1\\,z + \\boldsymbol{A}_2\\,z^2``, its
polynomial adjugate ``\\mathrm{adj}(\\boldsymbol{K})(z)`` and the determinant
``Q(z) = \\det\\boldsymbol{K}(z)``, then compute the three roots of ``Q`` in
the upper half plane, listed with their multiplicity. A reference whose
acoustic tensor is singular along the integration, so that ``Q`` loses its
degree or gains a real root, is refused with an `ArgumentError`.

All polynomial entries are `Polynomial{ComplexF64,:z}` (so that complex
roots can be handled uniformly).
"""
function _build_poly_system(
        C::AbstractArray{<:Real, 4},
        α₀ζ::AbstractVector{<:Real},
        α₁ζ::AbstractVector{<:Real}
    )
    _bump!(RESIDUE_SOLVES)
    A₀, A₁, A₂ = _build_acoustic_coeffs(C, α₀ζ, α₁ζ)

    poly(coefs) = Polynomial(ComplexF64.(coefs), :z)
    K_poly = Matrix{Polynomial{ComplexF64, :z}}(undef, 3, 3)
    @inbounds for j in 1:3, i in 1:3
        K_poly[i, j] = poly([A₀[i, j], A₁[i, j], A₂[i, j]])
    end

    # Complementary row/column indices for a fixed 3×3 pattern — a static
    # tuple lookup instead of `setdiff(1:3, [j])`, which allocates two
    # `Vector`s (plus the single-element literal) per one of the 9 entries.
    _other2(m::Int) = m == 1 ? (2, 3) : m == 2 ? (1, 3) : (1, 2)
    adj_poly = Matrix{Polynomial{ComplexF64, :z}}(undef, 3, 3)
    @inbounds for j in 1:3, i in 1:3
        rows = _other2(j)
        cols = _other2(i)
        minor = K_poly[rows[1], cols[1]] * K_poly[rows[2], cols[2]] -
            K_poly[rows[1], cols[2]] * K_poly[rows[2], cols[1]]
        adj_poly[i, j] = isodd(i + j) ? -minor : minor
    end

    Q = Polynomial(ComplexF64[0.0], :z)
    @inbounds for i in 1:3
        Q = Q + K_poly[1, i] * adj_poly[i, 1]
    end

    # The leading coefficient of `Q` is `det K(α₁)`, positive for a reference
    # that is strongly elliptic. Where it vanishes the Green operator, and with
    # it the Hill tensor, is not defined: a reference without shear stiffness,
    # say. Dropping it to carry on with a polynomial of lower degree, as was done,
    # only turned an undefined integral into a wrong number.
    Q_coeffs = coeffs(Q)
    (length(Q_coeffs) == 7 && abs(Q_coeffs[end]) >= 1.0e-14 * maximum(abs, Q_coeffs)) ||
        throw(
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
    a = real.(Q_coeffs)
    companion = zeros(Float64, 6, 6)
    @inbounds for i in 2:6
        companion[i, i - 1] = 1.0
    end
    @inbounds for i in 1:6
        companion[i, 6] = -a[i] / a[7]
    end
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

    return (K_poly = K_poly, adj_poly = adj_poly, Q = Q, roots_uhp = roots_uhp)
end

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
    _taylor_at!(b, a, c, n) -> b

Overwrite `b[1:n]` with the first `n` Taylor coefficients at `c` of the
polynomial of ascending coefficients `a` (zero beyond its degree), by `n` passes
of synthetic division. [`_taylor_at`](@ref) allocates the result.
"""
function _taylor_at!(b::AbstractVector{ComplexF64}, a::AbstractVector, c::ComplexF64, n::Int)
    d = length(a) - 1
    @inbounds for i in 1:max(n, d + 1)
        b[i] = i <= d + 1 ? a[i] : zero(ComplexF64)
    end
    @inbounds for j in 0:(min(n, d) - 1), m in (d - 1):-1:j
        b[m + 1] += c * b[m + 2]
    end
    return b
end

"""
    _taylor_at(a, c, n) -> Vector{ComplexF64}

The first `n` Taylor coefficients at `c` of the polynomial of ascending
coefficients `a`, zero beyond its degree.
"""
_taylor_at(a::AbstractVector, c::ComplexF64, n::Int) =
    _taylor_at!(zeros(ComplexF64, max(n, length(a))), a, c, n)[1:n]

# Distance from `c` to the nearest pole outside `members`, and to the branch
# point of the logarithm at 0 when it multiplies the integrand.
function _outer_distance(poles, members, c, logfactor::Bool)
    ρ = logfactor ? abs(c) : Inf
    @inbounds for j in eachindex(poles)
        j in members || (ρ = min(ρ, abs(poles[j] - c)))
    end
    return ρ
end

_cluster_center(poles, members) = sum(poles[j] for j in members) / length(members)
_cluster_radius(poles, members, c) = maximum(abs(poles[j] - c) for j in members)

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
    groups = [[j] for j in 1:nsum]
    nodes = copy(groups)
    while length(groups) > 1
        best, ia, ib = Inf, 0, 0
        for a in eachindex(groups), b in (a + 1):length(groups), i in groups[a], j in groups[b]
            d = abs(poles[i] - poles[j])
            d < best && ((best, ia, ib) = (d, a, b))
        end
        groups[ia] = vcat(groups[ia], groups[ib])
        deleteat!(groups, ib)
        push!(nodes, groups[ia])
    end
    taken = falses(nsum)
    clusters = Vector{Int}[]
    for node in Iterators.reverse(nodes)
        any(j -> taken[j], node) && continue
        if length(node) == 1 || _cluster_ratio(poles, node, logfactor) < τ
            push!(clusters, node)
            for j in node
                taken[j] = true
            end
        end
    end
    return clusters
end

# Σ_l h_l g_{l+k-1} regrouped by the Taylor coefficient of the numerator it
# multiplies: H_i = Σ_l g_{l+k-1-i} h_l, zero from i = k + L on.
function _regroup_weights(g::Vector{ComplexF64}, h::Vector{ComplexF64}, k::Int)
    L = length(h) - 1
    H = zeros(ComplexF64, L + k)
    @inbounds for i in 0:(L + k - 1), l in max(0, i - k + 1):L
        H[i + 1] += g[l + k - i] * h[l + 1]
    end
    return H
end

"""
    _cluster_weights(poles, members, lead; logfactor) -> (c, H_one, H_log)

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
"""
function _cluster_weights(
        poles::AbstractVector{ComplexF64}, members::AbstractVector{Int},
        lead::ComplexF64; logfactor::Bool
    )
    k = length(members)
    c = _cluster_center(poles, members)
    δmax = _cluster_radius(poles, members, c)
    # Terms of the series beyond l decay as (δmax / ρ)^l: enough of them for
    # the remainder to fall below rounding.
    r = δmax == 0 ? 0.0 : δmax / _outer_distance(poles, members, c, logfactor)
    L = r == 0 ? 0 : clamp(ceil(Int, log(1.0e-17) / log(r)), 1, 80)
    # h_l(δ), l = 0, …, L: the product of the geometric series 1 / (1 - δⱼ s).
    h = zeros(ComplexF64, L + 1)
    h[1] = 1
    @inbounds for j in members
        δ = poles[j] - c
        for l in 1:L
            h[l + 1] += δ * h[l]
        end
    end
    # Taylor coefficients at c of 1 / R, R = lead · Π_{j ∉ members} (w - wⱼ),
    # up to the order L + k - 1 the sum reaches: each factor divides the series.
    m = L + k
    g = zeros(ComplexF64, m)
    g[1] = 1 / lead
    @inbounds for j in eachindex(poles)
        j in members && continue
        ie = 1 / (c - poles[j])
        g[1] *= ie
        for p in 2:m
            g[p] = (g[p] - g[p - 1]) * ie
        end
    end
    H_one = _regroup_weights(g, h, k)
    logfactor || return c, H_one, ComplexF64[]
    # times log(c + t) = log c - Σ_{q ≥ 1} (-t / c)^q / q
    ℓ = Vector{ComplexF64}(undef, m)
    ℓ[1] = log(c)
    w = one(ComplexF64)
    @inbounds for q in 1:(m - 1)
        w *= -1 / c
        ℓ[q + 1] = -w / q
    end
    gl = zeros(ComplexF64, m)
    @inbounds for p in 0:(m - 1), q in 0:p
        gl[p + 1] += ℓ[q + 1] * g[p - q + 1]
    end
    return c, H_one, _regroup_weights(gl, h, k)
end
