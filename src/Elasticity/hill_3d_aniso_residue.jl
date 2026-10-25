# =============================================================================
#  hill_3d_aniso_residue.jl
#
#  The Hill tensor by residues, after Masson (2008). The surface integral over
#  the unit sphere is split into an outer integral over an angle φ and an inner
#  one which, by the substitution t = cot(θ/2), is that of a rational function of
#  t over (0, ∞) (Masson's Eq. (18)); the inner one is evaluated exactly by
#  residues and the outer one by quadrature. The generic polynomial bookkeeping
#  (acoustic tensor, adjugate, determinant, root finding) and the residue sums
#  over groups of poles are in `Core.green_residue.jl`. Float64 only.
# =============================================================================

# Eₙ(t) = t^(7-n) vⁿ, v = (t² - 1)/2, n = 0, …, 6, which is t⁷ ξⁿ with
# ξ(t) = (t² - 1)/(2t): a numerator p of degree ≤ 6 in ξ becomes
# t⁷ p(ξ(t)) = Σₙ pₙ Eₙ(t). Their first `M` Taylor coefficients at `c`, as the
# columns of a matrix, from the factored form: for a flat or a long ellipsoid
# the poles lie near t = ±1 or near 0, where the expanded polynomials lose every
# digit to cancellation. `v₀ = (c² - 1)/2` is passed in, computed without
# cancellation.
function _joukowski_taylor(c::Number, v₀::Number, M::Int)
    # A lone pole, by far the most frequent, needs the values Eₙ(c) alone.
    M == 1 && return reshape(c .^ (7:-1:1) .* v₀ .^ (0:6), 1, 7)
    v = MFH_Core._series_powers([v₀, c, one(c) / 2], 6, M)      # vⁿ
    t = MFH_Core._series_powers([c, one(c)], 7, M)              # (c + t)ᵏ
    return reduce(hcat, [MFH_Core._series_product(v[n + 1], t[8 - n], M) for n in 0:6])
end

# v - 1 and v + 1 at a preimage v of z: (v - 1)(v + 1) = 2 z v gives the smaller
# of the two without the difference that cancels.
function _unit_offsets(z::Complex, v::Complex)
    a, b = v - 1, v + 1
    return abs(a) < abs(b) ? (2 * z * v / b, b) : (a, 2 * z * v / a)
end

# The poles of the upper half plane: i twice,
# then the two preimages of each root, the larger directly and the other as its
# inverse (their product is -1), each with its offsets from ±1. All of them, the
# conjugates included, in `poles`. `1 + z²` is formed as `(z - i)(z + i)`, free
# of the cancellation the sum has near z = ±i.
function _hill_poles(roots_uhp::AbstractVector{<:Complex})
    i = one(eltype(roots_uhp)) * im
    s = @. sqrt((roots_uhp - i) * (roots_uhp + i))
    w = @. ifelse(abs(roots_uhp + s) >= abs(roots_uhp - s), roots_uhp + s, roots_uhp - s)
    z = repeat(roots_uhp; inner = 2)
    v = vec(permutedims([w -inv.(w)]))      # w₁, -1/w₁, w₂, -1/w₂, …
    offs = _unit_offsets.(z, v)
    upper = [i; i; v]
    return (;
        poles = [upper; conj.(upper)], nupper = length(upper),
        wm1 = [i - 1; i - 1; first.(offs)], wp1 = [i + 1; i + 1; last.(offs)],
    )
end

# The seven weights by sums over `clusters`, and the running error bound of
# that evaluation.
function _hill_line_sums(p, clusters, lead::Real)
    T = real(eltype(p.poles))
    leadc = Complex{T}(lead)
    two_pi = 2 * T(π)
    W = zeros(T, 7)
    err = zero(T)
    for members in clusters
        c, H_one, H_log = MFH_Core._cluster_weights(p.poles, members, leadc; logfactor = true)
        M = min(length(H_one), 14)
        v₀ = sum(view(p.wm1, members)) * sum(view(p.wp1, members)) / (2 * length(members)^2)
        E = transpose(_joukowski_taylor(c, v₀, M))       # 7 × M
        W .-= 2 .* (2 .* real.(E * view(H_log, 1:M)) .+ two_pi .* imag.(E * view(H_one, 1:M)))
        # The same sums on moduli bound the terms they add up (`_cluster_weights`).
        _, B_one, B_log = MFH_Core._cluster_weights(p.poles, members, leadc; logfactor = true, bound = Val(true))
        Eb = transpose(_joukowski_taylor(abs(c), abs(v₀), M))
        err += MFH_Core._RESIDUE_ROUNDING * eps(T) *
            maximum(2 .* (2 .* (Eb * view(B_log, 1:M)) .+ two_pi .* (Eb * view(B_one, 1:M))))
    end
    return W, err
end

# The same seven integrals by adaptive Gauss-Kronrod quadrature of their
# definition, with t⁶ Q(ξ(t)) = Σₖ qₖ (t² - 1)ᵏ t^(6-k) / 2ᵏ, a polynomial free of
# the cancellation of ξ near t = 0 or t = ∞, and t² - 1 formed as (t - 1)(t + 1).
function _hill_line_weights_quadrature(Q::Polynomial, rtol::Real, atol::Real)
    q = SVector{7}(coeffs(Q))
    function integrand(t)
        u = (t - 1) * (t + 1)
        D = (1 + t^2)^2 * sum(q[k + 1] * u^k * t^(6 - k) / 2^k for k in 0:6)
        v = u / 2
        return SVector{7}(ntuple(n -> 2 * t^(8 - n) * v^(n - 1) / D, Val(7)))
    end
    W, _ = QuadGK.quadgk(integrand, 0.0, 1.0, Inf; rtol = rtol, atol = atol, norm = MFH_Core._quadgk_norm)
    return Vector(W)
end

"""
    _hill_line_weights(Q, roots_uhp; rtol = 64eps()) -> Vector{Float64}

The seven integrals ``W_n = 2 \\int_0^\\infty E_n(t) / D(t)\\,\\mathrm{d}t``, with
``E_n(t) = t^{7-n} \\bigl((t^2 - 1)/2\\bigr)^n``,
``D(t) = (1 + t^2)^2\\, t^6\\, Q(\\xi(t))`` and ``\\xi(t) = (t^2 - 1)/(2t)``, so
that the inner integral of the Hill tensor at one angle φ, for a numerator
``p`` of coefficients ``p_n`` in ``\\xi``, is ``\\sum_n p_n W_n``
[masson2008](@cite), within `rtol` of their largest modulus.

The roots of ``D`` are ``\\pm i``, twice each, and the two preimages
``w = z \\pm \\sqrt{1 + z^2}`` of each root ``z`` of ``Q``, which lie in the
half plane of ``z``. The integral over ``(0, \\infty)`` is minus the sum of the
residues of ``\\log(w)\\,E_n/D`` at all of them, with the logarithm cut along the
positive axis. The residues at conjugate poles are conjugate, up to the
``2\\pi i`` the logarithm gains across the cut, so that the sum reduces to the upper
half plane:
``2\\,\\mathrm{Re} \\sum \\mathrm{Res} \\log(w) E_n/D + 2\\pi\\,\\mathrm{Im} \\sum \\mathrm{Res}\\, E_n/D``.
The poles are summed by groups, so that ``i`` and the roots that coincide with
it or with each other, exactly or nearly, need no case of their own.

When a root of ``Q`` comes close to ``\\pm i``, which happens for the directions
of a plane of isotropy of the matrix, the poles crowd around the double pole at
``i`` and the residue sums cancel, by four or five orders of magnitude next to
it: the representation by residues is ill-conditioned there, although the
integrals themselves are not. The running error bound of the sums detects it
(see `_cluster_weights`), and the integrals are then computed by quadrature of
their definition, to `rtol` of the scale the sums give.
"""
function _hill_line_weights(Q::Polynomial, roots_uhp::AbstractVector{ComplexF64}; rtol::Real = 64 * eps())
    p = _hill_poles(roots_uhp)
    clusters = MFH_Core._pole_clusters(p.poles, p.nupper; logfactor = true)
    lead = coeffs(Q)[end] / 64        # leading coefficient of D
    W, err = _hill_line_sums(p, clusters, lead)
    scale = maximum(abs, W)
    err <= rtol * scale && return W
    return _hill_line_weights_quadrature(Q, rtol, rtol * scale)
end

"""
    _hill_3d_aniso_residue(ell, C₀; abstol, reltol, maxiters) -> AbstractTens{4,3}

Hill polarization tensor of a 3-D ellipsoid in an arbitrarily
anisotropic matrix, evaluated by reducing the 2-D surface integral of
[willis1977](@citet) to a 1-D line quadrature via the
Cauchy residue theorem of [masson2008](@citet): at each angle ``\\varphi``
the inner integral is a finite sum of residues at the roots of the
determinant of the acoustic tensor (`_hill_line_weights`).

!!! note "Float64 only, by design"
    The residue algorithm finds polynomial roots (the eigenvalues of a
    companion matrix, by LAPACK) and therefore only accepts `Float64` coefficients — it does NOT propagate
    `ForwardDiff.Dual`, `Complex`, or symbolic element types.  For those,
    use the `NestedQuadGK` backend (`method = :nestedquadgk`), which is
    type-generic and AD-compatible.  The `:auto` dispatch already picks
    `NestedQuadGK` automatically whenever `eltype(C₀) !== Float64`
    (see `Core/dispatch.jl`); an explicit `method = :residues` under a
    non-`Float64` matrix or geometry is refused with an `ArgumentError` that
    says so.
"""
function _hill_3d_aniso_residue(
        ell::Ellipsoid{3}, C₀;
        abstol::Float64 = 1.0e-8,
        reltol::Float64 = 1.0e-6,
        maxiters::Int = 100_000
    )
    η = Float64(ell.semi_axes[2] / ell.semi_axes[1])
    ω = Float64(ell.semi_axes[3] / ell.semi_axes[1])

    # Components in the ellipsoid's frame, in `Float64`: this route finds
    # polynomial roots and is `Float64`-only by design (see the note above).
    C = Float64.(Array(TensND.components(C₀, ell.basis)))

    function hill_at_phi(φ::Float64)
        m1 = cos(φ)
        m2 = sin(φ) / η
        # ζ(z) = (m1, m2, z/ω) = α₀ + α₁·z with α₀=(m1,m2,0), α₁=(0,0,1/ω).
        α₀ζ = [m1, m2, 0.0]
        α₁ζ = [0.0, 0.0, 1.0 / ω]

        sys = MFH_Core._build_poly_system(C, α₀ζ, α₁ζ)
        # The inner sums an order and more below the tolerance of the outer
        # quadrature, which their rounding would otherwise keep from converging.
        W = _hill_line_weights(sys.Q, sys.roots_uhp; rtol = max(reltol / 16, 64 * eps()))

        # The 21 numerators (upper Voigt) are sums of four products
        # ζ_a ζ_b adj(K)_cd, each of degree ≤ 6, and the inner integral is a
        # bilinear form in their coefficients, through the Hankel matrix of the
        # weights: Λ(a, b, c, d) = Σ_{u,t} (ζ_a ζ_b)_u (adj_cd)_t W_{u+t}.
        Hk = [W[u + t - 1] for u in 1:3, t in 1:5]
        q = [
            SVector(α₀ζ[a] * α₀ζ[b], α₀ζ[a] * α₁ζ[b] + α₁ζ[a] * α₀ζ[b], α₁ζ[a] * α₁ζ[b])
                for a in 1:3, b in 1:3
        ]
        HA = Hk * reduce(hcat, sys.adj)
        Λ(a, b, c, d) = dot(q[a, b], view(HA, :, c + 3 * (d - 1)))
        return SVector{21}(
            map(_VOIGT_UPPER21) do (I, J)
                vi, vj = _VOIGT_IJ[I]
                vk, vl = _VOIGT_IJ[J]
                (Λ(vi, vl, vj, vk) + Λ(vi, vk, vj, vl) + Λ(vj, vl, vi, vk) + Λ(vj, vk, vi, vl)) / 4
            end
        )
    end

    MFH_Core._bump!(MFH_Core.QUADGK_OUTER)
    P_vals, _ = MFH_Core._counted_quadgk(
        hill_at_phi, 0.0, π;
        atol = abstol, rtol = reltol, maxevals = maxiters
    )

    P_vals = P_vals / π

    P_arr = _fill_hill21!(zeros(Float64, 3, 3, 3, 3), P_vals)

    return TensND.Tens(P_arr, ell.basis)
end
