# =============================================================================
#  hill_3d_aniso_residue.jl
#
#  The Hill tensor by residues, after Masson (2008). The surface integral over
#  the unit sphere is split into an outer integral over an angle φ and an inner
#  one which, by the substitution t = cot(θ/2), is that of a rational function of
#  t over (0, ∞) (Masson's Eq. (18)); the inner one is evaluated exactly by
#  residues and the outer one by quadrature. The generic polynomial bookkeeping
#  (acoustic tensor, adjugate, determinant, root finding) and the residue sums
#  over groups of poles are in `Core.green_residue.jl`. The roots are found in
#  Float64 on the values of the coefficients; the derivatives of dual numbers are
#  carried by the residue sums.
# =============================================================================

# Eₙ(t) = t^(7-n) vⁿ, v = (t² - 1)/2, n = 0, …, 6, which is t⁷ ξⁿ with
# ξ(t) = (t² - 1)/(2t): a numerator p of degree ≤ N in ξ becomes
# t^(N+1) p(ξ(t)) = Σₙ pₙ Eₙ(t), Eₙ(t) = t^(N+1-n) vⁿ with v = (t² - 1)/2. Their
# first `M` Taylor coefficients at `c`, as the columns of a matrix, from the
# factored form: for a flat or a long ellipsoid the poles lie near t = ±1 or
# near 0, where the expanded polynomials lose every digit to cancellation.
# `v₀ = (c² - 1)/2` is passed in, computed without cancellation. N = 6 for the
# weights themselves, 6(j + 1) for their derivatives of order j.
function _joukowski_taylor(c::Number, v₀::Number, M::Int, N::Int = 6)
    # A lone pole, by far the most frequent, needs the values Eₙ(c) alone.
    M == 1 && return reshape(c .^ ((N + 1):-1:1) .* v₀ .^ (0:N), 1, N + 1)
    v = MFH_Core._series_powers([v₀, c, one(c) / 2], N, M)      # vⁿ
    t = MFH_Core._series_powers([c, one(c)], N + 1, M)          # (c + t)ᵏ
    return reduce(hcat, [MFH_Core._series_product(v[n + 1], t[N + 2 - n], M) for n in 0:N])
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
# of the cancellation the sum has near z = ±i. Each preimage is listed `mult`
# times, its multiplicity as a root of the power `mult` of t⁶ Q(ξ(t)), the
# denominator of the derivatives of order `mult - 1` (see `_hill_line_weights`).
function _hill_poles(roots_uhp::AbstractVector{<:Complex}, mult::Int = 1)
    i = one(eltype(roots_uhp)) * im
    s = @. sqrt((roots_uhp - i) * (roots_uhp + i))
    w = @. ifelse(abs(roots_uhp + s) >= abs(roots_uhp - s), roots_uhp + s, roots_uhp - s)
    z = repeat(roots_uhp; inner = 2)
    v = vec(permutedims([w -inv.(w)]))      # w₁, -1/w₁, w₂, -1/w₂, …
    offs = repeat(_unit_offsets.(z, v); inner = mult)
    upper = [i; i; repeat(v; inner = mult)]
    return (;
        poles = [upper; conj.(upper)], nupper = length(upper),
        wm1 = [i - 1; i - 1; first.(offs)], wp1 = [i + 1; i + 1; last.(offs)],
    )
end

# The N + 1 weights 2∫₀^∞ Eₙ/D dt, n = 0, …, N, by sums over `clusters` of the
# poles `p` of D, of leading coefficient `lead`, and the running error bound of
# that evaluation. N = 6 and D = (1 + t²)² t⁶ Q(ξ(t)) for the weights themselves.
function _hill_line_sums(p, clusters, lead::Real, N::Int = 6)
    T = real(eltype(p.poles))
    leadc = Complex{T}(lead)
    two_pi = 2 * T(π)
    W = zeros(T, N + 1)
    err = zero(T)
    for members in clusters
        c, H_one, H_log = MFH_Core._cluster_weights(p.poles, members, leadc; logfactor = true)
        M = min(length(H_one), 2N + 2)      # Eₙ is of degree N + 1 + n ≤ 2N + 1
        v₀ = sum(view(p.wm1, members)) * sum(view(p.wp1, members)) / (2 * length(members)^2)
        E = transpose(_joukowski_taylor(c, v₀, M, N))       # (N + 1) × M
        W .-= 2 .* (2 .* real.(E * view(H_log, 1:M)) .+ two_pi .* imag.(E * view(H_one, 1:M)))
        # The same sums on moduli bound the terms they add up (`_cluster_weights`).
        _, B_one, B_log = MFH_Core._cluster_weights(p.poles, members, leadc; logfactor = true, bound = Val(true))
        Eb = transpose(_joukowski_taylor(abs(c), abs(v₀), M, N))
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
    _hill_line_weights(Q, roots_uhp; rtol = 64eps()) -> Vector

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

The roots are those of the values ``Q_0`` of the coefficients of ``Q``. When
these coefficients are dual numbers, ``Q = Q_0 + \\delta Q`` and the weights
carry their derivatives through
```math
W_n = \\sum_{j \\ge 0} (-1)^j \\sum_k \\bigl(\\delta Q^j\\bigr)_k\\, W^{(j)}_{n+k}, \\qquad
W^{(j)}_m = 2 \\int_0^\\infty \\frac{t^{7+6j-m}\\,\\bigl((t^2 - 1)/2\\bigr)^m}
{(1 + t^2)^2\\,\\bigl(t^6\\, Q_0(\\xi(t))\\bigr)^{j+1}}\\,\\mathrm{d}t ,
```
``(\\delta Q^j)_k`` the coefficients of the power ``j`` of ``\\delta Q``, a sum
that stops at the number of nested levels of dual numbers. The ``W^{(j)}_m`` are
plain numbers, given by the same residue sums over the same poles, each
preimage counted ``j + 1`` times, and held to the same error bound.
"""
function _hill_line_weights(Q::Polynomial, roots_uhp::AbstractVector{ComplexF64}; rtol::Real = 64 * eps())
    q = SVector{7}(coeffs(Q))
    q₀ = TensND.primal.(q)
    p = _hill_poles(roots_uhp)
    clusters = MFH_Core._pole_clusters(p.poles, p.nupper; logfactor = true)
    lead = q₀[7] / 64        # leading coefficient of D
    W₀, err = _hill_line_sums(p, clusters, lead)
    scale = maximum(abs, W₀)
    err <= rtol * scale || return _hill_line_weights_quadrature(Q, rtol, rtol * scale)
    W = convert(Vector{eltype(q)}, W₀)
    δq = Vector(q - q₀)
    δqʲ = δq
    # A recurrence on the powers of δQ, one order of derivation per pass.
    for j in 1:TensND.dual_depth(eltype(q))
        N = 6 * (j + 1)
        pⱼ = _hill_poles(roots_uhp, j + 1)
        Wⱼ, errⱼ = _hill_line_sums(pⱼ, MFH_Core._pole_clusters(pⱼ.poles, pⱼ.nupper; logfactor = true), lead^(j + 1), N)
        errⱼ <= rtol * maximum(abs, Wⱼ) || return _hill_line_weights_quadrature(Q, rtol, rtol * scale)
        W .+= (isodd(j) ? -1 : 1) .* [sum(δqʲ[k + 1] * Wⱼ[n + k + 1] for k in 0:(6j)) for n in 0:6]
        δqʲ = MFH_Core._series_product(δqʲ, δq, N + 1)
    end
    return W
end

"""
    _hill_3d_aniso_residue(ell, C₀; abstol, reltol, maxiters) -> AbstractTens{4,3}

Hill polarization tensor of a 3-D ellipsoid in an arbitrarily
anisotropic matrix, evaluated by reducing the 2-D surface integral of
[willis1977](@citet) to a 1-D line quadrature via the
Cauchy residue theorem of [masson2008](@citet): at each angle ``\\varphi``
the inner integral is a finite sum of residues at the roots of the
determinant of the acoustic tensor (`_hill_line_weights`).

!!! note "Derivatives"
    The roots of the determinant are found on the values of its coefficients,
    as the eigenvalues of a companion matrix in `Float64`, and are never
    differentiated: the derivatives carried by `ForwardDiff.Dual` semi-axes,
    orientation or stiffness, at any nesting level, come from residue sums over
    the same roots (`_hill_line_weights`). A symbolic or a complex stiffness
    has no route here and is refused with an `ArgumentError`; the `NestedQuadGK`
    backend (`method = :nestedquadgk`, which `:auto` selects) is generic.
"""
function _hill_3d_aniso_residue(
        ell::Ellipsoid{3}, C₀;
        abstol::Float64 = 1.0e-8,
        reltol::Float64 = 1.0e-6,
        maxiters::Int = 100_000
    )
    η = float(ell.semi_axes[2] / ell.semi_axes[1])
    ω = float(ell.semi_axes[3] / ell.semi_axes[1])

    # Components in the ellipsoid's frame, as floats or duals of floats.
    C = float.(Array(TensND.components(C₀, ell.basis)))
    T = promote_type(typeof(η), typeof(ω), eltype(C))

    function hill_at_phi(φ::Float64)
        m1 = cos(φ)
        m2 = sin(φ) / η
        # ζ(z) = (m1, m2, z/ω) = α₀ + α₁·z with α₀=(m1,m2,0), α₁=(0,0,1/ω).
        α₀ζ = [m1, m2, zero(m1)]
        α₁ζ = [zero(ω), zero(ω), inv(ω)]

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

    P_arr = _fill_hill21!(zeros(T, 3, 3, 3, 3), P_vals)

    return TensND.symmetric_tens(P_arr, ell.basis)
end
