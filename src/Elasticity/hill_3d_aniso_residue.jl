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
# t⁷ p(ξ(t)) = Σₙ pₙ Eₙ(t). Its first `M` Taylor coefficients at `c` go to the
# columns of `T`, from the factored form: for a flat or a long ellipsoid the poles
# lie near t = ±1 or near 0, where the expanded polynomials lose every digit to
# cancellation. `v₀ = (c² - 1)/2` is passed in, computed without cancellation.
function _joukowski_taylor!(T::AbstractMatrix{ComplexF64}, c::ComplexF64, v₀::ComplexF64, M::Int)
    # Powers of v(c + t) = v₀ + c t + t²/2 in the even columns of a scratch
    # layout: column n + 1 holds vⁿ, then is multiplied by (c + t)^(7 - n).
    fill!(T, zero(ComplexF64))
    @inbounds begin
        T[1, 1] = 1
        for n in 1:6, i in 1:M
            a = v₀ * T[i, n]
            i >= 2 && (a += c * T[i - 1, n])
            i >= 3 && (a += T[i - 2, n] / 2)
            T[i, n + 1] = a
        end
        for n in 0:6
            for _ in 1:(7 - n), i in M:-1:1
                T[i, n + 1] = c * T[i, n + 1] + (i >= 2 ? T[i - 1, n + 1] : zero(ComplexF64))
            end
        end
    end
    return T
end

"""
    _hill_line_weights(Q, roots_uhp) -> Vector{Float64}

The seven integrals ``W_n = 2 \\int_0^\\infty E_n(t) / D(t)\\,\\mathrm{d}t``, with
``E_n(t) = t^{7-n} \\bigl((t^2 - 1)/2\\bigr)^n``,
``D(t) = (1 + t^2)^2\\, t^6\\, Q(\\xi(t))`` and ``\\xi(t) = (t^2 - 1)/(2t)``, so
that the inner integral of the Hill tensor at one angle φ, for a numerator
``p`` of coefficients ``p_n`` in ``\\xi``, is ``\\sum_n p_n W_n``
[masson2008](@cite).

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
"""
function _hill_line_weights(Q::Polynomial, roots_uhp::AbstractVector{ComplexF64})
    # The poles of the upper half plane, with w - 1 and w + 1 for each: at a
    # preimage of z, (w - 1)(w + 1) = 2 z w, so the smaller factor comes from that
    # product rather than from a difference that cancels.
    upper = ComplexF64[im, im]
    wm1 = ComplexF64[im - 1, im - 1]
    wp1 = ComplexF64[im + 1, im + 1]
    for z in roots_uhp
        s = sqrt(1 + z^2)
        # The preimages have product -1: the larger one directly, the other as its
        # inverse.
        w = abs(z + s) >= abs(z - s) ? z + s : z - s
        for v in (w, -1 / w)
            a, b = v - 1, v + 1
            abs(a) < abs(b) ? (a = 2 * z * v / b) : (b = 2 * z * v / a)
            push!(upper, v)
            push!(wm1, a)
            push!(wp1, b)
        end
    end
    poles = vcat(upper, conj.(upper))
    lead = ComplexF64(coeffs(Q)[end] / 64)        # leading coefficient of D
    W = zeros(Float64, 7)
    T = zeros(ComplexF64, 14, 7)
    for members in MFH_Core._pole_clusters(poles, length(upper); logfactor = true)
        c, H_one, H_log = MFH_Core._cluster_weights(poles, members, lead; logfactor = true)
        M = min(length(H_one), 14)
        k = length(members)
        v₀ = sum(wm1[j] for j in members) * sum(wp1[j] for j in members) / (2 * k^2)
        _joukowski_taylor!(T, c, v₀, M)
        for n in 1:7
            s_log = s_one = zero(ComplexF64)
            @inbounds for i in 1:M
                s_log += T[i, n] * H_log[i]
                s_one += T[i, n] * H_one[i]
            end
            W[n] -= 2 * (2 * real(s_log) + 2π * imag(s_one))
        end
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

!!! note "Float64 only, by design"
    The residue algorithm finds polynomial roots (the eigenvalues of a
    companion matrix, by LAPACK) and therefore only accepts `Float64` coefficients — it does NOT propagate
    `ForwardDiff.Dual`, `Complex`, or symbolic element types.  For those,
    use the `NestedQuadGK` backend (`method = :nestedquadgk`), which is
    type-generic and AD-compatible.  The `:auto` dispatch already picks
    `NestedQuadGK` automatically whenever `eltype(C₀) !== Float64`
    (see `Core/dispatch.jl`); only an explicit `method = :residues` under a
    non-`Float64` matrix will error.
"""
function _hill_3d_aniso_residue(
        ell::Ellipsoid{3}, C₀;
        abstol::Float64 = 1.0e-8,
        reltol::Float64 = 1.0e-6,
        maxiters::Int = 100_000
    )
    η = Float64(ell.semi_axes[2] / ell.semi_axes[1])
    ω = Float64(ell.semi_axes[3] / ell.semi_axes[1])

    C₀_princ = TensND.change_tens(C₀, ell.basis)
    C = zeros(Float64, 3, 3, 3, 3)
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        C[i, j, k, l] = Float64(C₀_princ[i, j, k, l])
    end

    function hill_at_phi(φ::Float64)
        m1 = cos(φ)
        m2 = sin(φ) / η
        # ζ(z) = (m1, m2, z/ω) = α₀ + α₁·z with α₀=(m1,m2,0), α₁=(0,0,1/ω).
        α₀ζ = [m1, m2, 0.0]
        α₁ζ = [0.0, 0.0, 1.0 / ω]

        sys = MFH_Core._build_poly_system(C, α₀ζ, α₁ζ)
        W = _hill_line_weights(sys.Q, sys.roots_uhp)

        # The 21 numerators (upper Voigt) are sums of four products
        # ζ_a ζ_b adj(K)_cd, each of degree ≤ 6, and the inner integral is linear
        # in their coefficients: Λ[a, b, c, d] = Σₘ (ζ_a ζ_b adj_cd)ₘ Wₘ.
        Λ = zeros(Float64, 3, 3, 3, 3)
        @inbounds for d in 1:3, c in 1:3
            ad = coeffs(sys.adj_poly[c, d])
            for b in 1:3, a in 1:b
                q = (
                    α₀ζ[a] * α₀ζ[b],
                    α₀ζ[a] * α₁ζ[b] + α₁ζ[a] * α₀ζ[b],
                    α₁ζ[a] * α₁ζ[b],
                )
                s = 0.0
                for t in eachindex(ad), u in 1:3
                    s += q[u] * real(ad[t]) * W[t + u - 1]
                end
                Λ[a, b, c, d] = Λ[b, a, c, d] = s
            end
        end
        res = zeros(Float64, 21)
        @inbounds for (n, (I, J)) in enumerate(_VOIGT_UPPER21)
            vi, vj = _VOIGT_IJ[I]
            vk, vl = _VOIGT_IJ[J]
            res[n] = 0.25 * (
                Λ[vi, vl, vj, vk] + Λ[vi, vk, vj, vl] + Λ[vj, vl, vi, vk] + Λ[vj, vk, vi, vl]
            )
        end
        return res
    end

    MFH_Core._bump!(MFH_Core.QUADGK_OUTER)
    P_vals, _ = MFH_Core._counted_quadgk(
        hill_at_phi, 0.0, π;
        atol = abstol, rtol = reltol, maxevals = maxiters
    )

    P_vals ./= π

    P_arr = _fill_hill21!(zeros(Float64, 3, 3, 3, 3), P_vals)

    return TensND.Tens(P_arr, ell.basis)
end
