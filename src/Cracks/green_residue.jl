# =============================================================================
#  green_residue.jl
#
#  Cauchy residue evaluation of ``\\hat{\\mathbf Q}^{\\star}_{nn}(\\vec ξ^{\\star})``
#  for a flat crack. Thin wrapper on `Core._build_poly_system`. The poles are
#  found in Float64 on the values of the coefficients; the derivatives of dual
#  numbers are carried by the residue sums.
# =============================================================================

# Subtract from `result` the real part of i times the sum, over the roots of the
# upper half plane, of the residues of `numer / Q₀^mult`, entry by entry: `Q₀` of
# leading coefficient `lead₀` and of roots `roots_uhp` there, each of
# multiplicity `mult` in the denominator. The sum is taken by groups so that a
# multiple root, or roots that nearly coincide, are summed as a whole: for
# coincident roots the group sum is the residue at a multiple pole, and an
# isotropic matrix has a triple root. Returns the running error bound of these
# sums, channel by channel when the numerators are dual numbers.
function _subtract_residue_sums!(result::AbstractMatrix, roots_uhp::AbstractVector{ComplexF64}, mult::Int, lead₀::Real, numer)
    upper = repeat(roots_uhp; inner = mult)
    poles = vcat(upper, conj.(upper))
    lead = ComplexF64(lead₀^mult)
    err = zero(eltype(result))
    for members in MFH_Core._pole_clusters(poles, length(upper); logfactor = false)
        c, H, _ = MFH_Core._cluster_weights(poles, members, lead; logfactor = false)
        result .-= real.(im .* [transpose(MFH_Core._taylor_at(b, c, length(H))) * H for b in numer])
        # The same sums on moduli bound the terms they add up (`_cluster_weights`).
        _, B, _ = MFH_Core._cluster_weights(poles, members, lead; logfactor = false, bound = Val(true))
        err += MFH_Core._RESIDUE_ROUNDING * eps() * mapreduce(
            b -> transpose(MFH_Core._taylor_at(MFH_Core._abs_channels.(b), abs(c), length(B))) * B,
            MFH_Core._max_channels, numer
        )
    end
    return err
end

"""
    _Qnn_star_residue(C::AbstractArray{<:Real,4}, ξs::AbstractVector, n̂::AbstractVector)
            -> Matrix  (3×3, symmetric)

Per-``\\varphi`` kernel of the crack-plane line integral that produces
the COD tensor of an anisotropic matrix: the reduced kernel
``\\hat{\\boldsymbol{Q}}^{\\star}_{nn}(\\underline{\\xi}^{\\star})``
[kanaunlevin2009, barthelemyIJSS2026](@cite), an integral over the normal
coordinate ``\\xi_n`` of a rational function ``N/Q`` whose poles are the roots of
the acoustic polynomial ``Q``, evaluated here by the residue theorem.

The roots are those of the values ``Q_0`` of the coefficients. With dual
arguments, ``Q = Q_0 + \\delta Q`` and
``N/Q = \\sum_{j \\ge 0} N\\,(-\\delta Q)^j / Q_0^{j+1}``, a sum that stops at the
number of nested levels of dual numbers: every derivative is a residue sum over
the same roots, of multiplicity ``j + 1``.
"""
function _Qnn_star_residue(
        C::AbstractArray{<:Real, 4},
        ξs::AbstractVector,
        n̂::AbstractVector;
        rtol::Real = 64 * eps()
    )
    # ζ(z) = ξs + z·n̂  ⇒  α₀ζ = ξs, α₁ζ = n̂
    sys = MFH_Core._build_poly_system(C, ξs, n̂)
    # V(z)ᵢₚ = Σ_q Σ_α C_{iαpq} n̂_α ξ_q(z), with ξ(z) = ξs + z n̂ (degree 1), and
    # the numerator V · adj(K) · Vᵀ of degree 6, as coefficient vectors.
    Tn = [sum(C[i, α, p, q] * n̂[α] for α in 1:3) for i in 1:3, p in 1:3, q in 1:3]
    V = [SVector(sum(Tn[i, p, q] * ξs[q] for q in 1:3), sum(Tn[i, p, q] * n̂[q] for q in 1:3)) for i in 1:3, p in 1:3]
    M = [sum(MFH_Core._conv(V[i, p], sys.adj[p, j]) for p in 1:3) for i in 1:3, j in 1:3]
    numer = [sum(MFH_Core._conv(M[i, q], V[k, q]) for q in 1:3) for i in 1:3, k in 1:3]

    # Residue sums over the roots of the upper half plane: that of N/Q₀, then
    # one per order of derivation, of N (-δQ)ʲ / Q₀ʲ⁺¹.
    qc = SVector{7}(coeffs(sys.Q))
    q₀ = TensND.primal.(qc)
    result = zeros(eltype(qc), 3, 3)
    err = _subtract_residue_sums!(result, sys.roots_uhp, 1, q₀[7], numer)
    mδq = Vector(q₀ - qc)
    mδqʲ = mδq
    # A recurrence on the powers of -δQ.
    for j in 1:TensND.dual_depth(eltype(qc))
        numerⱼ = [MFH_Core._series_product(b, mδqʲ, 6j + 7) for b in numer]
        err += _subtract_residue_sums!(result, sys.roots_uhp, j + 1, q₀[7], numerⱼ)
        mδqʲ = MFH_Core._series_product(mδqʲ, mδq, 6j + 7)
    end

    # Residue sums that cancel, which clusters of roots close to merging would
    # make, are detected by their running error bound and replaced by the
    # quadrature of the same kernel: each derivative against its own scale.
    scale = mapreduce(MFH_Core._abs_channels, MFH_Core._max_channels, result)
    MFH_Core._channels_within(err, rtol * scale) ||
        return _Qnn_star_nestedquadgk(C, ξs, n̂; abstol = rtol * TensND.primal(scale), reltol = rtol)
    return (result + transpose(result)) / 2
end
