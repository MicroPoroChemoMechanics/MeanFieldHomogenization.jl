# =============================================================================
#  green_residue.jl
#
#  Cauchy residue evaluation of ``\\hat{\\mathbf Q}^{\\star}_{nn}(\\vec ξ^{\\star})``
#  for a flat crack. Thin wrapper on `Core._build_poly_system`.
#  Float64 only.
# =============================================================================

"""
    _Qnn_star_residue(C::AbstractArray{Float64,4}, ξs::AbstractVector, n̂::AbstractVector)
            -> Matrix{Float64}  (3×3, symmetric)

Per-``\\varphi`` kernel of the crack-plane line integral that produces
the COD tensor of an anisotropic matrix: the reduced kernel
``\\hat{\\boldsymbol{Q}}^{\\star}_{nn}(\\underline{\\xi}^{\\star})``
[kanaunlevin2009, barthelemyIJSS2026](@cite), an integral over the normal
coordinate ``\\xi_n`` of a rational function whose poles are the roots of the
acoustic polynomial, evaluated here by the residue theorem. Float64 only.
"""
function _Qnn_star_residue(
        C::AbstractArray{Float64, 4},
        ξs::AbstractVector,
        n̂::AbstractVector
    )
    # ζ(z) = ξs + z·n̂  ⇒  α₀ζ = ξs, α₁ζ = n̂
    sys = MFH_Core._build_poly_system(C, ξs, n̂)
    # V(z)ᵢₚ = Σ_q Σ_α C_{iαpq} n̂_α ξ_q(z), with ξ(z) = ξs + z n̂ (degree 1), and
    # the numerator V · adj(K) · Vᵀ of degree 6, as coefficient vectors.
    Tn = [sum(C[i, α, p, q] * n̂[α] for α in 1:3) for i in 1:3, p in 1:3, q in 1:3]
    V = [SVector(sum(Tn[i, p, q] * ξs[q] for q in 1:3), sum(Tn[i, p, q] * n̂[q] for q in 1:3)) for i in 1:3, p in 1:3]
    M = [sum(MFH_Core._conv(V[i, p], sys.adj[p, j]) for p in 1:3) for i in 1:3, j in 1:3]
    numer = [sum(MFH_Core._conv(M[i, q], V[k, q]) for q in 1:3) for i in 1:3, k in 1:3]

    # Residue sum over the roots of the upper half plane, by groups so that a
    # multiple root, or roots that nearly coincide, are summed as a whole: for
    # coincident roots the group sum is the residue at a multiple pole, and an
    # isotropic matrix has a triple root. Skipping such roots, as was done, lost
    # their contribution altogether.
    poles = vcat(sys.roots_uhp, conj.(sys.roots_uhp))
    lead = ComplexF64(coeffs(sys.Q)[end])
    result = zeros(Float64, 3, 3)
    for members in MFH_Core._pole_clusters(poles, 3; logfactor = false)
        c, H, _ = MFH_Core._cluster_weights(poles, members, lead; logfactor = false)
        result .-= real.(im .* [transpose(MFH_Core._taylor_at(b, c, length(H))) * H for b in numer])
    end

    return (result + transpose(result)) / 2
end
