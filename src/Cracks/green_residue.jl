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
    adj = sys.adj_poly
    Q = sys.Q

    # Tncon[i, p, q] = Σ_α C_{i α p q} n̂_α
    Tncon = zeros(Float64, 3, 3, 3)
    @inbounds for q in 1:3, p in 1:3, i in 1:3
        s = 0.0
        for α in 1:3
            s += C[i, α, p, q] * n̂[α]
        end
        Tncon[i, p, q] = s
    end

    poly(coefs) = Polynomial(ComplexF64.(coefs), :z)

    # V(z)_{ip} = Σ_q Tncon[i, p, q] ξ_q(z), with ξ(z) = ξs + z n̂
    V = Matrix{Polynomial{ComplexF64, :z}}(undef, 3, 3)
    @inbounds for p in 1:3, i in 1:3
        a0 = 0.0; a1 = 0.0
        for q in 1:3
            a0 += Tncon[i, p, q] * ξs[q]
            a1 += Tncon[i, p, q] * n̂[q]
        end
        V[i, p] = poly([a0, a1])
    end

    # M = V · adj
    M = Matrix{Polynomial{ComplexF64, :z}}(undef, 3, 3)
    @inbounds for j in 1:3, i in 1:3
        acc = Polynomial(ComplexF64[0.0], :z)
        for p in 1:3
            acc = acc + V[i, p] * adj[p, j]
        end
        M[i, j] = acc
    end
    # Bpoly = M · V^T
    Bpoly = Matrix{Polynomial{ComplexF64, :z}}(undef, 3, 3)
    @inbounds for k in 1:3, i in 1:3
        acc = Polynomial(ComplexF64[0.0], :z)
        for q in 1:3
            acc = acc + M[i, q] * V[k, q]
        end
        Bpoly[i, k] = acc
    end

    # Residue sum over the roots of the upper half plane, by groups so that a
    # multiple root, or roots that nearly coincide, are summed as a whole: for
    # coincident roots the group sum is the residue at a multiple pole, and an
    # isotropic matrix has a triple root. Skipping such roots, as was done, lost
    # their contribution altogether.
    poles = vcat(sys.roots_uhp, conj.(sys.roots_uhp))
    lead = ComplexF64(coeffs(Q)[end])
    T = zeros(ComplexF64, maximum(b -> length(coeffs(b)), Bpoly))
    result = zeros(Float64, 3, 3)
    for members in MFH_Core._pole_clusters(poles, 3; logfactor = false)
        c, H, _ = MFH_Core._cluster_weights(poles, members, lead; logfactor = false)
        @inbounds for i in 1:3, k in 1:3
            a = coeffs(Bpoly[i, k])
            m = min(length(H), length(a))
            MFH_Core._taylor_at!(T, a, c, m)
            s = zero(ComplexF64)
            for j in 1:m
                s += T[j] * H[j]
            end
            result[i, k] -= real(im * s)
        end
    end

    # Symmetrize
    @inbounds for i in 1:3, k in (i + 1):3
        avg = (result[i, k] + result[k, i]) / 2
        result[i, k] = avg
        result[k, i] = avg
    end
    return result
end
