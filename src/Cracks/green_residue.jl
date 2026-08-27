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
the COD tensor of an anisotropic matrix, evaluated by Cauchy residues
([masson2008](@cite) adapted to the crack limit, with the
limit-algorithm construction of [barthelemyIJSS2009](@cite)).
Float64 only.
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
    dQ = sys.dQ
    roots_uhp = sys.roots_uhp
    # Adaptive singular-root guard: skip roots where |Q'(zᵣ)| ≪ scale of Q.
    dQ_thresh = 1.0e-14 * maximum(abs, coeffs(Q))

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

    # Residue sum over UHP roots
    result = zeros(Float64, 3, 3)
    @inbounds for zr in roots_uhp
        dQr = dQ(zr)
        abs(dQr) < dQ_thresh && continue
        for i in 1:3, k in 1:3
            contrib = im * Bpoly[i, k](zr) / dQr
            result[i, k] -= real(contrib)
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
