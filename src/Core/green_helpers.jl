# =============================================================================
#  green_helpers.jl
#
#  Quadrature-agnostic building blocks of the crack-plane / half-sphere
#  Green-function evaluation.  Shared by every numerical back-end
#  (`green_nestedquadgk.jl`, `green_decuhr.jl`) in `Cracks/` and
#  `Elasticity/`.
#
#  `_A_and_Tn`          : φ-independent pre-computation (n̂-only).
#  `_phi_cache`         : φ-dependent, α-independent pre-computation.
#  `_qnn_pair_components!`: inner-α evaluation of
#           `[Q̂_{nn}(ζp) + Q̂_{nn}(ζm)] · ρ/sin²α`
#           from the pre-computed `(A, Vs, Ks, Kns, ca, sa)` quantities.
# =============================================================================

"""
    _A_and_Tn(C, n̂, ::Type{T}) -> (A, Tn)

Pre-compute the `n̂`-only quantities used by every evaluation of
``\\hat{\\boldsymbol{Q}}_{nn} = \\underline{n}\\cdot\\hat{\\mathbb{Q}}\\cdot\\underline{n}``:

  * `Tn`: ``(T_n)_{ipq} = \\sum_{\\alpha} C_{i\\alpha pq}\\,n_{\\alpha}`` (3×3×3 array)
  * `A`: ``A_{ip} = \\sum_q (T_n)_{ipq}\\,n_q`` (3×3, ``\\boldsymbol{A} = \\boldsymbol{V}(\\underline{n}) = \\boldsymbol{N}(\\underline{n})``,
    with ``\\boldsymbol{N}`` the acoustic tensor)

Element type is the supplied `T = promote_type(...)`.
"""
function _A_and_Tn(C::AbstractArray, n̂::AbstractVector, ::Type{T}) where {T}
    # Single choke point of every numerical Green back-end, and the only place
    # that needs `T` to be `isbits`: `MArray{…,T}(undef)` cannot even be
    # *constructed* for a boxed scalar such as `SymPy.Sym` or `Symbolics.Num`.
    # Without this check the failure surfaces as a bare
    # `setindex!() with non-isbitstype eltype …` from StaticArrays, several
    # frames deep, which says nothing about what to do instead.
    isbitstype(T) || throw(
        ArgumentError(
            "the numerical Green back-ends require an isbits scalar type, got $T. " *
                "Symbolic scalars only reach a closed form: for cracks that means an " *
                "isotropic reference, or a transversely isotropic one whose axis is the " *
                "crack normal. Check the alignment, or substitute numbers first."
        )
    )
    Tn_m = MArray{Tuple{3, 3, 3}, T}(undef)
    @inbounds for q in 1:3, p in 1:3, i in 1:3
        s = zero(T)
        for α in 1:3
            s += T(C[i, α, p, q]) * T(n̂[α])
        end
        Tn_m[i, p, q] = s
    end
    Tn = SArray(Tn_m)
    A_m = MMatrix{3, 3, T}(undef)
    @inbounds for p in 1:3, i in 1:3
        s = zero(T)
        for q in 1:3
            s += Tn[i, p, q] * T(n̂[q])
        end
        A_m[i, p] = s
    end
    return SMatrix(A_m), Tn
end

"""
    _phi_cache(C, Tn, n̂, ξshat, ::Type{T}) -> (Vs, Ks, Kns)

Pre-compute the three 3×3 matrices that depend only on `ξshat`, the
in-plane unit direction written ``\\underline{u}`` below:

  * `Vs`: ``(V_s)_{ip} = \\sum_q (T_n)_{ipq}\\,u_q``, i.e. ``\\boldsymbol{V}(\\underline{u})``
  * `Ks`: ``(K_s)_{ij} = \\sum_{k,l} C_{ikjl}\\,u_k u_l``, i.e. the acoustic tensor ``\\boldsymbol{N}(\\underline{u})``
  * `Kns`: ``(K_{ns})_{ij} = \\sum_{k,l} C_{ikjl}\\,(n_k u_l + u_k n_l)``

`Ks` is built on its upper triangle only and `Kns` is obtained as
`Vs + transpose(Vs)` — both follow from the major symmetry of `C` (see the
comments in the body).  `n̂` is consequently no longer read here; it is kept
in the signature because it is part of this seam's documented contract and
every caller already has it to hand.
"""
function _phi_cache(
        C::AbstractArray, Tn::AbstractArray,
        n̂::AbstractVector, ξshat::AbstractVector,
        ::Type{T}
    ) where {T}
    Vs_m = MMatrix{3, 3, T}(undef)
    Ks_m = MMatrix{3, 3, T}(undef)
    @inbounds for p in 1:3, i in 1:3
        s = zero(T)
        for q in 1:3
            s += Tn[i, p, q] * ξshat[q]
        end
        Vs_m[i, p] = s
    end
    # `Ks` is symmetric under the major symmetry `C_{ikjl} = C_{jlik}`, so only
    # the upper triangle is accumulated and mirrored (was: all 9 entries, half
    # of them computed twice).
    @inbounds for j in 1:3, i in 1:j
        sk = zero(T)
        for k in 1:3, l in 1:3
            sk += T(C[i, k, j, l]) * ξshat[k] * ξshat[l]
        end
        Ks_m[i, j] = sk
        Ks_m[j, i] = sk
    end
    # `Kns[i,j] = Σ_{k,l} C_{ikjl}(n̂_k ξ_l + ξ_k n̂_l)` needs NO loop at all.
    # Split the two terms:
    #   term 1 = Σ_{k,l} C_{ikjl} n̂_k ξ_l                       = Vs[i,j]
    #   term 2 = Σ_{k,l} C_{ikjl} ξ_k n̂_l
    #          = Σ_{k,l} C_{jlik} ξ_k n̂_l   (major symmetry)     = Vs[j,i]
    # so `Kns == Vs + transpose(Vs)`, at zero flops.  Verified to 2e-16 on
    # random minor+major-symmetric stiffnesses; the identity genuinely needs
    # the MAJOR symmetry (breaking it alone moves the result by O(0.25)), which
    # every reference stiffness reaching this function has.
    Vs = SMatrix(Vs_m)
    Kns = Vs + transpose(Vs)
    return Vs, SMatrix(Ks_m), Kns
end

"""
    _qnn_pair_components(A, Vs, Ks, Kns, ca, sa, scale) -> SMatrix{3,3,T}

``[\\hat{\\boldsymbol{Q}}_{nn}(\\underline{\\zeta}_p) + \\hat{\\boldsymbol{Q}}_{nn}(\\underline{\\zeta}_m)]``
multiplied by `scale`, from the pre-computed ``\\varphi``-only quantities
(`A, Vs, Ks, Kns`) and the ``\\alpha``-only trigs (`ca` ``= \\cos\\alpha``,
`sa` ``= \\sin\\alpha``), with a caller-supplied `scale` bundling the residual
``\\rho/\\sin^2\\alpha`` prefactor.

This is the innermost loop of every crack COD back-end.  It used to write into
a caller-owned `Matrix{T}` buffer and build ~10 heap `Matrix{T}` temporaries
per call (`Vp, Vm, Kp, Km` by broadcast, `iKp, iKm` from `_inv3`, then two
chained `*` products).  Everything is `SMatrix` now, so the whole body is
stack-resident and the function is pure.
"""
@inline function _qnn_pair_components(
        A::StaticMatrix{3, 3, T},
        Vs::StaticMatrix{3, 3, T},
        Ks::StaticMatrix{3, 3, T},
        Kns::StaticMatrix{3, 3, T},
        ca::T, sa::T,
        scale::T
    ) where {T}
    cs = ca * sa
    ca² = ca * ca
    sa² = sa * sa
    Vp = ca * A + sa * Vs
    Vm = sa * Vs - ca * A
    Kp = ca² * A + cs * Kns + sa² * Ks
    Km = ca² * A - cs * Kns + sa² * Ks
    Bp = (Vp * _inv3(Kp)) * transpose(Vp)
    Bm = (Vm * _inv3(Km)) * transpose(Vm)
    return (2 * A - Bp - Bm) * scale
end
