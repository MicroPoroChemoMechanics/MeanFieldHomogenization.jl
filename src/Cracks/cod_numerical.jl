# =============================================================================
#  cod_numerical.jl — numerical COD tensor for anisotropic matrices.
# =============================================================================

"""
    _cod_elliptic_numerical(c, C₀, backend; abstol, reltol, maxiters) -> Tens{2,3}

COD tensor of an elliptic crack in an arbitrarily anisotropic matrix.
The limit ``\\omega\\to 0`` of ``\\omega\\,\\mathbb Q^{-1}`` is resolved
by the first-order Taylor term of the Hill tensor
([Barthélémy 2009](@cite barthelemyIJSS2009)); the resulting integral
on the unit circle of the crack plane is evaluated by `backend`
(residue reduction [Masson 2008](@cite masson2008) or DECUHR cubature
[Espelid & Genz 1994](@cite espelid1994)).
"""
function _cod_elliptic_numerical(
        c::EllipticCrack{T}, C₀, backend;
        abstol::Real = 1.0e-8,
        reltol::Real = 1.0e-6,
        maxiters::Int = 100_000
    ) where {T <: Number}
    η = aspect_ratio(c)
    # Frame consistency — see `_crack_local_frame`.
    Carr, lhat, mhat, nhat, basis = _crack_local_frame(c, C₀)

    Tp = promote_type(T, eltype(Carr), eltype(lhat))
    ηp = Tp(η)

    lhat_p = Tp[Tp(lhat[1]), Tp(lhat[2]), Tp(lhat[3])]
    mhat_p = Tp[Tp(mhat[1]), Tp(mhat[2]), Tp(mhat[3])]
    nhat_p = Tp[Tp(nhat[1]), Tp(nhat[2]), Tp(nhat[3])]

    # Vector-valued outer QuadGK: 6 independent components of the 3×3 symmetric
    # matrix M(φ) are integrated at once so all share the same adaptive
    # subdivisions — makes S self-consistent and reduces residue evaluations
    # from 6× to 1× per φ point.
    function vec_at(φ)
        ξ_plane = Tp[
            ηp * cos(φ) * lhat_p[1] + sin(φ) * mhat_p[1],
            ηp * cos(φ) * lhat_p[2] + sin(φ) * mhat_p[2],
            ηp * cos(φ) * lhat_p[3] + sin(φ) * mhat_p[3],
        ]
        Mv = backend(Carr, ξ_plane, nhat_p; abstol = abstol, reltol = reltol, maxiters = maxiters)
        return Tp[Mv[1, 1], Mv[2, 2], Mv[3, 3], Mv[1, 2], Mv[1, 3], Mv[2, 3]]
    end

    MFH_Core._bump!(MFH_Core.QUADGK_OUTER)
    Tvec, _ = MFH_Core._counted_quadgk(
        vec_at, zero(Tp), 2 * Tp(π);
        atol = abstol, rtol = reltol, maxevals = maxiters
    )

    S = Matrix{Tp}(undef, 3, 3)
    S[1, 1] = Tvec[1]; S[2, 2] = Tvec[2]; S[3, 3] = Tvec[3]
    S[1, 2] = S[2, 1] = Tvec[4]
    S[1, 3] = S[3, 1] = Tvec[5]
    S[2, 3] = S[3, 2] = Tvec[6]

    Smat = S / (Tp(8) / Tp(3))
    Bmat = inv(Smat)
    Bsym = (Bmat + transpose(Bmat)) / 2
    return TensND.Tens(Tensors.SymmetricTensor{2, 3}((i, j) -> Tp(Bsym[i, j])), basis)
end

"""
    _cod_ribbon_numerical(c, C₀, backend; …)
"""
function _cod_ribbon_numerical(
        c::RibbonCrack{T}, C₀, backend;
        abstol::Real = 1.0e-8,
        reltol::Real = 1.0e-6,
        maxiters::Int = 100_000
    ) where {T <: Number}
    # Same frame-consistency rule as `_cod_elliptic_numerical`: `Carr` holds the
    # components of `C₀` in the crack basis, so the ribbon's in-plane direction
    # and its normal are the canonical `e₂` and `e₃` of that basis.
    Carr, _, mhat, nhat, basis = _crack_local_frame(c, C₀)

    Tp = promote_type(T, eltype(Carr), eltype(mhat))

    mhat_p = Tp.(mhat)
    nhat_p = Tp.(nhat)

    Qstar = backend(
        Carr, mhat_p, nhat_p;
        abstol = abstol, reltol = reltol, maxiters = maxiters
    )

    Bmat = inv(Qstar) * (Tp(π) / 4)
    Bsym = (Bmat + transpose(Bmat)) / 2
    return TensND.Tens(Tensors.SymmetricTensor{2, 3}((i, j) -> Tp(Bsym[i, j])), basis)
end

# Backend dispatch
_residue_backend(Carr, ξ_plane, nhat; kw...) = _Qnn_star_residue(Carr, ξ_plane, nhat)

_decuhr_backend(Carr, ξ_plane, nhat; abstol, reltol, maxiters) =
    _Qnn_star_decuhr(Carr, ξ_plane, nhat; abstol = abstol, reltol = reltol, maxiters = maxiters)

_nestedquadgk_backend(Carr, ξ_plane, nhat; abstol, reltol, maxiters) =
    _Qnn_star_nestedquadgk(Carr, ξ_plane, nhat; abstol = abstol, reltol = reltol, maxiters = maxiters)
