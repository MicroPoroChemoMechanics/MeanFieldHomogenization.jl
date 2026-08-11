# =============================================================================
#  green_nestedquadgk.jl — ForwardDiff-compatible evaluation of Q̂*_{nn}
#  and direct 2D cubature for the elliptic-crack COD, using **nested 1-D
#  QuadGK quadrature** (historical default shipped under the `_decuhr`
#  suffix).  Shared 2-D helpers (`_A_and_Tn`, `_phi_cache`,
#  `_qnn_pair_components`, `_inv3`) live in
#  `src/Core/green_helpers.jl`.  The true DECUHR-based counterpart lives
#  in `green_decuhr.jl`.
# =============================================================================

"""
    _Qnn_star_nestedquadgk(C, ξs, n̂; abstol, reltol, maxiters) -> Matrix

ForwardDiff-compatible evaluation of the crack-plane kernel via
nested 1-D QuadGK adaptive quadrature — used in the ``\\omega \\to 0``
limit algorithm of [Barthélémy 2009](@cite barthelemyIJSS2009).
"""
function _Qnn_star_nestedquadgk(
        C::AbstractArray{TC, 4},
        ξs::AbstractVector{Tξ},
        n̂::AbstractVector{Tnh};
        abstol::Real = 1.0e-8,
        reltol::Real = 1.0e-6,
        maxiters::Int = 100_000
    ) where {TC <: Number, Tξ <: Number, Tnh <: Number}
    T = promote_type(TC, Tξ, Tnh)

    ρ² = T(ξs[1])^2 + T(ξs[2])^2 + T(ξs[3])^2
    ρ = sqrt(ρ²)
    iszero(ρ) && return zeros(T, 3, 3)

    nhat = SVector{3, T}(T(n̂[1]), T(n̂[2]), T(n̂[3]))
    ξshat = SVector{3, T}(T(ξs[1]) / ρ, T(ξs[2]) / ρ, T(ξs[3]) / ρ)

    A, Tn = MFH_Core._A_and_Tn(C, nhat, T)
    Vs, Ks, Kns = MFH_Core._phi_cache(C, Tn, nhat, ξshat, T)

    function vec_at(α)
        ca = cos(α); sa = sin(α)
        M = MFH_Core._qnn_pair_components(A, Vs, Ks, Kns, ca, sa, inv(sa * sa))
        return SVector{6, T}(M[1, 1], M[2, 2], M[3, 3], M[1, 2], M[1, 3], M[2, 3])
    end

    Tvec, _ = QuadGK.quadgk(
        vec_at, zero(T), T(π) / 2;
        atol = abstol, rtol = reltol, maxevals = maxiters
    )

    fac = ρ / (2 * T(π))
    result = Matrix{T}(undef, 3, 3)
    result[1, 1] = Tvec[1] * fac; result[2, 2] = Tvec[2] * fac; result[3, 3] = Tvec[3] * fac
    result[1, 2] = result[2, 1] = Tvec[4] * fac
    result[1, 3] = result[3, 1] = Tvec[5] * fac
    result[2, 3] = result[3, 2] = Tvec[6] * fac
    return result
end

# -----------------------------------------------------------------------------
#  Direct 2-D cubature for elliptic COD — nested 1-D QuadGK version
# -----------------------------------------------------------------------------

"""
    _cod_elliptic_nestedquadgk_direct(c, C₀; abstol, reltol, maxiters)

Direct 2-D cubature of the elliptic-crack COD integrand via nested
1-D QuadGK (outer over φ, inner over α).  Same mathematical kernel as
[`_cod_elliptic_decuhr_direct`](@ref); included for comparison with
the DECUHR-based path.
"""
function _cod_elliptic_nestedquadgk_direct(
        c::EllipticCrack{T}, C₀;
        abstol::Real = 1.0e-8,
        reltol::Real = 1.0e-6,
        maxiters::Int = 100_000
    ) where {T <: Number}
    η = aspect_ratio(c)
    # Frame consistency — see `_crack_local_frame`.
    Carr, lhat, mhat, nhat, basis = _crack_local_frame(c, C₀)

    Tp = promote_type(T, eltype(Carr), eltype(lhat))
    ηp = Tp(η)

    nhat_p = Tp[Tp(nhat[1]), Tp(nhat[2]), Tp(nhat[3])]
    lhat_p = Tp[Tp(lhat[1]), Tp(lhat[2]), Tp(lhat[3])]
    mhat_p = Tp[Tp(mhat[1]), Tp(mhat[2]), Tp(mhat[3])]

    A, Tn = MFH_Core._A_and_Tn(Carr, nhat_p, Tp)

    inner_atol = abstol / 10

    function outer(φ)
        cφ = cos(φ); sφ = sin(φ)
        ρ = sqrt(ηp * ηp * cφ * cφ + sφ * sφ)
        invρ = inv(ρ)
        ξshat = SVector{3, Tp}(
            (ηp * cφ * lhat_p[1] + sφ * mhat_p[1]) * invρ,
            (ηp * cφ * lhat_p[2] + sφ * mhat_p[2]) * invρ,
            (ηp * cφ * lhat_p[3] + sφ * mhat_p[3]) * invρ,
        )
        Vs, Ks, Kns = MFH_Core._phi_cache(Carr, Tn, nhat_p, ξshat, Tp)

        function inner(α)
            ca = cos(α); sa = sin(α)
            M = MFH_Core._qnn_pair_components(A, Vs, Ks, Kns, ca, sa, ρ / (sa * sa))
            return SVector{6, Tp}(M[1, 1], M[2, 2], M[3, 3], M[1, 2], M[1, 3], M[2, 3])
        end

        inner_val, _ = QuadGK.quadgk(
            inner, zero(Tp), Tp(π) / 2;
            atol = inner_atol, rtol = reltol, maxevals = maxiters
        )
        return inner_val
    end

    Tvec, _ = QuadGK.quadgk(
        outer, zero(Tp), 2 * Tp(π);
        atol = abstol, rtol = reltol, maxevals = maxiters
    )

    fac = inv(2 * Tp(π))
    Tmat = Matrix{Tp}(undef, 3, 3)
    Tmat[1, 1] = Tvec[1] * fac; Tmat[2, 2] = Tvec[2] * fac; Tmat[3, 3] = Tvec[3] * fac
    Tmat[1, 2] = Tmat[2, 1] = Tvec[4] * fac
    Tmat[1, 3] = Tmat[3, 1] = Tvec[5] * fac
    Tmat[2, 3] = Tmat[3, 2] = Tvec[6] * fac

    Bmat = (Tp(8) / Tp(3)) .* MFH_Core._inv3(Tmat)
    Bsym = (Bmat .+ transpose(Bmat)) ./ 2
    return TensND.Tens(Tensors.SymmetricTensor{2, 3}((i, j) -> Tp(Bsym[i, j])), basis)
end
