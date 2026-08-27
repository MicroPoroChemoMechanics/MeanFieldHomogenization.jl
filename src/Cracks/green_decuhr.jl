# =============================================================================
#  green_decuhr.jl — DECUHR-based evaluation of Q̂*_{nn} (1-D α-integral)
#  and direct 2-D cubature for the elliptic-crack COD.  Uses
#  `Integrals.jl` with `DecuhrAlgorithm` (Espelid & Genz 1994).
#
#  The integrands are smooth on the integration domains (α ∈ [0, π/2]
#  for Q̂*; (φ, α) ∈ [0, 2π] × [0, π/2] for the elliptic COD).  We pass
#  `singul = 1, alpha = 0.0` to `DecuhrAlgorithm`, which degrades to
#  ordinary adaptive Gauss and bypasses the Float64-only `alpha`
#  auto-estimation path — keeping ForwardDiff compatibility.
#
#  Shared 2-D helpers (`_A_and_Tn`, `_phi_cache`, `_qnn_pair_components`,
#  `_inv3`) live in `src/Core/green_helpers.jl`.  The nested-QuadGK
#  counterpart lives in `green_nestedquadgk.jl`.
# =============================================================================

"""
    _Qnn_star_decuhr(C, ξs, n̂; abstol, reltol, maxiters) -> Matrix

Adaptive 1-D DECUHR (via `Integrals.DecuhrAlgorithm`) evaluation of
the crack-plane kernel used in the ``\\omega \\to 0`` limit algorithm
of [barthelemyIJSS2009](@cite).  Integrand is smooth,
`alpha = 0.0` is forced to keep ForwardDiff compatibility.
"""
function _Qnn_star_decuhr(
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

    function integrand(u, _)
        α = u[1]
        ca = cos(α); sa = sin(α)
        M = MFH_Core._qnn_pair_components(A, Vs, Ks, Kns, ca, sa, inv(sa * sa))
        return SVector{6, T}(M[1, 1], M[2, 2], M[3, 3], M[1, 2], M[1, 3], M[2, 3])
    end

    Tvec = MFH_Core._decuhr_cubature(
        integrand, [0.0], [π / 2];
        wrksub = max(5000, maxiters ÷ 32),
        abstol = abstol, reltol = reltol, maxiters = maxiters
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
#  Direct 2-D cubature for elliptic COD — DECUHR version
# -----------------------------------------------------------------------------

"""
    _cod_elliptic_decuhr_direct(c, C₀; abstol, reltol, maxiters)

Direct 2-D DECUHR cubature (via `Integrals.DecuhrAlgorithm`) over the
elliptic-crack COD integrand on `(φ, α) ∈ [0, 2π] × [0, π/2]`.
Smooth integrand — `alpha = 0.0` forced for ForwardDiff compatibility.
"""
function _cod_elliptic_decuhr_direct(
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

    function integrand(u, _)
        φ, α = u[1], u[2]
        cφ = cos(φ); sφ = sin(φ)
        ρ = sqrt(ηp * ηp * cφ * cφ + sφ * sφ)
        invρ = inv(ρ)
        ξshat = SVector{3, Tp}(
            (ηp * cφ * lhat_p[1] + sφ * mhat_p[1]) * invρ,
            (ηp * cφ * lhat_p[2] + sφ * mhat_p[2]) * invρ,
            (ηp * cφ * lhat_p[3] + sφ * mhat_p[3]) * invρ,
        )
        Vs, Ks, Kns = MFH_Core._phi_cache(Carr, Tn, nhat_p, ξshat, Tp)
        ca = cos(α); sa = sin(α)
        M = MFH_Core._qnn_pair_components(A, Vs, Ks, Kns, ca, sa, ρ / (sa * sa))
        return SVector{6, Tp}(M[1, 1], M[2, 2], M[3, 3], M[1, 2], M[1, 3], M[2, 3])
    end

    Tvec = MFH_Core._decuhr_cubature(
        integrand, [0.0, 0.0], [2π, π / 2];
        wrksub = max(5000, maxiters ÷ 32),
        abstol = abstol, reltol = reltol, maxiters = maxiters
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
