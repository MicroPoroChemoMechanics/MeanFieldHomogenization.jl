# =============================================================================
#  green_decuhr.jl — direct 2-D DECUHR cubature for the elliptic-crack COD.
#  Uses `Integrals.jl` with `DecuhrAlgorithm` (Espelid & Genz 1994), which
#  integrates over hyperrectangles of dimension two or more: the ribbon, whose
#  reduced kernel is a single 1-D integral, has no DECUHR path (see the ribbon
#  `_kernel` in `api.jl`).
#
#  The integrand is smooth on (φ, α) ∈ [0, 2π] × [0, π/2].  We pass
#  `singul = 1, alpha = 0.0` to `DecuhrAlgorithm`, which degrades to
#  ordinary adaptive Gauss and bypasses the Float64-only `alpha`
#  auto-estimation path — keeping ForwardDiff compatibility.
#
#  Shared 2-D helpers (`_A_and_Tn`, `_phi_cache`, `_qnn_pair_components`,
#  the 3×3 inverse is TensND's `inv3`) live in `src/Core/green_helpers.jl`.  The nested-QuadGK
#  counterpart lives in `green_nestedquadgk.jl`.
# =============================================================================

# -----------------------------------------------------------------------------
#  Direct 2-D cubature for elliptic COD — DECUHR version
# -----------------------------------------------------------------------------

"""
    _cod_elliptic_decuhr_direct(c, C₀; abstol, reltol, maxiters)

Direct 2-D DECUHR cubature (via `Integrals.DecuhrAlgorithm`) over the
elliptic-crack COD integrand on ``(\\varphi, \\alpha) \\in [0, 2\\pi]\\times[0, \\pi/2]``.
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

    Bmat = (Tp(8) / Tp(3)) .* TensND.inv3(Tmat)
    Bsym = (Bmat .+ transpose(Bmat)) ./ 2
    return TensND.Tens(Tensors.SymmetricTensor{2, 3}((i, j) -> Tp(Bsym[i, j])), basis)
end
