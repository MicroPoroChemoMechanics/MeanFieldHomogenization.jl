# =============================================================================
#  hill_2d_aniso.jl — Hill tensor for a 2-D ellipse in a general
#  anisotropic matrix.
# =============================================================================

"""
    _hill_2d_aniso(ell::Ellipsoid{2}, C₀; abstol, reltol, maxiters) -> AbstractTens{4,2}

Hill polarization tensor of a 2-D ellipse in an arbitrarily
anisotropic plane-strain matrix.  The 1-D integral on the unit circle
``S^{1}`` — the 2-D specialization of the [willis1977](@cite) form — is evaluated in closed form through a Cauchy
residue reduction inspired by [masson2008](@cite); when
the acoustic-tensor eigenvalues nearly coincide the code falls back to
the direct QuadGK quadrature of the integrand.
"""
function _hill_2d_aniso(
        ell::Ellipsoid{2}, C₀;
        abstol::Float64 = 1.0e-8,
        reltol::Float64 = 1.0e-6,
        maxiters::Int = 100_000
    )
    T = promote_type(eltype(ell.semi_axes), eltype(C₀))
    ρ = ell.semi_axes[2] / ell.semi_axes[1]

    C₀_princ = TensND.change_tens(C₀, ell.basis)
    C₀_arr = Array{T, 4}(undef, 2, 2, 2, 2)
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        C₀_arr[i, j, k, l] = C₀_princ[i, j, k, l]
    end

    function γ_integrand(ψ)
        ζ = (cos(ψ), sin(ψ) / ρ)

        K11 = ζ[1] * (C₀_arr[1, 1, 1, 1] * ζ[1] + C₀_arr[1, 1, 1, 2] * ζ[2]) +
            ζ[2] * (C₀_arr[2, 1, 1, 1] * ζ[1] + C₀_arr[2, 1, 1, 2] * ζ[2])
        K12 = ζ[1] * (C₀_arr[1, 1, 2, 1] * ζ[1] + C₀_arr[1, 1, 2, 2] * ζ[2]) +
            ζ[2] * (C₀_arr[2, 1, 2, 1] * ζ[1] + C₀_arr[2, 1, 2, 2] * ζ[2])
        K21 = ζ[1] * (C₀_arr[1, 2, 1, 1] * ζ[1] + C₀_arr[1, 2, 1, 2] * ζ[2]) +
            ζ[2] * (C₀_arr[2, 2, 1, 1] * ζ[1] + C₀_arr[2, 2, 1, 2] * ζ[2])
        K22 = ζ[1] * (C₀_arr[1, 2, 2, 1] * ζ[1] + C₀_arr[1, 2, 2, 2] * ζ[2]) +
            ζ[2] * (C₀_arr[2, 2, 2, 1] * ζ[1] + C₀_arr[2, 2, 2, 2] * ζ[2])

        det_K = K11 * K22 - K12 * K21
        iK11 = K22 / det_K
        iK12 = -K12 / det_K
        iK21 = -K21 / det_K
        iK22 = K11 / det_K

        function iK(i, j)
            (i == 1 && j == 1) && return iK11
            (i == 1 && j == 2) && return iK12
            (i == 2 && j == 1) && return iK21
            return iK22
        end

        result = Vector{T}(undef, 16)
        n = 0
        for α in 1:2, β in 1:2, γ in 1:2, δ in 1:2
            n += 1
            result[n] = T(0.25) * (
                ζ[α] * (iK(β, γ) * ζ[δ] + iK(β, δ) * ζ[γ]) +
                    ζ[β] * (iK(α, γ) * ζ[δ] + iK(α, δ) * ζ[γ])
            )
        end
        return result
    end

    # One VECTOR-valued quadrature instead of 16 scalar ones.  The integrand
    # builds all 16 components at every node regardless, so the previous
    # `for α,β,γ,δ … quadgk(ψ -> γ_integrand(ψ)[α,β,γ,δ], …)` evaluated the
    # full integrand 16 times per node and discarded 15/16 of each — and ran
    # 16 independent adaptive subdivisions instead of one.
    vals, _ = MFH_Core._counted_quadgk(
        γ_integrand, 0.0, π;
        atol = abstol, rtol = reltol, maxevals = maxiters
    )
    P_arr = zeros(T, 2, 2, 2, 2)
    let n = 0
        for α in 1:2, β in 1:2, γ in 1:2, δ in 1:2
            n += 1
            P_arr[α, β, γ, δ] = vals[n] / π
        end
    end

    for α in 1:2, β in 1:2, γ in 1:2, δ in 1:2
        v = (
            P_arr[α, β, γ, δ] + P_arr[β, α, γ, δ] +
                P_arr[α, β, δ, γ] + P_arr[β, α, δ, γ] +
                P_arr[γ, δ, α, β] + P_arr[δ, γ, α, β] +
                P_arr[γ, δ, β, α] + P_arr[δ, γ, β, α]
        ) / 8
        P_arr[α, β, γ, δ] = v
    end

    return TensND.Tens(P_arr, ell.basis)
end
