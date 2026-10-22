# =============================================================================
#  hill_2d_aniso.jl — Hill tensor for a 2-D ellipse in a general
#  anisotropic matrix.
# =============================================================================

"""
    _hill_2d_aniso(ell::Ellipsoid{2}, C₀; abstol, reltol, maxiters) -> AbstractTens{4,2}

Hill polarization tensor of a 2-D ellipse in an arbitrarily
anisotropic plane-strain matrix.  The 1-D integral on the unit circle
``S^{1}`` — the 2-D specialization of the form of [willis1977](@citet) — is evaluated in closed form through a Cauchy
residue reduction inspired by [masson2008](@citet); when
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

    C₀_arr = T.(Array(TensND.components(C₀, ell.basis)))   # in the ellipse's frame

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

        iK = ((iK11, iK12), (iK21, iK22))

        # The 16 components in the order α, β, γ, δ with δ fastest, on the
        # stack: one call per quadrature node.
        return SVector{16, T}(
            ntuple(Val(16)) do n
                δ = (n - 1) % 2 + 1
                γ = (n - 1) ÷ 2 % 2 + 1
                β = (n - 1) ÷ 4 % 2 + 1
                α = (n - 1) ÷ 8 + 1
                T(0.25) * (
                    ζ[α] * (iK[β][γ] * ζ[δ] + iK[β][δ] * ζ[γ]) +
                        ζ[β] * (iK[α][γ] * ζ[δ] + iK[α][δ] * ζ[γ])
                )
            end
        )
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
    # `vals` runs over (α, β, γ, δ) with δ fastest; `reshape` puts the first
    # index fastest, hence the reversal of the four indices.
    P_arr = permutedims(reshape(collect(vals) ./ π, 2, 2, 2, 2), (4, 3, 2, 1))

    # Project on the minor and major symmetries. Averaged in place, an entry
    # already overwritten entered the average of the next member of its orbit,
    # and the result was not exactly symmetric; each orbit is now averaged
    # once, in the order of its smallest index, and that value given to all
    # its members.
    P_sym = similar(P_arr)
    for α in 1:2, β in 1:2, γ in 1:2, δ in 1:2
        rep = minimum(_orbit8(α, β, γ, δ))
        P_sym[α, β, γ, δ] = sum(P_arr[I...] for I in _orbit8(rep...)) / 8
    end

    return TensND.Tens(P_sym, ell.basis)
end

# The eight index tuples a tensor with minor and major symmetries maps to one value.
@inline _orbit8(a, b, c, d) = (
    (a, b, c, d), (b, a, c, d), (a, b, d, c), (b, a, d, c),
    (c, d, a, b), (d, c, a, b), (c, d, b, a), (d, c, b, a),
)
