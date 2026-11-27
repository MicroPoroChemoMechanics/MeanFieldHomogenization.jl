# =============================================================================
#  hill_2d_iso.jl — Hill tensor for a 2-D ellipse in an isotropic matrix.
#
#  Closed form derived from the Eshelby tensor of an elliptic cylinder,
#  [mura1987; eq. 11.22](@cite), contracted with `C₀⁻¹`
#  (`P = S : C₀⁻¹`).  A stiffness of the plane is, in TensND's convention,
#  `C₀ = TensISO{2}(α, β) = α·𝕁₂ + β·𝕂₂` with `α = 2κ` and `β = 2μ`, `κ` the
#  in-plane bulk modulus: `κ = λ + μ` for the plane strain of a 3-D material,
#  whose Poisson ratio is then `ν = (α-β)/(2α)`.
#
#  Everything is parametrized by
#      s = κ/(κ+μ) = α/(α+β)      t = μ/(κ+μ) = β/(α+β) = 1 - s
#  which stay finite as `κ → ∞` (`s → 1`, `t → 0`), so the incompressible
#  case needs no separate formula beyond `1/α → 0`.
#
#  Cross-validated to machine precision against Mura's `S` and, over a sweep
#  in `(k, μ, ρ)`, against the general quadrature path `_hill_2d_aniso`
#  (see `test/Elasticity/test_hill_2d.jl`).
# =============================================================================

# Shared (s, t, 1/α) triple; `isinf` is only meaningful for float moduli, so
# a symbolic `α` always takes the generic branch.
@inline function _hill_2d_iso_coeffs(α, β)
    if isa(α, AbstractFloat) && isinf(α)
        return one(typeof(β)), zero(typeof(β)), zero(typeof(β))
    end
    d = α + β
    return α / d, β / d, 1 / α
end

"""
    _hill_2d_iso(ell::Ellipsoid{2}, C₀::TensISO{4,2}) -> AbstractTens{4,2}

Analytical Hill polarization tensor of a 2-D ellipse in an isotropic
matrix of the plane ``\\mathbb{C}_0 = 2\\kappa\\,\\mathbb{J} + 2\\mu\\,\\mathbb{K}``
(``\\kappa`` the in-plane bulk modulus, ``\\lambda + \\mu`` in plane strain),
obtained from the elliptic-cylinder Eshelby tensor of
[mura1987](@citet) through ``\\mathbb{P} = \\mathbb{S}^{\\mathrm{E}}:\\mathbb{C}_0^{-1}``.
For a disc, ``\\mathbb{P} = \\mathbb{J}/(2(\\kappa+\\mu)) + (\\kappa+2\\mu)/(4\\mu(\\kappa+\\mu))\\,\\mathbb{K}``.

Setting ``\\kappa = \\infty`` (`Inf`) gives the incompressible limit.
"""
function _hill_2d_iso(ell::Ellipsoid{2, Circular}, C₀)
    T = promote_type(eltype(ell.semi_axes), eltype(C₀))
    α, β = C₀.data
    # ρ = 1 specialization of the elliptic formulas below:
    #   P = P_J·𝕁₂ + P_K·𝕂₂,  P_J = 1/(α+β),  P_K = (α+2β)/(2β(α+β)).
    local P1111, P1122
    if isa(α, AbstractFloat) && isinf(α)
        den = 4β
        P1111 = one(T) / den
        P1122 = -one(T) / den
    else
        den = 4β * (α + β)
        P1111 = (α + 4β) / den
        P1122 = -α / den
    end
    return TensND.TensISO{2}(T(P1111 + P1122), T(P1111 - P1122))
end

function _hill_2d_iso(ell::Ellipsoid{2, Elliptic}, C₀)
    T = promote_type(eltype(ell.semi_axes), eltype(C₀))
    α, β = C₀.data
    ρ = T(ell.semi_axes[2] / ell.semi_axes[1])
    ρ2 = ρ * ρ
    up = 1 + ρ
    up2 = up * up

    s, t, inv_α = _hill_2d_iso_coeffs(α, β)

    # Mura (1987) eq. 11.22 — elliptic cylinder, semi-axes (1, ρ).
    S1111 = s * (ρ2 + 2ρ) / up2 + t * ρ / up
    S2222 = s * (1 + 2ρ) / up2 + t / up
    S1122 = s * ρ2 / up2 - t * ρ / up
    S2211 = s / up2 - t / up
    S1212 = s * (1 + ρ2) / (2 * up2) + t / 2

    # P = S : C₀⁻¹ with C₀⁻¹ = (1/β)·𝕀 + (1/α - 1/β)·𝕁₂ ; the 𝕁₂ part only
    # sees the row traces S_ijmm.
    c = (inv_α - 1 / β) / 2
    P1111 = S1111 / β + c * (S1111 + S1122)
    P1122 = S1122 / β + c * (S1111 + S1122)
    P2222 = S2222 / β + c * (S2211 + S2222)
    P1212 = S1212 / β

    P_arr = zeros(T, 2, 2, 2, 2)
    P_arr[1, 1, 1, 1] = P1111
    P_arr[2, 2, 2, 2] = P2222
    P_arr[1, 1, 2, 2] = P_arr[2, 2, 1, 1] = P1122
    for (a, b, c_, d) in ((1, 2, 1, 2), (1, 2, 2, 1), (2, 1, 1, 2), (2, 1, 2, 1))
        P_arr[a, b, c_, d] = P1212
    end
    return TensND.symmetric_tens(P_arr, ell.basis)
end
