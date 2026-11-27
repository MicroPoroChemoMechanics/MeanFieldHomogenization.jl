# =============================================================================
#  green_plane_aniso.jl — real-space Green operator of an *anisotropic*
#  elastic medium of the plane.
#
#  In the plane the displacement Green function of a stiffness C is the
#  integral over the unit circle (Fourier transform of K(ξ)⁻¹/|ξ|² in polar
#  coordinates; Mura 1987, §5)
#
#       G(x) = −1/(4π²) ∫₀^{2π} K(ξ)⁻¹ log|ξ·x| dφ + const ,
#       K_ij(ξ) = ξ_a C_aijb ξ_b ,   ξ = (cos φ, sin φ) ,
#
#  which reduces to −log r/(2πκ) for a scalar K = κ. The constant does not
#  enter the gradients. Two derivatives of log|ξ·x| give −ξ⊗ξ/(ξ·x)², a double
#  pole where ξ ⊥ x, and the second gradient is a Hadamard finite part:
#
#       ∂ₖ∂ₗ G_ij(x) = 1/(4π²) f.p.∫₀^{2π} h_ijkl(φ)/(ξ·x)² dφ ,
#       h_ijkl = K(ξ)⁻¹_ij ξ_k ξ_l .
#
#  h is even in ξ, so the integral is twice that over half a turn. With
#  n = x/r, t ⊥ n and ξ(ψ) = cos ψ t − sin ψ n, the pole is at ψ = 0,
#  (ξ·x)² = r² sin²ψ, and an integration by parts (the boundary terms vanish
#  with cot(±π/2), the finite part discards the divergent ones) leaves a
#  principal value, written on ψ > 0 as a regular integral:
#
#       f.p.∫_{−π/2}^{π/2} h/sin²ψ dψ = p.v.∫ h′ cot ψ dψ
#                                     = ∫₀^{π/2} [h′(ψ) − h′(−ψ)] cot ψ dψ ,
#
#  so that
#
#       ∂ₖ∂ₗ G_ij(x) = 1/(2π² r²) ∫₀^{π/2} [h′_ijkl(ψ) − h′_ijkl(−ψ)] cot ψ dψ .
#
#  The integrand is analytic on [0, π/2] (the difference is odd, cot has a
#  simple pole), and a Gauss-Legendre rule converges geometrically. h′ is
#  written in closed form, from (K⁻¹)′ = −K⁻¹ K′ K⁻¹ and
#  K′ = ξ′·C·ξ + ξ·C·ξ′: no derivative is taken numerically, and no root of
#  the characteristic polynomial of the Stroh formalism is needed, so that the
#  isotropic limit, where these roots coincide, is an ordinary case.
#
#  For the scalar K = κ and x along e₁: h₁₁ = sin²ψ/κ, the integrand is
#  4cos²ψ/κ, its integral π/κ, and ∂₁∂₁G = 1/(2πκr²), which is the second
#  derivative of −log r/(2πκ) along the radius.
#
#  SIGN and API: those of `green_operator.jl` and `green_aniso.jl`. The operator
#  is minus the second gradient, symmetrized on both pairs of indices.
# =============================================================================

# K(ξ)⁻¹ and its derivative along the circle, ξ′ = η: K_ij = ξ_a C_aijb ξ_b.
function _plane_acoustic_inverse(C, ξ, η)
    K = _sfill((i, j) -> sum(ξ[a] * C[a, i, j, b] * ξ[b] for a in 1:2, b in 1:2), Val((2, 2)))
    K′ = _sfill((i, j) -> sum((η[a] * ξ[b] + ξ[a] * η[b]) * C[a, i, j, b] for a in 1:2, b in 1:2), Val((2, 2)))
    K⁻¹ = inv(K)
    return K⁻¹, -(K⁻¹ * K′ * K⁻¹)
end

# h′_ijkl at the angle u = sψ from the pole, s = ±1: ξ(u) = cos u t − sin u n
# and η = dξ/du.
function _plane_green_integrand(C, t, n, sψ, cψ, s)
    ξ, η = cψ * t - (s * sψ) * n, -(s * sψ) * t - cψ * n
    K⁻¹, dK⁻¹ = _plane_acoustic_inverse(C, ξ, η)
    return _sfill(Val((2, 2, 2, 2))) do i, j, k, l
        dK⁻¹[i, j] * ξ[k] * ξ[l] + K⁻¹[i, j] * (η[k] * ξ[l] + ξ[k] * η[l])
    end
end

# H[i, j, k, l] = ∂ₖ∂ₗ G_ij(x), from the canonical components `C` of the stiffness.
function _plane_green_hessian(C, x::AbstractVector, nodes::Int)
    r² = x[1]^2 + x[2]^2
    iszero(r²) && throw(
        DomainError(x, "the Green function of an infinite medium is singular at the origin")
    )
    r = sqrt(r²)
    n = SVector{2}(x[1] / r, x[2] / r)
    t = SVector{2}(-n[2], n[1])
    ψ, w = gauss_legendre_nodes(nodes, 0.0, π / 2)
    H = zero(SArray{Tuple{2, 2, 2, 2}, promote_type(eltype(C), eltype(x))})
    @inbounds for m in eachindex(ψ)
        sψ, cψ = sincos(ψ[m])
        H = H + (w[m] * cψ / sψ) * (_plane_green_integrand(C, t, n, sψ, cψ, 1) - _plane_green_integrand(C, t, n, sψ, cψ, -1))
    end
    return H / (2 * π^2 * r²)
end

"""
    green_operator_aniso(C₀::AbstractTens{4,2}, x; nodes = 32) -> Tens{4,2}

Real-space Green operator of an anisotropic elastic medium **of the plane**, of
stiffness `C₀` (any tensor of the plane: orthotropic, square, general),

```math
\\mathbb{G}^0_{ijkl}(\\underline{x}) = -\\Big[\\frac{\\partial^2 G_{ik}}
  {\\partial x_j\\, \\partial x_l}(\\underline{x})\\Big]_{(ij)(kl)} ,
\\qquad
\\frac{\\partial^2 G_{ij}}{\\partial x_k\\,\\partial x_l}(\\underline{x})
= \\frac{1}{4\\pi^2}\\ \\mathrm{f.p.}\\!\\!\\int_0^{2\\pi}
  \\frac{K^{-1}_{ij}(\\underline{\\xi})\\,\\xi_k\\,\\xi_l}{(\\underline{\\xi}\\cdot\\underline{x})^2}\\,\\mathrm{d}\\varphi ,
```

with ``K_{ij}(\\underline{\\xi}) = \\xi_a C_{aijb}\\,\\xi_b`` the acoustic tensor on
the unit circle. The finite part is integrated by parts into a regular
integral over a quarter turn, on which a Gauss-Legendre rule of `nodes` points
converges geometrically ([Plane problems](@ref th-plane-green)). It is
homogeneous of degree ``-2`` in ``\\underline{x}``, and reduces to
[`green_operator_iso`](@ref) for an isotropic `C₀`, which no step of the
computation singles out.

Throws a `DomainError` at the origin. Type-generic in the stiffness and in the
point, `ForwardDiff.Dual` included.
"""
green_operator_aniso(C₀::TensND.AbstractTens{4, 2}, x::AbstractVector; nodes::Int = 32) =
    TensND.Tens(_green_operator_aniso(C₀, x, nodes))

function _green_operator_aniso(C₀::TensND.AbstractTens{4, 2}, x::AbstractVector, nodes::Int)
    C = SArray{Tuple{2, 2, 2, 2}}(TensND.canonical_components(C₀))
    H = _plane_green_hessian(C, x, nodes)
    return _sfill(Val((2, 2, 2, 2))) do i, j, k, l
        -(H[i, k, j, l] + H[j, k, i, l] + H[i, l, j, k] + H[j, l, i, k]) / 4
    end
end
