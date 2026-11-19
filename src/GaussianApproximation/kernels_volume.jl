# =============================================================================
#  kernels_volume.jl — the Green operator applied to one Gaussian of the volume.
#
#  With the polarization τ expanded on the Gaussians of a grid of step h, the
#  strain (or the gradient of the potential) is E - Σ_m 𝔾⁰ₕ(x - x_m) : τ_m, with
#
#      𝔾⁰ₕ = 𝔾⁰ * φ ,   φ(x) = exp(-|x|²/(H h²)) / (πH)^(3/2) ,
#
#  the Green operator of the reference medium smoothed by the Gaussian. 𝔾⁰ being
#  homogeneous of degree -3 and φ of integral h³, 𝔾⁰ₕ depends on x/h only.
#
#  In an isotropic medium 𝔾⁰ is made of second derivatives of 1/(4πr) and of
#  fourth derivatives of r/(8π). Smoothed by the Gaussian, the two potentials are
#  (ρ = |x|/h, their Laplacians being -φ and Φ₀)
#
#      Φ₀(ρ) = erf(ρ/√H) / (4πρ) ,
#      Φ₂(ρ) = [(ρ²/2 + H/4) erf(ρ/√H) + ρ √(H/π) exp(-ρ²/H)/2] / (4πρ) ,
#
#  and 𝔾⁰ₕ follows by the same derivatives:
#
#      conduction   𝔾⁰ₕ = -(1/k₀) grad grad Φ₀
#      elasticity   𝔾⁰ₕ = -(1/μ) [ (𝟏 ⊠ˢ grad grad Φ₀)ˢ - grad⁴ Φ₂ / (2(1-ν)) ]
# =============================================================================

_harmonic_potential(ρ, H) = erf(ρ / sqrt(H)) / (4 * oftype(ρ, π) * ρ)
_biharmonic_potential(ρ, H) =
    ((ρ^2 / 2 + H / 4) * erf(ρ / sqrt(H)) + ρ * sqrt(H / π) * exp(-ρ^2 / H) / 2) / (4 * oftype(ρ, π) * ρ)

# Derivatives of order 1 to 4 of a function of one variable, by nested duals.
function _derivatives(f::F, ρ) where {F}
    d₁(x) = ForwardDiff.derivative(f, x)
    d₂(x) = ForwardDiff.derivative(d₁, x)
    d₃(x) = ForwardDiff.derivative(d₂, x)
    return d₁(ρ), d₂(ρ), d₃(ρ), ForwardDiff.derivative(d₃, ρ)
end

"""
    radial_hessian(ρ, H) -> (α, β)

Second gradient of the smoothed harmonic potential ``\\Phi_0`` at the distance
`ρ` (in grid steps) of its node, in the direction ``\\underline{n}``:
``\\operatorname{grad}\\operatorname{grad}\\Phi_0 = \\alpha\\,\\boldsymbol{1} +
\\beta\\,\\underline{n}\\otimes\\underline{n}``. At the node,
``\\alpha = -1/(3(\\pi H)^{3/2})`` and ``\\beta = 0``.
"""
function radial_hessian(ρ, H)
    iszero(ρ) && return -inv(3 * sqrt(oftype(ρ / 1, π) * H)^3), zero(ρ / 1)
    ψ₁, ψ₂, _, _ = _derivatives(x -> _harmonic_potential(x, H), ρ)
    return ψ₁ / ρ, ψ₂ - ψ₁ / ρ
end

"""
    radial_fourth_gradient(ρ, H) -> (A, B, C)

Fourth gradient of the smoothed biharmonic potential ``\\Phi_2`` at the distance
`ρ` of its node, in the direction ``\\underline{n}``: with
``\\boldsymbol{N} = \\underline{n}\\otimes\\underline{n}`` and
``\\mathbb{I} = \\boldsymbol{1}\\stackrel{s}{\\boxtimes}\\boldsymbol{1}``,

```math
\\operatorname{grad}^4\\Phi_2 = A\\,(\\boldsymbol{1}\\otimes\\boldsymbol{1} + 2\\,\\mathbb{I})
 + B\\,\\bigl(\\boldsymbol{1}\\otimes\\boldsymbol{N} + \\boldsymbol{N}\\otimes\\boldsymbol{1}
 + 2\\,(\\boldsymbol{1}\\stackrel{s}{\\boxtimes}\\boldsymbol{N}
 + \\boldsymbol{N}\\stackrel{s}{\\boxtimes}\\boldsymbol{1})\\bigr)
 + C\\,\\boldsymbol{N}\\otimes\\boldsymbol{N} .
```

At the node, ``A = -1/(15(\\pi H)^{3/2})`` and ``B = C = 0``.
"""
function radial_fourth_gradient(ρ, H)
    iszero(ρ) && return -inv(15 * sqrt(oftype(ρ / 1, π) * H)^3), zero(ρ / 1), zero(ρ / 1)
    ψ₁, ψ₂, ψ₃, ψ₄ = _derivatives(x -> _biharmonic_potential(x, H), ρ)
    return (ψ₂ - ψ₁ / ρ) / ρ^2, (ψ₃ - 3ψ₂ / ρ + 3ψ₁ / ρ^2) / ρ,
        ψ₄ - 6ψ₃ / ρ + 15ψ₂ / ρ^2 - 15ψ₁ / ρ^3
end

# Unit vector of a difference of nodes, and its distance; any direction at the node.
function _direction(Δ::NTuple{3, Int})
    ρ = sqrt(float(sum(abs2, Δ)))
    return ρ, iszero(ρ) ? Vec{3}((1.0, 0.0, 0.0)) : Vec{3}(Δ ./ ρ)
end

"""
    smoothed_green_operator(K₀::TensISO{2,3}, Δ, H) -> SMatrix{3,3}
    smoothed_green_operator(C₀::TensISO{4,3}, Δ, H) -> SMatrix{6,6}

The Green operator of an isotropic medium applied to the Gaussian of a node,
``\\mathbb{G}^0_h = \\mathbb{G}^0 * \\varphi``, at the difference of nodes `Δ`
(three integers): its components in conduction, its Kelvin–Mandel matrix in
elasticity. Far from the node it is ``\\mathbb{G}^0`` itself at the point `Δ`.
"""
function smoothed_green_operator(K₀::TensND.TensISO{2, 3}, Δ::NTuple{3, Int}, H)
    k₀ = first(TensND.get_data(K₀))
    ρ, n = _direction(Δ)
    α, β = radial_hessian(ρ, H)
    return -SMatrix{3, 3}(α * one(SymmetricTensor{2, 3}) + β * symmetric(n ⊗ n)) / k₀
end

function smoothed_green_operator(C₀::TensND.TensISO{4, 3}, Δ::NTuple{3, Int}, H)
    μ, ν = TensND.mu_nu(C₀)
    ρ, n = _direction(Δ)
    α, β = radial_hessian(ρ, H)
    A, B, C = radial_fourth_gradient(ρ, H)
    𝟏, N = one(SymmetricTensor{2, 3}), symmetric(n ⊗ n)
    𝕀 = one(SymmetricTensor{4, 3})
    # 𝟏 ⊠ˢ N + N ⊠ˢ 𝟏, with (a ⊠ˢ b)_ijkl = (a_ik b_jl + a_il b_jk) / 2
    𝟏N = symmetric(otimesu(𝟏, N) + otimesl(𝟏, N) + otimesu(N, 𝟏) + otimesl(N, 𝟏)) / 2
    hessian = α * 𝕀 + β * 𝟏N / 2
    fourth = A * (𝟏 ⊗ 𝟏 + 2 * 𝕀) + B * (𝟏 ⊗ N + N ⊗ 𝟏 + 2 * 𝟏N) + C * (N ⊗ N)
    return -tomandel(SMatrix{6, 6}, hessian - fourth / (2 * (1 - ν))) / μ
end
