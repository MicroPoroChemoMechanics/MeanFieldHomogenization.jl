# =============================================================================
#  sif.jl — stress / displacement intensity factors.
# =============================================================================

"""
    sif(crack, C₀, Σ; y₀=nothing, method=:auto, kw...) -> (𝐊, (Kᴵ, Kᴵᴵ, Kᴵᴵᴵ))

Stress intensity factor vector ``\\hat{\\mathbf K}`` at a point of the
crack front, together with its ``(K_{I},K_{II},K_{III})`` decomposition
on ``(\\hat{\\mathbf n},\\hat{\\boldsymbol\\nu},\\hat{\\boldsymbol\\tau})``
([irwin1957](@cite),
 [kassir1968](@cite),
 [willis1968](@cite);
 energy release rate identity ``G = \\hat{\\mathbf K}\\cdot\\hat{\\mathbf N}``
 in [barnett1972](@cite),
 [rice1989](@cite)).

For a ribbon crack (``\\hat{\\boldsymbol\\nu}=\\pm\\hat{\\mathbf m}``)
``\\hat{\\mathbf K}^{\\mathcal R} = \\sqrt{\\pi b}\\,\\boldsymbol\\Sigma\\cdot\\hat{\\mathbf n}``
(independent of the matrix stiffness).
For an elliptic crack, ``\\hat{\\mathbf K}`` is obtained from the COD
tensor ``\\mathbf B^{\\mathcal E}`` of the actual crack and the COD
tensor ``\\mathbf B^{\\mathcal R}`` of the tangent ribbon crack at the
observation point:

```
K̂ = (3/8) π^{3/2} √b √(b ‖S† · ŷ₀★‖)
    · (B^𝓡(ν̂, n̂))⁻¹ · B^𝓔(m̂, n̂, η) · Σ·n̂ .
```

The central identity
``\\hat{\\mathbf K} = \\pi\\,(\\mathbf B^{\\mathcal R})^{-1}\\cdot\\hat{\\mathbf N}``
is purely local
([kanaun1981](@cite), [kunin1983](@cite),
 [kanaun2009](@cite)).
"""
function sif end

# Ribbon crack (paper eq. 737)
function sif(
        crack::RibbonCrack{T}, C₀, Σ;
        y₀ = nothing, method::Symbol = :auto, kw...
    ) where {T}
    b = crack.b
    l̂, m̂, n̂ = (TensND.tens_basis(crack_basis(crack), i) for i in 1:3)
    𝐊 = sqrt(T(π) * b) * (Σ ⋅ n̂)
    Kᴵ = 𝐊 ⋅ n̂
    Kᴵᴵ = 𝐊 ⋅ m̂
    Kᴵᴵᴵ = 𝐊 ⋅ l̂
    return 𝐊, (Kᴵ, Kᴵᴵ, Kᴵᴵᴵ)
end

# Elliptical crack (paper eq. 719)
function sif(
        crack::EllipticCrack{T}, C₀, Σ;
        y₀ = nothing, method::Symbol = :auto, kw...
    ) where {T}
    a = crack.a
    b = crack.b
    ℬ = crack_basis(crack)
    l̂, m̂, n̂ = (TensND.tens_basis(ℬ, i) for i in 1:3)

    𝐒_inv = inv(a) * (l̂ ⊗ l̂) + inv(b) * (m̂ ⊗ m̂)

    y0 = y₀ === nothing ? m̂ : y₀
    S⁻¹_y0 = 𝐒_inv ⋅ y0
    n_Sy = norm(S⁻¹_y0)
    ν̂ = TensND.change_tens(S⁻¹_y0 / n_Sy, ℬ)
    τ̂ = TensND.Tens(TensND.change_tens(n̂, ℬ) × TensND.change_tens(ν̂, ℬ), ℬ)

    ℬ_ν = TensND.Basis(
        hcat(
            TensND.components_canon(τ̂),
            TensND.components_canon(ν̂),
            TensND.components_canon(n̂)
        )
    )

    B_ℰ = cod_tensor(crack, C₀; method = method, kw...)
    ribbon_ref = RibbonCrack(b, ℬ_ν)
    B_ℛ = cod_tensor(ribbon_ref, C₀; method = method, kw...)

    𝐊 = (3 * T(π)^(T(3) / 2) * b / 8) * sqrt(b * n_Sy) *
        inv(B_ℛ) ⋅ B_ℰ ⋅ Σ ⋅ n̂

    Kᴵ = 𝐊 ⋅ n̂
    Kᴵᴵ = 𝐊 ⋅ ν̂
    Kᴵᴵᴵ = 𝐊 ⋅ τ̂
    return 𝐊, (Kᴵ, Kᴵᴵ, Kᴵᴵᴵ)
end

# =============================================================================
#  Displacement intensity factor
# =============================================================================

"""
    dif(crack, C₀, Σ; method=:auto, kw...) -> Tens{1,3}
"""
function dif(crack::MFH_Core.AbstractCrack, C₀, Σ; method::Symbol = :auto, kw...)
    B = cod_tensor(crack, C₀; method = method, kw...)
    n̂ = TensND.tens_basis(crack_basis(crack), 3)
    return B ⋅ Σ ⋅ n̂
end

# =============================================================================
#  Thermal (2nd-order) — heat-flux / temperature intensity factors.
# =============================================================================

# Heat-flux intensity factor `K_T` — thermal analog of the elasticity SIF.
# The driving vector is the transport twin of the remote stress, σ^∞ ≡ -q^∞ =
# K₀·∇T^∞ (see the sign block in `localization.jl`), and the crack-tip singular
# field scales as ~ K_T/√r:
#   - Ribbon:    K_T = √(π b) (n̂ · σ^∞)
#   - Elliptic:  K_T = (3π^{3/2} b/8) √(b n_S) (b^𝓔/b^𝓡) (n̂ · σ^∞)
# Only the mode I analog exists in the scalar-temperature case (no shear mode).

"""
    sif(crack::RibbonCrack, K₀::AbstractTens{2,3}, σ∞; kw...) -> Real

Thermal SIF (heat-flux intensity factor) of a ribbon crack:
``K_T = \\sqrt{\\pi b}\\;\\hat{\\mathbf n}\\cdot\\boldsymbol\\sigma^{\\infty}``.

The driving vector is the transport twin of the remote stress of the elastic
`sif`, i.e. ``\\boldsymbol\\sigma^\\infty \\equiv -\\mathbf q^\\infty
= \\mathbf K_0\\cdot\\nabla T^\\infty``, per the package convention. Passing
the physical flux ``\\mathbf q^\\infty`` instead returns ``-K_T``; the kernel
is linear in its third argument and takes no view of its own.
"""
function sif(
        crack::RibbonCrack{T},
        K₀::TensND.AbstractTens{2, 3},
        σ∞;
        y₀ = nothing, method::Symbol = :auto, kw...
    ) where {T}
    b = crack.b
    n̂ = TensND.tens_basis(crack_basis(crack), 3)
    return sqrt(T(π) * b) * (n̂ ⋅ σ∞)
end

# Elliptic crack — thermal
function sif(
        crack::EllipticCrack{T},
        K₀::TensND.AbstractTens{2, 3},
        σ∞;
        y₀ = nothing, method::Symbol = :auto, kw...
    ) where {T}
    a = crack.a
    b = crack.b
    ℬ = crack_basis(crack)
    l̂, m̂, n̂ = (TensND.tens_basis(ℬ, i) for i in 1:3)

    𝐒_inv = inv(a) * (l̂ ⊗ l̂) + inv(b) * (m̂ ⊗ m̂)

    y0 = y₀ === nothing ? m̂ : y₀
    S⁻¹_y0 = 𝐒_inv ⋅ y0
    n_Sy = norm(S⁻¹_y0)
    ν̂ = TensND.change_tens(S⁻¹_y0 / n_Sy, ℬ)
    τ̂ = TensND.Tens(TensND.change_tens(n̂, ℬ) × TensND.change_tens(ν̂, ℬ), ℬ)

    ℬ_ν = TensND.Basis(
        hcat(
            TensND.components_canon(τ̂),
            TensND.components_canon(ν̂),
            TensND.components_canon(n̂)
        )
    )

    b_ℰ = cod_tensor(crack, K₀; method = method, kw...)
    ribbon_ref = RibbonCrack(b, ℬ_ν)
    b_ℛ = cod_tensor(ribbon_ref, K₀; method = method, kw...)

    return (3 * T(π)^(T(3) / 2) * b / 8) * sqrt(b * n_Sy) *
        (b_ℰ / b_ℛ) * (n̂ ⋅ σ∞)
end

"""
    dif(crack, K₀::AbstractTens{2,3}, σ∞; method=:auto, kw...) -> Real

Temperature intensity factor (analog of displacement intensity
factor) for a flat crack driven by the remote vector
``\\boldsymbol\\sigma^{\\infty} \\equiv -\\mathbf q^{\\infty}
 = \\mathbf K_0\\cdot\\nabla T^{\\infty}``:

```
[[T]]_avg = b · (\\hat{\\mathbf n}·\\boldsymbol\\sigma^{\\infty}) .
```

Same convention as the thermal [`sif`](@ref): the driving vector is the
transport twin of the remote stress, so the formula is the elastic one symbol
for symbol. Returns a scalar (vs the `Tens{1,3}` returned by the elasticity
`dif`) since the temperature field is scalar.
"""
function dif(
        crack::MFH_Core.AbstractCrack,
        K₀::TensND.AbstractTens{2, 3},
        σ∞;
        method::Symbol = :auto, kw...
    )
    b = cod_tensor(crack, K₀; method = method, kw...)
    n̂ = TensND.tens_basis(crack_basis(crack), 3)
    return b * (n̂ ⋅ σ∞)
end
