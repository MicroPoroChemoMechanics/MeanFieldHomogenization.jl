# =============================================================================
#  sif.jl — stress / displacement intensity factors.
# =============================================================================

"""
    sif(crack, C₀, Σ; y₀=nothing, method=:auto, kw...) -> (𝐊, (Kᴵ, Kᴵᴵ, Kᴵᴵᴵ))

Stress intensity factor vector ``\\underline{K}`` at a point of the
crack front, together with its ``(K_{I},K_{II},K_{III})`` decomposition
on ``(\\underline{n},\\underline{\\nu},\\underline{\\tau})``
(after [irwin1957, kassir1968, willis1968](@citet);
 energy release rate identity ``G = \\underline{K}\\cdot\\underline{N}``
 in [barnett1972, rice1989](@citet)).

For a ribbon crack (``\\underline{\\nu}=\\pm\\underline{m}``)
``\\underline{K}^{\\mathcal{R}} = \\sqrt{\\pi b}\\,\\boldsymbol{\\Sigma}\\cdot\\underline{n}``
(independent of the matrix stiffness).
For an elliptic crack, ``\\underline{K}`` is obtained from the COD
tensor ``\\boldsymbol{B}^{\\mathcal{E}}`` of the actual crack and the COD
tensor ``\\boldsymbol{B}^{\\mathcal{R}}`` of the tangent ribbon crack at the
observation point:

```math
\\underline{K} = \\tfrac{3}{8}\\pi^{3/2}\\sqrt{b}\\;
  \\sqrt{b\\,\\|\\boldsymbol{S}^{\\dagger}\\!\\cdot\\underline{y}^{\\star}_{0}\\|}\\;
  \\bigl(\\boldsymbol{B}^{\\mathcal{R}}(\\underline{\\nu},\\underline{n})\\bigr)^{-1}
  \\cdot\\boldsymbol{B}^{\\mathcal{E}}(\\underline{m},\\underline{n},\\eta)
  \\cdot\\boldsymbol{\\Sigma}\\cdot\\underline{n} .
```

The central identity
``\\underline{K} = \\pi\\,(\\boldsymbol{B}^{\\mathcal{R}})^{-1}\\cdot\\underline{N}``
is purely local
[kanaun1981, kunin1983, kanaun2008](@cite).
"""
function sif end

# The point of the unit circle that locates the front point: `m̂` when not given.
_front_point(::Nothing, m̂) = m̂
_front_point(y₀, _) = y₀

# Ribbon crack (paper eq. 737)
Base.@constprop :aggressive function sif(
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

# Unit outward normal ν to the front of an elliptic crack at the point of
# direction `y₀` of the unit circle, and the scale √(b ‖S⁻¹·y₀‖) of its local
# intensity factors.
function _ellipse_front(crack::EllipticCrack, y₀)
    ℬ = crack_basis(crack)
    l̂, m̂ = TensND.tens_basis(ℬ, 1), TensND.tens_basis(ℬ, 2)
    S⁻¹_y0 = (inv(crack.a) * (l̂ ⊗ l̂) + inv(crack.b) * (m̂ ⊗ m̂)) ⋅ _front_point(y₀, m̂)
    n_Sy = norm(S⁻¹_y0)
    return TensND.express(S⁻¹_y0 / n_Sy, ℬ), sqrt(crack.b * n_Sy)
end

# The frame (τ, ν, n) of the ribbon tangent to a front of outward normal ν in the
# plane of normal n, as a TensND basis, and its vector τ = n × ν.
function _front_frame(ν̂, n̂)
    ℬ = TensND.get_basis(ν̂)
    τ̂ = TensND.Tens(TensND.express(n̂, ℬ) × ν̂, ℬ)
    return TensND.Basis(
            hcat(
                TensND.canonical_components(τ̂), TensND.canonical_components(ν̂),
                TensND.canonical_components(n̂)
            )
        ), τ̂
end

"""
    sif_from_dif(N, ν, n, C₀; kw...) -> (𝐊, (Kᴵ, Kᴵᴵ, Kᴵᴵᴵ))
    sif_from_dif(N, ν, n, K₀; kw...) -> K_T

Stress intensity factor vector at a point of a crack front from the local
displacement intensity factor ``\\underline{N}`` there:

```math
\\underline{K} = \\pi\\,\\bigl(\\boldsymbol{B}^{\\mathcal{R}}(\\underline{\\nu},
    \\underline{n})\\bigr)^{-1}\\cdot\\underline{N},
\\qquad
[\\![\\underline{u}]\\!] \\simeq 4\\sqrt{\\tfrac{2}{\\pi}}\\;\\underline{N}\\,\\sqrt{s}
\\quad (s \\to 0),
```

``s`` the distance to the front inside the crack, ``\\underline{\\nu}`` the unit
outward normal to the front in the plane of the crack, of normal
``\\underline{n}``, and ``\\boldsymbol{B}^{\\mathcal{R}}`` the COD tensor of the
ribbon crack tangent to the front in the matrix `C₀`. The relation is local
[kanaun1981, kunin1983, kanaunlevin2009](@cite): it holds at any regular point of
the front of a flat crack of any contour, in a matrix of any anisotropy. The
modes are read on ``(\\underline{n}, \\underline{\\nu}, \\underline{\\tau})``,
``\\underline{\\tau} = \\underline{n}\\times\\underline{\\nu}``.

In conduction (`K₀` of order 2), ``N`` is the scalar intensity factor of the
potential jump and ``K_T = \\pi N / b^{\\mathcal{R}}``.

`ν` and `n` are order-1 TensND tensors; `kw...` goes to [`cod_tensor`](@ref) of
the ribbon.
"""
function sif_from_dif(N::TensND.AbstractTens{1, 3}, ν̂, n̂, C₀::TensND.AbstractTens{4, 3}; kw...)
    ℬ_ν, τ̂ = _front_frame(ν̂, n̂)
    B_ℛ = cod_tensor(RibbonCrack(one(eltype(ν̂)), ℬ_ν), C₀; kw...)
    𝐊 = oftype(one(eltype(N)), π) * (inv(B_ℛ) ⋅ N)
    return 𝐊, (𝐊 ⋅ n̂, 𝐊 ⋅ ν̂, 𝐊 ⋅ τ̂)
end

function sif_from_dif(N::Number, ν̂, n̂, K₀::TensND.AbstractTens{2, 3}; kw...)
    ℬ_ν, _ = _front_frame(ν̂, n̂)
    return oftype(N, π) * N / cod_tensor(RibbonCrack(one(eltype(ν̂)), ℬ_ν), K₀; kw...)
end

# Elliptical crack (paper eq. 719): its local displacement intensity factor is
# (3√π/8) √b √(b ‖S⁻¹·y₀‖) 𝐁ᴱ·𝚺·n, then the local relation above.
Base.@constprop :aggressive function sif(
        crack::EllipticCrack{T}, C₀, Σ;
        y₀ = nothing, method::Symbol = :auto, kw...
    ) where {T}
    n̂ = TensND.tens_basis(crack_basis(crack), 3)
    ν̂, scale = _ellipse_front(crack, y₀)
    B_ℰ = cod_tensor(crack, C₀; method = method, kw...)
    N = (3 * sqrt(T(π)) * sqrt(crack.b) / 8) * scale * (B_ℰ ⋅ Σ ⋅ n̂)
    return sif_from_dif(N, ν̂, n̂, C₀; method = method, kw...)
end

# =============================================================================
#  Displacement intensity factor
# =============================================================================

"""
    dif(crack, C₀, Σ; method=:auto, kw...) -> Tens{1,3}
"""
Base.@constprop :aggressive function dif(crack::MFH_Core.AbstractCrack, C₀, Σ; method::Symbol = :auto, kw...)
    B = cod_tensor(crack, C₀; method = method, kw...)
    n̂ = TensND.tens_basis(crack_basis(crack), 3)
    return B ⋅ Σ ⋅ n̂
end

# =============================================================================
#  Thermal (2nd-order) — heat-flux / temperature intensity factors.
# =============================================================================

# Heat-flux intensity factor `K_T` — thermal analog of the elasticity SIF.
# The driving vector is the transport twin of the remote stress, σ^∞ ≡ -q^∞ =
# K₀·∇T^∞ (see the sign block in `concentration.jl`), and the crack-tip singular
# field scales as ~ K_T/√r:
#   - Ribbon:    K_T = √(π b) (n̂ · σ^∞)
#   - Elliptic:  K_T = (3π^{3/2} b/8) √(b n_S) (b^𝓔/b^𝓡) (n̂ · σ^∞)
# Only the mode I analog exists in the scalar-temperature case (no shear mode).

"""
    sif(crack::RibbonCrack, K₀::AbstractTens{2,3}, σ∞; kw...) -> Real

Thermal SIF (heat-flux intensity factor) of a ribbon crack:
``K_T = \\sqrt{\\pi b}\\;\\underline{n}\\cdot\\boldsymbol{\\sigma}^{\\infty}``.

The driving vector is the transport twin of the remote stress of the elastic
`sif`, i.e. ``\\boldsymbol{\\sigma}^{\\infty} \\equiv -\\underline{q}^{\\infty} =
\\boldsymbol{K}_0\\cdot\\underline{\\operatorname{grad}}\\,T^{\\infty}``, per the package convention. Passing
the physical flux ``\\underline{q}^{\\infty}`` instead returns ``-K_T``; the kernel
is linear in its third argument and takes no view of its own.
"""
Base.@constprop :aggressive function sif(
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
Base.@constprop :aggressive function sif(
        crack::EllipticCrack{T},
        K₀::TensND.AbstractTens{2, 3},
        σ∞;
        y₀ = nothing, method::Symbol = :auto, kw...
    ) where {T}
    n̂ = TensND.tens_basis(crack_basis(crack), 3)
    ν̂, scale = _ellipse_front(crack, y₀)
    b_ℰ = cod_tensor(crack, K₀; method = method, kw...)
    N = (3 * sqrt(T(π)) * sqrt(crack.b) / 8) * scale * b_ℰ * (n̂ ⋅ σ∞)
    return sif_from_dif(N, ν̂, n̂, K₀; method = method, kw...)
end

"""
    dif(crack, K₀::AbstractTens{2,3}, σ∞; method=:auto, kw...) -> Real

Temperature intensity factor (analog of displacement intensity
factor) for a flat crack driven by the remote vector
``\\boldsymbol{\\sigma}^{\\infty} \\equiv -\\underline{q}^{\\infty} =
 \\boldsymbol{K}_0\\cdot\\underline{\\operatorname{grad}}\\,T^{\\infty}``:

```math
[\\![T]\\!]_{\\mathrm{avg}} = b\\,(\\underline{n}\\cdot\\boldsymbol{\\sigma}^{\\infty}) .
```

Same convention as the thermal [`sif`](@ref): the driving vector is the
transport twin of the remote stress, so the formula is the elastic one symbol
for symbol. Returns a scalar (vs the `Tens{1,3}` returned by the elasticity
`dif`) since the temperature field is scalar.
"""
Base.@constprop :aggressive function dif(
        crack::MFH_Core.AbstractCrack,
        K₀::TensND.AbstractTens{2, 3},
        σ∞;
        method::Symbol = :auto, kw...
    )
    b = cod_tensor(crack, K₀; method = method, kw...)
    n̂ = TensND.tens_basis(crack_basis(crack), 3)
    return b * (n̂ ⋅ σ∞)
end
