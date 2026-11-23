# =============================================================================
#  plane_states_alv.jl — the two stiffnesses of the plane of an aging material.
#
#  A cell of the plane takes the relaxation operator of the plane. For an
#  isotropic aging material of three-dimensional operator 𝕣 = α∘𝕁 + β∘𝕂
#  (α = 3k, β = 2μ, operators of time on one grid), it is, as for an elastic
#  material, either its in-plane block (plane strain) or the in-plane relation
#  at zero out-of-plane stress (plane stress). The second condenses ε₃₃ out
#  with Volterra products and a Volterra inverse, whose order matters: the
#  operators of an aging material do not commute.
#
#  With λ = (α − β)/3 and 2μ = β, σ₃₃ = λ∘(ε₁₁ + ε₂₂) + (λ + β)∘ε₃₃ = 0 gives
#  ε₃₃ = −(λ + β)^{-∘}∘λ∘(ε₁₁ + ε₂₂), and then
#      σ_αβ = λ*∘(ε₁₁ + ε₂₂) δ_αβ + β∘ε_αβ,   λ* = λ − λ∘(λ + β)^{-∘}∘λ,
#  that is the operator of the plane 2(λ* + μ)∘𝕁 + 2μ∘𝕂 of TensISO{4,2}:
#  α₂ = 2λ* + β, β₂ = β. In plane strain ε₃₃ = 0 and α₂ = 2λ + β = (2α + β)/3.
#  For constant moduli λ* = 2λμ/(λ + 2μ), the elastic plane-stress λ.
# =============================================================================

# The operator of the plane of blocks (α₂, β₂), on the grid of R; the class is a
# unit tensor of the element type of the blocks, as `_alg_one` builds it.
function _plane_iso_operator(R::VolterraOperator, α₂::Matrix{T}, β₂::Matrix{T}) where {T}
    return VolterraOperator(TensISO{2}(one(T), one(T)), (α₂, β₂), R.grid)
end

"""
    plane_strain_stiffness(R::VolterraOperator) → VolterraOperator

The relaxation operator of the plane, in plane strain, of an isotropic aging
material of three-dimensional relaxation operator `R`: its in-plane block.
`R` is built from a law on a time grid, `VolterraOperator(law, times)`.
"""
function TensND.plane_strain_stiffness(R::VolterraOperator{<:TensISO{4, 3}})
    α, β = R.blocks
    return _plane_iso_operator(R, (2 .* α .+ β) ./ 3, β)
end

"""
    plane_stress_stiffness(R::VolterraOperator) → VolterraOperator

The relaxation operator of the plane, in plane stress, of an isotropic aging
material of three-dimensional relaxation operator `R`: the in-plane relation at
zero out-of-plane stress, the out-of-plane strain condensed out with Volterra
products and inverse,

```math
\\lambda^{*}=\\lambda-\\lambda\\circ(\\lambda+2\\mu)^{-\\circ}\\circ\\lambda ,
```

which is not the pointwise conversion of the moduli of an aging material. For
constant moduli it is the elastic plane-stress stiffness.
"""
function TensND.plane_stress_stiffness(R::VolterraOperator{<:TensISO{4, 3}})
    α, β = R.blocks
    λ = (α .- β) ./ 3
    λ★ = λ .- λ * (LowerTriangular(λ .+ β) \ λ)
    return _plane_iso_operator(R, 2 .* λ★ .+ β, β)
end
