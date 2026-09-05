# =============================================================================
#  maxwell.jl — Maxwell homogenization.
#
#       C_eff = C₀ + Σ ⊡ (I − P_d ⊡ Σ)⁻¹,
#       Σ     = Σ_i fᵢ Nᵢ  (+ density-weighted crack contributions)
#       P_d   = hill_tensor(rve.distribution_shape, C₀)
#
#  Reduces to Dilute when P_d → 0 (no interaction) and to Mori-Tanaka
#  when the distribution shape coincides with the inclusion shape.
# =============================================================================

"""
    _evaluate(rve, ::Maxwell, ::Val{p}; kw...) -> AbstractTens

Maxwell homogenization for property `:p`. Uses the RVE's
[`distribution_shape`](@ref) (`UniformDistribution` wrapper) as the
reference for the outer Hill polarization tensor `P_d`. Conductivity
(`:K`) is supported through the same recipe with 2nd-order tensors.

The dispatch on `rve.distribution_shape` is on its concrete subtype so
that a future `PairwiseDistribution` (Willis 1982) can plug in by
adding a new method without touching the public API.
"""
function _evaluate(rve::RVE, scheme::Maxwell, ::Val{p}; kw...) where {p}
    return _maxwell(rve, distribution_shape(rve, scheme), matrix_name(scheme, rve), Val(p); kw...)
end

function _maxwell(rve::RVE, ds::UniformDistribution, m::Symbol, ::Val{p}; kw...) where {p}
    P₀ = phase_property(rve, m, p)
    return _maxwell_kernel(rve, ds.shape, P₀, m, Val(p); kw...)
end

# 4th-order (elasticity)
function _maxwell_kernel(
        rve, shape, C₀::TensND.AbstractTens{4, 3}, m::Symbol, ::Val{p};
        kw...
    ) where {p}
    Σ = _accumulate_contributions(rve, C₀, p, m; kw...)
    P_d = hill_tensor(shape, C₀; kw...)
    I4 = _identity_like(C₀)
    return C₀ + Σ ⊡ inv(I4 - P_d ⊡ Σ)
end

# 2nd-order (conductivity)
function _maxwell_kernel(
        rve, shape, K₀::TensND.AbstractTens{2, 3}, m::Symbol, ::Val{p};
        kw...
    ) where {p}
    Σ = _accumulate_contributions(rve, K₀, p, m; kw...)
    P_d = hill_tensor(shape, K₀; kw...)
    I2 = _identity_like(K₀)
    return K₀ + Σ ⋅ inv(I2 - P_d ⋅ Σ)
end

# Helpers — shared with PCW and SC.

"""
    _accumulate_contributions(rve, P₀, prop::Symbol, matrix::Symbol; kw...) -> AbstractTens

Sum of [`_phase_stiffness_contribution`](@ref) over every phase other than
`matrix`, at reference `P₀`. Used by Maxwell, PCW, SC, ASC.
"""
function _accumulate_contributions(rve::RVE, P₀, prop::Symbol, matrix::Symbol; kw...)
    Σ = zero(P₀)
    for name in inclusion_phase_names(rve, matrix)
        Σ += _phase_stiffness_contribution(rve, name, prop, P₀; kw...)
    end
    return Σ
end
