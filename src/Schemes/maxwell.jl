# =============================================================================
#  maxwell.jl — Maxwell homogenization.
#
#       C_eff = C₀ + Σ ⊡ (I − P_d ⊡ Σ)⁻¹,
#       Σ     = Σ_i fᵢ Nᵢ  (+ density-weighted crack contributions)
#       P_d   = hill_tensor(rve.distribution_shape, C₀)
#
#  Agrees with Dilute to first order in the fractions, and reduces to
#  Mori-Tanaka when every inclusion phase has the Hill tensor of the
#  distribution shape (MT is C₀ + Σ ⊡ (I − Σᵢ fᵢ Pᵢ ⊡ Nᵢ)⁻¹).
# =============================================================================

"""
    _evaluate(rve, ::Maxwell, ::Val{p}; kw...) -> AbstractTens

Maxwell homogenization for property `:p`. Uses the RVE's
[`distribution_shape`](@ref) (`UniformDistribution` wrapper) as the
reference for the outer Hill polarization tensor ``\\mathbb{P}_{\\mathrm{d}}``. Conductivity
(`:K`) is supported through the same procedure with 2nd-order tensors.

The dispatch on `rve.distribution_shape` is on its concrete subtype so
that a future `PairwiseDistribution` [willis1982](@cite) can plug in by
adding a new method without touching the public API. The derivation, from
the far field of the inhomogeneities [sevostianovIJES2013](@cite), is in the
theory page [Maxwell](@ref th-scheme-maxwell).
"""
function _evaluate(rve::RVE, scheme::Maxwell, ::Val{p}; kw...) where {p}
    return _maxwell(rve, distribution_shape(rve, scheme), matrix_name(scheme, rve), Val(p); kw...)
end

function _maxwell(rve::RVE, ds::UniformDistribution, m::Symbol, ::Val{p}; kw...) where {p}
    P₀ = phase_property(rve, m, p)
    return _maxwell_kernel(rve, ds.shape, P₀, m, Val(p); kw...)
end

function _maxwell_kernel(rve, shape, P₀::TensND.AbstractTens, m::Symbol, ::Val{p}; kw...) where {p}
    Σ = _accumulate_contributions(rve, P₀, p, m; kw...)
    return _maxwell_closure(P₀, Σ, hill_tensor(shape, P₀; kw...))
end

"""
    _maxwell_closure(P₀, Σ, P_d) -> AbstractTens

Maxwell closure ``\\mathbb{P}_0 + \\boldsymbol{\\Sigma} : (\\mathbb{I} - \\mathbb{P}_{\\mathrm{d}} : \\boldsymbol{\\Sigma})^{-1}``
from the sum `Σ` of the contribution tensors per unit volume and the Hill
tensor `P_d` of the domain that holds the inclusions, at either tensor order.
The sum may come from an `RVE` (one inclusion at a time) or from a resolved
assembly, whose contributions account for the interactions.
"""
_maxwell_closure(P₀, Σ, P_d) = P₀ + _compose(Σ, _maxwell_remote_field(P₀, Σ, P_d))

"""
    _maxwell_remote_field(P₀, Σ, P_d) -> AbstractTens

Remote field of the inclusions per unit macroscopic one in Maxwell's scheme,
``(\\mathbb{I} - \\mathbb{P}_{\\mathrm{d}} : \\boldsymbol{\\Sigma})^{-1}``.
"""
_maxwell_remote_field(P₀, Σ, P_d) = inv(one(P₀) - _compose(P_d, Σ))

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
