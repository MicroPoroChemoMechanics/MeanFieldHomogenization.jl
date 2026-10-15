# =============================================================================
#  pcw.jl — Ponte-Castañeda & Willis (1995) scheme.
#
#  In the single-distribution-shape case, the PCW formula is algebraically
#  identical to Maxwell; the difference is in the physical interpretation
#  (PCW: ensemble average over a stochastic distribution function with the
#  given outer shape; Maxwell: matching far-field of an effective inclusion).
#  The identity is proved in the theory page on the two schemes.
#
#  The schemes diverge when the distribution shape depends on the pair of
#  phases (Ponte Castañeda & Willis 1995, §3), which is the future extension
#  hook left open by the `AbstractDistributionShape` hierarchy.
# =============================================================================

"""
    _evaluate(rve, ::PonteCastanedaWillis, ::Val{p}; kw...) -> AbstractTens

PCW homogenization for property `:p`. In the single-shape case
(`UniformDistribution`), the result coincides with
[`Maxwell`](@ref) but the docstring and reference differ —
[ponte1995](@citet) frames the formula in
terms of an ensemble average over a distribution function rather than a
single effective inclusion.

The pairwise form, with one distribution shape per pair of phases, would be
supported by adding a method `_pcw(rve, ::PairwiseDistribution, …)` without
touching this file or the public API. The identity with Maxwell is proved in
the theory page [Ponte Castañeda–Willis](@ref th-scheme-pcw), which derives the
estimate for any number of phases and pairwise distribution ellipsoids.
"""
function _evaluate(rve::RVE, scheme::PonteCastanedaWillis, ::Val{p}; kw...) where {p}
    return _pcw(rve, distribution_shape(rve, scheme), matrix_name(scheme, rve), Val(p); kw...)
end

function _pcw(rve::RVE, ds::UniformDistribution, m::Symbol, ::Val{p}; kw...) where {p}
    P₀ = phase_property(rve, m, p)
    return _pcw_kernel(rve, ds.shape, P₀, m, Val(p); kw...)
end

# 4th-order (elasticity)
function _pcw_kernel(
        rve, shape, C₀::TensND.AbstractTens{4, 3}, m::Symbol, ::Val{p};
        kw...
    ) where {p}
    Σ = _accumulate_contributions(rve, C₀, p, m; kw...)
    P_d = hill_tensor(shape, C₀; kw...)
    I4 = _identity_like(C₀)
    return C₀ + Σ ⊡ inv(I4 - P_d ⊡ Σ)
end

# 2nd-order (conductivity)
function _pcw_kernel(
        rve, shape, K₀::TensND.AbstractTens{2, 3}, m::Symbol, ::Val{p};
        kw...
    ) where {p}
    Σ = _accumulate_contributions(rve, K₀, p, m; kw...)
    P_d = hill_tensor(shape, K₀; kw...)
    I2 = _identity_like(K₀)
    return K₀ + Σ ⋅ inv(I2 - P_d ⋅ Σ)
end
