# =============================================================================
#  dilute.jl — dilute scheme on the stiffness side.
#
#       C_eff = C₀ + Σ_i Δᵢ
#
#  with Δᵢ = fᵢ · Nᵢ for solid inclusions (Nᵢ = (Cᵢ−C₀):A_εε^{(i)}) and
#  Δᵢ = (4π/3 or π) εᵢ · Nᵢ_crack for flat cracks (Nᵢ_crack = -C₀:Hᵢ:C₀).
#
#  First-order accurate in concentration. For a crack-only RVE the result
#  agrees with the leading order of the dual compliance form
#  ([`DiluteDual`](@ref)) but the two diverge at higher order.
# =============================================================================

"""
    _evaluate(rve, ::Dilute, ::Val{p}; kw...) -> AbstractTens

Dilute scheme on the stiffness side for property `:p`. Solid inclusions
contribute via the size-independent stiffness contribution tensor
``\\mathbb N``; cracks via their associated
``\\mathbb N_\\text{crack} = -\\mathbb C_0 : \\mathbb H : \\mathbb C_0``
weighted by the geometry-specific Budiansky prefactor (`4π/3` for an
elliptic crack, `π` for a ribbon crack).

References: [eshelby1957](@cite),
[kachanov2018](@cite).
"""
function _evaluate(rve::RVE, scheme::Dilute, ::Val{p}; kw...) where {p}
    m = matrix_name(scheme, rve)
    C₀ = phase_property(rve, m, p)
    ΔC = zero(C₀)
    for name in inclusion_phase_names(rve, m)
        ΔC += _phase_stiffness_contribution(rve, name, p, C₀; kw...)
    end
    return C₀ + ΔC
end
