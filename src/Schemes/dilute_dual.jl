# =============================================================================
#  dilute_dual.jl — dilute scheme on the compliance side.
#
#       S_eff = S₀ + Σ_i Δᵢ,    C_eff = inv(S_eff)
#
#  with Δᵢ = fᵢ · Hᵢ for solid inclusions and Δᵢ = (4π/3 or π) εᵢ · Hᵢ
#  for flat cracks. The natural form for cracks (Hᵢ stays finite while
#  the stiffness contribution diverges).
# =============================================================================

"""
    _evaluate(rve, ::DiluteDual, ::Val{p}; kw...) -> AbstractTens

Dilute scheme on the compliance side for property `:p`: averages the
size-independent compliance / resistivity contributions and inverts the
result. For mixed RVEs (solid inclusions + cracks) this is the
preferred form because the crack compliance contribution
[`compliance_contribution`](@ref)`(crack, C₀)` is finite while the
stiffness one is the rank-1 limit of a divergent eigenvalue.

Reference: [kachanov2018](@cite).
"""
function _evaluate(rve::RVE, scheme::DiluteDual, ::Val{p}; kw...) where {p}
    m = matrix_name(scheme, rve)
    P₀ = phase_property(rve, m, p)
    S₀ = inv(P₀)
    ΔS = zero(S₀)
    for name in inclusion_phase_names(rve, m)
        ΔS += _phase_compliance_contribution(rve, name, p, P₀; kw...)
    end
    return inv(S₀ + ΔS)
end
