# =============================================================================
#  reuss.jl — Reuss (uniform-stress) lower bound.
#
#       S_Reuss = Σ_i f_i  S_i,        C_Reuss = S_Reuss⁻¹.
#
#  Same crack-handling convention as Voigt: `CrackDensity` phases are
#  ignored (zero volume contribution).
# =============================================================================

"""
    _evaluate(rve, ::Reuss, ::Val{p}; kw...) -> AbstractTens

Reuss lower bound on the effective property `:p`. For a stiffness-like
property the algorithm averages the compliances and inverts:
``\\mathbb C_\\mathrm{Reuss} = (\\sum_i f_i \\mathbb S_i)^{-1}``.

The same logic applies to a 2nd-order conductivity tensor: the
"compliance" is then the resistivity ``\\mathbf R = \\mathbf K^{-1}``,
and Reuss returns ``\\mathbf K_\\mathrm{Reuss} = \\mathbf R_\\mathrm{Reuss}^{-1}``.

Phases carrying a [`CrackDensity`](@ref) are ignored, see [`Voigt`](@ref).

Reference: [hill1965](@cite).
"""
function _evaluate(rve::RVE, ::Reuss, ::Val{p}; kw...) where {p}
    f_m = matrix_volume_fraction(rve)
    Seff = f_m * inv(matrix_property(rve, p))
    for name in inclusion_phase_names(rve)
        a = rve.amounts[name]
        if a isa VolumeFraction
            Seff += scale_by_amount(a, _phase_reuss_property(rve, name, p, Seff))
        end
    end
    return inv(Seff)
end
