# =============================================================================
#  order_aliases.jl — the names of elasticity and of conduction.
#
#  The generic functions of `abstractions.jl` serve both tensor orders. Their
#  names in elasticity and in conduction forward to them and accept a call of
#  their own order only, read off the order of the result: under the name of
#  the other physics, a call would return a correct tensor under a misleading
#  name.
# =============================================================================

"""
    _order_checked(Val(order), alias, generic, result)

`result` when it is a tensor of order `order`; otherwise an `ArgumentError`
naming the generic function, which serves every order.
"""
_order_checked(::Val{N}, ::Symbol, ::Symbol, x::TensND.AbstractTens{N}) where {N} = x
function _order_checked(::Val{N}, alias::Symbol, generic::Symbol, x) where {N}
    throw(
        ArgumentError(
            "`$(alias)` is the name of `$(generic)` for tensors of order $(N), and " *
                "this call is not of that order; call `$(generic)`, which serves every order"
        )
    )
end

"""
    strain_strain_concentration(args...; kw...)

Name in elasticity of [`gradient_gradient_concentration`](@ref), which it takes the arguments of:
the dilute strain concentration tensor ``\\mathbb{A}_{\\varepsilon\\varepsilon}``, for fourth-order stiffnesses only.
"""
strain_strain_concentration(args...; kw...) =
    _order_checked(Val(4), :strain_strain_concentration, :gradient_gradient_concentration, gradient_gradient_concentration(args...; kw...))

"""
    stress_strain_concentration(args...; kw...)

Name in elasticity of [`flux_gradient_concentration`](@ref), which it takes the arguments of:
the mean stress per unit remote strain, ``\\mathbb{A}_{\\sigma\\varepsilon}``, for fourth-order stiffnesses only.
"""
stress_strain_concentration(args...; kw...) =
    _order_checked(Val(4), :stress_strain_concentration, :flux_gradient_concentration, flux_gradient_concentration(args...; kw...))

"""
    strain_stress_concentration(args...; kw...)

Name in elasticity of [`gradient_flux_concentration`](@ref), which it takes the arguments of:
the mean strain per unit remote stress, ``\\mathbb{A}_{\\varepsilon\\sigma}``, for fourth-order stiffnesses only.
"""
strain_stress_concentration(args...; kw...) =
    _order_checked(Val(4), :strain_stress_concentration, :gradient_flux_concentration, gradient_flux_concentration(args...; kw...))

"""
    stress_stress_concentration(args...; kw...)

Name in elasticity of [`flux_flux_concentration`](@ref), which it takes the arguments of:
the stress concentration tensor ``\\mathbb{A}_{\\sigma\\sigma}``, for fourth-order stiffnesses only.
"""
stress_stress_concentration(args...; kw...) =
    _order_checked(Val(4), :stress_stress_concentration, :flux_flux_concentration, flux_flux_concentration(args...; kw...))

"""
    stiffness_contribution(args...; kw...)

Name in elasticity of [`property_contribution`](@ref), which it takes the arguments of:
the stiffness contribution ``\\mathbb{N}``, for fourth-order stiffnesses only.
"""
stiffness_contribution(args...; kw...) =
    _order_checked(Val(4), :stiffness_contribution, :property_contribution, property_contribution(args...; kw...))

"""
    compliance_contribution(args...; kw...)

Name in elasticity of [`inverse_property_contribution`](@ref), which it takes the arguments of:
the compliance contribution ``\\mathbb{H}``, for fourth-order stiffnesses only.
"""
compliance_contribution(args...; kw...) =
    _order_checked(Val(4), :compliance_contribution, :inverse_property_contribution, inverse_property_contribution(args...; kw...))

"""
    conductivity_contribution(args...; kw...)

Name in conduction of [`property_contribution`](@ref), which it takes the arguments of:
the conductivity contribution ``\\boldsymbol{N}``, for second-order conductivities only.
"""
conductivity_contribution(args...; kw...) =
    _order_checked(Val(2), :conductivity_contribution, :property_contribution, property_contribution(args...; kw...))

"""
    resistivity_contribution(args...; kw...)

Name in conduction of [`inverse_property_contribution`](@ref), which it takes the arguments of:
the resistivity contribution ``\\boldsymbol{R}``, for second-order conductivities only.
"""
resistivity_contribution(args...; kw...) =
    _order_checked(Val(2), :resistivity_contribution, :inverse_property_contribution, inverse_property_contribution(args...; kw...))

"""
    delta_stiffness(args...; kw...)

Name in elasticity of [`delta_property`](@ref), which it takes the arguments of:
the dilute correction of the effective stiffness, for fourth-order stiffnesses only.
"""
delta_stiffness(args...; kw...) =
    _order_checked(Val(4), :delta_stiffness, :delta_property, delta_property(args...; kw...))

"""
    delta_compliance(args...; kw...)

Name in elasticity of [`delta_inverse_property`](@ref), which it takes the arguments of:
the dilute correction of the effective compliance, for fourth-order stiffnesses only.
"""
delta_compliance(args...; kw...) =
    _order_checked(Val(4), :delta_compliance, :delta_inverse_property, delta_inverse_property(args...; kw...))

"""
    delta_conductivity(args...; kw...)

Name in conduction of [`delta_property`](@ref), which it takes the arguments of:
the dilute correction of the effective conductivity, for second-order conductivities only.
"""
delta_conductivity(args...; kw...) =
    _order_checked(Val(2), :delta_conductivity, :delta_property, delta_property(args...; kw...))

"""
    delta_resistivity(args...; kw...)

Name in conduction of [`delta_inverse_property`](@ref), which it takes the arguments of:
the dilute correction of the effective resistivity, for second-order conductivities only.
"""
delta_resistivity(args...; kw...) =
    _order_checked(Val(2), :delta_resistivity, :delta_inverse_property, delta_inverse_property(args...; kw...))
