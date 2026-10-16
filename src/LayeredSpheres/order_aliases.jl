# =============================================================================
#  order_aliases.jl — the names of elasticity for the fields of a layered
#  sphere. The generic functions serve both orders; these names forward to
#  them and accept an elastic call only, whose result is a fourth-order
#  concentration tensor or a second-order strain or stress (a conduction call
#  returns a second-order tensor or a vector instead).
# =============================================================================

"""
    local_strain_strain_concentration(args...; kw...)

Name in elasticity of [`local_gradient_gradient_concentration`](@ref), which it takes the arguments of:
the pointwise strain concentration tensor, for a fourth-order matrix stiffness only.
"""
local_strain_strain_concentration(args...; kw...) = MFH_Core._order_checked(
    Val(4), :local_strain_strain_concentration, :local_gradient_gradient_concentration, local_gradient_gradient_concentration(args...; kw...)
)

"""
    local_stress_strain_concentration(args...; kw...)

Name in elasticity of [`local_flux_gradient_concentration`](@ref), which it takes the arguments of:
the pointwise stress per unit remote strain, for a fourth-order matrix stiffness only.
"""
local_stress_strain_concentration(args...; kw...) = MFH_Core._order_checked(
    Val(4), :local_stress_strain_concentration, :local_flux_gradient_concentration, local_flux_gradient_concentration(args...; kw...)
)

"""
    local_strain_stress_concentration(args...; kw...)

Name in elasticity of [`local_gradient_flux_concentration`](@ref), which it takes the arguments of:
the pointwise strain per unit remote stress, for a fourth-order matrix stiffness only.
"""
local_strain_stress_concentration(args...; kw...) = MFH_Core._order_checked(
    Val(4), :local_strain_stress_concentration, :local_gradient_flux_concentration, local_gradient_flux_concentration(args...; kw...)
)

"""
    local_stress_stress_concentration(args...; kw...)

Name in elasticity of [`local_flux_flux_concentration`](@ref), which it takes the arguments of:
the pointwise stress per unit remote stress, for a fourth-order matrix stiffness only.
"""
local_stress_stress_concentration(args...; kw...) = MFH_Core._order_checked(
    Val(4), :local_stress_stress_concentration, :local_flux_flux_concentration, local_flux_flux_concentration(args...; kw...)
)

"""
    local_strain(args...; kw...)

Name in elasticity of [`local_gradient`](@ref), which it takes the arguments of:
the local strain, for a fourth-order matrix stiffness only.
"""
local_strain(args...; kw...) = MFH_Core._order_checked(
    Val(2), :local_strain, :local_gradient, local_gradient(args...; kw...)
)

"""
    local_stress(args...; kw...)

Name in elasticity of [`local_flux`](@ref), which it takes the arguments of:
the local stress, for a fourth-order matrix stiffness only.
"""
local_stress(args...; kw...) = MFH_Core._order_checked(
    Val(2), :local_stress, :local_flux, local_flux(args...; kw...)
)

"""
    layer_strain_average(args...; kw...)

Name in elasticity of [`layer_gradient_average`](@ref), which it takes the arguments of:
the mean strain of a layer, for a fourth-order matrix stiffness only.
"""
layer_strain_average(args...; kw...) = MFH_Core._order_checked(
    Val(2), :layer_strain_average, :layer_gradient_average, layer_gradient_average(args...; kw...)
)

"""
    layer_stress_average(args...; kw...)

Name in elasticity of [`layer_flux_average`](@ref), which it takes the arguments of:
the mean stress of a layer, for a fourth-order matrix stiffness only.
"""
layer_stress_average(args...; kw...) = MFH_Core._order_checked(
    Val(2), :layer_stress_average, :layer_flux_average, layer_flux_average(args...; kw...)
)

"""
    sphere_strain_average(args...; kw...)

Name in elasticity of [`sphere_gradient_average`](@ref), which it takes the arguments of:
the mean strain of the composite sphere, for a fourth-order matrix stiffness only.
"""
sphere_strain_average(args...; kw...) = MFH_Core._order_checked(
    Val(2), :sphere_strain_average, :sphere_gradient_average, sphere_gradient_average(args...; kw...)
)

"""
    sphere_stress_average(args...; kw...)

Name in elasticity of [`sphere_flux_average`](@ref), which it takes the arguments of:
the mean stress of the composite sphere, for a fourth-order matrix stiffness only.
"""
sphere_stress_average(args...; kw...) = MFH_Core._order_checked(
    Val(2), :sphere_stress_average, :sphere_flux_average, sphere_flux_average(args...; kw...)
)
