# =============================================================================
#  averages.jl — averages of the gradient (strain, temperature gradient) and
#  of the flux (stress, k·∇T) over a layer and over the material of the
#  cylinder, under a given remote loading.
# =============================================================================

const _Order = Union{_Order4, _Order2}

"""
    layer_gradient_average(cyl::LayeredCylinder, P₀, X∞, layer)

Average over the material of layer `layer` of the strain under the remote
strain `X∞` (reference stiffness `P₀`), or of the temperature gradient under
the remote gradient `X∞` (reference conductivity `P₀`). The opening of an
imperfect interface is not included;
[`gradient_gradient_concentration`](@ref)`(cyl, P₀; layer, external, internal)`
adds it.
"""
layer_gradient_average(cyl::LayeredCylinder, P₀::_Order, X∞, layer::Int) =
    _apply(gradient_gradient_concentration(cyl, P₀; layer), X∞)

"""
    layer_flux_average(cyl::LayeredCylinder, P₀, X∞, layer)

Average over the material of layer `layer` of the stress
``\\mathbb C_k:\\boldsymbol\\varepsilon`` under the remote strain `X∞`, or of
``\\boldsymbol k_k\\cdot\\nabla T`` under the remote gradient `X∞`.
"""
layer_flux_average(cyl::LayeredCylinder, P₀::_Order, X∞, layer::Int) =
    _apply(flux_gradient_concentration(cyl, P₀; layer), X∞)

"""
    cylinder_gradient_average(cyl, P₀, X∞)
    cylinder_flux_average(cyl, P₀, X∞)

Averages over the material of the whole cylinder,
``\\sum_k f_k\\,\\langle\\cdot\\rangle_k``: they leave out the opening of the spring
(or Kapitza) interfaces and the surface stress (or flux) of the membranes (or
conducting surfaces), which the tensors of the schemes,
[`gradient_gradient_concentration`](@ref)`(cyl, P₁, P₀)` and
[`flux_gradient_concentration`](@ref)`(cyl, P₁, P₀)`, include. Names in
elasticity: [`cylinder_strain_average`](@ref) and
[`cylinder_stress_average`](@ref).
"""
function cylinder_gradient_average(cyl::LayeredCylinder{T, N}, P₀::_Order, X∞) where {T, N}
    f = _layer_fractions(cyl)
    return sum(f[k] * layer_gradient_average(cyl, P₀, X∞, k) for k in 1:N)
end

"""
    cylinder_flux_average(cyl, P₀, X∞)

Average over the material of the whole cylinder of the stress under the remote
strain `X∞`, or of ``\\boldsymbol k\\cdot\\nabla T`` under the remote gradient `X∞`:
``\\sum_k f_k\\,\\langle\\cdot\\rangle_k``, without the surface stress (or flux) of the
membranes (or conducting surfaces), which
[`flux_gradient_concentration`](@ref)`(cyl, P₁, P₀)` includes. Name in
elasticity: [`cylinder_stress_average`](@ref).
"""
function cylinder_flux_average(cyl::LayeredCylinder{T, N}, P₀::_Order, X∞) where {T, N}
    f = _layer_fractions(cyl)
    return sum(f[k] * layer_flux_average(cyl, P₀, X∞, k) for k in 1:N)
end

"""
    cylinder_strain_average(args...; kw...)

Name in elasticity of [`cylinder_gradient_average`](@ref), which it takes the
arguments of, for a fourth-order reference stiffness only.
"""
cylinder_strain_average(args...; kw...) = MFH_Core._order_checked(
    Val(2), :cylinder_strain_average, :cylinder_gradient_average,
    cylinder_gradient_average(args...; kw...)
)

"""
    cylinder_stress_average(args...; kw...)

Name in elasticity of [`cylinder_flux_average`](@ref), which it takes the
arguments of, for a fourth-order reference stiffness only.
"""
cylinder_stress_average(args...; kw...) = MFH_Core._order_checked(
    Val(2), :cylinder_stress_average, :cylinder_flux_average,
    cylinder_flux_average(args...; kw...)
)
