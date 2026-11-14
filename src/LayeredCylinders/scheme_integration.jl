# =============================================================================
#  scheme_integration.jl — public tensors of a `LayeredCylinder`: what the
#  mean-field schemes consume (concentration, average flux, contribution, layer
#  averages for the bounds) and the per-layer concentrations.
#
#  As a layered sphere, a layered cylinder has no Hill tensor: it enters the
#  schemes through its concentration tensors. The phase property `P₁` of the
#  generic signatures is ignored, the moduli being those of the layers.
# =============================================================================

const _Order4 = TensND.AbstractTens{4, 3}
const _Order2 = TensND.AbstractTens{2, 3}

_check_layer(cyl::LayeredCylinder{T, N}, layer::Int) where {T, N} =
    1 ≤ layer ≤ N || throw(BoundsError(cyl, layer))

# ── Elasticity ───────────────────────────────────────────────────────────────

"""
    gradient_gradient_concentration(cyl::LayeredCylinder, C₁, C₀; external = true) -> TensTI{4}

Strain concentration tensor of the whole cylinder in the reference `C₀`
(isotropic, or transversely isotropic about the cylinder axis):
``\\langle\\boldsymbol{\\varepsilon}\\rangle_\\Omega = \\mathbb{A}_\\Omega:\\boldsymbol{\\varepsilon}^{\\infty}``.
It is transversely isotropic about the axis and has no major symmetry. The
average includes the opening of the inner [`SpringInterface`](@ref)s, and that
of the outer one when `external` is `true` (the default, which the schemes
use), as for a [`LayeredSphere`](@ref MeanFieldHomogenization.LayeredSpheres.LayeredSphere). `C₁` is ignored.
"""
function gradient_gradient_concentration(
        cyl::LayeredCylinder, ::_Order4, C₀::_Order4; external::Bool = true, kw...
    )
    return _ti_tensor(_whole_strain(_elastic_solution(cyl, C₀), external), cyl.axis)
end

"""
    flux_gradient_concentration(cyl::LayeredCylinder, C₁, C₀; external = true) -> TensTI{4}

Average stress of the whole cylinder per unit remote strain,
``\\langle\\boldsymbol{\\sigma}\\rangle_\\Omega = \\mathbb{B}_\\Omega:\\boldsymbol{\\varepsilon}^{\\infty}``:
the stress of the layers plus the surface stress of the
[`MembraneInterface`](@ref)s, that of the outer one when `external` is `true`.
`C₁` is ignored.
"""
function flux_gradient_concentration(
        cyl::LayeredCylinder, ::_Order4, C₀::_Order4; external::Bool = true, kw...
    )
    return _ti_tensor(_whole_stress(_elastic_solution(cyl, C₀), external), cyl.axis)
end

"""
    property_contribution(cyl::LayeredCylinder, C₁, C₀; external = true) -> TensTI{4}
    property_contribution(cyl::LayeredCylinder, C₀; external = true) -> TensTI{4}

Stiffness contribution tensor of the cylinder,
``\\mathbb{N} = \\mathbb{B}_\\Omega - \\mathbb{C}_0:\\mathbb{A}_\\Omega``, with the tensors of
[`flux_gradient_concentration`](@ref) and
[`gradient_gradient_concentration`](@ref) for the same `external`. `C₁` is
ignored.
"""
function Core.property_contribution(
        cyl::LayeredCylinder, ::_Order4, C₀::_Order4; external::Bool = true, kw...
    )
    return _ti_tensor(_contribution(_elastic_solution(cyl, C₀), external), cyl.axis)
end

Core.property_contribution(cyl::LayeredCylinder, C₀::_Order4; kw...) =
    Core.property_contribution(cyl, C₀, C₀; kw...)

function Core.concentration_and_contribution(
        cyl::LayeredCylinder, ::_Order4, C₀::_Order4; external::Bool = true, kw...
    )
    sol = _elastic_solution(cyl, C₀)
    return (
        _ti_tensor(_whole_strain(sol, external), cyl.axis),
        _ti_tensor(_contribution(sol, external), cyl.axis),
    )
end

function Core.concentration_and_flux_average(
        cyl::LayeredCylinder, ::_Order4, C₀::_Order4; external::Bool = true, kw...
    )
    sol = _elastic_solution(cyl, C₀)
    return (
        _ti_tensor(_whole_strain(sol, external), cyl.axis),
        _ti_tensor(_whole_stress(sol, external), cyl.axis),
    )
end

"""
    gradient_gradient_concentration(cyl::LayeredCylinder, C₀; layer, external = false, internal = false) -> TensTI{4}

Strain concentration tensor of layer `layer`: the average strain of its
material per unit remote strain, plus the opening of the spring interface on
its outer boundary when `external`, on its inner boundary when `internal`.
With `external = true` on every layer the layers partition the cylinder:
``\\sum_k f_k\\,\\mathbb{A}_k = \\mathbb{A}_\\Omega``.
"""
function gradient_gradient_concentration(
        cyl::LayeredCylinder, C₀::_Order4;
        layer::Int, external::Bool = false, internal::Bool = false, kw...
    )
    _check_layer(cyl, layer)
    return _ti_tensor(_layer_strain(_elastic_solution(cyl, C₀), layer, external, internal), cyl.axis)
end

"""
    flux_gradient_concentration(cyl::LayeredCylinder, C₀; layer) -> TensTI{4}

Average stress of the material of layer `layer` per unit remote strain,
``\\mathbb{C}_k:\\mathbb{A}_k``. With second-order tensors, its average
``\\boldsymbol{k}_k\\cdot\\boldsymbol{A}_k``.
"""
function flux_gradient_concentration(cyl::LayeredCylinder, C₀::_Order4; layer::Int, kw...)
    _check_layer(cyl, layer)
    return _ti_tensor(_layer_stress(_elastic_solution(cyl, C₀), layer), cyl.axis)
end

# ── Conduction ───────────────────────────────────────────────────────────────

"""
    gradient_gradient_concentration(cyl::LayeredCylinder, K₁, K₀; external = true) -> TensTI{2}

Gradient concentration tensor of the whole cylinder in the reference `K₀`
(isotropic, or transversely isotropic about the cylinder axis). The axial
gradient is uniform, so its axial coefficient is 1. The transverse average
includes the temperature jumps of the inner [`KapitzaInterface`](@ref)s, and
that of the outer one when `external` is `true`. `K₁` is ignored.
"""
function gradient_gradient_concentration(
        cyl::LayeredCylinder, ::_Order2, K₀::_Order2; external::Bool = true, kw...
    )
    return _ti_tensor(_whole_gradient(_conduction_solution(cyl, K₀), external), cyl.axis)
end

"""
    flux_gradient_concentration(cyl::LayeredCylinder, K₁, K₀; external = true) -> TensTI{2}

Average of ``\\boldsymbol{k}\\cdot\\nabla T`` over the whole cylinder per unit remote
gradient, plus the flux carried by the
[`SurfaceConductiveInterface`](@ref)s, that of the outer one when `external` is
`true`. `K₁` is ignored.
"""
function flux_gradient_concentration(
        cyl::LayeredCylinder, ::_Order2, K₀::_Order2; external::Bool = true, kw...
    )
    return _ti_tensor(_whole_flux(_conduction_solution(cyl, K₀), external), cyl.axis)
end

function Core.property_contribution(
        cyl::LayeredCylinder, ::_Order2, K₀::_Order2; external::Bool = true, kw...
    )
    return _ti_tensor(_conduction_contribution(_conduction_solution(cyl, K₀), external), cyl.axis)
end

Core.property_contribution(cyl::LayeredCylinder, K₀::_Order2; kw...) =
    Core.property_contribution(cyl, K₀, K₀; kw...)

function Core.concentration_and_contribution(
        cyl::LayeredCylinder, ::_Order2, K₀::_Order2; external::Bool = true, kw...
    )
    sol = _conduction_solution(cyl, K₀)
    return (
        _ti_tensor(_whole_gradient(sol, external), cyl.axis),
        _ti_tensor(_conduction_contribution(sol, external), cyl.axis),
    )
end

function Core.concentration_and_flux_average(
        cyl::LayeredCylinder, ::_Order2, K₀::_Order2; external::Bool = true, kw...
    )
    sol = _conduction_solution(cyl, K₀)
    return (
        _ti_tensor(_whole_gradient(sol, external), cyl.axis),
        _ti_tensor(_whole_flux(sol, external), cyl.axis),
    )
end

function gradient_gradient_concentration(
        cyl::LayeredCylinder, K₀::_Order2;
        layer::Int, external::Bool = false, internal::Bool = false, kw...
    )
    _check_layer(cyl, layer)
    return _ti_tensor(
        _layer_gradient(_conduction_solution(cyl, K₀), layer, external, internal), cyl.axis
    )
end

function flux_gradient_concentration(cyl::LayeredCylinder, K₀::_Order2; layer::Int, kw...)
    _check_layer(cyl, layer)
    return _ti_tensor(_layer_flux(_conduction_solution(cyl, K₀), layer), cyl.axis)
end

# ── Layer averages of the properties (Voigt and Reuss bounds) ────────────────

_layer_tensor(cyl::LayeredCylinder, k::Int, ::Val{4}) =
    _stiffness_tensor(_hill_moduli(layer_modulus(cyl, k), cyl.axis), cyl.axis)
_layer_tensor(cyl::LayeredCylinder, k::Int, ::Val{2}) =
    _ti_tensor(_conductivities(layer_modulus(cyl, k), cyl.axis), cyl.axis)

layer_property_average(cyl::LayeredCylinder, ::TensND.AbstractTens{O, 3}) where {O} =
    layer_property_average(cyl, Val(O))
layer_inverse_property_average(cyl::LayeredCylinder, ::TensND.AbstractTens{O, 3}) where {O} =
    layer_inverse_property_average(cyl, Val(O))

# Every layer is transversely isotropic about the axis of the cylinder: the
# average is that of the Walpole coefficients, about that axis. (A sum of
# `TensTI` would compare their axes at run time.)
function _coaxial_average(cyl::LayeredCylinder{T, N}, tensors::Tuple) where {T, N}
    f = _layer_fractions(cyl)
    data = reduce((x, y) -> x .+ y, ntuple(k -> f[k] .* TensND.get_data(tensors[k]), Val(N)))
    return _rebuild_ti(first(tensors), data, cyl.axis)
end
_rebuild_ti(::TensND.TensTI{4}, ℓ, axis) = TensND.TensTI{4}(ℓ..., axis)
_rebuild_ti(::TensND.TensTI{2}, c, axis) = TensND.TensTI{2}(c..., axis)

layer_property_average(cyl::LayeredCylinder{T, N}, order::Val) where {T, N} =
    _coaxial_average(cyl, ntuple(k -> _layer_tensor(cyl, k, order), Val(N)))

layer_inverse_property_average(cyl::LayeredCylinder{T, N}, order::Val) where {T, N} =
    _coaxial_average(cyl, ntuple(k -> inv(_layer_tensor(cyl, k, order)), Val(N)))
