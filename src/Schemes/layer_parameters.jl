# =============================================================================
#  layer_parameters.jl — parameters of one layer of a layered sphere or cylinder.
#
#  The moduli of a `LayeredSphere` or of a `LayeredCylinder` are held by its
#  geometry, one per layer, and
#  not by the property of the phase: a coefficient of a layer is designated by
#  the phase, the rank of the layer and the name of the coefficient. For a layer
#  transversely isotropic about the radius (a `TensND.ChartTens`), the names are those of
#  the components in the local frame, (e_r, e_θ, e_φ) on the sphere and
#  (e_r, e_θ, e_z) on the cylinder, which turns with the point:
#  the derivative with respect to `:Crrrr` is taken in that frame.
#
#  `LayerDirection` is to a layer what `PropertyDirection` is to a phase: the
#  abscissa of its modulus along a fixed tensor, isotropic or radial. The
#  second moments of the strain in a layer follow (`second_moments.jl`), those
#  of its components in the local frame included.
# =============================================================================

using ..LayeredSpheres: LayeredSphere, layer_modulus, layer_volume_fraction
using ..LayeredCylinders: LayeredCylinder
const RadialField = MFH_Core.RadialField

# ── The coefficients of a field transversely isotropic about the radius ─────
#
# By its Walpole coefficients about the radial direction, or by its components
# in the local frame: (e_r, e_θ, e_φ) on a sphere, (e_r, e_θ, e_z) on a cylinder.
# The two components across the radius, C_θθθθ and C_θθφφ (or C_θθzz), share ℓ₂
# and ℓ₅, so that setting one of them at fixed other components changes both.

const _RADIAL_WALPOLE = (:ℓ₁, :ℓ₂, :ℓ₃, :ℓ₅, :ℓ₆)
const _RADIAL_CONDUCTIVITIES = (:tangential, :radial)
_component_names(::TensND.SphericalChart) = (:Crrrr, :Cθθθθ, :Cθθφφ, :Crrθθ, :Crθrθ)
_component_names(::TensND.CylindricalChart) = (:Crrrr, :Cθθθθ, :Cθθzz, :Crrθθ, :Crθrθ)

# The components in the local frame, from Hill's moduli about the radius, and back.
function _radial_components(c::RadialField{4})
    (; n, l, k, m, p) = MFH_Core.radial_hill_moduli(c)
    return NamedTuple{_component_names(c.chart)}((n, k + m, k - m, l, p))
end
function _from_components(c::RadialField{4}, C)
    Crrrr, Cₜₜₜₜ, Cₜₜₛₛ, Crrₜₜ, Crₜrₜ = values(C)
    return TensND.radial_ti(c.chart; n = Crrrr, l = Crrₜₜ, k = (Cₜₜₜₜ + Cₜₜₛₛ) / 2, m = (Cₜₜₜₜ - Cₜₜₛₛ) / 2, p = Crₜrₜ)
end

_radial_rank(names, sel) = findfirst(==(sel), names)
_not_radial(sel, names) = _unknown_coefficient(sel, "a tensor transversely isotropic about the radius", join(repr.(names), ", "))

function get_coefficient(c::RadialField{4}, sel::Symbol; symmetry::Nothing = nothing)
    i = _radial_rank(_RADIAL_WALPOLE, sel)
    i === nothing || return TensND.get_data(c.pattern)[i]
    sel in _component_names(c.chart) || _not_radial(sel, (_component_names(c.chart)..., _RADIAL_WALPOLE...))
    return _radial_components(c)[sel]
end

function set_coefficient(c::RadialField{4}, sel::Symbol, v; symmetry::Nothing = nothing)
    i = _radial_rank(_RADIAL_WALPOLE, sel)
    i === nothing || return TensND.radial_ti(c.chart, Base.setindex(TensND.get_data(c.pattern), v, i)...)
    sel in _component_names(c.chart) || _not_radial(sel, (_component_names(c.chart)..., _RADIAL_WALPOLE...))
    return _from_components(c, merge(_radial_components(c), NamedTuple{(sel,)}((v,))))
end

function _conductivity_rank(sel)
    i = _radial_rank(_RADIAL_CONDUCTIVITIES, sel)
    return i === nothing ? _not_radial(sel, _RADIAL_CONDUCTIVITIES) : i
end
get_coefficient(c::RadialField{2}, sel::Symbol; symmetry::Nothing = nothing) = TensND.get_data(c.pattern)[_conductivity_rank(sel)]
set_coefficient(c::RadialField{2}, sel::Symbol, v; symmetry::Nothing = nothing) =
    TensND.radial_ti(c.chart, Base.setindex(TensND.get_data(c.pattern), v, _conductivity_rank(sel))...)

# ── The lenses ───────────────────────────────────────────────────────────────

"""
    LayerParameter(phase, layer, selector)

Lens on one coefficient of the modulus of the layer `layer` of the layered
sphere or cylinder of the phase `phase`: `:k`, `:μ` for an isotropic layer (`:scalar` for
its conductivity); for a layer transversely isotropic about the radius
(`TensND.radial_ti(SphericalChart(), …)`), a component in the local frame, `:Crrrr`, `:Cθθθθ`, `:Cθθφφ` (`:Cθθzz` on a
cylinder), `:Crrθθ`, `:Crθrθ`, a Walpole coefficient `:ℓ₁`, `:ℓ₂`, `:ℓ₃`, `:ℓ₅`, `:ℓ₆`, or a conductivity
`:radial`, `:tangential`. The other coefficients of the same family stay fixed.
Prefer the helper [`layer_param`](@ref).
"""
struct LayerParameter <: AbstractParameter
    phase::Symbol
    layer::Int
    selector::Symbol
end

"""
    LayerDirection(phase, layer, direction)

Lens on the abscissa of the modulus of one layer of a layered sphere or
cylinder along a
fixed tensor, isotropic or transversely isotropic about the radius
(`TensND.radial_ti(SphericalChart(), …)`): zero in the cell, as
[`PropertyDirection`](@ref) is for the property of a phase. An isotropic layer
moved along a radial field becomes one. Prefer the helper
[`layer_param`](@ref).
"""
struct LayerDirection{B} <: AbstractParameter
    phase::Symbol
    layer::Int
    direction::B
end

"""
    layer_param(phase::Symbol, layer::Int, selector::Symbol) -> LayerParameter
    layer_param(phase::Symbol, layer::Int, direction) -> LayerDirection

Helper. A coefficient of the modulus of one layer of the layered sphere or
cylinder of `phase`, or its abscissa along a tensor: `layer_param(:I, 2, :Crrrr)` is the
radial stiffness of the second layer, in the frame that turns with the point.
"""
layer_param(phase::Symbol, layer::Int, sel::Symbol) = LayerParameter(phase, layer, sel)
layer_param(phase::Symbol, layer::Int, direction::Union{TensND.TensISO, RadialField}) = LayerDirection(phase, layer, direction)

const _LayerLens = Union{LayerParameter, LayerDirection}

# A graded layer has no coefficient: its profile is a function, differentiated
# with respect to what it depends on by `sensitivity`.
_no_graded(p) = throw(
    ArgumentError(
        "layer $(p.layer) of :$(p.phase) has a modulus that is a function of the radius: " *
            "it has no coefficient to designate. Differentiate with respect to a parameter of its " *
            "profile by `sensitivity`, on a function that builds the cell"
    )
)
const _VaryingLayer = Union{MFH_Core.GradedLayer, MFH_Core.PowerLawLayer}
_layer_scalar(::_VaryingLayer, p::LayerParameter) = _no_graded(p)
_layer_scalar(::_VaryingLayer, p::LayerDirection) = _no_graded(p)
_with_layer_scalar(::_VaryingLayer, p::LayerParameter, v) = _no_graded(p)
_with_layer_scalar(::_VaryingLayer, p::LayerDirection, v) = _no_graded(p)
_layer_scalar(t, p::LayerParameter) = get_coefficient(t, p.selector)
_with_layer_scalar(t, p::LayerParameter, v) = set_coefficient(t, p.selector, v)
_layer_scalar(t, p::LayerDirection) = zero(promote_type(eltype(t), eltype(p.direction)))
_with_layer_scalar(t, p::LayerDirection, v) = _moved(t, v * p.direction)
# The modulus of a layer moved by `d`: an isotropic layer moved along a field
# that turns with the point becomes such a field, of the same chart.
_moved(t::TensND.TensISO, d::TensND.TensISO) = t + d
_moved(t::TensND.TensISO, d::TensND.ChartTens) = TensND.radial_ti(d.chart, t) + d
_moved(t::TensND.ChartTens, d::TensND.TensISO) = t + TensND.radial_ti(t.chart, d)
_moved(t::TensND.ChartTens, d::TensND.ChartTens) = t + d
_moved(t, d) = throw(
    ArgumentError(
        "a $(nameof(typeof(t))) layer cannot be moved along a $(nameof(typeof(d))): the two are " *
            "isotropic tensors or fields transversely isotropic about the radius of the same chart"
    )
)

# The layered inclusion of a phase, and the same with its moduli replaced.
const _Layered = Union{LayeredSphere, LayeredCylinder}
_with_moduli(g::LayeredSphere, moduli) = LayeredSphere(g.radii, moduli; interfaces = g.interfaces)
_with_moduli(g::LayeredCylinder, moduli) = LayeredCylinder(g.radii, moduli; interfaces = g.interfaces, axis = g.axis)

function _layered_inclusion(rve::RVE, phase::Symbol, layer::Int)
    haskey(rve.phases, phase) || throw(ArgumentError("no phase named :$phase in RVE"))
    g = rve.phases[phase].geometry
    g isa _Layered || throw(
        ArgumentError(
            "phase :$phase is a $(nameof(typeof(g))): a layer parameter designates a layer of a " *
                "LayeredSphere or of a LayeredCylinder"
        )
    )
    1 ≤ layer ≤ length(g.moduli) || throw(ArgumentError("the inclusion of :$phase has $(length(g.moduli)) layers; got layer $layer"))
    return g
end
_layered_inclusion(rve::RVE, p::_LayerLens) = _layered_inclusion(rve, p.phase, p.layer)

get_param(rve::RVE, p::_LayerLens) = _layer_scalar(layer_modulus(_layered_inclusion(rve, p), p.layer), p)

function set_param(rve::RVE, p::_LayerLens, value)
    g = _layered_inclusion(rve, p)
    moduli = Base.setindex(g.moduli, _with_layer_scalar(g.moduli[p.layer], p, value), p.layer)
    phase = rve.phases[p.phase]
    phases = Dict{Symbol, Phase}(rve.phases)
    phases[p.phase] = Phase(_with_moduli(g, moduli), phase.properties)
    return _rebuild_rve(rve; phases = phases)
end

# ── Second moments in a layer ────────────────────────────────────────────────

_direction_lens(layer::Tuple{Symbol, Int}, property, 𝔹) = LayerDirection(layer..., 𝔹)
_moment_fraction(rve::RVE, layer::Tuple{Symbol, Int}) =
    volume_fraction(rve, layer[1]) * layer_volume_fraction(_layered_inclusion(rve, layer...), layer[2])
