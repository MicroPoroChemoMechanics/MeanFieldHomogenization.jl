# =============================================================================
#  layer_parameters.jl — parameters of one layer of a layered sphere.
#
#  The moduli of a `LayeredSphere` are held by its geometry, one per layer, and
#  not by the property of the phase: a coefficient of a layer is designated by
#  the phase, the rank of the layer and the name of the coefficient. For a layer
#  transversely isotropic about the radius (a `TensND.ChartTens`), the names are those of
#  the components in the local frame (e_r, e_θ, e_φ), which turns with the point:
#  the derivative with respect to `:Crrrr` is taken in that frame.
#
#  `LayerDirection` is to a layer what `PropertyDirection` is to a phase: the
#  abscissa of its modulus along a fixed tensor, isotropic or radial. The
#  second moments of the strain in a layer follow (`second_moments.jl`), those
#  of its components in the local frame included.
# =============================================================================

using ..LayeredSpheres: RadialField, LayeredSphere, layer_modulus, layer_volume_fraction

# ── The coefficients of a field transversely isotropic about the radius ─────
#
# By its Walpole coefficients about the radial direction, or by its components
# in the local frame (e_r, e_θ, e_φ): C_θθθθ and C_θθφφ share ℓ₂ and ℓ₅, so that
# setting one of them at fixed other components changes both.

const _RADIAL_WALPOLE = (:ℓ₁, :ℓ₂, :ℓ₃, :ℓ₅, :ℓ₆)
const _RADIAL_COMPONENTS = (:Crrrr, :Cθθθθ, :Cθθφφ, :Crrθθ, :Crθrθ)
const _RADIAL_CONDUCTIVITIES = (:tangential, :radial)

# The components in the local frame, from the Walpole coefficients, and back.
function _radial_components(c::RadialField{4})
    ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆ = TensND.get_data(c.pattern)
    return (Crrrr = ℓ₁, Cθθθθ = (ℓ₂ + ℓ₅) / 2, Cθθφφ = (ℓ₂ - ℓ₅) / 2, Crrθθ = ℓ₃ / sqrt(convert(TensND.sqrt2_eltype(typeof(ℓ₃)), 2)), Crθrθ = ℓ₆ / 2)
end
_from_components(c::RadialField{4}, C) = TensND.radial_ti(
    c.chart; n = C.Crrrr, l = C.Crrθθ, k = (C.Cθθθθ + C.Cθθφφ) / 2, m = (C.Cθθθθ - C.Cθθφφ) / 2, p = C.Crθrθ
)

_radial_rank(names, sel) = findfirst(==(sel), names)
_not_radial(sel, names) = _unknown_coefficient(sel, "a tensor transversely isotropic about the radius", join(repr.(names), ", "))

function get_coefficient(c::RadialField{4}, sel::Symbol; symmetry::Nothing = nothing)
    i = _radial_rank(_RADIAL_WALPOLE, sel)
    i === nothing || return TensND.get_data(c.pattern)[i]
    sel in _RADIAL_COMPONENTS || _not_radial(sel, (_RADIAL_COMPONENTS..., _RADIAL_WALPOLE...))
    return _radial_components(c)[sel]
end

function set_coefficient(c::RadialField{4}, sel::Symbol, v; symmetry::Nothing = nothing)
    i = _radial_rank(_RADIAL_WALPOLE, sel)
    i === nothing || return TensND.radial_ti(c.chart, Base.setindex(TensND.get_data(c.pattern), v, i)...)
    sel in _RADIAL_COMPONENTS || _not_radial(sel, (_RADIAL_COMPONENTS..., _RADIAL_WALPOLE...))
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
sphere of the phase `phase`: `:k`, `:μ` for an isotropic layer (`:scalar` for
its conductivity); for a layer transversely isotropic about the radius
(`TensND.radial_ti(SphericalChart(), …)`), a component in the local frame, `:Crrrr`, `:Cθθθθ`, `:Cθθφφ`, `:Crrθθ`, `:Crθrθ`,
a Walpole coefficient `:ℓ₁`, `:ℓ₂`, `:ℓ₃`, `:ℓ₅`, `:ℓ₆`, or a conductivity
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

Lens on the abscissa of the modulus of one layer of a layered sphere along a
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

Helper. A coefficient of the modulus of one layer of the layered sphere of
`phase`, or its abscissa along a tensor: `layer_param(:I, 2, :Crrrr)` is the
radial stiffness of the second layer, in the frame that turns with the point.
"""
layer_param(phase::Symbol, layer::Int, sel::Symbol) = LayerParameter(phase, layer, sel)
layer_param(phase::Symbol, layer::Int, direction::Union{TensND.TensISO, RadialField}) = LayerDirection(phase, layer, direction)

const _LayerLens = Union{LayerParameter, LayerDirection}

_layer_scalar(t, p::LayerParameter) = get_coefficient(t, p.selector)
_with_layer_scalar(t, p::LayerParameter, v) = set_coefficient(t, p.selector, v)
_layer_scalar(t, p::LayerDirection) = zero(promote_type(eltype(t), eltype(p.direction)))
_with_layer_scalar(t, p::LayerDirection, v) = _moved(t, v * p.direction)
_moved(t::TensND.TensISO, d::TensND.TensISO) = t + d
_moved(t, d) = LayeredSpheres._radial_field(t) + LayeredSpheres._radial_field(d)

# The sphere of a phase, and the same with one modulus replaced.
function _layered_sphere(rve::RVE, phase::Symbol, layer::Int)
    haskey(rve.phases, phase) || throw(ArgumentError("no phase named :$phase in RVE"))
    sphere = rve.phases[phase].geometry
    sphere isa LayeredSphere || throw(
        ArgumentError("phase :$phase is a $(nameof(typeof(sphere))): a layer parameter designates a layer of a LayeredSphere")
    )
    1 ≤ layer ≤ length(sphere.moduli) || throw(ArgumentError("the sphere of :$phase has $(length(sphere.moduli)) layers; got layer $layer"))
    return sphere
end
_layered_sphere(rve::RVE, p::_LayerLens) = _layered_sphere(rve, p.phase, p.layer)

get_param(rve::RVE, p::_LayerLens) = _layer_scalar(layer_modulus(_layered_sphere(rve, p), p.layer), p)

function set_param(rve::RVE, p::_LayerLens, value)
    sphere = _layered_sphere(rve, p)
    moduli = Base.setindex(sphere.moduli, _with_layer_scalar(sphere.moduli[p.layer], p, value), p.layer)
    phase = rve.phases[p.phase]
    phases = Dict{Symbol, Phase}(rve.phases)
    phases[p.phase] = Phase(LayeredSphere(sphere.radii, moduli; interfaces = sphere.interfaces), phase.properties)
    return _rebuild_rve(rve; phases = phases)
end

# ── Second moments in a layer ────────────────────────────────────────────────

_direction_lens(layer::Tuple{Symbol, Int}, property, 𝔹) = LayerDirection(layer..., 𝔹)
_moment_fraction(rve::RVE, layer::Tuple{Symbol, Int}) =
    volume_fraction(rve, layer[1]) * layer_volume_fraction(_layered_sphere(rve, layer...), layer[2])
