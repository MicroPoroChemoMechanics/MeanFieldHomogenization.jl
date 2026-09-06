# =============================================================================
#  parameters.jl — parameter lenses on a `Laminate`.
#
#  `PropertyParameter` is reused verbatim (its `phase` field naming a LAYER
#  here); two lenses are specific to a laminate — the layer thickness and the
#  scalar fields of an imperfect interface. `NestedParameter` (declared in
#  `Core/cells.jl`) composes with all of them, so `derivative` / `gradient` /
#  `jacobian` cross a whole multiscale chain in one ForwardDiff pass.
# =============================================================================

"""
    ThicknessParameter(layer::Symbol)
    thickness(layer::Symbol)

Lens on the thickness `h_i` of one layer of a [`Laminate`](@ref).

Differentiating with respect to a thickness is *not* the same as
differentiating with respect to a volume fraction: changing `h_i` changes
both `f_i` (through `L`) and the period, hence the `1/L` weight of every
imperfect interface. With perfect interfaces the two coincide up to the chain
rule; with a spring interface the thickness derivative also carries the size
effect, which is usually what one wants.
"""
struct ThicknessParameter <: MFH_Core.AbstractParameter
    layer::Symbol
end

"""
    thickness(layer::Symbol) -> ThicknessParameter

Convenience constructor for [`ThicknessParameter`](@ref), mirroring `amount`,
`property` and `geometry`.
"""
thickness(layer::Symbol) = ThicknessParameter(layer)

"""
    InterfaceParameter(index::Int, field::Symbol)
    interface_param(index::Int, field::Symbol)

Lens on one scalar parameter of the `index`-th interface of a
[`Laminate`](@ref) — `:kn`, `:kt` (spring **stiffnesses**) or equivalently
`:sn`, `:st` (the matching compliances, which is what
[`SpringInterface`](@ref) stores); `:κs`, `:μs` (surface moduli);
`:resistance` (Kapitza); `:conductance` (surface-conductive).

Differentiating with respect to `:kn` and with respect to `:sn` are both
legitimate and give reciprocal sensitivities; pick the one your model is
parameterized on.

Interface `index` sits **on top of** layer `index` in stacking order.
"""
struct InterfaceParameter <: MFH_Core.AbstractParameter
    index::Int
    field::Symbol
end

"""
    interface_param(index::Int, field::Symbol) -> InterfaceParameter

Convenience constructor for [`InterfaceParameter`](@ref).
"""
interface_param(index::Int, field::Symbol) = InterfaceParameter(index, field)

# ── ThicknessParameter ──────────────────────────────────────────────────────

MFH_Core.get_param(lam::Laminate, p::ThicknessParameter) =
    layer_thickness(lam, p.layer)

function MFH_Core.set_param(lam::Laminate{T}, p::ThicknessParameter, value) where {T}
    haskey(lam.thicknesses, p.layer) ||
        throw(ArgumentError("no layer named :$(p.layer) in Laminate"))
    new_h = Dict{Symbol, Any}(lam.thicknesses)
    new_h[p.layer] = value
    Tnew = promote_type(T, typeof(value))
    return _rebuild_laminate(lam; thicknesses = new_h, T = Tnew)
end

# ── PropertyParameter (shared with the RVE) ─────────────────────────────────

function MFH_Core.get_param(lam::Laminate, p::PropertyParameter)
    t = layer_property(lam, p.phase, p.property)
    return TensND.get_data(t)[Schemes._resolve_selector(t, p.selector)]
end

function MFH_Core.set_param(lam::Laminate, p::PropertyParameter, value)
    old_t = layer_property(lam, p.phase, p.property)
    i = Schemes._resolve_selector(old_t, p.selector)
    new_t = Schemes._replace_data_at(old_t, i, value)
    return MFH_Core.cell_set_property(lam, p.phase, p.property, new_t)
end

# ── InterfaceParameter ──────────────────────────────────────────────────────

function MFH_Core.get_param(lam::Laminate, p::InterfaceParameter)
    itf = layer_interface(lam, p.index)
    # `propertynames`, not `fieldnames`: a `SpringInterface` stores compliances
    # and exposes the stiffnesses `:kn`, `:kt` as conversions, and both must be
    # addressable as parameters.
    props = propertynames(itf)
    p.field in props || throw(
        ArgumentError(
            "interface $(p.index) is a $(nameof(typeof(itf))), which has no parameter " *
                ":$(p.field); available: $(isempty(props) ? "none" : join(props, ", "))"
        )
    )
    return getproperty(itf, p.field)
end

function MFH_Core.set_param(lam::Laminate, p::InterfaceParameter, value)
    old = layer_interface(lam, p.index)
    new_itf = _replace_interface_field(old, p.field, value)
    new_interfaces = Vector{AbstractInterface}(lam.interfaces)
    new_interfaces[p.index] = new_itf
    return _rebuild_laminate(lam; interfaces = new_interfaces)
end

"""
    _replace_interface_field(itf, field::Symbol, value) -> AbstractInterface

Rebuild an interface with one scalar field replaced, promoting the element
type to absorb `typeof(value)` — the route by which a `ForwardDiff.Dual`
reaches an interface compliance.
"""
function _replace_interface_field(itf::AbstractInterface, field::Symbol, value)
    fields = fieldnames(typeof(itf))
    field in fields || throw(
        ArgumentError(
            "interface $(nameof(typeof(itf))) has no field :$(field); available: $(fields)"
        )
    )
    vals = map(f -> f === field ? value : getfield(itf, f), fields)
    T = promote_type(map(typeof, vals)...)
    # Rebuild through the type-parameterized constructor, which always takes the
    # STORED fields in declaration order.  Going through the bare name would
    # feed a `SpringInterface`'s compliances to its stiffness constructor.
    return Base.typename(typeof(itf)).wrapper{T}(map(x -> convert(T, x), vals)...)
end

# A spring stores compliances; setting a stiffness sets the reciprocal, and
# `ForwardDiff` differentiates straight through the `inv`.
function _replace_interface_field(itf::SpringInterface, field::Symbol, value)
    sn, st = spring_compliances(itf)
    if field === :kn
        sn = inv(value)
    elseif field === :kt
        st = inv(value)
    elseif field === :sn
        sn = value
    elseif field === :st
        st = value
    else
        throw(
            ArgumentError(
                "SpringInterface has no parameter :$(field); available: kn, kt, sn, st"
            )
        )
    end
    return _spring_from_compliances(sn, st)
end

# ── AmountParameter: rejected, with a pointer ───────────────────────────────

# A laminate stores THICKNESSES; the fractions are derived (`f_i = h_i / L`).
# Silently reinterpreting an `AmountParameter` as a thickness would be a
# footgun — the two differ by the period, and only the thickness carries the
# interface size effect. Fail loudly instead.
function MFH_Core.get_param(::Laminate, p::AmountParameter)
    throw(
        ArgumentError(
            "a Laminate stores thicknesses, not amounts: the volume fractions are " *
                "derived as f_i = h_i / L. Use ThicknessParameter(:$(p.phase)) " *
                "(sugar: `thickness(:$(p.phase))`) instead of AmountParameter."
        )
    )
end

MFH_Core.set_param(lam::Laminate, p::AmountParameter, value) = MFH_Core.get_param(lam, p)
