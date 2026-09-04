# =============================================================================
#  parameters.jl — Parameter lenses for autodiff sensitivities.
#
#  An `AbstractParameter` *lens* designates a single scalar input of a
#  homogenization computation : a phase volume fraction, a coefficient of a
#  property tensor, a scalar geometry field of an inclusion, or a field of
#  the (Maxwell / PCW) distribution shape. The `get_param` / `set_param`
#  functions read and immutably replace the designated scalar in a `RVE` ;
#  `set_param` returns a *new* `RVE` instance whose affected fields are
#  reconstructed with element-type promotion, leaving the original
#  untouched.
#
#  This indirection underpins the friendly user-facing API exposed in
#  `sensitivities.jl` : `derivative(rve, scheme, p)` is a thin
#  `ForwardDiff.derivative` over the closure
#  `x -> homogenize(set_param(rve, p, x), …)`.
#
#  This file does not depend on ForwardDiff — it remains usable even when
#  the weak extension is not loaded.
# =============================================================================

# =============================================================================
#  Lens hierarchy
# =============================================================================

# `AbstractParameter`, `get_param` and `set_param` are declared in
# `Core/cells.jl` — the lenses are shared by every `AbstractHomogenizationCell`
# (an `RVE` here, a `Laminate` in `Laminates/parameters.jl`), and
# `NestedParameter` composes them across a whole multiscale chain. Only the
# RVE-specific lenses and methods live in this file.

"""
    AmountParameter(phase::Symbol)

Lens on the amount (volume fraction or crack density) of phase `phase`.
Prefer the helper [`amount`](@ref).
"""
struct AmountParameter <: AbstractParameter
    phase::Symbol
end

"""
    PropertyParameter(phase, property, selector)

Lens on a scalar coefficient of a property tensor :

- `phase::Symbol` — phase name (`:M` for the matrix by default, or any
  inclusion phase).
- `property::Symbol` — property key in the phase (`:C`, `:K`, …).
- `selector::Union{Symbol,Int}` — coefficient selector :
  - `Symbol` for canonical names (`:bulk` / `:shear` for `TensISO{4,3}` ;
    `:scalar` for `TensISO{2}` ; `:ℓ₁`..`:ℓ₆` for `TensTI{4}` ;
    `:transverse` / `:axial` for `TensTI{2}`),
  - `Int` for the positional index into `get_data(tensor)` (universal
    fallback).

Prefer the helper [`property`](@ref).
"""
struct PropertyParameter <: AbstractParameter
    phase::Symbol
    property::Symbol
    selector::Union{Symbol, Int}
end

"""
    GeometryParameter(phase, field, index = nothing)

Lens on a scalar geometry field of a phase. For an `NTuple` field (e.g.
`semi_axes`), `index` selects one element of the tuple ; otherwise leave
`index = nothing`.

Prefer the helper [`geometry`](@ref).
"""
struct GeometryParameter <: AbstractParameter
    phase::Symbol
    field::Symbol
    index::Union{Int, Nothing}
end

GeometryParameter(phase::Symbol, field::Symbol) = GeometryParameter(phase, field, nothing)

"""
    DistributionShapeParameter(field, index = nothing)

Lens on a scalar field of the (Maxwell / PCW) distribution shape.
Prefer the helper [`shape_param`](@ref).
"""
struct DistributionShapeParameter <: AbstractParameter
    field::Symbol
    index::Union{Int, Nothing}
end

DistributionShapeParameter(field::Symbol) = DistributionShapeParameter(field, nothing)

# =============================================================================
#  User helpers (sugar)
# =============================================================================

"""
    amount(phase::Symbol) -> AmountParameter

Helper. Equivalent to `AmountParameter(phase)`.
"""
amount(phase::Symbol) = AmountParameter(phase)

"""
    property(phase::Symbol, property::Symbol, selector) -> PropertyParameter

Helper. Equivalent to `PropertyParameter(phase, property, selector)`.
"""
property(phase::Symbol, prop::Symbol, sel::Union{Symbol, Int}) =
    PropertyParameter(phase, prop, sel)

"""
    geometry(phase::Symbol, field::Symbol, index=nothing) -> GeometryParameter

Helper. Equivalent to `GeometryParameter(phase, field, index)`.
"""
geometry(phase::Symbol, field::Symbol, index::Union{Int, Nothing} = nothing) =
    GeometryParameter(phase, field, index)

"""
    shape_param(field::Symbol, index=nothing) -> DistributionShapeParameter

Helper. Equivalent to `DistributionShapeParameter(field, index)`.
"""
shape_param(field::Symbol, index::Union{Int, Nothing} = nothing) =
    DistributionShapeParameter(field, index)

# =============================================================================
#  Named selectors → `get_data` indices
# =============================================================================

# TensISO{2, dim, T, 1} — single scalar `λ`
_resolve_selector(::TensND.TensISO{2}, sel::Symbol) =
    sel === :scalar || sel === :λ ? 1 :
    throw(ArgumentError("TensISO{2} only accepts selector :scalar (or :λ); got :$(sel)"))

# TensISO{4, dim, T, 2} — data `(α, β) = (3K, 2μ)` or `(λ, μ)` per convention
_resolve_selector(::TensND.TensISO{4}, sel::Symbol) =
    sel === :bulk      || sel === :α || sel === :K ? 1 :
    sel === :shear     || sel === :β || sel === :μ ? 2 :
    throw(ArgumentError("TensISO{4} accepts :bulk/:α/:K (1) or :shear/:β/:μ (2); got :$(sel)"))

# TensTI{2, T, 2} — `(a, b)` = (transverse, axial)
_resolve_selector(::TensND.TensTI{2}, sel::Symbol) =
    sel === :transverse || sel === :a ? 1 :
    sel === :axial      || sel === :b ? 2 :
    throw(ArgumentError("TensTI{2} accepts :transverse/:a (1) or :axial/:b (2); got :$(sel)"))

# TensTI{4, T, 5} — major-symmetric : (ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆) with ℓ₄ = ℓ₃
_resolve_selector(::TensND.TensTI{4, T, 5}, sel::Symbol) where {T} = begin
    sel === :ℓ₁ || sel === :l1 ? 1 :
        sel === :ℓ₂ || sel === :l2 ? 2 :
        sel === :ℓ₃ || sel === :l3 || sel === :ℓ₄ || sel === :l4 ? 3 :
        sel === :ℓ₅ || sel === :l5 ? 4 :
        sel === :ℓ₆ || sel === :l6 ? 5 :
        throw(ArgumentError("TensTI{4} (major-sym) accepts :ℓ₁..:ℓ₆ (with ℓ₃=ℓ₄); got :$(sel)"))
end

# TensTI{4, T, 6} — general : (ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆)
_resolve_selector(::TensND.TensTI{4, T, 6}, sel::Symbol) where {T} = begin
    sel === :ℓ₁ || sel === :l1 ? 1 :
        sel === :ℓ₂ || sel === :l2 ? 2 :
        sel === :ℓ₃ || sel === :l3 ? 3 :
        sel === :ℓ₄ || sel === :l4 ? 4 :
        sel === :ℓ₅ || sel === :l5 ? 5 :
        sel === :ℓ₆ || sel === :l6 ? 6 :
        throw(ArgumentError("TensTI{4} general accepts :ℓ₁..:ℓ₆ ; got :$(sel)"))
end

_resolve_selector(_, sel::Int) = sel

_resolve_selector(t::TensND.AbstractTens, sel::Symbol) = throw(
    ArgumentError(
        "no named selector :$(sel) defined for tensor of type $(typeof(t)); use an Int positional index"
    )
)

# =============================================================================
#  Tensor reconstruction with one coefficient replaced
# =============================================================================
#
#  For each supported tensor type, `_replace_data_at(t, i, v)` returns a
#  new tensor of the *same structural type*, with the i-th scalar of
#  `get_data` replaced by `v`, and the element type promoted via
#  `promote_type(eltype(t), typeof(v))`.

"""
    _replace_data_at(t, i, v) -> typeof_promoted(t)

Return a new tensor of the same structural type as `t`, with the scalar
at index `i` (in `get_data`) replaced by `v`. The element type is
promoted to absorb `typeof(v)`.
"""
_replace_data_at(t, i, v) = throw(
    ArgumentError(
        "_replace_data_at not implemented for tensor of type $(typeof(t)); add a method"
    )
)

function _replace_data_at(t::TensND.TensISO{order, dim, T, N}, i::Int, v::Tv) where {order, dim, T, N, Tv}
    Tnew = promote_type(T, Tv)
    new_data = ntuple(k -> k == i ? convert(Tnew, v) : convert(Tnew, t.data[k]), N)
    return TensND.TensISO{dim}(new_data...)
end

function _replace_data_at(t::TensND.TensTI{order, T, N}, i::Int, v::Tv) where {order, T, N, Tv}
    Tnew = promote_type(T, Tv)
    new_data = ntuple(k -> k == i ? convert(Tnew, v) : convert(Tnew, t.data[k]), N)
    new_n = ntuple(k -> convert(Tnew, t.n[k]), 3)
    return TensND.TensTI{order, Tnew, N}(new_data, new_n)
end

# =============================================================================
#  Inclusion-geometry reconstruction with one field replaced
# =============================================================================
#
#  Type promotion is handled locally : `_replace_geom_field` returns an
#  instance whose `T` parameter has been promoted to absorb the new
#  scalar. For user-defined inclusion types, a generic `@generated`
#  fallback reflects on `fieldnames(typeof(geom))` and reconstructs via
#  the parametric inner constructor — the auto-generated pattern Julia
#  provides for structs without custom non-parametric inner constructors.

"""
    _replace_geom_field(geom, ::Val{name}, index, value) -> typeof_promoted(geom)

Reconstruct an inclusion geometry, replacing field `name` (optionally at
tuple index `index`) by `value`. Type promotion is applied uniformly to
all `<:Number` sibling fields so the resulting struct's parametric T is
consistent.

Specific methods are provided for `Ellipsoid`. User-defined inclusion
types fall back to the generic `@generated` reconstruction, which works
as long as the type follows the standard Julia parametric-constructor
pattern.
"""
function _replace_geom_field end

# Generic fallback : direct call to the type constructor with the original
# fields, swapping the target. Numeric (`<:Number`) sibling fields are
# uniformly promoted to a common `T` to avoid type mismatches with
# parametric constructors that share T across multiple fields
# (e.g. `MyBlob{T<:Number}` with two `T`-typed fields).
@generated function _replace_geom_field(geom, ::Val{name}, ::Nothing, value) where {name}
    fields = fieldnames(geom)
    name in fields || return :(
        throw(
            ArgumentError(
                "geometry type $(typeof(geom)) has no field :$($(QuoteNode(name)))"
            )
        )
    )
    types = [fieldtype(geom, f) for f in fields]
    # Identify numeric-valued fields whose type may need promotion.
    nums = [t <: Number for t in types]
    args = Any[]
    for (i, f) in enumerate(fields)
        if f === name
            push!(args, :(__T(value)))
        elseif nums[i]
            push!(args, :(__T(getfield(geom, $(QuoteNode(f))))))
        else
            push!(args, :(getfield(geom, $(QuoteNode(f)))))
        end
    end
    UA = geom.name.wrapper
    # Build promotion expression: __T = promote_type of all <:Number siblings
    # with typeof(value), so each numeric field receives the same eltype.
    sibling_types = Expr(:call, :promote_type)
    push!(sibling_types.args, :(typeof(value)))
    for (i, f) in enumerate(fields)
        if nums[i] && f !== name
            push!(sibling_types.args, :(typeof(getfield(geom, $(QuoteNode(f))))))
        end
    end
    return :(
        let __T = $(sibling_types)
            $(UA)($(args...))
        end
    )
end

@generated function _replace_geom_field(geom, ::Val{name}, idx::Int, value) where {name}
    fields = fieldnames(geom)
    name in fields || return :(
        throw(
            ArgumentError(
                "geometry type $(typeof(geom)) has no field :$($(QuoteNode(name)))"
            )
        )
    )
    types = [fieldtype(geom, f) for f in fields]
    nums = [t <: Number for t in types]
    args = Any[]
    for (i, f) in enumerate(fields)
        if f === name
            # tuple field: replace at idx, promote across siblings + value
            push!(args, :(_promote_tuple_at(getfield(geom, $(QuoteNode(f))), idx, value, __T)))
        elseif nums[i]
            push!(args, :(__T(getfield(geom, $(QuoteNode(f))))))
        else
            push!(args, :(getfield(geom, $(QuoteNode(f)))))
        end
    end
    UA = geom.name.wrapper
    sibling_types = Expr(:call, :promote_type)
    push!(sibling_types.args, :(typeof(value)))
    for (i, f) in enumerate(fields)
        if nums[i] && f !== name
            push!(sibling_types.args, :(typeof(getfield(geom, $(QuoteNode(f))))))
        end
    end
    return :(
        let __T = $(sibling_types)
            $(UA)($(args...))
        end
    )
end

# Tuple element replacement with explicit promoted type (used by @generated
# version above when the target field is itself a tuple).
function _promote_tuple_at(t::NTuple{N}, i::Int, v, ::Type{Tnew}) where {N, Tnew}
    return ntuple(k -> k == i ? convert(Tnew, v) : convert(Tnew, t[k]), N)
end

# ─── Specializations for built-in inclusion types ────────────────────────────
# Parametric structs whose type parameters are computed from the values
# (Ellipsoid, EllipticCrack, RibbonCrack) need explicit reconstruction
# with the right parameter binding ; the @generated fallback fails for
# Ellipsoid because `S` (the shape trait) cannot be inferred from the
# inner-constructor arguments.

# Ellipsoid{dim, S, T, B} — recomputes `T` **and** `S`.
#
# The shape trait is not incidental metadata: the Hill/Eshelby kernels
# dispatch on it, so carrying the old `S` over to new semi-axes silently
# returns the *old shape's* answer. Changing the third semi-axis of a
# `Spherical` ellipsoid to 0.2 must yield an `Oblate` one, not a sphere
# that merely stores oblate numbers. Reconstructing through the outer
# constructor also restores the descending-axis sort, the matching basis
# permutation, and the degenerate-limit redirection (Cylinder / crack).
function _replace_geom_field(
        geom::Ellipsoid{dim, S, T, B},
        ::Val{:semi_axes},
        idx::Int, value::Tv
    ) where {dim, S, T, B, Tv}
    Tnew = promote_type(T, Tv)
    new_axes = ntuple(k -> k == idx ? convert(Tnew, value) : convert(Tnew, geom.semi_axes[k]), dim)
    return Ellipsoid(new_axes..., geom.basis)
end

function _replace_geom_field(
        geom::Ellipsoid{dim, S, T, B},
        ::Val{:semi_axes},
        ::Nothing,
        value::NTuple{dim}
    ) where {dim, S, T, B}
    # `eltype` rather than a `Tv` type parameter: as a parameter it was bound by
    # nothing when `dim` was zero. For a homogeneous tuple the two agree exactly.
    Tnew = promote_type(T, eltype(value))
    new_axes = ntuple(k -> convert(Tnew, value[k]), dim)
    return Ellipsoid(new_axes..., geom.basis)
end

# =============================================================================
#  RVE reconstruction with amount-eltype promotion
# =============================================================================

# Promote an `AbstractAmount{T}` to `AbstractAmount{Tnew}` preserving the
# subtype (VolumeFraction stays VolumeFraction, CrackDensity stays CrackDensity).
_promote_amount(a::VolumeFraction{T}, ::Type{Tnew}) where {T, Tnew} =
    VolumeFraction{Tnew}(convert(Tnew, a.value))
_promote_amount(a::CrackDensity{T}, ::Type{Tnew}) where {T, Tnew} =
    CrackDensity{Tnew}(convert(Tnew, a.value))
# A `Remainder` carries no value, hence no element type to promote: it stays
# itself, and the fraction it stands for takes the element type of the sum it
# is subtracted from.
_promote_amount(a::Remainder, ::Type) = a

"""
    _rebuild_rve(rve; phases=rve.phases, amounts=rve.amounts,
                 symmetrize=rve.symmetrize,
                 distribution_shape=rve.distribution_shape,
                 T=_declared_eltype(rve)) -> RVE

Immutable reconstruction of a `RVE` with a subset of fields replaced.
Preserves `phase_names` (insertion order) and the declared element-type
floor `T`, and recomputes `S` from `distribution_shape`.

The floor is preserved rather than recomputed because amounts carry
their own element types: replacing one amount by a `Dual` does not
require re-typing the others, they promote where the values meet.
"""
function _rebuild_rve(
        rve::RVE;
        phases = rve.phases,
        amounts = rve.amounts,
        symmetrize = rve.symmetrize,
        distribution_shape = rve.distribution_shape,
        T = _declared_eltype(rve)
    )
    S = typeof(distribution_shape)
    # `amounts` is already a `Dict{Symbol,AbstractAmount}` on every call path
    # (`rve.amounts` itself, `_amounts_with_replacement`, `promote_rve`), so it
    # is stored as-is — copying it here would allocate a second dict per
    # `set_param`, i.e. per autodiff step.
    new_amounts = amounts isa Dict{Symbol, AbstractAmount} ? amounts :
        Dict{Symbol, AbstractAmount}(amounts)
    # Property / geometry / distribution-shape lenses leave the amounts alone,
    # which is the common case under autodiff — carry the cached fraction over
    # instead of summing the dict again.
    f_sum = (new_amounts === rve.amounts && T === _declared_eltype(rve)) ?
        rve.f_sum : _compute_fraction_sum(new_amounts, T)
    # `rest_name` always carries over: no lens changes the *type* of an amount
    # (`set_param` preserves `VolumeFraction` / `CrackDensity` explicitly), so
    # the phase absorbing the complement cannot move under one.
    return RVE{T, S}(
        copy(rve.phase_names),
        phases, new_amounts,
        symmetrize, distribution_shape, rve.closure, rve.rest_name, f_sum,
    )
end

# Declared element-type floor (as opposed to `eltype(rve)`, the effective
# element type promoted with the stored amounts — see `rve.jl`).
_declared_eltype(::Type{<:RVE{T}}) where {T} = T
_declared_eltype(rve::RVE) = _declared_eltype(typeof(rve))

Base.eltype(::Type{<:RVE{T}}) where {T} = T

# Same dict but with one entry replaced by `new_amount`. No cross-promotion
# of the untouched entries: the amounts dict is heterogeneous by design.
function _amounts_with_replacement(
        amounts::AbstractDict, name::Symbol,
        new_amount::AbstractAmount
    )
    new_dict = Dict{Symbol, AbstractAmount}(amounts)
    new_dict[name] = new_amount
    return new_dict
end

# =============================================================================
#  get_param / set_param — public lens entry points
# =============================================================================

# `get_param` / `set_param` are declared in `Core/cells.jl`; the methods below
# are the RVE-specific ones.

# ── AmountParameter ──────────────────────────────────────────────────────────

function get_param(rve::RVE, p::AmountParameter)
    haskey(rve.amounts, p.phase) ||
        throw(ArgumentError("phase :$(p.phase) has no amount in RVE"))
    a = rve.amounts[p.phase]
    # The complement has no declared value; its resolved one is the natural
    # reading. Every other amount reports what was DECLARED, not what the
    # closure resolves it to — this is the inverse of `set_param`, and under
    # `RescaledFractions` the two differ (see `volume_fraction`).
    a isa Remainder && return remainder_volume_fraction(rve)
    return amount_value(a)
end

function set_param(rve::RVE, p::AmountParameter, value)
    haskey(rve.amounts, p.phase) ||
        throw(ArgumentError("phase :$(p.phase) has no amount in RVE"))
    old = rve.amounts[p.phase]
    old isa Remainder && throw(
        ArgumentError(
            "phase :$(p.phase) takes up the volume complement (1 - Σ f) and has no " *
                "declared amount to differentiate; differentiate with respect to an " *
                "explicit amount instead."
        )
    )
    v = _amount_promote(_declared_eltype(rve), value)
    new_amount = old isa VolumeFraction ? VolumeFraction(v) : CrackDensity(v)
    new_amounts = _amounts_with_replacement(rve.amounts, p.phase, new_amount)
    return _rebuild_rve(rve; amounts = new_amounts)
end

# ── PropertyParameter ────────────────────────────────────────────────────────

function get_param(rve::RVE, p::PropertyParameter)
    t = phase_property(rve, p.phase, p.property)
    i = _resolve_selector(t, p.selector)
    return TensND.get_data(t)[i]
end

function set_param(rve::RVE, p::PropertyParameter, value)
    haskey(rve.phases, p.phase) ||
        throw(ArgumentError("no phase named :$(p.phase) in RVE"))
    phase = rve.phases[p.phase]
    haskey(phase.properties, p.property) ||
        throw(ArgumentError("phase :$(p.phase) does not carry property :$(p.property)"))
    old_t = phase.properties[p.property]
    i = _resolve_selector(old_t, p.selector)
    new_t = _replace_data_at(old_t, i, value)

    # Build new properties dict (keeping all other entries pointing to the same
    # objects). It must be typed `Any`, exactly like `Phase.properties`: a
    # property value is not necessarily an `AbstractTens` — it can be a
    # `ViscoLaw` (ALV) or a `Homogenized` (declaratively nested cell), and
    # narrowing here threw on both.
    new_props = Dict{Symbol, Any}(phase.properties)
    new_props[p.property] = new_t
    new_phase = Phase(phase.geometry, new_props)

    new_phases = Dict{Symbol, Phase}(rve.phases)
    new_phases[p.phase] = new_phase

    return _rebuild_rve(rve; phases = new_phases)
end

# ── GeometryParameter ────────────────────────────────────────────────────────

function get_param(rve::RVE, p::GeometryParameter)
    haskey(rve.phases, p.phase) ||
        throw(ArgumentError("no phase named :$(p.phase) in RVE"))
    geom = rve.phases[p.phase].geometry
    val = getfield(geom, p.field)
    return p.index === nothing ? val : val[p.index]
end

function set_param(rve::RVE, p::GeometryParameter, value)
    haskey(rve.phases, p.phase) ||
        throw(ArgumentError("no phase named :$(p.phase) in RVE"))
    phase = rve.phases[p.phase]
    new_geom = _replace_geom_field(phase.geometry, Val(p.field), p.index, value)
    new_phase = Phase(new_geom, phase.properties)

    new_phases = Dict{Symbol, Phase}(rve.phases)
    new_phases[p.phase] = new_phase

    return _rebuild_rve(rve; phases = new_phases)
end

# ── DistributionShapeParameter ───────────────────────────────────────────────

function get_param(rve::RVE, p::DistributionShapeParameter)
    ds = rve.distribution_shape
    ds isa UniformDistribution || throw(
        ArgumentError(
            "distribution shape parameter only supported for UniformDistribution; got $(typeof(ds))"
        )
    )
    val = getfield(ds.shape, p.field)
    return p.index === nothing ? val : val[p.index]
end

function set_param(rve::RVE, p::DistributionShapeParameter, value)
    ds = rve.distribution_shape
    ds isa UniformDistribution || throw(
        ArgumentError(
            "distribution shape parameter only supported for UniformDistribution; got $(typeof(ds))"
        )
    )
    new_shape = _replace_geom_field(ds.shape, Val(p.field), p.index, value)
    new_ds = UniformDistribution(new_shape)
    return _rebuild_rve(rve; distribution_shape = new_ds)
end

# =============================================================================
#  _set_many — type-stable batch update used by `gradient`/`jacobian`
# =============================================================================

"""
    _set_many(rve, params, values) -> RVE

Apply several `set_param` calls to the same RVE in one shot. Used inside
`gradient`/`jacobian` to guarantee a single reconstruction pass and
uniform type-stability across all the lenses involved.

The default implementation composes individual `set_param` calls; for
`AbstractParameter`s that target the same field type (e.g. several
amounts), a future optimization could fuse the passes. Correctness is
preserved by composition.
"""
function _set_many(
        rve::AbstractHomogenizationCell,
        params::AbstractVector{<:AbstractParameter}, values::AbstractVector
    )
    length(params) == length(values) || throw(
        ArgumentError(
            "params and values must have the same length"
        )
    )
    out = rve
    for (p, v) in zip(params, values)
        out = set_param(out, p, v)
    end
    return out
end
