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
    AbstractPropertyParameter <: AbstractParameter

A lens on the tensor of a property (`property` of the member `phase` of a
cell): one of its coefficients ([`PropertyParameter`](@ref)) or its abscissa
along a direction ([`PropertyDirection`](@ref)). A cell reads and replaces the
tensor; what the scalar is in that tensor is decided by the lens.
"""
abstract type AbstractPropertyParameter <: AbstractParameter end

"""
    PropertyParameter(phase, property, selector, symmetry = nothing)

Lens on one coefficient of a property tensor (`property` of the phase `phase`,
`:C`, `:K`, …), in a symmetry class that contains the tensor:

  - `symmetry = nothing`: the tensor's own class, `selector` one of its named
    coefficients (`:k`, `:μ`, `:α`, `:β` of an isotropic stiffness, `:ℓ₁` … `:ℓ₆`
    of a transversely isotropic one, `:C1111` … of an orthotropic one, …), an
    index of `get_data`, or a Kelvin-Mandel component `:Mij` (`:Mi` at order 2)
    in the canonical basis;
  - `symmetry = (:TI, n)`, `(:ORTHO, frame)`, `(:CUBIC, frame)`: a class that
    contains the tensor, the tensor being written in it first; `n` is the unit
    vector of the axis and `frame` an orthonormal basis. The orientation may be
    left out (`symmetry = :TI`) when the tensor carries it: a `TensTI` for
    `:TI`, a `TensOrtho` or a `TensCubic` for `:ORTHO`, a `TensCubic` for
    `:CUBIC`;
  - `symmetry = :ANISO` or `(:ANISO, basis)`: the Kelvin-Mandel components in
    `basis` (the canonical one by default), `selector = :Mij`.

The other coefficients of the class stay fixed when this one is set, so that a
derivative is the partial derivative in that class. Prefer the helper
[`property`](@ref); the names of every class are in the manual page
[Sensitivities](@ref man-sensitivities-lenses).
"""
struct PropertyParameter{S} <: AbstractPropertyParameter
    phase::Symbol
    property::Symbol
    selector::Union{Symbol, Int}
    symmetry::S
end

PropertyParameter(phase::Symbol, prop::Symbol, sel::Union{Symbol, Int}) = PropertyParameter(phase, prop, sel, nothing)

"""
    PropertyDirection(phase, property, direction)

Lens on the abscissa `t` of a property tensor along a fixed tensor: setting it
replaces the tensor ``\\mathbb{C}`` of the phase by
``\\mathbb{C} + t\\,\\mathbb{B}``, `direction` being ``\\mathbb{B}``, a tensor
of the order and the dimension of the property. Its value in a cell is zero, and
a derivative with respect to it is the derivative of the result along
``\\mathbb{B}``, whatever the symmetry classes of the two tensors. Prefer the
helper [`property`](@ref); [`second_moment`](@ref) is built on it.
"""
struct PropertyDirection{B <: TensND.AbstractTens} <: AbstractPropertyParameter
    phase::Symbol
    property::Symbol
    direction::B
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
    property(phase::Symbol, property::Symbol, selector; symmetry = nothing) -> PropertyParameter

Helper. Equivalent to `PropertyParameter(phase, property, selector, symmetry)`:
`property(:I, :C, :k)` is the bulk modulus of an isotropic phase,
`property(:I, :C, :ℓ₁; symmetry = (:TI, n))` its first Walpole coefficient as a
transversely isotropic tensor of axis `n`, `property(:I, :C, :M12)` its
Kelvin-Mandel component (1, 2).
"""
property(phase::Symbol, prop::Symbol, sel::Union{Symbol, Int}; symmetry = nothing) =
    PropertyParameter(phase, prop, sel, symmetry)

"""
    property(phase::Symbol, property::Symbol, direction::AbstractTens) -> PropertyDirection

Helper. Equivalent to `PropertyDirection(phase, property, direction)`: the
abscissa of the property along the tensor `direction`, zero in the cell.
`derivative(rve, scheme, property(:I, :C, 𝔹))` is the derivative of the
effective stiffness when the stiffness of `:I` moves along `𝔹`.
"""
property(phase::Symbol, prop::Symbol, direction::TensND.AbstractTens) = PropertyDirection(phase, prop, direction)

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
#  The coefficient a `PropertyParameter` designates
# =============================================================================
#
#  A coefficient belongs to a symmetry class that contains the tensor: its own
#  class by default, or an enclosing one named by `symmetry`, transversely
#  isotropic about an axis, orthotropic or cubic in a frame, or anisotropic,
#  read in the Kelvin-Mandel components of a basis. The tensor is first written
#  in that class, exactly, since the class contains it; the coefficient is then
#  read, or replaced by a number while the other coefficients of the class stay
#  fixed. A dual number put there gives, by ForwardDiff, the derivative with
#  respect to that coefficient, whatever the class the tensor was given in.

# ── The named coefficients of each class ─────────────────────────────────────
# `(i, f)`: the coefficient is `get_data(t)[i] / f`. Isotropic stiffness of
# dimension d, `α 𝕁 + β 𝕂`: α = d k (k the bulk modulus of the space, λ + μ in
# the plane), β = 2μ.

_coefficient(::TensND.TensISO{4, dim}, sel::Symbol) where {dim} =
    sel === :k || sel === :bulk ? (1, dim) :
    sel === :μ || sel === :shear ? (2, 2) :
    sel === :α ? (1, 1) :
    sel === :β ? (2, 1) :
    _unknown_coefficient(sel, "an isotropic tensor of order 4", ":k (≡ :bulk), :μ (≡ :shear), :α, :β")

_coefficient(::TensND.TensISO{2}, sel::Symbol) =
    sel === :scalar ? (1, 1) : _unknown_coefficient(sel, "an isotropic tensor of order 2", ":scalar")

_coefficient(::TensND.TensTI{2}, sel::Symbol) =
    sel === :transverse ? (1, 1) :
    sel === :axial ? (2, 1) :
    _unknown_coefficient(sel, "a transversely isotropic tensor of order 2", ":transverse, :axial")

const _WALPOLE = (:ℓ₁, :ℓ₂, :ℓ₃, :ℓ₄, :ℓ₅, :ℓ₆)
const _WALPOLE_ASCII = (:l1, :l2, :l3, :l4, :l5, :l6)
_walpole_rank(sel::Symbol) = something(findfirst(==(sel), _WALPOLE), findfirst(==(sel), _WALPOLE_ASCII), 0)

# With major symmetry ℓ₃ = ℓ₄ is one coefficient, stored third.
function _coefficient(::TensND.TensTI{4, T, 5}, sel::Symbol) where {T}
    r = _walpole_rank(sel)
    r == 0 && _unknown_coefficient(sel, "a transversely isotropic tensor of order 4 with major symmetry", ":ℓ₁ … :ℓ₆ (ℓ₃ = ℓ₄)")
    return (r ≤ 4 ? min(r, 3) : r - 1, 1)
end
function _coefficient(::TensND.TensTI{4, T, 6}, sel::Symbol) where {T}
    r = _walpole_rank(sel)
    r == 0 && _unknown_coefficient(sel, "a transversely isotropic tensor of order 4", ":ℓ₁ … :ℓ₆")
    return (r, 1)
end

const _ORTHO9 = (:C1111, :C2222, :C3333, :C1122, :C1133, :C2233, :C2323, :C1313, :C1212)
const _ORTHO12 = (_ORTHO9..., :C2211, :C3311, :C3322)
_coefficient(::TensND.TensOrtho{4, T, 9}, sel::Symbol) where {T} = _named_rank(sel, _ORTHO9, "an orthotropic tensor of order 4")
_coefficient(::TensND.TensOrtho{4, T, 12}, sel::Symbol) where {T} = _named_rank(sel, _ORTHO12, "an orthotropic tensor of order 4 without major symmetry")
_coefficient(::TensND.TensOrtho{2}, sel::Symbol) = _named_rank(sel, (:a1, :a2, :a3), "an orthotropic tensor of order 2")
_coefficient(::TensND.TensCubic, sel::Symbol) = _named_rank(sel, (:α, :β, :γ), "a cubic tensor")

_coefficient(::Any, i::Int) = (i, 1)
_coefficient(t, sel::Symbol) = _unknown_coefficient(sel, "a tensor of type $(nameof(typeof(t)))", "a Kelvin-Mandel component :Mij, or an index of `get_data`")

function _named_rank(sel::Symbol, names, what)
    r = findfirst(==(sel), names)
    r === nothing && _unknown_coefficient(sel, what, join(string.(":", names), ", "))
    return (r, 1)
end

_unknown_coefficient(sel, what, names) = throw(
    ArgumentError(
        ":$sel is not a coefficient of $what, whose coefficients are $names. A coefficient " *
            "of an enclosing class is named with `symmetry`: `property(phase, prop, :ℓ₁; " *
            "symmetry = (:TI, n))`, `(:ORTHO, frame)`, `(:CUBIC, frame)`, or a Kelvin-Mandel " *
            "component, `property(phase, prop, :M12)`."
    )
)

# ── Kelvin-Mandel components ─────────────────────────────────────────────────
# `:Mij` is the component (i, j) of the Kelvin-Mandel matrix of a tensor of
# order 4, `:Mi` the component i of the Kelvin-Mandel vector of a tensor of
# order 2, in a basis (the canonical one by default). The matrix stays
# symmetric: `:Mij` with i ≠ j sets (i, j) and (j, i) together.

struct _MandelView{A <: AbstractArray, B <: TensND.AbstractBasis}
    components::A
    basis::B
end

function _mandel_index(sel::Symbol)
    m = match(r"^M(\d)(\d)?$", String(sel))
    m === nothing && return nothing
    i = parse(Int, m.captures[1])
    return m.captures[2] === nothing ? (i,) : (i, parse(Int, m.captures[2]))
end

function _mandel_index(V::_MandelView, sel::Symbol)
    idx = _mandel_index(sel)
    (idx !== nothing && length(idx) == ndims(V.components) && all(i -> 1 ≤ i ≤ size(V.components, 1), idx)) ||
        _not_a_component(V, sel)
    return idx
end
_mandel_index(V::_MandelView, sel::Int) = _not_a_component(V, sel)
_not_a_component(V, sel) = throw(
    ArgumentError(
        "$(repr(sel)) is not a Kelvin-Mandel component of this tensor: name " *
            (ndims(V.components) == 2 ? "`:Mij`, 1 ≤ i, j ≤ $(size(V.components, 1))" : "`:Mi`, 1 ≤ i ≤ $(size(V.components, 1))")
    )
)

_canonical_basis(t::TensND.AbstractTens{order, dim}) where {order, dim} = TensND.CanonicalBasis{dim, eltype(t)}()
_mandel_view(t, b = _canonical_basis(t)) = _MandelView(TensND.mandel(t, b), b)

# ── The tensor written in the class of the coefficient ──────────────────────

# With no class named, a name `:Mij` is a Kelvin-Mandel component (a run-time
# value of the symbol), any other name or index a coefficient of the own class.
_class_view(t, sel::Symbol, ::Nothing) = _mandel_index(sel) === nothing ? t : _mandel_view(t)
_class_view(t, ::Int, ::Nothing) = t
_class_view(t, sel, symmetry::Symbol) = _class_view(t, sel, (symmetry,))
_class_view(t, sel, symmetry::Tuple) = _enclosing(t, Val(first(symmetry)), Base.tail(symmetry)...)

_enclosing(t, ::Val{:ANISO}, b...) = _mandel_view(t, b...)
# A class named without its orientation takes that of the tensor when its type
# carries one for that class: the tensor is already written in it, or its frame
# is that of the class. No orientation is searched, so that dual numbers and
# symbols go through.
_enclosing(t::TensND.TensTI, ::Val{:TI}) = t
_enclosing(t::TensND.TensOrtho, ::Val{:ORTHO}) = t
_enclosing(t::TensND.TensCubic, ::Val{:CUBIC}) = t
_enclosing(t::TensND.TensCubic, class::Val{:ORTHO}) = _in_space_class(t, class, t.frame)
_enclosing(t, class::Val, args...) = _in_space_class(t, class, args...)
_in_space_class(t::TensND.AbstractTens{order, 3}, class::Val, args...) where {order} = _exact_projection(t, class, args...)
_in_space_class(::TensND.AbstractTens{order, 2}, ::Val{S}, args...) where {order, S} = throw(
    ArgumentError(
        "symmetry = :$S: in the plane a coefficient is one of the tensor's own class or a " *
            "Kelvin-Mandel component (`symmetry = :ANISO`, or a name `:Mij`)"
    )
)

# The projection onto a class that contains the tensor returns the tensor. Not
# containing it is refused: the derivative would be taken at another tensor.
_exact_projection(t, class::Val, args...) = _project_contained(Val(is_hard_numeric(eltype(t))), t, class, args...)
function _project_contained(::Val{true}, t, class::Val{S}, args...) where {S}
    B, _, drel = TensND.project(t, class, args...)
    drel ≤ 1.0e-10 || _not_contained(S, t, " (relative distance $drel)")
    return B
end
# A symbolic distance cannot be compared: the isotropic tensor, which every class
# contains, is the one case decided without it.
_project_contained(::Val{false}, t::TensND.TensISO, class::Val, args...) = first(TensND.project(t, class, args...))
_project_contained(::Val{false}, t, ::Val{S}, args...) where {S} =
    _not_contained(S, t, ", which a symbolic tensor cannot show unless it is isotropic")
_not_contained(S, t, why) = throw(
    ArgumentError(
        "symmetry = :$S does not contain this $(nameof(typeof(t)))$why: name a class that " *
            "contains the tensor, or a Kelvin-Mandel component"
    )
)

# ── Reading and replacing the coefficient ────────────────────────────────────

function _read_coefficient(B, sel)
    i, f = _coefficient(B, sel)
    return TensND.get_data(B)[i] / f
end
_read_coefficient(V::_MandelView, sel) = V.components[_mandel_index(V, sel)...]

function _write_coefficient(B, sel, v)
    i, f = _coefficient(B, sel)
    return TensND.rebuild(B, Base.setindex(TensND.get_data(B), f * v, i))
end
function _write_coefficient(V::_MandelView, sel, v)
    idx = _mandel_index(V, sel)
    A = Array{promote_type(eltype(V.components), typeof(v))}(V.components)
    A[idx...] = v
    A[reverse(idx)...] = v
    return TensND.from_mandel(A, V.basis)
end

"""
    get_coefficient(t, selector; symmetry = nothing)

The coefficient `selector` of the tensor `t`, in the symmetry class `symmetry`
that contains it (its own class by default): `get_coefficient(C, :k)` is the
bulk modulus of an isotropic stiffness, `get_coefficient(C, :ℓ₁; symmetry =
(:TI, n))` its first Walpole coefficient about `n`, `get_coefficient(C, :M12)`
its Kelvin-Mandel component (1, 2). The selectors and the classes are those of
[`property`](@ref), listed in the manual page
[Sensitivities](@ref man-sensitivities-lenses).

# Examples
```jldoctest
julia> using MeanFieldHomogenization

julia> C = iso_stiffness(10.0, 4.0);

julia> get_coefficient(C, :k), get_coefficient(C, :α), get_coefficient(C, :μ)
(10.0, 30.0, 4.0)

julia> get_coefficient(C, :M44) ≈ 2 * 4.0      # the Kelvin-Mandel component of the shear
true
```
"""
get_coefficient(t, sel; symmetry = nothing) = _read_coefficient(_class_view(t, sel, symmetry), sel)

"""
    set_coefficient(t, selector, v; symmetry = nothing)

The tensor of the class `symmetry` (the class of `t` by default) whose
coefficient `selector` is `v`, its other coefficients being those of `t`. With a
dual number `v`, a derivative with respect to that coefficient; the class must
contain `t`.

# Examples
```jldoctest
julia> using MeanFieldHomogenization, TensND

julia> C = set_coefficient(iso_stiffness(10.0, 4.0), :μ, 5.0);

julia> get_coefficient(C, :k), get_coefficient(C, :μ)
(10.0, 5.0)
```
"""
set_coefficient(t, sel, v; symmetry = nothing) = _write_coefficient(_class_view(t, sel, symmetry), sel, v)

# The scalar a property lens designates in its tensor, and the tensor in which
# that scalar is replaced: a coefficient of a class, or the abscissa along a
# direction, zero at the tensor itself.
property_coefficient(t, p::PropertyParameter) = get_coefficient(t, p.selector; symmetry = p.symmetry)
with_property_coefficient(t, p::PropertyParameter, v) = set_coefficient(t, p.selector, v; symmetry = p.symmetry)
property_coefficient(t::TensND.AbstractTens{order, dim}, p::PropertyDirection{<:TensND.AbstractTens{order, dim}}) where {order, dim} =
    zero(promote_type(eltype(t), eltype(p.direction)))
with_property_coefficient(t::TensND.AbstractTens{order, dim}, p::PropertyDirection{<:TensND.AbstractTens{order, dim}}, v) where {order, dim} =
    t + v * p.direction
property_coefficient(t, p::PropertyDirection) = _not_a_direction(t, p)
with_property_coefficient(t, p::PropertyDirection, v) = _not_a_direction(t, p)
_not_a_direction(t, p) = throw(
    ArgumentError(
        "property :$(p.property) of :$(p.phase) is a $(nameof(typeof(t))) and the direction a " *
            "$(nameof(typeof(p.direction))) of order $(ndims(p.direction)): a direction is a tensor " *
            "of the order and the dimension of the property"
    )
)

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
    new_amounts = convert(Dict{Symbol, AbstractAmount}, amounts)
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
    return _declared_amount(a, rve)
end

_declared_amount(a::AbstractAmount, rve) = amount_value(a)
_declared_amount(::Remainder, rve) = remainder_volume_fraction(rve)

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
    new_amount = _with_value(old, v)
    new_amounts = _amounts_with_replacement(rve.amounts, p.phase, new_amount)
    return _rebuild_rve(rve; amounts = new_amounts)
end

# ── PropertyParameter ────────────────────────────────────────────────────────

function get_param(rve::RVE, p::AbstractPropertyParameter)
    return property_coefficient(phase_property(rve, p.phase, p.property), p)
end

function set_param(rve::RVE, p::AbstractPropertyParameter, value)
    haskey(rve.phases, p.phase) ||
        throw(ArgumentError("no phase named :$(p.phase) in RVE"))
    phase = rve.phases[p.phase]
    haskey(phase.properties, p.property) ||
        throw(ArgumentError("phase :$(p.phase) does not carry property :$(p.property)"))
    new_t = with_property_coefficient(phase.properties[p.property], p, value)

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

"""
    _geom_field(geom, name) -> value

Read a named geometry parameter. `getfield` by default, and overridable, which
is what lets a geometry expose parameters that are not literal fields — a type
holding a shape *object* rather than its scalars, say. The reader and the
rebuilder, `_replace_geom_field`, must agree on the same name set or a
derivative silently differentiates the wrong thing.
"""
_geom_field(geom, name::Symbol) = getfield(geom, name)

function get_param(rve::RVE, p::GeometryParameter)
    haskey(rve.phases, p.phase) ||
        throw(ArgumentError("no phase named :$(p.phase) in RVE"))
    geom = rve.phases[p.phase].geometry
    val = _geom_field(geom, p.field)
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

# A lens on a shape the RVE never declared is a different mistake from a lens
# on a distribution kind that has no scalar fields, and deserves a different
# message: the first is fixed on the constructor, the second cannot be fixed
# at all.
function _uniform_distribution_or_throw(rve::RVE)
    ds = rve.distribution_shape
    ds === nothing && throw(
        ArgumentError(
            "this RVE declares no distribution shape, so there is no parameter " *
                "to read or set. Declare one on the constructor, e.g. " *
                "`RVE(; distribution_shape = Spheroid(0.3))`."
        )
    )
    ds isa UniformDistribution || throw(
        ArgumentError(
            "distribution shape parameter only supported for UniformDistribution; got $(typeof(ds))"
        )
    )
    return ds
end

function get_param(rve::RVE, p::DistributionShapeParameter)
    ds = _uniform_distribution_or_throw(rve)
    val = getfield(ds.shape, p.field)
    return p.index === nothing ? val : val[p.index]
end

function set_param(rve::RVE, p::DistributionShapeParameter, value)
    ds = _uniform_distribution_or_throw(rve)
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
