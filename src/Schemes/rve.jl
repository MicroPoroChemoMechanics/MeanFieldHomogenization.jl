# =============================================================================
#  rve.jl — Representative Volume Element (RVE) data model.
#
#  An `RVE` aggregates a *matrix* phase with one or more *inclusion* phases,
#  storing for each phase its geometry (an `AbstractInclusion`), its material
#  properties (a `Symbol => AbstractTens` map: `:C` for stiffness, `:K` for
#  conductivity, …) and an *amount* — a `VolumeFraction` for ellipsoidal
#  inclusions, a `CrackDensity` for cracks.  Volume fractions are stored
#  **at the RVE level**, not on the inclusions: a single inclusion is still
#  usable for localization-tensor calculations (`hill_tensor`,
#  `strain_strain_loc`, …) without any fraction-related machinery.
#
#  The matrix amount is implicit (`f_matrix = 1 - Σ f_inc`) and crack
#  densities are excluded from that sum (their volume contribution → 0
#  in the penny limit).
#
#  PCW / Maxwell additionally need a *distribution shape* describing the
#  outer envelope of the phase distribution; this is stored in the
#  `distribution_shape` field through an `AbstractDistributionShape`
#  hierarchy that allows future extension to pairwise distributions
#  ([Willis 1982](@cite willis1982)) without breaking the public API.
# =============================================================================

# =============================================================================
#  Amounts: volume fraction (ellipsoidal phases) and crack density (cracks)
# =============================================================================

"""
    AbstractAmount{T<:Number}

Supertype for the *quantity* attached to a phase in a `RVE`. Two concrete
subtypes:

- [`VolumeFraction`](@ref) — for solid (ellipsoidal) inclusions and the
  matrix; obeys the unit-sum constraint
  `f_matrix = 1 - Σ_other VolumeFraction.value`.
- [`CrackDensity`](@ref) — for flat cracks (Budiansky-O'Connell density);
  does **not** participate in the unit-sum constraint, since the volume
  contribution of a flat crack vanishes in the penny limit while the
  density remains finite.

The type parameter `T` is the element type of the stored value
(`Float64`, `ForwardDiff.Dual`, `Complex{Float64}`, …) and propagates
through every scheme that consumes the amount. Each amount carries its
own `T`: a `RVE` may hold a `Float64` fraction next to a
`ForwardDiff.Dual` one, the element types being promoted where the
values actually meet (see [`RVE`](@ref)).
"""
abstract type AbstractAmount{T <: Number} end

"""
    VolumeFraction(f) <: AbstractAmount

Volume fraction of a solid inclusion (or of the matrix).
"""
struct VolumeFraction{T <: Number} <: AbstractAmount{T}
    value::T
end

"""
    CrackDensity(ε) <: AbstractAmount

Budiansky-O'Connell crack density of a population of flat cracks.
"""
struct CrackDensity{T <: Number} <: AbstractAmount{T}
    value::T
end

"""
    amount_value(a::AbstractAmount) -> Number

Return the scalar value carried by an `AbstractAmount`.
"""
amount_value(a::AbstractAmount) = a.value

"""
    scale_by_amount(a::AbstractAmount, X) -> typeof(a.value * X)

Multiply `X` by the amount's value **through a function barrier**.

`amounts` is heterogeneous, so `rve.amounts[name]` is statically an
`AbstractAmount` and `amount_value(a)` alone would return `Any` — one
boxed scalar, then a second dynamic call for the product. Passing `a`
itself lets the dispatch land on its concrete type (`VolumeFraction{F}`),
inside which `a.value::F` is inferred and the product is specialized: one
dynamic dispatch instead of a boxing plus two, on a path every scheme
walks once per phase.
"""
scale_by_amount(a::AbstractAmount, X) = a.value * X
#
# Only the `f * X` shape is worth a barrier. The crack paths pass the amount
# as the last argument of a `delta_*` helper; wrapping those in a varargs
# barrier was measured to *cost* ~1.4 KB per call on `mt.crack.penny` (the
# varargs tuple) for no gain, since those cases run at tens of microseconds.
# They keep the plain `amount_value(a)`.

Base.eltype(::Type{<:AbstractAmount{T}}) where {T} = T
Base.eltype(a::AbstractAmount) = eltype(typeof(a))

"""
    _sums_to_unit(a::AbstractAmount) -> Bool

Whether the amount counts towards the matrix-fraction complement
`f_matrix = 1 - Σ_phase _sums_to_unit·value`. `true` for
[`VolumeFraction`](@ref), `false` for [`CrackDensity`](@ref).
"""
_sums_to_unit(::VolumeFraction) = true
_sums_to_unit(::CrackDensity) = false

# =============================================================================
#  Symmetrize: orientation-distribution projection of a phase's contribution
# =============================================================================

"""
    AbstractSymmetrize

Specifies how a phase's *localization tensor* (and the derived stiffness /
compliance / conductivity / resistivity contributions) is averaged over an
orientation distribution before being used in the homogenization formula.

Three concrete subtypes are shipped :

- [`NoSymmetrize`](@ref) (default) — keep the contribution as computed for
  the single-orientation inclusion stored in the phase.
- [`IsoSymmetrize`](@ref) — average over **all** rotations (uniform spatial
  distribution of orientations) ; produces an isotropic projection.
- [`TISymmetrize`](@ref) — average over rotations around a specified axis
  (uniaxial uniform distribution) ; produces a transversely-isotropic
  projection.

This mirrors C++ ECHOES's `symmetrize=[ISO]` / `symmetrize=[TI]` keyword on
`ellipsoid()`, but moved to the *RVE* side (just like volume fractions) :
the same inclusion type can be re-used in different RVEs with different
distribution assumptions, and a single inclusion remains usable for
localization-tensor calculations without any RVE.
"""
abstract type AbstractSymmetrize end

"""
    NoSymmetrize() <: AbstractSymmetrize

Default. The localization tensor is used as computed for the single
orientation defined by the inclusion's basis.
"""
struct NoSymmetrize <: AbstractSymmetrize end

"""
    IsoSymmetrize() <: AbstractSymmetrize

The localization tensor is averaged over a *uniform spatial distribution*
of orientations, equivalent to projecting onto the isotropic basis
`(J, K_proj)` for 4th-order tensors and onto the spherical part for
2nd-order tensors. Produces an isotropic phase contribution regardless of
the inclusion's actual shape.
"""
struct IsoSymmetrize <: AbstractSymmetrize end

"""
    TISymmetrize(axis = (0, 0, 1); matrix_projection = :iso) <: AbstractSymmetrize

The localization tensor is averaged **exactly** over rotations about `axis`
(uniaxial uniform distribution).  The average preserves the full
axially-invariant structure — including the non-major-symmetric components
of concentration tensors — and returns a `TensND.TensTI{4,T,8}`
(resp. `TensTI{2,T,3}` at 2nd order).

`matrix_projection` controls how the *reference medium* is pre-projected
before the localization tensor of this phase is computed :

- `:iso` (default) — project the reference to its isotropic average.  Every
  inclusion orientation then has an analytical, ForwardDiff-compatible Hill
  branch.  This is an approximation whenever the reference is not isotropic;
  it is exact at the isotropic fixed point of a self-consistent iteration.
- `:none` — use the reference as is.  Non-coaxial anisotropic references
  route through the general-anisotropy Hill branch (`NestedQuadGK`,
  ForwardDiff-compatible but quadrature-priced).
- `:ti` — project the reference to its best-fit TI form about `axis`.
  Only valid when the phase's inclusions are coaxial with `axis`
  (the TI-coaxial analytical Hill branch applies).
"""
struct TISymmetrize{T <: Number} <: AbstractSymmetrize
    axis::NTuple{3, T}
    matrix_projection::Symbol
    function TISymmetrize(axis::NTuple{3, T}, matrix_projection::Symbol) where {T <: Number}
        matrix_projection in (:iso, :none, :ti) || throw(
            ArgumentError(
                "matrix_projection must be :iso, :none or :ti; got :$(matrix_projection)"
            )
        )
        return new{T}(axis, matrix_projection)
    end
end
TISymmetrize(axis::NTuple{3, <:Number}; matrix_projection::Symbol = :iso) =
    TISymmetrize(axis, matrix_projection)
TISymmetrize(; matrix_projection::Symbol = :iso) =
    TISymmetrize((0.0, 0.0, 1.0), matrix_projection)
TISymmetrize(axis::AbstractVector; matrix_projection::Symbol = :iso) =
    TISymmetrize(NTuple{3}(Tuple(axis)), matrix_projection)

# Coercer for kwargs : accept a Symbol shortcut, an `AbstractSymmetrize`, or
# nothing (no projection).
_to_symmetrize(::Nothing) = NoSymmetrize()
_to_symmetrize(s::AbstractSymmetrize) = s
function _to_symmetrize(s::Symbol)
    s === :none && return NoSymmetrize()
    s === :iso  && return IsoSymmetrize()
    s === :ISO  && return IsoSymmetrize()
    s === :ti   && return TISymmetrize()
    s === :TI   && return TISymmetrize()
    throw(ArgumentError("unknown symmetrize Symbol :$(s); expected :none, :iso or :ti (or pass an AbstractSymmetrize instance for non-default axis)"))
end

# =============================================================================
#  Distribution shape: PCW / Maxwell outer-envelope descriptor
# =============================================================================

"""
    AbstractDistributionShape

Supertype for the *outer envelope* of the phase distribution used by the
[`Maxwell`](@ref) and [`PonteCastanedaWillis`](@ref) schemes.

Currently a single concrete subtype is shipped:

- [`UniformDistribution`](@ref) — a single shape applied to every
  inclusion phase (Maxwell 1873 ; Ponte-Castañeda & Willis 1995).

Future extension (placeholder, *not* implemented in this PR): a
`PairwiseDistribution` carrying a per-pair `(i, j) ↦ shape` mapping
([Willis 1982](@cite willis1982)).  Adding it will only require a new
concrete subtype + matching `_evaluate(rve, ::Maxwell|::PonteCastanedaWillis, …)`
methods — no public-API change.
"""
abstract type AbstractDistributionShape end

"""
    UniformDistribution(shape::AbstractInclusion) <: AbstractDistributionShape

Single distribution shape applied to every inclusion phase. The default
constructor `UniformDistribution()` returns a unit sphere (isotropic
distribution, recovers Mori-Tanaka in the limit `P_d = P_inc`).
"""
struct UniformDistribution{S <: AbstractInclusion} <: AbstractDistributionShape
    shape::S
end
UniformDistribution() = UniformDistribution(Ellipsoid(1.0))

"""
    distribution_shape_of(d::UniformDistribution) -> AbstractInclusion

Return the inclusion describing the (single) distribution envelope.
"""
distribution_shape_of(d::UniformDistribution) = d.shape

"""
    _to_distribution_shape(x) -> AbstractDistributionShape

Coerce `x` to a concrete `AbstractDistributionShape`. Accepts:

- `nothing` → `UniformDistribution(Ellipsoid(1.0))` (default sphere),
- an `AbstractInclusion` → wrapped as `UniformDistribution(x)`,
- an `AbstractDistributionShape` → passed through.
"""
_to_distribution_shape(::Nothing) = UniformDistribution()
_to_distribution_shape(s::AbstractInclusion) = UniformDistribution(s)
_to_distribution_shape(s::AbstractDistributionShape) = s

# =============================================================================
#  Phase: geometry + material properties
# =============================================================================

"""
    Phase(geometry::AbstractInclusion, properties::Dict{Symbol,<:AbstractTens})

A single phase of a [`RVE`](@ref): one inclusion *geometry* (ellipsoid,
crack, …) together with one or several material *property tensors*
indexed by symbol (`:C` for stiffness, `:K` for conductivity, …).

The geometry is field-typed `AbstractInclusion` (rather than parametric
`Phase{I}`) so that a heterogeneous RVE mixing ellipsoids and cracks can
be stored in a single `Dict{Symbol,Phase}` without losing information at
construction time. Specialization happens at the dispatch site
(`hill_tensor(phase.geometry, …)`, `cod_tensor(phase.geometry, …)`).
"""
mutable struct Phase
    geometry::AbstractInclusion
    properties::Dict{Symbol, Any}
end

Phase(geometry::AbstractInclusion, properties::AbstractDict) =
    Phase(geometry, Dict{Symbol, Any}(properties...))

# =============================================================================
#  RVE: ordered collection of phases + matrix tag + distribution shape
# =============================================================================

"""
    RVE{T<:Number, S<:Union{Nothing,AbstractDistributionShape}}

Multi-phase representative volume element. Fields:

- `matrix_name::Symbol` — name of the phase that plays the role of the
  matrix (its amount is implicit, computed as `1 - Σ_inc f_inc`).
- `phase_names::Vector{Symbol}` — phases in insertion order (the matrix
  is the first entry).
- `phases::Dict{Symbol,Phase}` — geometry + properties of each phase.
- `amounts::Dict{Symbol,AbstractAmount}` — volume fraction or crack
  density of each non-matrix phase. Each entry keeps its own element
  type. The matrix entry, if present, is ignored when computing
  `matrix_volume_fraction`.
- `distribution_shape::S` — outer envelope used by Maxwell / PCW;
  defaults to a unit sphere wrapped in [`UniformDistribution`](@ref).
- `f_matrix` — cached `1 - Σ f_inc`, maintained by [`add_phase!`](@ref)
  and read by [`matrix_volume_fraction`](@ref). Do not write `amounts`
  directly: that would leave the cache stale.

`T` is the *declared* amount element type, i.e. a **floor for
promotion**, not a constraint: it seeds `zero`/`one` for an RVE whose
amounts are all narrower (an `Int` fraction still yields a `Float64`
matrix fraction under the default `T = Float64`), and any amount handed
to [`add_phase!`](@ref) that is *wider* than `T` is stored as such
rather than converted down. Amounts, moduli and geometries therefore
each carry their own element type, promoted only where the values meet:

```julia
rve = RVE(:M)                                          # nothing to declare
add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_complex))    # complex moduli
add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => C_complex);
           fraction = 0.3)                             # real fraction
add_phase!(rve, :J, Ellipsoid(1.0), Dict(:C => C1);
           fraction = dual_x)                          # Dual fraction
```

`eltype(rve)` reports the *effective* element type (the promotion of the
floor with every stored amount); `eltype(RVE{T,S})` reports the declared
floor `T`.

Construction is two-step:

```julia
rve = RVE(:M; distribution_shape = nothing)   # or RVE{ComplexF64}(:M)
add_matrix!(rve, ellipsoid_matrix, Dict(:C => C0))
add_phase!(rve, :I1, ellipsoid_inc, Dict(:C => C1); fraction = 0.2)
add_phase!(rve, :CRACK, penny_crack, Dict(:C => C0); density = 0.05)
```

See also [`add_matrix!`](@ref), [`add_phase!`](@ref),
[`matrix_volume_fraction`](@ref), [`validate_rve`](@ref).
"""
mutable struct RVE{T <: Number, S <: Union{Nothing, AbstractDistributionShape}} <:
    AbstractHomogenizationCell
    matrix_name::Symbol
    phase_names::Vector{Symbol}
    phases::Dict{Symbol, Phase}
    amounts::Dict{Symbol, AbstractAmount}
    symmetrize::Dict{Symbol, AbstractSymmetrize}
    distribution_shape::S
    # Cached `1 - Σ f_inc`. Heterogeneous amounts make the sum's element type
    # a runtime property, so recomputing it per call costs three boxed
    # temporaries — measurable on the sub-µs schemes, which call it once per
    # `homogenize`. `add_phase!` (the only writer of `amounts` on a live RVE)
    # refreshes it with the very same loop, so the arithmetic and the dict
    # iteration order are unchanged and the value stays bit-identical.
    f_matrix::Any
end

"""
    RVE(matrix_name::Symbol; T = Float64, distribution_shape = nothing)
    RVE{T}(matrix_name::Symbol; distribution_shape = nothing)

Construct an empty RVE. The matrix phase is referenced by `matrix_name`
but **not** added — call [`add_matrix!`](@ref) next.

The two forms are strictly equivalent; both are optional. `T` declares
the amount element-type *floor* (default `Float64`) and is only needed
to force a wider type on amounts that are themselves narrow — complex
moduli, `ForwardDiff.Dual` or symbolic amounts propagate on their own,
without declaration (see [`RVE`](@ref)).
"""
function RVE(
        matrix_name::Symbol;
        T::Type{<:Number} = Float64,
        distribution_shape = nothing
    )
    ds = _to_distribution_shape(distribution_shape)
    return RVE{T, typeof(ds)}(
        matrix_name,
        Symbol[],
        Dict{Symbol, Phase}(),
        Dict{Symbol, AbstractAmount}(),
        Dict{Symbol, AbstractSymmetrize}(),
        ds,
        one(T),
    )
end

RVE{T}(matrix_name::Symbol; distribution_shape = nothing) where {T <: Number} =
    RVE(matrix_name; T = T, distribution_shape = distribution_shape)

function RVE{T, S}(
        matrix_name::Symbol; distribution_shape = nothing
    ) where {T <: Number, S <: Union{Nothing, AbstractDistributionShape}}
    rve = RVE(matrix_name; T = T, distribution_shape = distribution_shape)
    rve isa RVE{T, S} || throw(
        ArgumentError(
            "distribution_shape yields $(typeof(rve.distribution_shape)), not the requested $S"
        )
    )
    return rve
end

"""
    _amount_promote(::Type{T}, v) -> Number

Store `v` as an amount value under the declared floor `T`: widened to
`promote_type(T, typeof(v))`, never narrowed down to `T`. This is what
lets a `Dual` or complex amount live in a plain `RVE(:M)`.
"""
_amount_promote(::Type{T}, v::Number) where {T <: Number} =
    convert(promote_type(T, typeof(v)), v)
# Non-`Number` scalars (symbolic backends that do not subtype `Number`)
# are stored verbatim rather than rejected.
_amount_promote(::Type{<:Number}, v) = v

"""
    eltype(rve::RVE) -> Type

Effective amount element type: the promotion of the declared floor
`T` with the element type of every stored amount. Use
`eltype(typeof(rve))` for the declared floor alone.
"""
Base.eltype(rve::RVE{T}) where {T} =
    mapfoldl(eltype, promote_type, values(rve.amounts); init = T)

"""
    promote_rve(rve, ::Type{T}) -> RVE

Return a copy of `rve` whose declared floor is `T` and whose amounts are
all converted to `promote_type(T, ·)`. Rarely needed — amounts promote
themselves where they are consumed — but useful to force an element type
on an RVE built elsewhere.
"""
function promote_rve(rve::RVE, ::Type{T}) where {T <: Number}
    new_amounts = Dict{Symbol, AbstractAmount}()
    for (k, a) in rve.amounts
        new_amounts[k] = _promote_amount(a, promote_type(T, eltype(a)))
    end
    return _rebuild_rve(rve; amounts = new_amounts, T = T)
end

# A single method: the `rve::RVE{T}` identity overload is never selected over
# this one (`Type{RVE{T}}` is itself a `UnionAll` in `S`), so the no-op case is
# handled by an explicit guard rather than by dispatch.
Base.convert(::Type{RVE{T}}, rve::RVE) where {T <: Number} =
    eltype(typeof(rve)) === T ? rve : promote_rve(rve, T)

# =============================================================================
#  Mutators
# =============================================================================

"""
    add_matrix!(rve, geometry, properties::AbstractDict; symmetrize = nothing)

Register the matrix phase. Must be called before any [`add_phase!`](@ref).
The matrix has no explicit amount (its volume fraction is implicit).

Pass a `symmetrize = :iso | :ti | TISymmetrize(axis) | NoSymmetrize()` kwarg
to declare an orientation-distribution projection of the matrix's
localization tensor (see [`AbstractSymmetrize`](@ref)).
"""
function add_matrix!(
        rve::RVE, geometry::AbstractInclusion, properties::AbstractDict;
        symmetrize = nothing
    )
    name = rve.matrix_name
    haskey(rve.phases, name) &&
        throw(ArgumentError("matrix phase :$(name) already registered"))
    rve.phases[name] = Phase(geometry, properties)
    pushfirst!(rve.phase_names, name)
    sym = _to_symmetrize(symmetrize)
    if !(sym isa NoSymmetrize)
        rve.symmetrize[name] = sym
    end
    return rve
end

"""
    add_phase!(rve, name::Symbol, geometry, properties::AbstractDict;
               fraction = nothing, density = nothing, symmetrize = nothing)

Register an inclusion phase with the given `geometry` and material
`properties`. Exactly one of `fraction` (for ellipsoidal inclusions and
solid inhomogeneities) or `density` (for cracks) must be supplied;
`fraction` produces a [`VolumeFraction`](@ref), `density` a
[`CrackDensity`](@ref).

Both `fraction` and `density` are stored under the RVE's declared
element-type floor `T`: widened to `promote_type(T, typeof(value))`,
never narrowed. A complex, `ForwardDiff.Dual` or symbolic amount is
therefore accepted by a plain `RVE(:M)`, and phases may carry amounts of
different element types.

The optional `symmetrize` kwarg declares an orientation-distribution
projection of this phase's localization tensor : `:iso` (uniform spatial
distribution → isotropic projection), `:ti` (uniaxial uniform around
z-axis), `TISymmetrize(axis)` (around an arbitrary axis), or pass an
explicit [`AbstractSymmetrize`](@ref) instance. The default
[`NoSymmetrize`](@ref) keeps the inclusion's actual single-orientation
tensor.
"""
function add_phase!(
        rve::RVE{T}, name::Symbol, geometry::AbstractInclusion,
        properties::AbstractDict;
        fraction = nothing, density = nothing,
        symmetrize = nothing
    ) where {T}
    name === rve.matrix_name &&
        throw(ArgumentError("name :$(name) is reserved for the matrix phase"))
    haskey(rve.phases, name) &&
        throw(ArgumentError("phase :$(name) is already registered"))
    (fraction === nothing) == (density === nothing) &&
        throw(ArgumentError("specify exactly one of `fraction=…` or `density=…`"))

    rve.phases[name] = Phase(geometry, properties)
    push!(rve.phase_names, name)
    rve.amounts[name] = if fraction !== nothing
        VolumeFraction(_amount_promote(T, fraction))
    else
        CrackDensity(_amount_promote(T, density))
    end
    rve.f_matrix = _compute_matrix_volume_fraction(rve.amounts, T)
    sym = _to_symmetrize(symmetrize)
    if !(sym isa NoSymmetrize)
        rve.symmetrize[name] = sym
    end
    return rve
end

# =============================================================================
#  In-place updates of an already-registered phase
#
#  `set_param` is the sanctioned way to change an amount: it is a pure lens,
#  returns a NEW RVE, and that is precisely what makes ForwardDiff work through
#  it. The price is a fresh `Dict` of amounts, a fresh `Dict` of phases and a
#  copy of `phase_names` per call — irrelevant for a sensitivity study, wasteful
#  in a loop that re-evaluates the same RVE at every Gauss point of every Newton
#  iteration of every time step.
#
#  The two setters below are the mutating counterpart for that loop. They are
#  NOT a replacement for `set_param`: they bypass `promote_rve`, so they do not
#  widen the RVE's declared element-type floor, and a `Dual` stored this way
#  will propagate but was never promoted with the rest.
# =============================================================================

"""
    set_amount!(rve::RVE, name::Symbol, value) -> RVE

Overwrite the amount of an already-registered phase **in place**, keeping its
kind ([`VolumeFraction`](@ref) or [`CrackDensity`](@ref)) and refreshing the
`f_matrix` cache when needed.

Intended for the inner loop of a Gauss-point constitutive law, where the same
RVE is re-evaluated thousands of times with a slightly different crack density
or porosity. Everywhere else — and always for anything differentiated — prefer
the immutable [`set_param`](@ref) lens.

Changing the *kind* of an amount is rejected: every scheme branches on
`a isa VolumeFraction`, so turning a crack density into a volume fraction
silently changes which formula the phase goes through. Register a different
phase instead.

```julia
set_amount!(rve, :CRACK, 0.12)     # new crack density
homogenize(rve, SelfConsistent())  # sees the new value
```

!!! note "Only volume fractions touch the cache"
    `f_matrix = 1 - Σ f_inc` sums [`VolumeFraction`](@ref) entries only, so a
    [`CrackDensity`](@ref) write cannot stale it — cracks carry no volume. The
    cache is still recomputed for a volume-fraction write, with the same loop
    [`add_phase!`](@ref) uses, so the value stays bit-identical to what a fresh
    RVE would hold.

See also [`set_geometry!`](@ref), [`set_param`](@ref), [`add_phase!`](@ref).
"""
function set_amount!(rve::RVE{T}, name::Symbol, value) where {T}
    haskey(rve.amounts, name) ||
        throw(ArgumentError("no phase named :$(name) carries an amount in this RVE"))
    old = rve.amounts[name]
    new = if old isa VolumeFraction
        VolumeFraction(_amount_promote(T, value))
    else
        CrackDensity(_amount_promote(T, value))
    end
    rve.amounts[name] = new
    # Only a volume fraction can move the implicit matrix fraction.
    if _sums_to_unit(new)
        rve.f_matrix = _compute_matrix_volume_fraction(rve.amounts, T)
    end
    return rve
end

"""
    set_geometry!(rve::RVE, name::Symbol, geometry::AbstractInclusion) -> RVE

Replace the geometry of an already-registered phase **in place**.

[`Phase`](@ref) is mutable and nothing on the RVE is cached from a geometry, so
this invalidates no state. As with [`set_amount!`](@ref), it is the loop-friendly
counterpart of the immutable [`set_param`](@ref) lens and bypasses element-type
promotion.
"""
function set_geometry!(rve::RVE, name::Symbol, geometry::AbstractInclusion)
    haskey(rve.phases, name) ||
        throw(ArgumentError("no phase named :$(name) in RVE"))
    rve.phases[name].geometry = geometry
    return rve
end

"""
    set_property!(rve::RVE, name::Symbol, key::Symbol, value) -> RVE

Set (or add) the property `key` of an already-registered phase **in place**.

The typical use is toggling a crack family between open and closed through its
`:K_interface` spring stiffness without rebuilding the RVE. Nothing is cached
from a property, so this invalidates no state.
"""
function set_property!(rve::RVE, name::Symbol, key::Symbol, value)
    haskey(rve.phases, name) ||
        throw(ArgumentError("no phase named :$(name) in RVE"))
    rve.phases[name].properties[key] = value
    return rve
end

# =============================================================================
#  Accessors
# =============================================================================

"""
    matrix_phase(rve::RVE) -> Phase

Return the matrix `Phase`. Errors if the matrix has not been registered
(call [`add_matrix!`](@ref) first).
"""
function matrix_phase(rve::RVE)
    haskey(rve.phases, rve.matrix_name) ||
        throw(ArgumentError("matrix phase :$(rve.matrix_name) is not yet registered — call add_matrix! first"))
    return rve.phases[rve.matrix_name]
end

"""
    inclusion_phase_names(rve::RVE) -> Vector{Symbol}

Names of the non-matrix phases in insertion order.
"""
inclusion_phase_names(rve::RVE) =
    Symbol[n for n in rve.phase_names if n != rve.matrix_name]

"""
    phase_property(rve, name::Symbol, key::Symbol) -> AbstractTens

Return the property tensor `key` (e.g. `:C`, `:K`) of the phase named
`name`.

If the stored value is a [`Homogenized`](@ref) — a declaratively nested cell
— it is resolved here, memoized for the duration of the enclosing
`homogenize` call, so that every scheme kernel sees a plain tensor without
knowing that nesting exists. Use [`phase_property_raw`](@ref) to inspect the
stored value without triggering that resolution.
"""
function phase_property(rve::RVE, name::Symbol, key::Symbol)
    haskey(rve.phases, name) ||
        throw(ArgumentError("no phase named :$(name) in RVE"))
    p = rve.phases[name]
    haskey(p.properties, key) ||
        throw(ArgumentError("phase :$(name) does not carry property :$(key)"))
    return resolve_property(p.properties[key], key)
end

"""
    phase_property_raw(rve, name::Symbol, key::Symbol)

The value stored under `key` on phase `name`, **without** resolving a
[`Homogenized`](@ref). Type inspections must use this — resolving would run a
whole inner homogenization just to look at a type.
"""
function phase_property_raw(rve::RVE, name::Symbol, key::Symbol)
    haskey(rve.phases, name) ||
        throw(ArgumentError("no phase named :$(name) in RVE"))
    p = rve.phases[name]
    haskey(p.properties, key) ||
        throw(ArgumentError("phase :$(name) does not carry property :$(key)"))
    return p.properties[key]
end

"""
    matrix_property(rve, key::Symbol) -> AbstractTens

Shortcut for `phase_property(rve, rve.matrix_name, key)`.
"""
matrix_property(rve::RVE, key::Symbol) = phase_property(rve, rve.matrix_name, key)

"""
    volume_fraction(rve, name::Symbol) -> Number

Volume fraction of phase `name`. Returns a zero of the phase's own
element type (promoted with the RVE floor) if the phase carries a
[`CrackDensity`](@ref) instead of a [`VolumeFraction`](@ref).
"""
function volume_fraction(rve::RVE{T}, name::Symbol) where {T}
    name === rve.matrix_name && return matrix_volume_fraction(rve)
    a = rve.amounts[name]
    return a isa VolumeFraction ? amount_value(a) : zero(promote_type(T, eltype(a)))
end

"""
    crack_density(rve, name::Symbol) -> Number

Crack density of phase `name`. Returns a zero of the phase's own element
type (promoted with the RVE floor) if the phase carries a
[`VolumeFraction`](@ref) instead of a [`CrackDensity`](@ref).
"""
function crack_density(rve::RVE{T}, name::Symbol) where {T}
    haskey(rve.amounts, name) || return zero(T)
    a = rve.amounts[name]
    return a isa CrackDensity ? amount_value(a) : zero(promote_type(T, eltype(a)))
end

"""
    phase_symmetrize(rve, name::Symbol) -> AbstractSymmetrize

Return the orientation-distribution projection declared for phase `name`.
Defaults to [`NoSymmetrize`](@ref) if none was set.
"""
phase_symmetrize(rve::RVE, name::Symbol) =
    get(rve.symmetrize, name, NoSymmetrize())

"""
    matrix_volume_fraction(rve::RVE) -> Number

Implicit matrix volume fraction `1 - Σ_inc f_inc` (only
[`VolumeFraction`](@ref) entries contribute; [`CrackDensity`](@ref)
entries are ignored).

The value is cached on the RVE and refreshed by [`add_phase!`](@ref);
this is a read of the `f_matrix` field, not a recomputation.

The accumulator behind it is seeded with `zero(T)` (the declared floor)
and then promoted by the stored amounts, so the result carries the
effective element type — `Dual` as soon as one fraction is `Dual`,
`Complex` as soon as one is complex. The unit is taken from the
accumulator rather than from `T`, so a symbolic RVE yields `1 - f` and
not `1.0 - f`.
"""
matrix_volume_fraction(rve::RVE) = rve.f_matrix

# Recomputation behind the `f_matrix` cache. Only `add_phase!` and
# `_rebuild_rve` call it — never a scheme.
function _compute_matrix_volume_fraction(amounts::AbstractDict, ::Type{T}) where {T}
    f_inc = zero(T)
    for (_, a) in amounts
        if _sums_to_unit(a)
            f_inc = f_inc + amount_value(a)
        end
    end
    return one(f_inc) - f_inc
end

# =============================================================================
#  Validation
# =============================================================================

"""
    validate_rve(rve::RVE)

Sanity-check the RVE: matrix registered, all amounts non-negative, sum
of `VolumeFraction` entries ≤ 1.  Throws `ArgumentError` on
hard failures; emits `@warn` if `f_inc > 1` (non-physical RVE — useful
for symbolic / Dual exploration but flagged).
"""
function validate_rve(rve::RVE)
    haskey(rve.phases, rve.matrix_name) ||
        throw(ArgumentError("RVE has no matrix phase :$(rve.matrix_name); call add_matrix! first"))
    for (name, a) in rve.amounts
        v = amount_value(a)
        if v isa Real && v < 0
            throw(ArgumentError("phase :$(name) has negative amount $(v)"))
        end
    end
    fm = matrix_volume_fraction(rve)
    if fm isa Real && fm < 0
        @warn "RVE has matrix volume fraction $(fm) < 0 — total inclusion volume fraction exceeds 1"
    end
    return rve
end

# =============================================================================
#  The `AbstractHomogenizationCell` contract (declared in `Core/cells.jl`)
# =============================================================================
#
# `validate_rve` is left strictly as it was — including its requirement of a
# registered matrix phase, which is precisely the part that does NOT
# generalize to a matrix-free cell. That requirement is why the indirection
# below exists rather than a relaxed `validate_rve`.

validate_cell(rve::RVE) = validate_rve(rve)

cell_member_names(rve::RVE) = rve.phase_names

cell_container_property(rve::RVE, name::Symbol, key::Symbol) =
    phase_property_raw(rve, name, key)

function cell_set_property(rve::RVE, name::Symbol, key::Symbol, value)
    haskey(rve.phases, name) ||
        throw(ArgumentError("no phase named :$(name) in RVE"))
    ph = rve.phases[name]
    new_props = Dict{Symbol, Any}(ph.properties)
    new_props[key] = value
    new_phases = Dict{Symbol, Phase}(rve.phases)
    new_phases[name] = Phase(ph.geometry, new_props)
    return _rebuild_rve(rve; phases = new_phases)
end

# =============================================================================
#  Pretty printing
# =============================================================================

function Base.show(io::IO, ::MIME"text/plain", rve::RVE{T, S}) where {T, S}
    Te = eltype(rve)
    tag = Te === T ? "$T" : "$T → $Te"
    println(io, "RVE{$tag} with ", length(rve.phase_names), " phase(s)")
    println(io, "  matrix : :$(rve.matrix_name)")
    for name in rve.phase_names
        name === rve.matrix_name && continue
        a = rve.amounts[name]
        kind = a isa VolumeFraction ? "f" : "ε"
        println(io, "  inclusion : :$(name)   $kind = $(amount_value(a))")
    end
    print(io, "  distribution_shape : ", rve.distribution_shape)
    return
end

Base.show(io::IO, rve::RVE{T}) where {T} =
    print(io, "RVE{$T}(:$(rve.matrix_name), $(length(rve.phase_names)) phases)")
