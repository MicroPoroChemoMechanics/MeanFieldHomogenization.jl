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
#  ([willis1982](@cite)) without breaking the public API.
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

"""
    Remainder() <: AbstractAmount

Marker amount of the phase whose volume fraction is *derived* rather than
declared — the one that takes up whatever the others leave.

It carries no value: under [`ComplementFraction`](@ref) the value is
`1 - Σ f`, a property of the RVE and not of the phase. Read the resolved
number with [`volume_fraction`](@ref); `amount_value` deliberately throws,
so a loop that assumed every amount carries a number fails by name instead
of silently skipping the phase.

Its element type is `Union{}`, which promotes away: an RVE holding a
`Remainder` next to `Float64` and `ForwardDiff.Dual` fractions has exactly
the `eltype` it would have without it.
"""
struct Remainder <: AbstractAmount{Union{}} end

_sums_to_unit(::Remainder) = false      # it *is* the complement, it feeds no sum

amount_value(::Remainder) = throw(
    ArgumentError(
        "a Remainder amount carries no declared value — it is the volume left " *
            "over by the other phases. Read the resolved number with " *
            "`volume_fraction(rve, name)`."
    )
)

scale_by_amount(::Remainder, X) = throw(
    ArgumentError(
        "a Remainder amount cannot scale a tensor on its own; weight by " *
            "`volume_fraction(rve, name)` instead."
    )
)

# =============================================================================
#  Fraction closure: how the declared volume fractions are made to sum to one
# =============================================================================

"""
    AbstractFractionClosure

How an [`RVE`](@ref) turns the volume fractions the caller *declared* into the
fractions the schemes *use*. Three concrete policies ship:

- [`ComplementFraction`](@ref) — one phase, declared `fraction = :rest`,
  absorbs `1 - Σ f`;
- [`RescaledFractions`](@ref) — every declared fraction is divided by their
  sum, so relative proportions may be given;
- [`StrictFractions`](@ref) — the declared fractions must already sum to one.

The policy is inferred when it is not given: `ComplementFraction()` as soon as
a phase is declared with `fraction = :rest`, `StrictFractions()` otherwise.

[`CrackDensity`](@ref) never takes part. A flat crack has no volume, so it is
outside the unit-sum constraint, is never renormalized, and is never absorbed
into a complement.
"""
abstract type AbstractFractionClosure end

"""
    StrictFractions(; atol = 1e-10) <: AbstractFractionClosure

The declared volume fractions must sum to one within `atol`; anything else is
an error naming the sum. The default policy for an RVE in which no phase is
declared with `fraction = :rest` — a polycrystal, a granular aggregate, any
microstructure in which no phase is a leftover.
"""
struct StrictFractions{T <: Real} <: AbstractFractionClosure
    atol::T
end
StrictFractions(; atol::Real = 1.0e-10) = StrictFractions(atol)

"""
    ComplementFraction(; on_negative = :warn) <: AbstractFractionClosure

The single phase declared `fraction = :rest` takes the volume the others leave,
`1 - Σ f`.

`on_negative` says what happens when that complement comes out negative, i.e.
when the declared inclusion fractions already exceed one: `:warn` (default,
the historical behavior — a non-physical RVE is still useful for symbolic or
`ForwardDiff` exploration) or `:error`.
"""
struct ComplementFraction <: AbstractFractionClosure
    on_negative::Symbol
    function ComplementFraction(on_negative::Symbol)
        on_negative in (:warn, :error) || throw(
            ArgumentError(
                "ComplementFraction.on_negative must be :warn or :error; " *
                    "got :$(on_negative)"
            )
        )
        return new(on_negative)
    end
end
ComplementFraction(; on_negative::Symbol = :warn) = ComplementFraction(on_negative)

"""
    RescaledFractions() <: AbstractFractionClosure

Every declared volume fraction is divided by their sum, so that only their
*ratios* matter: `2, 3, 5` and `0.2, 0.3, 0.5` describe the same RVE.

No phase may then be declared `fraction = :rest` — there is no complement left
to absorb. The rescaling is a plain division, so it differentiates: under this
policy `∂C/∂f_i` is the derivative along the normalized simplex, and raising
one fraction lowers the others.
"""
struct RescaledFractions <: AbstractFractionClosure end

# Coercer for the `closure` kwarg: `nothing` means "infer at declaration time".
_to_closure(::Nothing) = nothing
_to_closure(c::AbstractFractionClosure) = c
function _to_closure(s::Symbol)
    s === :strict     && return StrictFractions()
    s === :complement && return ComplementFraction()
    s === :rest       && return ComplementFraction()
    s === :rescale    && return RescaledFractions()
    s === :normalize  && return RescaledFractions()
    throw(
        ArgumentError(
            "unknown closure Symbol :$(s); expected :strict, :complement or " *
                ":rescale (or pass an AbstractFractionClosure for non-default options)"
        )
    )
end

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
    TISymmetrize(axis = (0, 0, 1); reference_projection = :iso) <: AbstractSymmetrize

The localization tensor is averaged **exactly** over rotations about `axis`
(uniaxial uniform distribution).  The average preserves the full
axially-invariant structure — including the non-major-symmetric components
of concentration tensors — and returns a `TensND.TensTI{4,T,8}`
(resp. `TensTI{2,T,3}` at 2nd order).

`reference_projection` controls how the *reference medium* is pre-projected
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
    reference_projection::Symbol
    function TISymmetrize(axis::NTuple{3, T}, reference_projection::Symbol) where {T <: Number}
        reference_projection in (:iso, :none, :ti) || throw(
            ArgumentError(
                "reference_projection must be :iso, :none or :ti; got :$(reference_projection)"
            )
        )
        return new{T}(axis, reference_projection)
    end
end
TISymmetrize(axis::NTuple{3, <:Number}; reference_projection::Symbol = :iso) =
    TISymmetrize(axis, reference_projection)
TISymmetrize(; reference_projection::Symbol = :iso) =
    TISymmetrize((0.0, 0.0, 1.0), reference_projection)
TISymmetrize(axis::AbstractVector; reference_projection::Symbol = :iso) =
    TISymmetrize(NTuple{3}(Tuple(axis)), reference_projection)

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
([willis1982](@cite)).  Adding it will only require a new
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
#  RVE: ordered collection of phases + fraction closure + distribution shape
# =============================================================================

"""
    RVE{T<:Number, S<:Union{Nothing,AbstractDistributionShape}}

Multi-phase representative volume element — a **description of a
microstructure**, and nothing more. No phase is singled out: which one, if any,
plays the reference medium of a homogenization scheme is stated on that scheme
(see [`matrix_name`](@ref)), because the same RVE is a matrix/inclusion
composite under Mori-Tanaka and a matrix-free aggregate under the
self-consistent scheme.

Fields:

- `phase_names::Vector{Symbol}` — phases in declaration order.
- `phases::Dict{Symbol,Phase}` — geometry + properties of each phase.
- `amounts::Dict{Symbol,AbstractAmount}` — [`VolumeFraction`](@ref),
  [`CrackDensity`](@ref) or [`Remainder`](@ref) of each phase. Each entry
  keeps its own element type.
- `distribution_shape::S` — outer envelope used by Maxwell / PCW;
  defaults to a unit sphere wrapped in [`UniformDistribution`](@ref).
- `closure` — the [`AbstractFractionClosure`](@ref) turning the declared
  fractions into the ones the schemes use; `nothing` until inferred.
- `rest_name` — the phase declared `fraction = :rest`, if any.
- `f_sum` — cached `Σ f`, maintained by [`add_phase!`](@ref) and
  [`set_amount!`](@ref). Do not write `amounts` directly: that would leave
  the cache stale.

`T` is the *declared* amount element type, i.e. a **floor for
promotion**, not a constraint: it seeds `zero`/`one` for an RVE whose
amounts are all narrower (an `Int` fraction still yields a `Float64`
matrix fraction under the default `T = Float64`), and any amount handed
to [`add_phase!`](@ref) that is *wider* than `T` is stored as such
rather than converted down. Amounts, moduli and geometries therefore
each carry their own element type, promoted only where the values meet:

```julia
rve = RVE()                                          # nothing to declare
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_complex); fraction = :rest)    # complex moduli
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
rve = RVE(; distribution_shape = nothing)   # or RVE{ComplexF64}()
add_phase!(rve, :M, ellipsoid_matrix, Dict(:C => C0); fraction = :rest)
add_phase!(rve, :I1, ellipsoid_inc, Dict(:C => C1); fraction = 0.2)
add_phase!(rve, :CRACK, penny_crack, Dict(:C => C0); density = 0.05)
```

See also [`add_phase!`](@ref), [`volume_fraction`](@ref),
[`AbstractFractionClosure`](@ref), [`validate_rve`](@ref).
"""
mutable struct RVE{T <: Number, S <: Union{Nothing, AbstractDistributionShape}} <:
    AbstractHomogenizationCell
    phase_names::Vector{Symbol}
    phases::Dict{Symbol, Phase}
    amounts::Dict{Symbol, AbstractAmount}
    symmetrize::Dict{Symbol, AbstractSymmetrize}
    distribution_shape::S
    # `nothing` means "not yet decided": the policy is inferred at declaration
    # time (see `AbstractFractionClosure`) and frozen from then on.
    closure::Union{Nothing, AbstractFractionClosure}
    # Derived index, not a role: the single phase carrying a `Remainder`, if
    # any. Kept so the invariant "at most one" is O(1) to enforce and so the
    # hot accessors need no dict scan.
    rest_name::Union{Nothing, Symbol}
    # Cached `Σ f` over the amounts that take part in the unit sum.
    # Heterogeneous amounts make the sum's element type a runtime property, so
    # recomputing it per call costs three boxed temporaries — measurable on the
    # sub-µs schemes, which read it once per `homogenize`. `add_phase!` (the
    # only writer of `amounts` on a live RVE) refreshes it with the very same
    # loop, so the arithmetic and the dict iteration order are unchanged.
    #
    # It stores Σ, not `1 - Σ`: `ComplementFraction` wants `1 - Σ`, `Rescale`
    # and `Strict` want Σ, and `one(f_sum) - f_sum` reproduces the historical
    # value bit for bit — the unit is taken from the accumulator, so a symbolic
    # RVE still yields `1 - f` and not `1.0 - f`.
    f_sum::Any
end

"""
    RVE(; T = Float64, distribution_shape = nothing, closure = nothing)
    RVE{T}(; distribution_shape = nothing, closure = nothing)

Construct an empty RVE, then declare every phase with [`add_phase!`](@ref) —
including the one taking up the volume complement, `fraction = :rest`:

```julia
rve = RVE()
add_phase!(rve, :cem, Ellipsoid(1.0), Dict(:C => C_cem); fraction = :rest)
add_phase!(rve, :agg, Ellipsoid(1.0), Dict(:C => C_agg); fraction = 0.4)

homogenize(rve, MoriTanaka(:cem), :C)   # :cem is the reference medium
homogenize(rve, SelfConsistent(), :C)   # no phase is
```

`closure` selects the [`AbstractFractionClosure`](@ref); left unset it is
inferred. The two constructor forms are strictly equivalent; both are
optional. `T` declares
the amount element-type *floor* (default `Float64`) and is only needed
to force a wider type on amounts that are themselves narrow — complex
moduli, `ForwardDiff.Dual` or symbolic amounts propagate on their own,
without declaration (see [`RVE`](@ref)).
"""
function RVE(;
        T::Type{<:Number} = Float64,
        distribution_shape = nothing,
        closure = nothing
    )
    ds = _to_distribution_shape(distribution_shape)
    return RVE{T, typeof(ds)}(
        Symbol[],
        Dict{Symbol, Phase}(),
        Dict{Symbol, AbstractAmount}(),
        Dict{Symbol, AbstractSymmetrize}(),
        ds,
        _to_closure(closure),
        nothing,
        zero(T),
    )
end

RVE{T}(; distribution_shape = nothing, closure = nothing) where {T <: Number} =
    RVE(; T = T, distribution_shape = distribution_shape, closure = closure)

function RVE{T, S}(
        ; distribution_shape = nothing, closure = nothing
    ) where {T <: Number, S <: Union{Nothing, AbstractDistributionShape}}
    rve = RVE(; T = T, distribution_shape = distribution_shape, closure = closure)
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
lets a `Dual` or complex amount live in a plain `RVE()`.
"""
_amount_promote(::Type{T}, v::Number) where {T <: Number} =
    convert(promote_type(T, typeof(v)), v)
# Non-`Number` scalars (symbolic backends that do not subtype `Number`)
# are stored verbatim rather than rejected.
_amount_promote(::Type{<:Number}, v) = v

"""
    _to_amount(::Type{T}, fraction, density) -> AbstractAmount

Turn the `fraction=` / `density=` kwargs of [`add_phase!`](@ref) into a stored
amount. `fraction = :rest` (or `Remainder()`) declares the phase that takes up
whatever the others leave — see [`AbstractFractionClosure`](@ref).
"""
_to_amount(::Type{T}, ::Nothing, density) where {T <: Number} =
    CrackDensity(_amount_promote(T, density))
_to_amount(::Type{<:Number}, ::Remainder, ::Nothing) = Remainder()
function _to_amount(::Type{<:Number}, fraction::Symbol, ::Nothing)
    fraction === :rest && return Remainder()
    throw(
        ArgumentError(
            "unknown fraction Symbol :$(fraction); expected :rest, or a number"
        )
    )
end
_to_amount(::Type{T}, fraction, ::Nothing) where {T <: Number} =
    VolumeFraction(_amount_promote(T, fraction))

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
    add_phase!(rve, :M, geometry, properties::AbstractDict; fraction = :rest, symmetrize = nothing)

Register the matrix phase. Must be called before any [`add_phase!`](@ref).
The matrix has no explicit amount (its volume fraction is implicit).

Pass a `symmetrize = :iso | :ti | TISymmetrize(axis) | NoSymmetrize()` kwarg
to declare an orientation-distribution projection of the matrix's
localization tensor (see [`AbstractSymmetrize`](@ref)).
"""

# Register `name` as the phase absorbing the volume complement, enforcing the
# two invariants that make the closure well posed: at most one such phase, and
# not under a policy that leaves no complement to absorb.
function _claim_remainder!(rve::RVE, name::Symbol)
    rve.rest_name === nothing || throw(
        ArgumentError(
            "phase :$(rve.rest_name) already absorbs the volume complement, and an " *
                "RVE has at most one. Give :$(name) an explicit fraction, or build " *
                "the RVE with `closure = :rescale` to renormalize relative proportions."
        )
    )
    rve.closure isa RescaledFractions && throw(
        ArgumentError(
            "closure = RescaledFractions() rescales the declared fractions to unit " *
                "sum, so there is no complement for :$(name) to absorb; drop " *
                "`fraction = :rest`."
        )
    )
    rve.closure === nothing && (rve.closure = ComplementFraction())
    rve.rest_name = name
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
therefore accepted by a plain `RVE()`, and phases may carry amounts of
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
    haskey(rve.phases, name) &&
        throw(ArgumentError("phase :$(name) is already registered"))
    (fraction === nothing) == (density === nothing) &&
        throw(ArgumentError("specify exactly one of `fraction=…` or `density=…`"))

    amount = _to_amount(T, fraction, density)
    amount isa Remainder && _claim_remainder!(rve, name)
    rve.phases[name] = Phase(geometry, properties)
    push!(rve.phase_names, name)
    rve.amounts[name] = amount
    rve.f_sum = _compute_fraction_sum(rve.amounts, T)
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
    old isa Remainder && throw(
        ArgumentError(
            "phase :$(name) takes up the volume complement and has no declared " *
                "amount to set; change an explicit fraction instead."
        )
    )
    new = if old isa VolumeFraction
        VolumeFraction(_amount_promote(T, value))
    else
        CrackDensity(_amount_promote(T, value))
    end
    rve.amounts[name] = new
    # Only a volume fraction can move the sum the closure is applied to.
    if _sums_to_unit(new)
        rve.f_sum = _compute_fraction_sum(rve.amounts, T)
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
    phase_names(rve::RVE) -> Vector{Symbol}

Names of **every** phase, in declaration order. This is what a scheme that
distinguishes no phase — the self-consistent family, the bounds — iterates
over; schemes built on a reference medium use
[`inclusion_phase_names`](@ref) instead.
"""
phase_names(rve::RVE) = rve.phase_names

"""
    inclusion_phase_names(rve::RVE, matrix::Symbol) -> Vector{Symbol}

Names of the phases other than `matrix`, in declaration order — the phases a
matrix-based scheme localizes in its reference medium.

The phase to exclude has to be named: the reference medium is a property of the
*scheme*, and an RVE holds no such designation of its own. Use
[`phase_names`](@ref) for the schemes that distinguish none.
"""
inclusion_phase_names(rve::RVE, matrix::Symbol) =
    Symbol[n for n in rve.phase_names if n != matrix]

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
    volume_fraction(rve, name::Symbol) -> Number

Volume fraction of phase `name`, **as the schemes see it**: the RVE's
[`AbstractFractionClosure`](@ref) has already been applied, so a
[`Remainder`](@ref) reads as `1 - Σ f` and, under
[`RescaledFractions`](@ref), a declared fraction reads as its renormalized
value. This is the one accessor a scheme should use.

The *declared* value — what a parameter lens round-trips — is
`get_param(rve, AmountParameter(name))` instead. The two differ under
`RescaledFractions`, and only there.

A [`CrackDensity`](@ref) phase reads as zero: a flat crack carries no volume.
"""
function volume_fraction(rve::RVE{T}, name::Symbol) where {T}
    haskey(rve.amounts, name) ||
        throw(ArgumentError("no phase named :$(name) in RVE"))
    a = rve.amounts[name]
    a isa CrackDensity && return zero(promote_type(T, eltype(a)))
    a isa Remainder && return remainder_volume_fraction(rve)
    return _closure_scale(_closure(rve), amount_value(a), rve.f_sum)
end

# Under `RescaledFractions` a declared fraction is divided by their sum; every
# other policy uses it as declared. A plain division, so it differentiates.
_closure_scale(::RescaledFractions, v, s) = v / s
_closure_scale(::AbstractFractionClosure, v, _s) = v

"""
    remainder_volume_fraction(rve::RVE) -> Number

The volume left over by the declared fractions, `1 - Σ f` — what the phase
declared `fraction = :rest` takes up.

[`CrackDensity`](@ref) entries do not contribute: a flat crack's volume
vanishes in the penny limit while its density stays finite.

This is a read of the cached `f_sum` field plus one subtraction, not a
recomputation. The unit comes from the accumulator rather than from the
declared floor, so a symbolic RVE yields `1 - f` and not `1.0 - f`.
"""
remainder_volume_fraction(rve::RVE) = one(rve.f_sum) - rve.f_sum

"""
    remainder_phase_name(rve::RVE) -> Union{Nothing, Symbol}

The phase declared `fraction = :rest`, or `nothing` when the RVE designates
none.
"""
remainder_phase_name(rve::RVE) = rve.rest_name

# The effective policy: `nothing` means none was declared and no phase claimed
# a complement, which is exactly the strict case.
_closure(rve::RVE) = rve.closure === nothing ? StrictFractions() : rve.closure

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


# Recomputation behind the `f_sum` cache. Only `add_phase!`, `set_amount!` and
# `_rebuild_rve` call it — never a scheme.
function _compute_fraction_sum(amounts::AbstractDict, ::Type{T}) where {T}
    f = zero(T)
    for (_, a) in amounts
        if _sums_to_unit(a)
            f = f + amount_value(a)
        end
    end
    return f
end

# =============================================================================
#  Validation
# =============================================================================

"""
    validate_rve(rve::RVE)

Sanity-check the RVE: at least one phase, no negative amount, and the
fraction-closure policy satisfied. Throws `ArgumentError` on hard failures;
emits `@warn` for a negative complement under the default
`ComplementFraction(on_negative = :warn)` — a non-physical RVE stays useful
for symbolic or `Dual` exploration, but is flagged.
"""
function validate_rve(rve::RVE)
    isempty(rve.phase_names) &&
        throw(ArgumentError("RVE has no phase; call add_phase! first"))
    for (name, a) in rve.amounts
        a isa Remainder && continue      # no declared value to check
        v = amount_value(a)
        # `is_hard_numeric`, not `v isa Real`: `Symbolics.Num` subtypes `Real`
        # and answers no comparison — see the same guard in `add_layer!`.
        if is_hard_numeric(v) && v < 0
            throw(ArgumentError("phase :$(name) has negative amount $(v)"))
        end
    end
    _validate_closure(_closure(rve), rve)
    return rve
end

"""
    _validate_closure(closure, rve) -> RVE

Check the RVE against its fraction-closure policy. Each policy has exactly one
way to be ill-posed, and the message says how to reach a well-posed RVE rather
than only what is wrong.
"""
function _validate_closure(c::StrictFractions, rve::RVE)
    rve.rest_name === nothing || throw(
        ArgumentError(
            "closure = StrictFractions(), but phase :$(rve.rest_name) is declared " *
                "with `fraction = :rest`"
        )
    )
    s = rve.f_sum
    if is_hard_numeric(s) && abs(s - one(s)) > c.atol
        throw(
            ArgumentError(
                "volume fractions sum to $(s), not 1 (atol = $(c.atol)). Declare one " *
                    "phase with `fraction = :rest` to absorb the complement, or build " *
                    "the RVE with `closure = :rescale` to renormalize relative " *
                    "proportions."
            )
        )
    end
    return rve
end

function _validate_closure(c::ComplementFraction, rve::RVE)
    rve.rest_name === nothing && throw(
        ArgumentError(
            "closure = ComplementFraction(), but no phase is declared with " *
                "`fraction = :rest`"
        )
    )
    f = remainder_volume_fraction(rve)
    if is_hard_numeric(f) && f < 0
        msg = "phase :$(rve.rest_name) takes up the volume complement 1 - Σ f = " *
            "$(f) < 0 — the declared fractions already exceed 1"
        c.on_negative === :error ? throw(ArgumentError(msg)) : @warn msg
    end
    return rve
end

function _validate_closure(::RescaledFractions, rve::RVE)
    s = rve.f_sum
    if is_hard_numeric(s) && s <= zero(s)
        throw(
            ArgumentError(
                "closure = RescaledFractions() needs a strictly positive fraction " *
                    "sum to divide by; got Σ f = $(s)"
            )
        )
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
    # Every phase is shown the same way, with the fraction the schemes will
    # actually use. No phase is singled out as "the matrix": which one plays
    # the reference medium is the scheme's business, not the RVE's.
    for name in rve.phase_names
        a = rve.amounts[name]
        if a isa CrackDensity
            println(io, "  phase :$(name)   ε = $(amount_value(a))")
        elseif a isa Remainder
            println(io, "  phase :$(name)   f = $(volume_fraction(rve, name))  (rest)")
        else
            println(io, "  phase :$(name)   f = $(volume_fraction(rve, name))")
        end
    end
    println(io, "  closure : ", _closure(rve))
    print(io, "  distribution_shape : ", rve.distribution_shape)
    return
end

Base.show(io::IO, rve::RVE{T}) where {T} =
    print(io, "RVE{$T}(", length(rve.phase_names), " phases)")
