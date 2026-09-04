# =============================================================================
#  cells.jl — the homogenization *cell* contract.
#
#  A "cell" is a morphological description of a heterogeneous unit whose
#  effective property `homogenize` can compute.  Two ship with the package:
#
#    * `Schemes.RVE`         — matrix + named phases, the container of every
#      Eshelby-based scheme (bounds, dilute, Mori-Tanaka, Maxwell, PCW,
#      self-consistent, differential): random morphologies, an auxiliary
#      inclusion problem and a reference medium;
#    * `Laminates.Laminate`  — a periodic stack of parallel layers with no
#      matrix and no reference medium, solved exactly.
#
#  Following the sub-module convention, every generic of the contract is
#  *declared* here without a body, so that `Schemes`, `Laminates`,
#  `Viscoelasticity` and out-of-package user code all attach their methods to
#  one canonical function.
#
#  This file also carries the **declarative multiscale** seam (`Homogenized`,
#  `NestedParameter`): a property value may itself be another cell plus a
#  scheme, resolved lazily so that a whole multiscale chain differentiates in
#  a single `ForwardDiff` pass.  See `docs/src/manual/multiscale.md`.
# =============================================================================

using Base.ScopedValues: ScopedValue, with

"""
    AbstractHomogenizationCell

Supertype of every *cell* accepted by [`homogenize`](@ref): a morphological
description of a heterogeneous unit whose effective property can be computed.

Two concrete cells ship with `MeanFieldHomogenization`:

- `RVE` — a matrix phase plus named inclusion phases, the container behind
  every scheme built on the Eshelby auxiliary problem (`Voigt`, `Reuss`,
  `Dilute`, `MoriTanaka`, `Maxwell`, `PonteCastanedaWillis`,
  `SelfConsistent`, `AsymmetricSelfConsistent`, `DifferentialScheme`);
- `Laminate` — a periodic unit cell of parallel layers, with no matrix and no
  reference medium, whose `Laminated` solution is exact rather than an
  estimate.

A new cell type must provide

- `validate_cell(cell)` — structural sanity check, called by `homogenize`
  (optionally `validate_cell(cell, scheme)` for scheme-dependent requirements);
- at least one `_evaluate(cell, scheme, ::Val{property}; kw...)` method;
- [`cell_member_names`](@ref), [`cell_container_property`](@ref) and
  [`cell_set_property`](@ref) to take part in declarative nesting and in the
  parameter lenses.

See also [`Homogenized`](@ref), [`NestedParameter`](@ref).
"""
abstract type AbstractHomogenizationCell end

"""
    AbstractParameter

Supertype of parameter lenses. Concrete subtypes designate a single scalar
input of a homogenization computation:

- `AmountParameter` — volume fraction or crack density of a phase.
- `PropertyParameter` — a scalar coefficient of a property tensor (`:C`,
  `:K`, …) of a phase or a layer, designated either by a named selector
  (`:bulk`, `:shear`, …) or by a positional index.
- `GeometryParameter` — a scalar geometry field of a phase (semi-axis of an
  ellipsoid, radius of a layer, …).
- `DistributionShapeParameter` — a scalar field of the PCW / Maxwell
  distribution shape.
- `ThicknessParameter` — the thickness of one layer of a laminate.
- `InterfaceParameter` — a compliance or surface modulus of one imperfect
  interface of a laminate.
- [`NestedParameter`](@ref) — a lens reaching *into* a declaratively nested
  cell, so that a multiscale chain differentiates end to end.

[`get_param`](@ref) / [`set_param`](@ref) read and replace the designated
scalar in a cell. See also the convenience constructors `amount`, `property`,
`geometry`, `shape_param`, `thickness`, `interface_param` and
[`nested`](@ref).
"""
abstract type AbstractParameter end

"""
    homogenize(cell, scheme, property::Symbol; kw...)

Compute the effective `property` of `cell` under `scheme`. Methods live in
`Schemes/homogenize.jl`; the generic is declared here so that every cell type
attaches to the same function.
"""
function homogenize end

"""
    validate_cell(cell)
    validate_cell(cell, scheme)

Structural sanity check of a homogenization cell, called by
[`homogenize`](@ref) before evaluation. `validate_cell(::RVE)` forwards to
`validate_rve`, `validate_cell(::Laminate)` to `validate_laminate`.

The two-argument form additionally checks what only the *scheme* can require.
A matrix-based estimate cannot be evaluated until one phase is designated as
its reference medium, whereas the bounds and the self-consistent scheme
distinguish no phase and accept a cell that designates none — so the
requirement belongs here, keyed on the scheme, and not in the cell's own
structural check. It defaults to the one-argument form, so a cell or a scheme
that has nothing extra to say need not implement it.
"""
function validate_cell end

"""
    _evaluate(cell, scheme, ::Val{property}; kw...)

Internal entry point implemented once per (cell, scheme) pair. Declared here
so that `Schemes` and `Laminates` extend one common function.
"""
function _evaluate end

"""
    get_param(cell, p::AbstractParameter) -> Number

Read the scalar designated by lens `p` in `cell`.
"""
function get_param end

"""
    set_param(cell, p::AbstractParameter, value)

Return a *new* cell in which the scalar designated by lens `p` has been
replaced by `value`. The element type of the affected fields is promoted to
absorb `typeof(value)`; every other field is preserved, and the original is
never mutated.
"""
function set_param end

"""
    _cell_promote(::Type{T}, v) -> Number

Store a scalar cell datum (an amount, a thickness) under the declared floor
`T`: widened to `promote_type(T, typeof(v))`, **never** narrowed down to `T`.
This is what lets a `Dual` or a symbolic value live in a cell declared with
the default `Float64` floor. Non-`Number` scalars (symbolic backends that do
not subtype `Number`) are stored verbatim rather than rejected.
"""
_cell_promote(::Type{T}, v::Number) where {T <: Number} =
    convert(promote_type(T, typeof(v)), v)
_cell_promote(::Type{<:Number}, v) = v

"""
    cell_member_names(cell) -> Vector{Symbol}

Names of the cell's members in insertion order — the phases of an `RVE`
(matrix included), the layers of a `Laminate`.
"""
function cell_member_names end

"""
    cell_container_property(cell, name::Symbol, key::Symbol)

**Raw** stored value of property `key` on member `name` — a
[`Homogenized`](@ref) is returned as such, *not* resolved. This is what type
inspections (`has_visco_property`) and the nesting lenses must use; the
resolving accessors are `phase_property` / `layer_property`.
"""
function cell_container_property end

"""
    cell_set_property(cell, name::Symbol, key::Symbol, value)

Return a new cell in which property `key` of member `name` has been replaced
by `value`, without mutating the original.
"""
function cell_set_property end

# =============================================================================
#  Declarative multiscale nesting
# =============================================================================

"""
    Homogenized(cell, scheme; property = nothing, kw...)

A property value that **is** another homogenization problem.

Stored in a phase's or a layer's `properties` dict under the usual key (`:C`,
`:K`, …) and resolved *lazily*, when `homogenize` needs it — so a whole
multiscale chain is one object, `ForwardDiff` traverses it in a single
forward pass, and a symbolic backend sees one expression tree.

`property = nothing` (the default) means **inherit the key it is stored
under**: the same `Homogenized` then answers `:C` with the inner cell's
effective stiffness and `:K` with its effective conductivity. Pass an
explicit `property` to read a different key from the inner cell.

`kw...` are forwarded to the inner `homogenize` (`method = :nestedquadgk`,
`abstol`, …).

```julia
inner = Laminate(; normal = (0, 0, 1))
add_layer!(inner, :A, Dict(:C => C_A); fraction = 0.4)
add_layer!(inner, :B, Dict(:C => C_B); fraction = 0.6)

rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_matrix); fraction = :rest)
add_phase!(rve, :agg, Ellipsoid(1.0),
           Dict(:C => Homogenized(inner, Laminated())); fraction = 0.3)

homogenize(rve, MoriTanaka(), :C)     # resolves `inner` on the way
```

Within one `homogenize` call each `(Homogenized, key)` pair is evaluated
exactly once, however many times a scheme reads the property — an iterative
self-consistent loop included. The memoization is task-local and torn down
when the call returns, so no value ever leaks from one autodiff step to the
next.

Nesting a `Homogenized` inside an ageing-viscoelastic chain is not supported:
the inner result would have to be re-expressible as a `ViscoLaw`.

See also [`NestedParameter`](@ref), [`resolve_property`](@ref).
"""
struct Homogenized{C <: AbstractHomogenizationCell, S, K <: NamedTuple}
    cell::C
    scheme::S
    property::Union{Nothing, Symbol}
    kwargs::K
end

function Homogenized(
        cell::AbstractHomogenizationCell, scheme;
        property::Union{Nothing, Symbol} = nothing, kw...
    )
    return Homogenized(cell, scheme, property, NamedTuple(kw))
end

function Base.show(io::IO, h::Homogenized)
    key = h.property === nothing ? "⟨inherited⟩" : ":$(h.property)"
    return print(io, "Homogenized(", nameof(typeof(h.cell)), ", ", h.scheme, ", ", key, ")")
end

# ── The call-scoped memoization ─────────────────────────────────────────────
#
# A `ScopedValue` is task-local and unwound at scope exit.  A global cache
# would be neither: it would not be thread-safe (schemes are run from
# `@threads` loops in the benchmark harness), and it would retain
# `ForwardDiff.Dual` values across evaluations, so two nested `derivative`
# calls with different tags could collide and silently return a stale-tag
# `Dual` — the worst class of autodiff bug.
#
# The outer key is an `IdDict` on the *identity* of the `Homogenized`: two
# structurally equal values produced by different `set_param` branches of the
# same autodiff step must not share a result.

const _NESTED_CACHE = ScopedValue{Union{Nothing, IdDict{Any, Dict{Symbol, Any}}}}(nothing)
const _NESTED_DEPTH = ScopedValue{Int}(0)

"""
    MAX_NESTING

Depth ceiling for declaratively nested cells. Exceeding it raises rather than
looping forever — the usual cause is a cell nested inside itself.
"""
const MAX_NESTING = 8

"""
    _with_nested_cache(f)

Run `f()` under a fresh task-local memoization cache for declaratively nested
cells — unless one is already active, in which case `f()` runs in the
existing scope so that sibling `Homogenized` values share their results.

Opened once by `homogenize`, and spanning the whole evaluation including
iterative solvers: `SelfConsistent` reads the phase properties once per
iteration, and without the cache a nested cell would be re-homogenized on
every one of them.
"""
@inline function _with_nested_cache(f)
    _NESTED_CACHE[] === nothing || return f()
    return with(f, _NESTED_CACHE => IdDict{Any, Dict{Symbol, Any}}())
end

"""
    resolve_property(value, key::Symbol)

Resolve a stored property value: the identity on every ordinary value, and
the inner effective property for a [`Homogenized`](@ref) (memoized within the
enclosing `homogenize` call).

This is the single seam through which declarative nesting enters; the
resolving accessors `phase_property` and `layer_property` call it, so every
scheme kernel sees a plain tensor without knowing nesting exists.
"""
resolve_property(value, ::Symbol) = value

function resolve_property(h::Homogenized, key::Symbol)
    prop = h.property === nothing ? key : h.property
    cache = _NESTED_CACHE[]
    cache === nothing && return _resolve_uncached(h, prop)
    per_key = get!(() -> Dict{Symbol, Any}(), cache, h)
    return get!(() -> _resolve_uncached(h, prop), per_key, prop)
end

function _resolve_uncached(h::Homogenized, prop::Symbol)
    d = _NESTED_DEPTH[]
    d ≥ MAX_NESTING && error(
        "Homogenized: nesting deeper than $(MAX_NESTING) levels while resolving " *
            ":$(prop) on a $(typeof(h.cell)) — a cell is probably nested inside itself"
    )
    return with(_NESTED_DEPTH => d + 1) do
        homogenize(h.cell, h.scheme, prop; h.kwargs...)
    end
end

"""
    has_nested_property(cell, key::Symbol) -> Bool

Whether any member of `cell` carries a [`Homogenized`](@ref) under `key`.
Purely introspective — it never resolves anything.
"""
function has_nested_property(cell::AbstractHomogenizationCell, key::Symbol)
    for name in cell_member_names(cell)
        v = try
            cell_container_property(cell, name, key)
        catch
            continue
        end
        v isa Homogenized && return true
    end
    return false
end

# ── The traversing lens ─────────────────────────────────────────────────────

"""
    NestedParameter(member, property, inner)
    nested(member, property, inner)

Lens addressing a scalar **inside a declaratively nested cell**. `member` is
the phase (or layer) name in the outer cell, `property` the key under which a
[`Homogenized`](@ref) is stored, and `inner` a lens into that `Homogenized`'s
cell — itself possibly a `NestedParameter`, so any depth composes.

To reach the bulk modulus of layer `:L1` of the laminate nested under the
`:C` property of phase `:agg`:

```julia
nested(:agg, :C, property(:L1, :C, :bulk))
```

`set_param` rebuilds the whole chain immutably — outer cell, `Homogenized`
wrapper and inner cell — so `ForwardDiff` sees one straight-line computation
and the original objects are untouched. This is what lets
`derivative`/`gradient`/`jacobian` cross every scale of a multiscale model
without any hand-written closure.
"""
struct NestedParameter{P <: AbstractParameter} <: AbstractParameter
    member::Symbol
    property::Symbol
    inner::P
end

"""
    nested(member, property, inner) -> NestedParameter

Convenience constructor for [`NestedParameter`](@ref), which documents the
lens itself: `member` is the phase or layer name in the outer cell, `property`
the key under which a [`Homogenized`](@ref) is stored, and `inner` a lens into
that `Homogenized`'s cell.

```julia
nested(:agg, :C, property(:L1, :C, :bulk))
```
"""
nested(member::Symbol, property::Symbol, inner::AbstractParameter) =
    NestedParameter(member, property, inner)

@inline function _nested_home(cell::AbstractHomogenizationCell, p::NestedParameter)
    h = cell_container_property(cell, p.member, p.property)
    h isa Homogenized || throw(
        ArgumentError(
            "NestedParameter: :$(p.member).:$(p.property) holds a $(typeof(h)), " *
                "not a Homogenized — this lens only addresses declaratively nested cells"
        )
    )
    return h
end

get_param(cell::AbstractHomogenizationCell, p::NestedParameter) =
    get_param(_nested_home(cell, p).cell, p.inner)

function set_param(cell::AbstractHomogenizationCell, p::NestedParameter, value)
    h = _nested_home(cell, p)
    h′ = Homogenized(set_param(h.cell, p.inner, value), h.scheme, h.property, h.kwargs)
    return cell_set_property(cell, p.member, p.property, h′)
end
