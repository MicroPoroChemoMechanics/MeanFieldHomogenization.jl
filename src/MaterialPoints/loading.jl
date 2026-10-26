# =============================================================================
#  loading.jl — a loading path, component by component.
#
#  Each component of each field is driven either by its gradient or by its
#  flux, as a test machine does: a tensile test drives ε₁₁ and leaves the
#  lateral faces free (σ₂₂ = σ₃₃ = 0), a creep test drives σ₁₁, a drained test on
#  a porous sample drives the pore pressure p, an undrained one the fluid
#  content φ. A component the user does not name has a zero flux.
# =============================================================================

# Tensor components in Kelvin–Mandel order; the twins of the shears name the
# same component.
const _COMPONENTS = ("11", "22", "33", "23", "13", "12")
const _SYMMETRIC_TWINS = Dict("32" => "23", "31" => "13", "21" => "12")

"""
    LoadingPath(times; ε11 = …, σ22 = …, p = …, …)

A loading path for a material point: the `times` at which the response is
computed, and, for each component of each field of the law, whether it is
driven by its **gradient** or by its **flux**. A value is a number (held
constant) or a function of time.

- A tensor field is driven component by component: `ε11`, …, `ε12` for the
  strain, `σ11`, …, `σ12` for the stress (tensor components: `ε12` is
  ``\\varepsilon_{12}``, not the engineering shear ``2\\varepsilon_{12}``).
- A scalar field by its name: `p` for the pore pressure, `φ` for the change of
  fluid content of a [`BiotPoroelasticity`](@ref) law.
- A component named in neither form has a **zero flux**: a free face.
- `external` gives the variables the law reads without returning their flux,
  by name, as numbers or functions of time: `external = (T = t -> 293 + 0.5t,)`
  for a temperature (see [`point_update`](@ref)).

```julia
tension   = LoadingPath(range(0, 1; length = 101); ε11 = t -> 2e-3 * t)   # uniaxial stress
creep     = LoadingPath(range(0, 100; length = 201); σ11 = 150.0)          # constant stress
drained   = LoadingPath(0:0.1:1; ε11 = t -> 1e-3 * t, p = 0)              # pore pressure held at zero
undrained = LoadingPath(0:0.1:1; ε11 = t -> 1e-3 * t)                     # φ = 0: no fluid exchanged
```

The first time is the initial instant: the response there is computed from the
initial state with a step of zero length. The names are checked against the
law's fields when the path is run.
"""
struct LoadingPath{T, C <: NamedTuple, X <: NamedTuple}
    times::Vector{T}
    controls::C
    external::X
end

function LoadingPath(times::AbstractVector{<:Real}; external::NamedTuple = (;), controls...)
    length(times) ≥ 2 || throw(ArgumentError("a loading path needs at least two times; got $(length(times))"))
    all(>(0), diff(times)) || throw(ArgumentError("the times of a loading path must increase strictly"))
    return LoadingPath(collect(float.(times)), map(_of_time, NamedTuple(controls)), map(_of_time, external))
end

# A prescribed value: a number is held constant, a function of time is used as is.
_of_time(v::Number) = Returns(v)
_of_time(f) = f

# The names of the components of one field of size n: `ε11`, … for a tensor,
# the name itself for a scalar, `q1`, `q2`, … otherwise.
_component_names(name::Symbol, n::Int) =
    n == 6 ? Tuple(Symbol(name, ij) for ij in _COMPONENTS) :
    n == 1 ? (name,) : Tuple(Symbol(name, k) for k in 1:n)

# A shear named by its twin (ε21 for ε12) is the same component.
function _canonical_name(name::Symbol)
    s = String(name)
    length(s) ≥ 2 || return name
    tail = s[prevind(s, lastindex(s), 1):end]
    haskey(_SYMMETRIC_TWINS, tail) || return name
    return Symbol(s[1:prevind(s, lastindex(s), 2)], _SYMMETRIC_TWINS[tail])
end

# Factor from a tensor component to its Kelvin–Mandel counterpart.
_mandel_factors(n::Int) = n == 6 ? (1.0, 1.0, 1.0, sqrt(2.0), sqrt(2.0), sqrt(2.0)) : ntuple(_ -> 1.0, n)

"""
    _resolve(path, gradients, fluxes) -> (driven, targets, factors)

The path read against the fields of a law: for each entry of the flat vectors,
whether it is driven by its gradient, the function of time giving its target
(gradient or flux), and the factor turning the user's tensor component into the
Mandel one.
"""
function _resolve(path::LoadingPath, gradients::NamedTuple, fluxes::NamedTuple)
    # Built once per run, behind the function barrier of `point_response`: the
    # targets are functions of different types, held as `Any` until the tuple.
    driven, targets, factors = Bool[], Any[], Float64[]
    used = Set{Symbol}()
    given = Dict(_canonical_name(k) => k for k in keys(path.controls))
    for (gname, fname, r) in zip(keys(gradients), keys(fluxes), values(gradients))
        n = _width(r)
        for (gc, fc, factor) in zip(_component_names(gname, n), _component_names(fname, n), _mandel_factors(n))
            haskey(given, gc) && haskey(given, fc) && throw(
                ArgumentError("$gc and $fc are conjugate: drive the component by one of them, not both")
            )
            key = haskey(given, gc) ? gc : haskey(given, fc) ? fc : nothing
            push!(driven, haskey(given, gc))
            push!(targets, key === nothing ? Returns(0.0) : path.controls[given[key]])
            push!(factors, factor)
            key === nothing || push!(used, key)
        end
    end
    unknown = setdiff(keys(given), used)
    isempty(unknown) || throw(
        ArgumentError(
            "unknown component(s) $(join(sort!(collect(unknown)), ", ")): the law's fields are " *
                "$(keys(gradients)) (gradients) and $(keys(fluxes)) (fluxes)"
        )
    )
    M = length(driven)
    return SVector{M, Bool}(driven), Tuple(targets), SVector{M, Float64}(factors)
end
