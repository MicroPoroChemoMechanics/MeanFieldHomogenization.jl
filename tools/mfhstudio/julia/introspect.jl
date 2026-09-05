# =============================================================================
#  introspect.jl — the MFH feature catalog, discovered at run time.
#
#  Hard-coding the list of schemes, interfaces and inclusion types in the web
#  UI would make the interface silently fall behind MeanFieldHomogenization. Everything
#  the UI offers is therefore enumerated from the loaded package, in the same
#  spirit as `echoes2mfh check-drift`.
# =============================================================================

using MeanFieldHomogenization
using TensND
using InteractiveUtils: subtypes

"""
    concrete_subtypes(T) -> Vector{Type}

Every instantiable leaf under `T`, depth-first. Abstract intermediate layers
are traversed but not reported.
"""
function concrete_subtypes(T::Type)
    out = Type[]
    for S in subtypes(T)
        if isabstracttype(S)
            append!(out, concrete_subtypes(S))
        else
            push!(out, S)
        end
    end
    return out
end

# ── Constructor keywords ─────────────────────────────────────────────────────
#
# The scheme constructors carry the solver options (`abstol`, `maxiters`,
# `select_best`, `nsteps`, …). Reading them off the methods keeps the UI's
# option list exact rather than approximate.

function ctor_keywords(T::Type)
    names = Set{Symbol}()
    for m in methods(T)
        for kw in Base.kwarg_decl(m)
            s = String(kw)
            # `kwargs...` shows up as a trailing `...` name
            endswith(s, "...") && continue
            push!(names, kw)
        end
    end
    return sort!(collect(names); by = String)
end

# Several schemes take their solver options through a `kwargs...` bag
# (`SelfConsistent(; algorithm = …, kwargs...)`), so `kwarg_decl` cannot see
# them — and the bag accepts *anything*, so probing the constructor proves
# nothing either: `SelfConsistent(; nsteps = 3)` succeeds while `nsteps` is
# meaningless there.
#
# What the schemes do publish is the list of keys they actually read. Reading
# those constants keeps the interface exactly in step with the schemes, and
# the fallback below means a rename degrades to "no options offered" rather
# than to a wrong list.
const OPTION_KEY_CONSTANTS = Dict(
    "SelfConsistent" => :_SC_SOLVER_KWARGS,
    "AsymmetricSelfConsistent" => :_SC_SOLVER_KWARGS,
    "DifferentialScheme" => :_DIFF_RESERVED_OPTIONS,
)

# Defaults are **per scheme family**, because they genuinely differ: the
# self-consistent solvers stop at `1e-12 / 1e-8` while the differential scheme
# hands `1e-8 / 1e-6` to `OrdinaryDiffEq`. A single flat table reported the
# ODE numbers for every scheme, so the panel showed `abstol = 1e-8` as the
# default of `SelfConsistent` when the solver actually uses `1e-12` — a wrong
# number in the one place a user looks to find out what they are overriding.
const OPTION_DEFAULTS_COMMON = Dict{Symbol, Any}(
    :damping => 0.0, :select_best => false, :verbose => false,
    :nsteps => 100, :formulation => "stiffness",
)

const OPTION_DEFAULTS_BY_SCHEME = Dict(
    # `Schemes._solve_sc(::AndersonDefault, …)` / `(::NewtonDefault, …)`
    "SelfConsistent" => Dict{Symbol, Any}(
        :abstol => 1.0e-12, :reltol => 1.0e-8, :maxiters => 100,
    ),
    "AsymmetricSelfConsistent" => Dict{Symbol, Any}(
        :abstol => 1.0e-12, :reltol => 1.0e-8, :maxiters => 100,
    ),
    # `DifferentialScheme(; …)`, forwarded to `solve`
    "DifferentialScheme" => Dict{Symbol, Any}(
        :abstol => 1.0e-8, :reltol => 1.0e-6,
    ),
)

"""
    option_default(S, key) -> Any

Documented default of option `key` for scheme `S`, or `nothing` when the
scheme does not publish one.
"""
function option_default(S::Type, key::Symbol)
    per = get(OPTION_DEFAULTS_BY_SCHEME, String(nameof(S)), nothing)
    per !== nothing && haskey(per, key) && return per[key]
    return get(OPTION_DEFAULTS_COMMON, key, nothing)
end

"""
    consumed_options(S) -> Vector{Symbol}

The option keys scheme `S` actually reads, taken from the constant it
declares for the purpose. Empty when the scheme declares none.
"""
function consumed_options(S::Type)
    name = String(nameof(S))
    key = get(OPTION_KEY_CONSTANTS, name, nothing)
    key === nothing && return Symbol[]
    mod = parentmodule(S)
    isdefined(mod, key) || return Symbol[]
    return collect(Symbol, getfield(mod, key))
end

"""
Options that hold an object rather than a number (`algorithm`, `trajectory`,
`alg`). They are reported so the UI can display the default, but not offered
as free-form inputs.
"""
const OPAQUE_OPTIONS = Set([:algorithm, :trajectory, :alg])

function scheme_entry(S::Type)
    consumed = consumed_options(S)
    declared = ctor_keywords(S)

    opts = Dict{String, Any}[]
    for k in consumed
        push!(
            opts, Dict(
                "name" => String(k),
                "default" => option_default(S, k),
                "editable" => !(k in OPAQUE_OPTIONS),
            )
        )
    end
    for k in declared
        k in consumed && continue
        push!(
            opts, Dict(
                "name" => String(k), "default" => nothing,
                "editable" => !(k in OPAQUE_OPTIONS),
            )
        )
    end
    sort!(opts; by = d -> d["name"])

    return Dict(
        "name" => String(nameof(S)),
        "options" => opts,
        # A scheme with nothing to configure is a singleton: `Voigt()`.
        "singleton" => all(d -> !d["editable"], opts),
    )
end

"""
    catalog() -> Dict

The facts only the installed package can supply: which schemes exist and
which solver options each one reads. The interface's form definitions are not
here -- they are a UI concern and live in `mfhstudio/catalog.py`, so the
interface stays usable while this side is still loading.
"""
function catalog()
    schemes = [scheme_entry(S) for S in concrete_subtypes(MeanFieldHomogenization.HomogenizationScheme)]
    sort!(schemes; by = d -> d["name"])
    return Dict(
        "mfh_version" => string(pkgversion(MeanFieldHomogenization)),
        "julia_version" => string(VERSION),
        "schemes" => schemes,
    )
end
