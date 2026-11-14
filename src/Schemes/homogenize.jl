# =============================================================================
#  homogenize.jl — central public entry point for homogenization schemes.
#
#  Each concrete scheme implements a method
#       _evaluate(rve::RVE, ::ConcreteScheme, ::Val{property}; kw...)
#  in its own file (`voigt.jl`, `dilute.jl`, …). The `Val{property}` form
#  enables compile-time specialization on the property name (`:C`, `:K`)
#  without paying the Symbol-dispatch cost at runtime.
# =============================================================================

"""
    homogenize(cell, scheme::HomogenizationScheme, property::Symbol; kw...)
    homogenize(cell, scheme::Symbol, property::Symbol; kw...)

Compute the effective `property` of `cell` under the chosen `scheme`.

`cell` is any [`AbstractHomogenizationCell`](@ref): an [`RVE`](@ref) (matrix
plus named phases, for every scheme built on the Eshelby auxiliary problem)
or a `Laminate` (periodic stack of parallel layers, solved exactly by
`Laminated`). A scheme that a cell does not support reports it explicitly
rather than dispatching elsewhere.

`property` is a *required* `Symbol` argument naming which stored phase
property is homogenized. The order of the result (4th-order, 2nd-order,
…) follows from the order of the tensor stored under that name in each
phase ; the symbol itself is just a key — any user-chosen name
(`:stiffness`, `:conductivity`, `:κ`, …) works as long as every phase
carries a tensor under that key. Common conventions are `:C` for elastic
stiffness and `:K` for conductivity / diffusivity.

The scheme can be passed either as a *type instance* (full control of
options — `MoriTanaka()`, `SelfConsistent(algorithm = NewtonDefault(), abstol = 1e-12)`,
…) or as a `Symbol` shortcut for the default constructor. The canonical
Symbol aliases are lowercase (`:mt`, `:sc`, `:voigt`, …) to match the
algorithm-method symbols (`:auto`, `:residues`, `:decuhr`, …) used by
the underlying Hill-tensor backends; CamelCase and upper-case forms are
also accepted (see [`SCHEME_ALIAS`](@ref)).

Extra `kw...` are forwarded to the scheme's `_evaluate` method
(typically `abstol` / `reltol` / `maxiters` for iterative schemes,
`method = :auto | :decuhr | …` for the underlying Hill-tensor backend).
"""
function homogenize(
        cell::AbstractHomogenizationCell, scheme::HomogenizationScheme,
        property::Symbol; kw...
    )
    validate_cell(cell, scheme)
    _check_layered_host(cell, scheme, property)
    # The scope spans the WHOLE evaluation, iterative solvers included: a
    # self-consistent loop reads the phase properties once per iteration, and
    # without the memoization a declaratively nested cell would be
    # re-homogenized on every one of them. `_with_nested_cache` reuses an
    # already-active scope, so sibling and deeper cells share it.
    return MFH_Core._with_nested_cache() do
        result = _evaluate(cell, scheme, Val(property); kw...)
        _warn_average_domain(cell, scheme, property, result)
        result
    end
end

# An iterative scheme compares successive values (a fixed point) or controls
# its step on them (an adaptive ODE solve), which a symbolic modulus cannot
# answer: `Num` raises a TypeError deep in the loop, `Sym` a MethodError, or
# the solve does not end. Refused at the entry instead, with the way out, on
# the element type of a tensor or a state the scheme computes anyway (the seed
# of a self-consistent iteration, the initial state of the differential ODE):
# a check decided on a type. A `ForwardDiff.Dual` is comparable: derivatives
# go through. The scheme is named by a `Symbol`, the throw kept out of line:
# a scheme object passed down was boxed on the numerical path, 32 bytes for
# each self-consistent estimate.
@inline _require_comparable(t::TensND.AbstractTens, name::Symbol) = _require_comparable(eltype(t), name)
@inline _require_comparable(::Type{T}, name::Symbol) where {T} =
    is_hard_numeric(real(T)) ? nothing : _throw_not_comparable(T, name)
@noinline function _throw_not_comparable(T, name::Symbol)
    throw(
        ArgumentError(
            "$name iterates on the effective property, comparing successive values, " *
                "which symbolic moduli ($(T)) cannot be compared by: give numbers (duals carry the " *
                "derivatives through), or take an explicit scheme (Voigt, Reuss, Dilute, DiluteDual, " *
                "MoriTanaka) for a closed form"
        )
    )
end

# Backward-compatible kwarg form, kept so existing scripts and tests using
# the old `homogenize(rve, scheme; property = …)` signature keep working.
# New code is encouraged to pass the property name as a positional argument
# (the user must always state explicitly which property is being homogenized).
# A host given the shape of a one-layer layered cylinder carries its property
# twice, declared and in its layer: the schemes that use the host as the
# reference read the former, those that solve its own problem the latter.
# They must be the same property, or two schemes would silently answer for two
# different matrices.
_check_layered_host(cell, scheme, property) = nothing
function _check_layered_host(rve::RVE, scheme::HomogenizationScheme, property::Symbol)
    r = _layered_host_name(rve, scheme)
    (r === nothing || !haskey(rve.phases, r)) && return nothing  # an absent phase is reported by the scheme
    geom = rve.phases[r].geometry
    geom isa LayeredCylinders.LayeredCylinder && LayeredCylinders.layer_count(geom) == 1 || return nothing
    declared = get(rve.phases[r].properties, property, nothing)
    declared === nothing && return nothing
    _same_property(declared, LayeredCylinders.layer_modulus(geom, 1)) || throw(
        ArgumentError(
            "the host phase :$(r) is a one-layer LayeredCylinder whose layer differs from its " *
                "declared property :$(property): the schemes that take the host as their " *
                "reference read the declared property, the others its layer. Declare the " *
                "property of the layer"
        )
    )
    return nothing
end

# The phase a scheme takes as its matrix, when it has one: the one it names,
# else the remainder phase.
_layered_host_name(rve::RVE, scheme::HomogenizationScheme) =
    something(scheme_matrix(scheme), remainder_phase_name(rve), Some(nothing))

# Two properties compared, where they compare: tensors by their canonical
# components, block matrices entry by entry; anything else (a law, a symbol) is
# left to the caller.
_same_property(a::TensND.AbstractTens, b::TensND.AbstractTens) =
    _same_property(TensND.canonical_components(a), TensND.canonical_components(b))
_same_property(a::AbstractArray, b::AbstractArray) =
    !(is_hard_numeric(eltype(a)) && is_hard_numeric(eltype(b))) || size(a) == size(b) && isapprox(a, b; rtol = 1.0e-12)
_same_property(a, b) = true

function homogenize(
        cell::AbstractHomogenizationCell, scheme::HomogenizationScheme;
        property::Symbol = :C, kw...
    )
    return homogenize(cell, scheme, property; kw...)
end

"""
    SCHEME_ALIAS :: Dict{Symbol, Type{<:HomogenizationScheme}}

Maps Symbol shortcuts to concrete scheme types for the convenience
overload `homogenize(rve, ::Symbol)`.

The canonical aliases are **all lowercase** (`:voigt`, `:reuss`,
`:dilute`, `:dilute_dual`, `:mori_tanaka`, `:mt`, `:maxwell`, `:pcw`,
`:sc`, `:asc`, `:differential`, `:diff`) for consistency with the
algorithm-method symbols accepted elsewhere in the package
(`:auto`, `:residues`, `:decuhr`, `:nestedquadgk`, `:analytical`).
The CamelCase forms (`:MoriTanaka`, `:Differential`) and the
ECHOES-compatible upper-case codes (`:MT`, `:DIFF`, …) are kept as
extra aliases for ease of porting.
"""
const SCHEME_ALIAS = Dict{Symbol, Type{<:HomogenizationScheme}}(
    # Voigt
    :voigt => Voigt, :Voigt => Voigt, :VOIGT => Voigt, :v => Voigt, :V => Voigt,
    # Reuss
    :reuss => Reuss, :Reuss => Reuss, :REUSS => Reuss, :r => Reuss, :R => Reuss,
    # Dilute
    :dilute => Dilute, :Dilute => Dilute, :dil => Dilute, :Dil => Dilute, :DIL => Dilute,
    # Dilute Dual
    :dilute_dual => DiluteDual, :DiluteDual => DiluteDual,
    :dild => DiluteDual, :DilD => DiluteDual, :DILD => DiluteDual,
    # Mori-Tanaka
    :mori_tanaka => MoriTanaka, :moritanaka => MoriTanaka, :MoriTanaka => MoriTanaka,
    :mt => MoriTanaka, :MT => MoriTanaka,
    # Maxwell
    :maxwell => Maxwell, :Maxwell => Maxwell, :max => Maxwell, :Max => Maxwell, :MAX => Maxwell,
    # Ponte-Castañeda & Willis
    :pcw => PonteCastanedaWillis, :PCW => PonteCastanedaWillis,
    :PonteCastanedaWillis => PonteCastanedaWillis,
    :ponte_castaneda_willis => PonteCastanedaWillis,
    # Cluster model (Molinari & El Mouden) — acts on a `ParticleAssembly`
    :cluster => ClusterModel, :ClusterModel => ClusterModel,
    :cluster_model => ClusterModel, :CLUSTER => ClusterModel,
    # Equivalent inclusion method (Brisard et al.) — acts on a `ParticleAssembly`
    :eim => EquivalentInclusion, :EIM => EquivalentInclusion,
    :EquivalentInclusion => EquivalentInclusion,
    :equivalent_inclusion => EquivalentInclusion,
    # Self-Consistent
    :self_consistent => SelfConsistent, :SelfConsistent => SelfConsistent,
    :sc => SelfConsistent, :SC => SelfConsistent,
    # Asymmetric Self-Consistent
    :asymmetric_self_consistent => AsymmetricSelfConsistent,
    :AsymmetricSelfConsistent => AsymmetricSelfConsistent,
    :asc => AsymmetricSelfConsistent, :ASC => AsymmetricSelfConsistent,
    # Differential
    :differential => DifferentialScheme, :Differential => DifferentialScheme,
    :diff => DifferentialScheme, :Diff => DifferentialScheme, :DIFF => DifferentialScheme,
    # Laminated (exact periodic multilayer — applies to a `Laminate` cell)
    :laminated => Laminated, :Laminated => Laminated, :LAMINATED => Laminated,
    :laminate => Laminated, :lam => Laminated, :LAM => Laminated,
    :multilayer => Laminated,
)

function homogenize(cell::AbstractHomogenizationCell, scheme::Symbol, property::Symbol; kw...)
    haskey(SCHEME_ALIAS, scheme) ||
        throw(ArgumentError("unknown scheme :$(scheme); see MeanFieldHomogenization.Schemes.SCHEME_ALIAS"))
    return homogenize(cell, SCHEME_ALIAS[scheme](), property; kw...)
end

function homogenize(cell::AbstractHomogenizationCell, scheme::Symbol; property::Symbol = :C, kw...)
    return homogenize(cell, scheme, property; kw...)
end

# =============================================================================
#  Default fallback — explicit "not yet implemented" error so that an
#  unimplemented scheme does not silently dispatch elsewhere.
# =============================================================================

"""
    _evaluate(cell, scheme, ::Val{p}; kw...) -> AbstractTens

Internal entry point that each concrete (cell, scheme) pair implements. This
generic fallback throws an explicit `ErrorException` so that a missing
specialization is reported clearly instead of dispatching to the wrong
method — in particular when a scheme is applied to a cell it does not serve
(`MoriTanaka` needs a matrix, so it does not apply to a `Laminate`;
`Laminated` needs an ordered stack of layers, so it does not apply to an
`RVE`).
"""
function _evaluate(
        cell::AbstractHomogenizationCell, scheme::HomogenizationScheme,
        ::Val{p}; kw...
    ) where {p}
    error(
        "homogenize: scheme $(typeof(scheme)) does not implement property :$(p) " *
            "for a $(nameof(typeof(cell)))"
    )
end
