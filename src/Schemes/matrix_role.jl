# =============================================================================
#  matrix_role.jl — which phase plays the reference medium, and for whom.
#
#  A "matrix" is not a property of a microstructure: it is a property of the
#  morphological model applied to it. The very same RVE — a bag of phases with
#  their shapes, properties and volume fractions — is a matrix/inclusion
#  composite under Mori-Tanaka and a matrix-free aggregate under the
#  self-consistent scheme. So the reference medium is named on the *scheme*,
#  and the RVE stays purely descriptive.
#
#  Two functions carry that contract:
#
#    `scheme_matrix(scheme)`  — what the caller asked for, or `nothing`;
#    `matrix_name(scheme, rve)` — the phase actually used, defaults resolved
#                                 and undecidable cases reported by name.
#
#  Schemes that distinguish no phase (`Voigt`, `Reuss`, `SelfConsistent`,
#  `Laminated`) need neither, and the generic fallbacks below let them — and
#  any third-party scheme — work without declaring anything.
# =============================================================================

"""
    scheme_matrix(scheme) -> Union{Nothing, Symbol}

The phase the caller named as the reference medium of `scheme`, or `nothing`
when the scheme has no such notion or the caller left the choice open.

Generic fallback: `nothing`. A scheme carrying a `matrix` field overrides it.
"""
scheme_matrix(::HomogenizationScheme) = nothing
scheme_matrix(
    s::Union{
        Dilute, DiluteDual, MoriTanaka, Maxwell, PonteCastanedaWillis,
        DifferentialScheme, AsymmetricSelfConsistent,
    }
) = s.matrix

"""
    requires_matrix(scheme) -> Bool

Whether `scheme` computes localization tensors in a reference medium taken
from one distinguished phase, and therefore cannot be evaluated until that
phase is decided.

`false` for the bounds, for the self-consistent scheme and for `Laminated`;
`true` for the matrix-based estimates.
"""
requires_matrix(::HomogenizationScheme) = false
requires_matrix(
    ::Union{
        Dilute, DiluteDual, MoriTanaka, Maxwell, PonteCastanedaWillis,
        DifferentialScheme, AsymmetricSelfConsistent,
    }
) = true

"""
    matrix_name(scheme, rve) -> Symbol

The phase of `rve` that plays the *reference medium* of `scheme`.

Named explicitly on the scheme when the caller said so (`MoriTanaka(:cem)`);
otherwise the phase that absorbs the volume complement, when the RVE
designates one. Choosing a matrix is a modeling decision — a different choice
is a different composite, not a rounding difference — so an RVE that
designates none and a scheme that names none is an error, and the message
lists the candidates rather than picking for the caller.
"""
function matrix_name(scheme::HomogenizationScheme, rve::RVE)
    declared = scheme_matrix(scheme)
    if declared !== nothing
        haskey(rve.phases, declared) || throw(
            ArgumentError(
                "$(nameof(typeof(scheme))) names :$(declared) as its matrix, but this RVE " *
                    "has no such phase; its phases are $(rve.phase_names)"
            )
        )
        return declared
    end
    r = remainder_phase_name(rve)
    r === nothing && throw(
        ArgumentError(
            "$(nameof(typeof(scheme))) needs a matrix phase — the reference medium " *
                "its localization tensors are computed in — and this RVE designates " *
                "none. Name it on the scheme, e.g. " *
                "$(nameof(typeof(scheme)))(:" *
                "$(isempty(rve.phase_names) ? :phase : first(rve.phase_names)))" *
                ", or declare one phase with `fraction = :rest`. " *
                "Candidates: $(rve.phase_names)."
        )
    )
    return r
end

"""
    host_phase_name(rve, declared, who::AbstractString) -> Symbol

The phase that plays the continuous host of `rve` for a consumer that is *not*
a homogenization scheme — the solid skeleton of a fracture network, the
homogeneous solid of micro-poromechanics.

`declared` is what the caller named, or `nothing` to fall back on the phase
that absorbs the volume complement. `who` names the caller in the error
message: an RVE that designates no complement leaves the host undecided, and
which phase is the skeleton is a modeling statement no container can make on
the caller's behalf.
"""
function host_phase_name(rve::RVE, declared::Union{Nothing, Symbol}, who::AbstractString)
    if declared !== nothing
        haskey(rve.phases, declared) || throw(
            ArgumentError(
                "$(who): this RVE has no phase :$(declared); its phases are $(rve.phase_names)"
            )
        )
        return declared
    end
    r = remainder_phase_name(rve)
    r === nothing && throw(
        ArgumentError(
            "$(who): this RVE designates no phase as the continuous host. Name one " *
                "explicitly, or declare a phase with `fraction = :rest`. " *
                "Candidates: $(rve.phase_names)."
        )
    )
    return r
end

# ── The distribution shape a scheme requires of the RVE ─────────────────────
#
# Same shape of contract as the matrix above, for a different reason. The
# distribution shape is *not* a modeling choice that belongs on the scheme: it
# is a two-point statistic of the microstructure (Willis 1982), which is why it
# stays a field of the `RVE` alongside the phase geometries. What it shares with
# the matrix is that a scheme needing it cannot be evaluated until it is
# decided, and that guessing produces a different composite rather than a
# rounding difference.

"""
    requires_distribution_shape(scheme) -> Bool

Whether `scheme` builds its estimate on the shape of the spatial distribution
of the inclusions, and therefore cannot be evaluated until the RVE declares one.

`true` for [`Maxwell`](@ref) and [`PonteCastanedaWillis`](@ref); `false` for
every other scheme, which ignores the field the way [`Voigt`](@ref) ignores a
phase's geometry.
"""
requires_distribution_shape(::HomogenizationScheme) = false
requires_distribution_shape(::Union{Maxwell, PonteCastanedaWillis}) = true

"""
    distribution_shape(rve, scheme) -> AbstractDistributionShape

The outer envelope of the spatial distribution that `scheme` reads off `rve`.

An RVE that declares none is an error, and deliberately so. A spherical
distribution — the value this used to default to — makes both
[`Maxwell`](@ref) and [`PonteCastanedaWillis`](@ref) reduce **exactly** to
Mori-Tanaka. Supplying one silently therefore answered a scheme whose whole
purpose is a non-spherical distribution with the estimate it generalizes, and
did so without a word. A sphere remains a perfectly legitimate choice; it just
has to be the caller's, not the container's.
"""
function distribution_shape(rve::RVE, scheme::HomogenizationScheme)
    ds = rve.distribution_shape
    ds === nothing && throw(
        ArgumentError(
            "$(nameof(typeof(scheme))) needs the shape of the spatial " *
                "distribution of the inclusions — the two-point statistics its " *
                "estimate is built on — and this RVE declares none. Declare it on " *
                "the RVE, e.g. `RVE(; distribution_shape = Spheroid(0.3))`. " *
                "A spherical distribution is a legitimate choice, but it makes " *
                "this scheme coincide exactly with Mori-Tanaka, so it has to be " *
                "stated: `RVE(; distribution_shape = Ellipsoid(1.0))`."
        )
    )
    return ds
end

"""
    reference_property(rve, scheme, key::Symbol) -> AbstractTens

Property `key` of the phase that plays `scheme`'s reference medium — the
scheme-aware replacement for reading a property off "the matrix" of an RVE.
"""
reference_property(rve::RVE, scheme::HomogenizationScheme, key::Symbol) =
    phase_property(rve, matrix_name(scheme, rve), key)

# ── Scheme-aware validation ─────────────────────────────────────────────────

# Default: a scheme adds nothing to what the cell already checks about itself.
validate_cell(cell::AbstractHomogenizationCell, ::HomogenizationScheme) =
    validate_cell(cell)

"""
    validate_cell(rve::RVE, scheme) -> RVE

[`validate_rve`](@ref) plus the requirements that are the scheme's and not the
RVE's: a matrix-based estimate must be able to name its reference medium, and a
distribution-aware estimate must find a declared distribution shape. Resolving
both here means the caller gets the explanatory [`matrix_name`](@ref) /
[`distribution_shape`](@ref) error before any kernel runs, rather than a
`KeyError` deep inside an iteration.
"""
function validate_cell(rve::RVE, scheme::HomogenizationScheme)
    validate_rve(rve)
    requires_matrix(scheme) && matrix_name(scheme, rve)
    requires_distribution_shape(scheme) && distribution_shape(rve, scheme)
    return rve
end
