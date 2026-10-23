# =============================================================================
#  average_domain.jl — the domain of an orientation average.
#
#  An orientation average (`symmetrize`) is defined only when the reference
#  medium of the auxiliary Eshelby problems is invariant under the rotations it
#  averages over: only then do averaging the concentration tensors and averaging
#  the contributions give the same scheme. Outside that domain the reference is
#  still projected (`_project_matrix`) and the estimate is an approximation, which
#  `homogenize` reports with a warning rather than refusing.
# =============================================================================

# The reference of the Eshelby problems of `scheme`: its matrix for a
# matrix-based estimate (the differential scheme starts from it and keeps its
# symmetry), the fixed point for a self-consistent one, none for the bounds.
_eshelby_reference(scheme::HomogenizationScheme, rve, prop, result) =
    requires_matrix(scheme) ? phase_property(rve, matrix_name(scheme, rve), prop) : nothing
_eshelby_reference(::Union{SelfConsistent, AsymmetricSelfConsistent}, rve, prop, result) = result

_reference_label(scheme::HomogenizationScheme, rve) = "the matrix :$(matrix_name(scheme, rve))"
_reference_label(::Union{SelfConsistent, AsymmetricSelfConsistent}, rve) = "the self-consistent estimate"

# Relative distance of `P₀` to its exact average under `sym`, on primal values:
# only its size matters, and a dual reference would otherwise carry its partials
# through the average. Zero, without computation, for a reference whose type
# shows the invariance.
_symmetry_defect(::TensND.TensISO, ::AbstractSymmetrize) = 0.0
_symmetry_defect(P₀::TensND.TensTI, sym::TISymmetrize) =
    _coaxial(TensND.axis(P₀), sym.axis) ? 0.0 : _average_distance(P₀, sym)
_symmetry_defect(P₀, sym::AbstractSymmetrize) = _average_distance(P₀, sym)
function _average_distance(P₀, sym)
    A = map(_primal_value, TensND.components_canon(P₀))
    return norm(A - TensND.components_canon(_project_matrix(TensND.Tens(A), sym))) / norm(A)
end

_symmetry_name(::IsoSymmetrize) = "every rotation (`symmetrize = :iso`)"
_symmetry_name(sym::TISymmetrize) = "rotations about $(sym.axis) (`TISymmetrize`)"
_symmetry_part(::IsoSymmetrize) = "isotropic average"
_symmetry_part(::TISymmetrize) = "average about that axis"

# The value of a number that may be a dual, nested or not.
_primal_value(x::ForwardDiff.Dual) = _primal_value(ForwardDiff.value(x))
_primal_value(x) = x

# Far above the rounding of a reference that has the symmetry, far below any
# anisotropy that matters.
const _AVERAGE_DOMAIN_RTOL = 1.0e-6

_warn_average_domain(cell, scheme, prop, result) = nothing
function _warn_average_domain(rve::RVE, scheme::HomogenizationScheme, prop::Symbol, result)
    # The common case, no averaged phase, reads the phase table and nothing else.
    isempty(rve.symmetrize) && return nothing
    _check_reference_symmetry(rve, scheme, _eshelby_reference(scheme, rve, prop, result))
    return nothing
end
# No reference (the bounds), a non-tensor property, or an isotropic reference:
# nothing to check.
_check_reference_symmetry(rve, scheme, ::Any) = nothing
_check_reference_symmetry(rve, scheme, ::TensND.TensISO) = nothing
function _check_reference_symmetry(rve, scheme, P₀::TensND.AbstractTens)
    # A symbolic reference cannot be compared with a tolerance.
    is_hard_numeric(real(eltype(P₀))) || return nothing
    checked = AbstractSymmetrize[]
    for name in rve.phase_names
        sym = phase_symmetrize(rve, name)
        (sym isa NoSymmetrize || sym in checked) && continue
        push!(checked, sym)
        defect = _symmetry_defect(P₀, sym)
        defect ≤ _AVERAGE_DOMAIN_RTOL && continue
        @warn "Phase :$name is averaged over $(_symmetry_name(sym)), but the reference " *
            "medium of the Eshelby problems of $(nameof(typeof(scheme))), " *
            "$(_reference_label(scheme, rve)), is not invariant under them (relative " *
            "distance $(round(Float64(real(defect)); sigdigits = 2)) to its " *
            "$(_symmetry_part(sym))). The average is defined only for a reference with that " *
            "symmetry; the reference is projected onto it, so the estimate is an " *
            "approximation. Give the reference that symmetry, or drop `symmetrize` on :$name."
    end
    return nothing
end
