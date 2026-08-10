# =============================================================================
#  dispatch_pair.jl
#
#  `_resolve_pair_algo(Val(method), incl_a, incl_b, P₀)` — the counterpart of
#  `_resolve_algo` for *two-inclusion* kernels (the interaction tensors of the
#  equivalent inclusion method and of the cluster model).
#
#  It is a separate function rather than a four-argument method of
#  `_resolve_algo` because the truth table is genuinely different: a closed
#  form exists only when *both* regions are balls (3D) or disks (2D) and the
#  reference medium is isotropic, whereas the one-inclusion table has a
#  closed form for every ellipsoid.
#
#  Rules
#  -----
#  - two balls / two disks + isotropic reference        → Analytical
#  - any other pair of ellipsoids + isotropic reference → Multipole
#  - explicit `:quadrature`                             → NestedQuadGK
#  - explicit `:multipole`                              → Multipole
#  - explicit `:analytical`                             → Analytical (errors in
#                                                         the kernel table if no
#                                                         closed form exists)
#
#  Anisotropic references are *not* resolved here: the real-space Green
#  operator of a generally anisotropic medium is not implemented, so the
#  entry point rejects them with a message that names the limitation instead
#  of dispatching to a kernel that would silently use an isotropic kernel.
#
#  As everywhere else in the package, new rules are added as new methods —
#  never by editing an existing one.
# =============================================================================

"""
    Multipole <: AbstractAlgorithm

Truncated multipole expansion of a two-inclusion interaction integral
(Brisard, Dormieux & Sab 2014, §4.2): the regular part of the Green operator
is Taylor-expanded about the line of centers and the resulting monomials are
integrated over the two regions in closed form. Exact for balls; an
asymptotic series in `(size / separation)` for a general ellipsoid.
"""
struct Multipole <: AbstractAlgorithm end

# ─── Ball / disk pairs in an isotropic reference — closed form ───────────────
#
# The shape predicate is a trait query rather than a type query so that any
# inclusion advertising a spherical (resp. circular) shape reaches the closed
# form, whatever concrete type carries it.

_pair_has_closed_form(incl_a, incl_b) =
    _is_ball_like(incl_a) && _is_ball_like(incl_b)

_is_ball_like(::AbstractInclusion) = false

# ─── Entry point ─────────────────────────────────────────────────────────────

"""
    _resolve_pair_algo(Val(method), incl_a, incl_b, P₀) -> AbstractAlgorithm

Translate a method symbol into the concrete [`AbstractAlgorithm`](@ref) used
to evaluate the interaction tensor between `incl_a` and `incl_b` in the
reference medium `P₀`. See the file header for the truth table.
"""
function _resolve_pair_algo(::Val{:auto}, incl_a, incl_b, ::TensND.TensISO)
    return _pair_has_closed_form(incl_a, incl_b) ? Analytical() : Multipole()
end

_resolve_pair_algo(::Val{:analytical}, _, _, ::TensND.AbstractTens) = Analytical()
_resolve_pair_algo(::Val{:multipole}, _, _, ::TensND.AbstractTens) = Multipole()
_resolve_pair_algo(::Val{:quadrature}, _, _, ::TensND.AbstractTens) = NestedQuadGK()

# Anisotropic reference — refuse rather than silently mis-evaluate.
function _resolve_pair_algo(::Val, _, _, P₀::TensND.AbstractTens)
    throw(
        ArgumentError(
            "interaction_tensor: no real-space Green operator is available for a " *
                "reference medium of type $(nameof(typeof(P₀))). Two-inclusion " *
                "interaction kernels currently require an isotropic reference " *
                "(`TensISO`); the anisotropic Green operator (Barnett-Willis angular " *
                "integral, or Pan-Chou for transverse isotropy) is not implemented."
        )
    )
end
