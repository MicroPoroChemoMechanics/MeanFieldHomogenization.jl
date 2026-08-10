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
#  - two balls / two disks + isotropic reference        → Analytical (exact)
#  - any other pair of ellipsoids + isotropic reference → Multipole
#  - any pair + anisotropic reference (3D, or 2D cond.) → Multipole
#  - any pair + anisotropic reference, 2D elasticity    → Multipole, and
#                                                         `green_operator` then
#                                                         refuses (Stroh)
#  - explicit `:quadrature`                             → NestedQuadGK
#  - explicit `:multipole`                              → Multipole
#  - explicit `:analytical`                             → Analytical (errors in
#                                                         the kernel table if no
#                                                         closed form exists)
#
#  Anisotropic references ARE supported (since `Core/green_aniso.jl` landed):
#  they resolve to `Multipole`, because the exactness of the closed forms rests
#  on the isotropic Green function being biharmonic — which a general
#  anisotropic one is not — so even a ball pair needs the truncated expansion
#  there.  The single remaining gap is plane-strain elasticity with an
#  anisotropic reference, which would need the Stroh formalism and is refused
#  with a message naming the limitation.
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

# Anisotropic reference: the closed forms do not apply — their exactness rests
# on the isotropic Green function being biharmonic, which a general anisotropic
# one is not — so even a ball pair goes through the truncated expansion.
_resolve_pair_algo(::Val{:auto}, _, _, ::TensND.AbstractTens{4, 3}) = Multipole()
_resolve_pair_algo(::Val{:auto}, _, _, ::TensND.AbstractTens{2, 3}) = Multipole()
_resolve_pair_algo(::Val{:auto}, _, _, ::TensND.AbstractTens{2, 2}) = Multipole()

# Plane-strain elasticity: resolved like any other case here. When the
# reference is genuinely anisotropic the refusal comes from `green_operator`,
# which owns that message — putting a throwing method on this argument
# position instead would be ambiguous with the explicit `:quadrature` /
# `:multipole` rules below, which are more specific on the method symbol.
_resolve_pair_algo(::Val{:auto}, _, _, ::TensND.AbstractTens{4, 2}) = Multipole()

_resolve_pair_algo(::Val{:analytical}, _, _, ::TensND.AbstractTens) = Analytical()
_resolve_pair_algo(::Val{:multipole}, _, _, ::TensND.AbstractTens) = Multipole()
_resolve_pair_algo(::Val{:quadrature}, _, _, ::TensND.AbstractTens) = NestedQuadGK()

# Catch-all for an unknown method symbol on a supported reference.
_resolve_pair_algo(::Val, _, _, ::TensND.AbstractTens) = Multipole()
