# =============================================================================
#  dispatch.jl
#
#  Centralized `_resolve_algo(Val(method), incl, C₀)` dispatch — single
#  place where the symbol `:auto` / `:residues` / `:decuhr` is translated
#  into an [`AbstractAlgorithm`](@ref) instance, taking into account the
#  material symmetry class (TensND type of `C₀`) *and* the inclusion
#  class (`AbstractEllipsoidalInclusion` vs `AbstractCrack`).
#
#  Rules
#  -----
#  - Isotropic matrix (`TensISO`) of any order, any dimension  → Analytical
#  - 2D stiffness / conductivity (any inclusion)               → Analytical
#  - 3D conductivity (2nd-order tensor)                        → Analytical
#  - 3D anisotropic elasticity (4th-order `AbstractTens{4,3}`) :
#      * `AbstractEllipsoidalInclusion`, `:auto`  → DECUHR if its extension is
#                                                   loaded, else NestedQuadGK
#      * `AbstractEllipsoidalInclusion`, `:residues` → Residue (explicit only)
#      * `AbstractEllipsoidalInclusion`, `:decuhr` / `:nestedquadgk` → as named
#      * `AbstractCrack`, TI + aligned with n̂     → Analytical
#      * `AbstractCrack`, `:auto`                 → same cubature rule as above
#      * `AbstractCrack`, `:residues`             → Residue (explicit only)
#
#  The rules that depend on the *inclusion* class are injected at the end
#  of the file so that they can refer to the abstract types defined in
#  `abstractions.jl` without introducing cycles.
#
#  Every dispatch method lists BOTH the inclusion and the TensND type as
#  method parameters.  Matching all three axes (method symbol × inclusion
#  class × material symmetry) in a single signature avoids the
#  classical "diamond" ambiguity between the inclusion-refined and the
#  symmetry-refined specializations.
# =============================================================================

# ─── TensISO (any kind) → Analytical — most specific on symmetry ───────────

_resolve_algo(::Val, ::AbstractInclusion, ::TensND.TensISO) = Analytical()
_resolve_algo(::Val{:auto}, ::AbstractInclusion, ::TensND.TensISO) = Analytical()
_resolve_algo(::Val{:residues}, ::AbstractInclusion, ::TensND.TensISO) = Analytical()
_resolve_algo(::Val{:decuhr}, ::AbstractInclusion, ::TensND.TensISO) = Analytical()
_resolve_algo(::Val{:nestedquadgk}, ::AbstractInclusion, ::TensND.TensISO) = Analytical()

# Specific inclusion + TensISO — keep the same rule (needed to avoid
# ambiguity with the inclusion-refined 3D-aniso methods below).
_resolve_algo(::Val, ::AbstractEllipsoidalInclusion, ::TensND.TensISO) = Analytical()
_resolve_algo(::Val{:auto}, ::AbstractEllipsoidalInclusion, ::TensND.TensISO) = Analytical()
_resolve_algo(::Val{:residues}, ::AbstractEllipsoidalInclusion, ::TensND.TensISO) = Analytical()
_resolve_algo(::Val{:decuhr}, ::AbstractEllipsoidalInclusion, ::TensND.TensISO) = Analytical()
_resolve_algo(::Val{:nestedquadgk}, ::AbstractEllipsoidalInclusion, ::TensND.TensISO) = Analytical()

_resolve_algo(::Val, ::AbstractCrack, ::TensND.TensISO) = Analytical()
_resolve_algo(::Val{:auto}, ::AbstractCrack, ::TensND.TensISO) = Analytical()
_resolve_algo(::Val{:residues}, ::AbstractCrack, ::TensND.TensISO) = Analytical()
_resolve_algo(::Val{:decuhr}, ::AbstractCrack, ::TensND.TensISO) = Analytical()
_resolve_algo(::Val{:nestedquadgk}, ::AbstractCrack, ::TensND.TensISO) = Analytical()

# ─── 2D elasticity → Analytical ──────────────────────────────────────────────

_resolve_algo(::Val, ::AbstractInclusion, ::TensND.AbstractTens{4, 2}) = Analytical()
_resolve_algo(::Val, ::AbstractEllipsoidalInclusion, ::TensND.AbstractTens{4, 2}) = Analytical()

# ─── Conductivity (2nd-order) → Analytical ───────────────────────────────────

_resolve_algo(::Val, ::AbstractInclusion, ::TensND.AbstractTens{2, 3}) = Analytical()
_resolve_algo(::Val, ::AbstractEllipsoidalInclusion, ::TensND.AbstractTens{2, 3}) = Analytical()
_resolve_algo(::Val, ::AbstractInclusion, ::TensND.AbstractTens{2, 2}) = Analytical()
_resolve_algo(::Val, ::AbstractEllipsoidalInclusion, ::TensND.AbstractTens{2, 2}) = Analytical()

# ─── 3D anisotropic elasticity ───────────────────────────────────────────────

# `:auto` picks a CUBATURE, never the residue algorithm — the same choice
# ECHOES makes with its `NUMINT` default.  The residue path is faster
# (~4 ms vs ~11 ms on a triaxial ellipsoid) but it is not robust: its
# acoustic polynomial degenerates whenever the reference is anisotropic in
# *type* and isotropic in *value*, and it then returns `NaN` / throws a
# `DomainError` rather than a number.  That reference is not exotic — it is
# exactly what the differential and self-consistent schemes feed back at
# their first step, and `Dilute` / `MoriTanaka` reach it too as soon as an
# anisotropically-typed matrix happens to hold isotropic values.  A default
# must not fail on a legitimate input, so `Residue` is now available on
# explicit `method = :residues` only.
#
# Between the two cubatures, `DECUHR` is the cheaper one (~11 ms vs ~31 ms)
# but lives behind a weak dependency, so it is used when its extension is
# active and the type-generic `NestedQuadGK` otherwise.  Loading `DECUHR`
# therefore changes the default backend — hence the accuracy the effective
# properties are computed to (~1e-9 against ~1e-14); pass `method`
# explicitly wherever that must not depend on the session.
#
# Non-`Float64` coefficient types (`ForwardDiff.Dual`, `Complex`, symbolic)
# keep the type-generic `NestedQuadGK`: it is what makes AD through a
# self-consistent iteration with a generically-anisotropic running estimate
# possible, and `DECUHR`'s support beyond `Dual` is not established.
function _aniso_default_algo(C₀::TensND.AbstractTens)
    eltype(TensND.get_array(C₀)) === Float64 || return NestedQuadGK()
    return _decuhr_available() ? DECUHR() : NestedQuadGK()
end

_decuhr_available() =
    Base.get_extension(parentmodule(@__MODULE__), :MeanFieldHomogenizationDECUHRExt) !== nothing

# Ellipsoidal inclusions — 3D anisotropic default (see `_aniso_default_algo`)
_resolve_algo(::Val{:auto}, ::AbstractEllipsoidalInclusion, C₀::TensND.AbstractTens{4, 3}) = _aniso_default_algo(C₀)
_resolve_algo(::Val{:residues}, ::AbstractEllipsoidalInclusion, ::TensND.AbstractTens{4, 3}) = Residue()
_resolve_algo(::Val{:decuhr}, ::AbstractEllipsoidalInclusion, ::TensND.AbstractTens{4, 3}) = DECUHR()
_resolve_algo(::Val{:nestedquadgk}, ::AbstractEllipsoidalInclusion, ::TensND.AbstractTens{4, 3}) = NestedQuadGK()

# ─── Cylinder dispatch — anisotropic elasticity (3D) routes to the dedicated
# 1D transverse-plane quadrature.  The residue algorithm is not applicable
# to a cylinder (acoustic polynomial degenerates), so `:residues` silently
# falls back to `CylinderQuadrature`.  The rules are injected by the
# Elasticity sub-module (see `Elasticity.jl`) to avoid a Core→Elasticity
# dependency — this file only declares the infrastructure.

# Generic inclusion fallback (also used by `AbstractCrack` before the
# TI-aligned refinement injected from the `Cracks` sub-module).
_resolve_algo(::Val{:auto}, ::AbstractInclusion, C₀::TensND.AbstractTens{4, 3}) = _aniso_default_algo(C₀)
_resolve_algo(::Val{:residues}, ::AbstractInclusion, ::TensND.AbstractTens{4, 3}) = Residue()
_resolve_algo(::Val{:decuhr}, ::AbstractInclusion, ::TensND.AbstractTens{4, 3}) = DECUHR()
_resolve_algo(::Val{:nestedquadgk}, ::AbstractInclusion, ::TensND.AbstractTens{4, 3}) = NestedQuadGK()

# Plain catch-all for the cases where a sub-module passes `method` symbols
# we don't know about (e.g. a user extension): default to Analytical.
_resolve_algo(::Val, ::AbstractInclusion, ::TensND.AbstractTens) = Analytical()

# ─── Public helper ───────────────────────────────────────────────────────────

"""
    _resolve_algo(Val(method), incl, C₀) -> AbstractAlgorithm

Translate a method symbol (`:auto`, `:residues`, `:decuhr`) into the
concrete [`AbstractAlgorithm`](@ref) instance, taking the inclusion
class and the TensND symmetry class of `C₀` into account.  Centralized
here so that every high-level entry point (`hill_tensor`,
`cod_tensor`, `compliance_contribution`, `sif`, `dif`) shares a single
truth table — new dispatch rules are added as new methods of this
function.
"""
_resolve_algo
