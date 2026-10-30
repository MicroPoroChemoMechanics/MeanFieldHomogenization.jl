# =============================================================================
#  symmetrize.jl — orientation-distribution treatment of a phase's tensors.
#
#  TWO distinct mechanisms (exact rotation average vs best-fit projection);
#  they must never be conflated :
#
#  1. EXACT rotation-group averaging (runtime, inside scheme kernels) —
#     `_apply_symmetrize(t, sym)` delegates to `Core.isotropify` /
#     `Core.transverse_isotropify`.  Valid for minor-symmetric tensors with
#     or without major symmetry ; the TI average returns the full 8-parameter
#     axially-invariant tensor (`TensND.TensTI{4,T,8}`), preserving ℓ₃ ≠ ℓ₄
#     and the antisymmetric azimuthal couplings (ℓ₇, ℓ₈) that concentration
#     tensors generally carry.  At 2nd order the antisymmetric in-plane part
#     is preserved (`TensTI{2,T,3}`).
#
#  2. BEST-FIT projection (reporting / parameter extraction ONLY) —
#     `best_fit_ti(t, axis)` (→ major-symmetric `TensTI{4,T,5}`) and
#     `best_fit_iso(t)`.  This is the orthogonal projection onto the
#     symmetric Walpole span, the analog of echoes' `.paramsym(sym=TI)`.
#     Do NOT use it inside scheme kernels : it silently drops the
#     non-major-symmetric content of concentration tensors.
#
#  References : Walpole (1981) for the TI basis ; echoes
#  the exact azimuthal-average closed form.
# =============================================================================

"""
    _apply_symmetrize(t::AbstractTens, sym::AbstractSymmetrize) -> AbstractTens

Apply the **exact** orientation average declared by `sym` to `t`.

`NoSymmetrize` is a passthrough ; `IsoSymmetrize` returns the SO(3) average
(`TensISO`) ; `TISymmetrize(axis)` returns the azimuthal average about
`axis` (`TensTI{4,T,8}` / `TensTI{2,T,3}`, non-major-symmetric content
preserved). Implemented for tensor orders 4 (elasticity) and 2
(conductivity).
"""
_apply_symmetrize(t::TensND.AbstractTens, ::NoSymmetrize) = t

_apply_symmetrize(t::TensND.AbstractTens{4, 3}, ::IsoSymmetrize) =
    MFH_Core.isotropify(t)
_apply_symmetrize(t::TensND.AbstractTens{2, 3}, ::IsoSymmetrize) =
    MFH_Core.isotropify(t)

_apply_symmetrize(t::TensND.AbstractTens{4, 3}, sym::TISymmetrize) =
    MFH_Core.transverse_isotropify(t, sym.axis)
_apply_symmetrize(t::TensND.AbstractTens{2, 3}, sym::TISymmetrize) =
    MFH_Core.transverse_isotropify(t, sym.axis)

# =============================================================================
#  Reference-medium projection for the concentration-tensor computation.
# =============================================================================

"""
    _project_matrix(P₀::AbstractTens, sym::AbstractSymmetrize) -> AbstractTens

The reference medium `P₀` in which the concentration tensors of a phase declaring
the orientation average `sym` are computed: its exact average over the same
rotations (`Core.isotropify` for `IsoSymmetrize`, `Core.transverse_isotropify`
about the axis for `TISymmetrize`), and `P₀` itself without an average.

The average is defined only for a reference invariant under it, which the
projection then leaves unchanged; one the type already shows to be invariant (an
isotropic tensor for every average, a transversely isotropic one about the same
axis for `TISymmetrize`) is returned as it is, at no cost. Outside that domain,
`homogenize` warns (see `_warn_average_domain`).
"""
_project_matrix(P₀::TensND.AbstractTens, ::NoSymmetrize) = P₀
_project_matrix(P₀::TensND.AbstractTens, ::IsoSymmetrize) = MFH_Core.isotropify(P₀)
_project_matrix(P₀::TensND.TensISO, ::IsoSymmetrize) = P₀
_project_matrix(P₀::TensND.TensISO, ::TISymmetrize) = P₀
_project_matrix(P₀::TensND.AbstractTens, sym::TISymmetrize) = MFH_Core.transverse_isotropify(P₀, sym.axis)
_project_matrix(P₀::TensND.TensTI, sym::TISymmetrize) =
    _coaxial(TensND.axis(P₀), sym.axis) ? P₀ : MFH_Core.transverse_isotropify(P₀, sym.axis)

# Whether two axes are parallel; a fast path only (the projection is exact either
# way), so a symbolic axis simply takes the projection.
_coaxial(n, a) = _coaxial(Val(is_hard_numeric(real(promote_type(eltype(n), eltype(a))))), n, a)
_coaxial(::Val{false}, n, a) = false
function _coaxial(::Val{true}, n, a)
    d = abs(sum(map(*, n, a)))
    return d ≥ (1 - 1.0e-12) * sqrt(sum(abs2, n) * sum(abs2, a))
end

# =============================================================================
#  Best-fit projections (reporting / parameter extraction only)
# =============================================================================
#
#  `best_fit_iso` / `best_fit_ti` / `best_fit_ortho` were three one-line
#  wrappers over `TensND.project(…)[1]` and have moved to TensND itself,
#  next to `project` and the `*_params_from_KM` conversions they belong with.
#  They are re-exported here because they are part of this package's reporting
#  surface — `best_fit_ti(C, n)` is how a scheme result is turned into
#  publishable TI parameters, the analog of echoes' `.paramsym(sym = TI)`.
#
#  The distinction with the block above is the one thing not to get wrong, and
#  it is worth restating at the point of use: `_apply_symmetrize` averages,
#  these fit. Inside a scheme kernel the orientation average is
#  `Core.transverse_isotropify` / `Core.isotropify`, which keep `ℓ₃ ≠ ℓ₄` and
#  the antisymmetric azimuthal couplings a concentration tensor carries;
#  `best_fit_ti` forces major symmetry and drops them, silently. Never call it
#  from a kernel.

using TensND: best_fit_iso, best_fit_ti, best_fit_ortho
