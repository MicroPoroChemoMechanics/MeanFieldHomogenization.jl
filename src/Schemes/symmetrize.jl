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
#  Reference-medium projection for the localization-tensor computation.
# =============================================================================

"""
    _project_matrix(P₀::AbstractTens, sym::AbstractSymmetrize) -> AbstractTens

Project the reference medium `P₀` before computing the localization tensor
of a phase declaring the orientation distribution `sym`.

- `NoSymmetrize` : passthrough.
- `IsoSymmetrize` : isotropic average of `P₀` (the inclusion's Hill tensor
  in an isotropic matrix always has an analytical branch).
- `TISymmetrize` : controlled by `sym.matrix_projection` :
    * `:iso` (default) — isotropic average of `P₀`.  Approximation whenever
      `P₀` is not isotropic ; exact at the isotropic fixed point of the SC
      iteration (where the reference converges to its isotropic average).
      Rationale : an inclusion family at polar angle θ ≠ 0 from the
      symmetrize axis is *not* coaxial with a TI reference, so the
      TI-coaxial analytical Hill branch does not apply ; the isotropic
      projection guarantees an analytical, ForwardDiff-compatible branch
      for every orientation.
    * `:none` — no projection ; non-coaxial anisotropic references route
      through the general-anisotropy `NestedQuadGK` Hill branch
      (ForwardDiff-compatible, quadrature-priced).
    * `:ti` — best-fit TI projection of `P₀` about `sym.axis`
      (`TensTI{4,T,5}`) ; only meaningful when the phase's inclusions are
      coaxial with the axis.

The phase contribution is in all cases still exactly averaged by the
outgoing `_apply_symmetrize`, so the outer symmetry semantics are
preserved.
"""
_project_matrix(P₀::TensND.AbstractTens, ::NoSymmetrize) = P₀
_project_matrix(P₀::TensND.AbstractTens, ::IsoSymmetrize) = MFH_Core.isotropify(P₀)
function _project_matrix(P₀::TensND.AbstractTens, sym::TISymmetrize)
    mp = sym.matrix_projection
    mp === :none && return P₀
    mp === :ti && return best_fit_ti(P₀, sym.axis)
    return MFH_Core.isotropify(P₀)
end

# =============================================================================
#  Best-fit projections (reporting / parameter extraction only)
# =============================================================================
#
#  `best_fit_iso` / `best_fit_ti` / `best_fit_ortho` were three one-line
#  wrappers over `TensND.proj_tens(…)[1]` and have moved to TensND itself,
#  next to `proj_tens` and the `*_params_from_KM` conversions they belong with.
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
