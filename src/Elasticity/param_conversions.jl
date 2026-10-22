# =============================================================================
#  param_conversions.jl — physical-parameter interpretations of the raw
#  symmetry-class coefficients extracted by TensND (`get_data`, `get_ℓ8`,
#  `best_fit_iso`/`best_fit_ti`/`best_fit_ortho`).
#
#  TensND only knows the projector coefficients — (α, β) on {𝕁, 𝕂} for an
#  isotropic tensor, the Walpole (ℓ₁..ℓ₆) for a TI tensor, the 9 Cᵢⱼ for an
#  orthotropic one — it has no notion of "stiffness" vs "compliance": a
#  `TensISO` could equally represent either. The physical INTERPRETATION
#  belongs here, in MeanFieldHomogenization, which knows which role a given tensor
#  plays in a given computation.
#
#  For a COMPLIANCE tensor S = C⁻¹, do NOT duplicate every function below
#  with a "_compliance" variant: `inv` on the structured TensND types
#  already gives the exact reciprocal on the same projector/Walpole/Cᵢⱼ
#  basis (e.g. `inv(TensISO(α,β)) == TensISO(1/α,1/β)`), so simply call
#  `k_mu(inv(S))`, `E_nu(inv(S))`, etc.
# =============================================================================

# ── Isotropic: (k, μ) / (E, ν) ────────────────────────────────────────────────
#
# `k_mu`, `iso_stiffness`, `E_nu` and `iso_stiffness_E_nu` are TensND's: generic
# tensor algebra, imported by `Elasticity` and re-exported by the package.

# ── Transversely isotropic: Hoenig (1978) parameters ─────────────────────────
#
# TensND's `arg_TI_Hoenig` / `tens_TI_Hoenig` carry the same five numbers on the
# COMPLIANCE, as (E, ν₁, ν₂, H, Γ); these two read and build the STIFFNESS, with
# MFH's names and order (E1, h, nu1, nu2, gamma). The closed forms are TensND's.

"""
    hoenig_params(t::TensND.TensTI{4}) -> (E1, h, nu1, nu2, gamma)
    hoenig_params(t::TensND.AbstractTens{4,3}, axis) -> (E1, h, nu1, nu2, gamma)

Hoenig (1978) parametrization of a transversely isotropic **stiffness**
tensor: `E1` the in-plane Young's modulus, `h` the axial/in-plane modulus
ratio, `nu1` the in-plane Poisson's ratio, `nu2` the out-of-plane Poisson's
ratio, and `gamma` the shear anisotropy ratio: TensND's
`TensND.arg_TI_Hoenig` of the compliance `inv(t)`.

For a tensor that is not already `TensTI`, the 2-argument form projects onto
the TI span about `axis` first (`TensND.best_fit_ti`). For a compliance tensor,
use `hoenig_params(inv(S))` / `hoenig_params(inv(S), axis)`.
"""
function hoenig_params(t::TensND.TensTI{4})
    E, ν₁, ν₂, H, Γ = TensND.arg_TI_Hoenig(inv(t))
    return (E1 = E, h = H, nu1 = ν₁, nu2 = ν₂, gamma = Γ)
end
hoenig_params(t::TensND.AbstractTens{4, 3}, axis) = hoenig_params(TensND.best_fit_ti(t, axis))

"""
    hoenig_stiffness(E1, h, nu1, nu2, gamma, axis) -> TensND.TensTI{4,T,5}

Build the TI stiffness tensor from its Hoenig (1978) parameters and symmetry
axis — the reciprocal of [`hoenig_params`](@ref): the inverse of TensND's
`TensND.tens_TI_Hoenig` compliance.
"""
hoenig_stiffness(E1, h, nu1, nu2, gamma, axis) =
    inv(TensND.tens_TI_Hoenig(E1, nu1, nu2, h, gamma, axis))
