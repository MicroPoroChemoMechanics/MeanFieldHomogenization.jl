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

"""
    k_mu(C::TensND.TensISO{4}) -> (k, mu)

Bulk and shear modulus of an isotropic **stiffness** tensor
``\\mathbb{C} = 3k\\,\\mathbb{J} + 2\\mu\\,\\mathbb{K}``. For a compliance tensor `S`, use `k_mu(inv(S))`.

For a tensor that is not already a `TensISO`, project first:
`k_mu(best_fit_iso(C))`.
"""
function k_mu(C::TensND.TensISO{4})
    α, β = TensND.get_data(C)
    return α / 3, β / 2
end

"""
    iso_stiffness(k, mu) -> TensND.TensISO{4}

Build the isotropic stiffness tensor ``\\mathbb{C} = 3k\\,\\mathbb{J} + 2\\mu\\,\\mathbb{K}`` from ``(k, \\mu)`` — the
reciprocal of [`k_mu`](@ref).
"""
iso_stiffness(k, mu) = TensND.TensISO{3}(3 * k, 2 * mu)

"""
    E_nu(C::TensND.TensISO{4}) -> (E, nu)

Young's modulus and Poisson's ratio of an isotropic **stiffness** tensor.
For a compliance tensor `S`, use `E_nu(inv(S))`.
"""
function E_nu(C::TensND.TensISO{4})
    k, mu = k_mu(C)
    E = 9 * k * mu / (3 * k + mu)
    nu = (3 * k - 2 * mu) / (2 * (3 * k + mu))
    return E, nu
end

"""
    iso_stiffness_E_nu(E, nu) -> TensND.TensISO{4}

Build the isotropic stiffness tensor from Young's modulus and Poisson's
ratio — the reciprocal of [`E_nu`](@ref).
"""
function iso_stiffness_E_nu(E, nu)
    k = E / (3 * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return iso_stiffness(k, mu)
end

# ── Transversely isotropic: Hoenig (1978) parameters ─────────────────────────
#
# Walpole ↔ Hoenig, verified against the reference `tensor(array([c1..c5]))`
# constructor: the Hoenig array [c1,c2,c3,c4,c5] IS the Walpole
# (ℓ1,ℓ2,ℓ3,ℓ5,ℓ6) tuple directly (same ordering, no re-indexing) —
# cross-checked by building the 6×6 Kelvin-Mandel matrix both ways and
# comparing entry by entry.
#
#   d  = 1 - ν₁ - 2h·ν₂²
#   ℓ₁ = h·E₁·(1-ν₁)/d          ℓ₂ = E₁/d              ℓ₃ = √2·h·ν₂·E₁/d
#   ℓ₅ = E₁/(1+ν₁)              ℓ₆ = γ·ℓ₅
#
# Reciprocal (solved in closed form, round-trip verified to machine
# precision):
#
#   K  = 1 - ℓ₃²/(ℓ₁·ℓ₂)
#   ν₁ = (K - ℓ₅/ℓ₂) / (K + ℓ₅/ℓ₂)
#   h  = ℓ₁ / (ℓ₂·(1-ν₁))
#   ν₂ = ℓ₃·(1-ν₁) / (√2·ℓ₁)
#   E₁ = ℓ₅·(1+ν₁)
#   γ  = ℓ₆/ℓ₅

"""
    hoenig_params(t::TensND.TensTI{4}) -> (E1, h, nu1, nu2, gamma)
    hoenig_params(t::TensND.AbstractTens{4,3}, axis) -> (E1, h, nu1, nu2, gamma)

Hoenig (1978) parametrization of a transversely isotropic **stiffness**
tensor: `E1` the in-plane Young's modulus, `h` the axial/in-plane modulus
ratio, `nu1` the in-plane Poisson's ratio, `nu2` the out-of-plane Poisson's
ratio, and `gamma` the shear anisotropy ratio.

For a tensor that is not already `TensTI`, the 2-argument form projects onto
the TI span about `axis` first (via `TensND.proj_tens(Val(:TI), t, axis)`,
the same machinery backing `best_fit_ti`). For a compliance tensor, use
`hoenig_params(inv(S))` / `hoenig_params(inv(S), axis)`.
"""
function hoenig_params(t::TensND.TensTI{4})
    ℓ = TensND.get_ℓ8(t)
    ℓ1, ℓ2, ℓ3, ℓ5, ℓ6 = ℓ[1], ℓ[2], ℓ[3], ℓ[5], ℓ[6]
    K = 1 - ℓ3^2 / (ℓ1 * ℓ2)
    nu1 = (K - ℓ5 / ℓ2) / (K + ℓ5 / ℓ2)
    h = ℓ1 / (ℓ2 * (1 - nu1))
    nu2 = ℓ3 * (1 - nu1) / (sqrt(2) * ℓ1)
    E1 = ℓ5 * (1 + nu1)
    gamma = ℓ6 / ℓ5
    return (E1 = E1, h = h, nu1 = nu1, nu2 = nu2, gamma = gamma)
end
hoenig_params(t::TensND.AbstractTens{4, 3}, axis) =
    hoenig_params(TensND.proj_tens(Val(:TI), t, axis)[1])

"""
    hoenig_stiffness(E1, h, nu1, nu2, gamma, axis) -> TensND.TensTI{4,T,5}

Build the TI stiffness tensor from its Hoenig (1978) parameters and symmetry
axis — the reciprocal of [`hoenig_params`](@ref).
"""
function hoenig_stiffness(E1, h, nu1, nu2, gamma, axis)
    d = 1 - nu1 - 2 * h * nu2^2
    ℓ1 = h * E1 * (1 - nu1) / d
    ℓ2 = E1 / d
    ℓ3 = sqrt(2) * h * nu2 * E1 / d
    ℓ5 = E1 / (1 + nu1)
    ℓ6 = gamma * ℓ5
    return TensND.TensTI{4}(ℓ1, ℓ2, ℓ3, ℓ5, ℓ6, axis)
end
