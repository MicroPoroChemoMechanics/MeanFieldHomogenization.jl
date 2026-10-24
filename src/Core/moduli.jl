# =============================================================================
#  moduli.jl
#
#  The modulus readers of the isotropic and transversely isotropic classes are
#  TensND's: `E_nu`, `k_mu`, `mu_nu`, `poisson_ratio` (3-D and plane-strain 2-D)
#  and `ti_args_hoenig`. The names this package used to export for them are kept,
#  deprecated, and point there.
#
#  `extract_iso_moduli` of a 2-D tensor read it with the 3-D formulas
#  (`k = α/3`), which gave the Green operator of a 2-D medium the Poisson ratio
#  `(α − β)/(2α + β)` where its Hill tensor uses the plane-strain `(α − β)/(2α)`;
#  `TensND.E_nu` returns the latter, the one of the 3-D material whose
#  plane-strain stiffness the 2-D tensor is.
# =============================================================================

Base.@deprecate extract_iso_moduli(C₀) TensND.E_nu(C₀)
Base.@deprecate extract_iso_conductivity(K₀) first(TensND.get_data(K₀))

function extract_ti_moduli(C₀::TensND.AbstractTens{4, 3}, n̂)
    Base.depwarn(
        "`extract_ti_moduli(C₀, n̂)` is deprecated: use `TensND.ti_args_hoenig(inv(C₀))` " *
            "(after `TensND.best_fit_ti(C₀, n̂)` for a tensor that is not a `TensTI`), " *
            "or `hoenig_params`.",
        :extract_ti_moduli,
    )
    C = C₀ isa TensND.TensTI{4} ? C₀ : TensND.best_fit_ti(C₀, collect(TensND._extract_vec(n̂)))
    E, ν₁, ν₂, H, Γ = TensND.ti_args_hoenig(inv(C))
    return (E = E, H = H, ν₁ = ν₁, ν₂ = ν₂, Γ = Γ)
end
