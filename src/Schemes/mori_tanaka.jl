# =============================================================================
#  mori_tanaka.jl — Mori-Tanaka scheme.
#
#  Each inclusion experiences the *average matrix strain* as far field. The
#  effective stiffness reads (for solid inclusions only)
#
#       C_MT = C₀ + (Σ_i fᵢ Nᵢ) : (f_m · I + Σ_i fᵢ A_dil^{(i)})⁻¹
#
#  with A_dil^{(i)} = (I + P_0 : (C_i − C_0))⁻¹ the dilute strain
#  concentration tensor and Nᵢ = (Cᵢ − C₀) : A_dil^{(i)}.
#
#  For cracks (`CrackDensity`) the strain concentration tensor is singular;
#  we add their density-weighted stiffness reduction
#  -C₀ : H : C₀ to the numerator (consistent with Kachanov 1992 in the
#  dilute crack limit) and skip the denominator term — physically the
#  crack volume → 0 limit anyway sends f_crack to zero.
# =============================================================================

"""
    _evaluate(rve, ::MoriTanaka, ::Val{p}; kw...) -> AbstractTens

Mori-Tanaka scheme for property `:p`
([mori1973](@cite);
[christensen1990](@cite)). Dispatches on the order of
the matrix property tensor — 4th order for elasticity (`:C`), 2nd order
for conductivity (`:K`).
"""
function _evaluate(rve::RVE, ::MoriTanaka, ::Val{p}; kw...) where {p}
    P₀ = matrix_property(rve, p)
    return _mt_dispatch(rve, P₀, Val(p); kw...)
end

_mt_dispatch(rve, C₀::TensND.AbstractTens{4, 3}, ::Val{p}; kw...) where {p} =
    _mt_4(rve, C₀, Val(p); kw...)
_mt_dispatch(rve, K₀::TensND.AbstractTens{2, 3}, ::Val{p}; kw...) where {p} =
    _mt_2(rve, K₀, Val(p); kw...)

# ── 4th-order (elasticity) ──────────────────────────────────────────────────
function _mt_4(rve, C₀::TensND.AbstractTens{4, 3}, ::Val{p}; kw...) where {p}
    f_m = matrix_volume_fraction(rve)
    Iref = _identity_like(C₀)
    A_avg = f_m * Iref          # ⟨A_dil⟩, matrix carries A_dil = I
    Nsum = zero(C₀)
    for name in inclusion_phase_names(rve)
        a = rve.amounts[name]
        if a isa VolumeFraction
            # Apply per-phase orientation symmetrize via the bundled helper,
            # which shares the single localization solve between `A_dil` and
            # the contribution `N` (they used to be computed independently,
            # i.e. two `hill_tensor` calls with identical arguments).
            # `N` goes through the trait so that internally heterogeneous
            # inclusions (LayeredSphere) sum over their layers instead of
            # using a phase property that does not represent them, and the
            # helper symmetrizes the PRODUCT (C_i − C₀):A, not just A.
            A_dil, N = _phase_dilute_and_contribution(rve, name, p, C₀; kw...)
            A_avg += scale_by_amount(a, A_dil)
            Nsum += N
        else  # CrackDensity — ECHOES `B · A^{-1}` form.
            # Strain-Stress contribution: A_crack = ε·sym(H_c)·C₀.
            # Stress-Stress contribution: 0 (traction-free).
            # Stiffness contribution into Nsum: ΔC_crack = -ε·C₀·sym(H_c)·C₀
            # (same as the additive form).  The non-trivial change is the
            # crack term in the denominator A_avg, which prevents the
            # additive form's spurious percolation at moderate density.
            H, N = _phase_compliance_and_contribution(rve, name, p, C₀; kw...)
            A_avg += H ⊡ C₀
            Nsum += N
        end
    end
    return C₀ + Nsum ⊡ inv(A_avg)
end

# ── 2nd-order (conductivity) ────────────────────────────────────────────────
function _mt_2(rve, K₀::TensND.AbstractTens{2, 3}, ::Val{p}; kw...) where {p}
    f_m = matrix_volume_fraction(rve)
    Iref = _identity_like(K₀)
    A_avg = f_m * Iref
    Nsum = zero(K₀)
    for name in inclusion_phase_names(rve)
        a = rve.amounts[name]
        if a isa VolumeFraction
            A_dil, N = _phase_dilute_and_contribution(rve, name, p, K₀; kw...)
            A_avg += scale_by_amount(a, A_dil)
            Nsum += N
        else  # CrackDensity — ECHOES `B · A^{-1}` form.
            R, N = _phase_compliance_and_contribution(rve, name, p, K₀; kw...)
            A_avg += R ⋅ K₀
            Nsum += N
        end
    end
    return K₀ + Nsum ⋅ inv(A_avg)
end

# ── Identity tensor matching the algebra of the property tensor ─────────────

"""
    _identity_like(P) -> AbstractTens

Identity tensor of the same order/dimension as `P`. Used as the `A_dil`
weight of the matrix in average-strain schemes (Mori-Tanaka,
self-consistent, …).
"""
_identity_like(C::TensND.AbstractTens{4, 3}) =
    TensND.tens_Id4(Val(3), Val(eltype(C)))
_identity_like(K::TensND.AbstractTens{2, 3}) =
    TensND.tens_Id2(Val(3), Val(eltype(K)))
