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
[mori1973, christensen1990](@cite). Dispatches on the order of
the matrix property tensor — 4th order for elasticity (`:C`), 2nd order
for conductivity (`:K`).
"""
function _evaluate(rve::RVE, scheme::MoriTanaka, ::Val{p}; kw...) where {p}
    m = matrix_name(scheme, rve)
    P₀ = phase_property(rve, m, p)
    return _mt_dispatch(rve, P₀, m, Val(p); kw...)
end

_mt_dispatch(rve, C₀::TensND.AbstractTens{4, 3}, m::Symbol, ::Val{p}; kw...) where {p} =
    _mt_4(rve, C₀, m, Val(p); kw...)
_mt_dispatch(rve, K₀::TensND.AbstractTens{2, 3}, m::Symbol, ::Val{p}; kw...) where {p} =
    _mt_2(rve, K₀, m, Val(p); kw...)

# ── 4th-order (elasticity) ──────────────────────────────────────────────────
function _mt_4(rve, C₀::TensND.AbstractTens{4, 3}, m::Symbol, ::Val{p}; kw...) where {p}
    f_m = volume_fraction(rve, m)
    Iref = _identity_like(C₀)
    A_avg = f_m * Iref          # ⟨A_dil⟩, matrix carries A_dil = I
    Nsum = zero(C₀)
    for name in inclusion_phase_names(rve, m)
        ΔA, N = _mt_terms(rve.amounts[name], rve, name, p, C₀; kw...)
        A_avg += ΔA
        Nsum += N
    end
    return C₀ + Nsum ⊡ inv(A_avg)
end

# ── 2nd-order (conductivity) ────────────────────────────────────────────────
function _mt_2(rve, K₀::TensND.AbstractTens{2, 3}, m::Symbol, ::Val{p}; kw...) where {p}
    f_m = volume_fraction(rve, m)
    Iref = _identity_like(K₀)
    A_avg = f_m * Iref
    Nsum = zero(K₀)
    for name in inclusion_phase_names(rve, m)
        ΔA, N = _mt_terms(rve.amounts[name], rve, name, p, K₀; kw...)
        A_avg += ΔA
        Nsum += N
    end
    return K₀ + Nsum ⋅ inv(A_avg)
end

# The terms a phase adds to ⟨A⟩ and to the contribution sum, in one loop over
# the phases so that the sums keep their order.
#
# A phase with a volume: its dilute concentration weighted by its fraction. The
# bundled helper shares one localization solve between `A_dil` and `N`, sums
# the layers of an internally heterogeneous inclusion (LayeredSphere) instead
# of reading a phase property that does not represent it, and symmetrizes the
# product (C_i − C₀):A, not A alone.
function _mt_terms(::AbstractAmount, rve, name, p, P₀; kw...)
    A_dil, N = _phase_dilute_and_contribution(rve, name, p, P₀; kw...)
    return volume_fraction(rve, name) * A_dil, N
end

# A crack family, in the ECHOES `B · A⁻¹` form: strain-stress term
# ε sym(H_c) : C₀, no stress-stress term (traction-free), and the stiffness
# contribution −ε C₀ : sym(H_c) : C₀ as in the additive form. The crack term in
# the denominator ⟨A⟩ is what prevents the spurious percolation of the
# additive form at moderate density.
function _mt_terms(::CrackDensity, rve, name, p, C₀::TensND.AbstractTens{4, 3}; kw...)
    H, N = _phase_compliance_and_contribution(rve, name, p, C₀; kw...)
    return H ⊡ C₀, N
end

function _mt_terms(::CrackDensity, rve, name, p, K₀::TensND.AbstractTens{2, 3}; kw...)
    R, N = _phase_compliance_and_contribution(rve, name, p, K₀; kw...)
    return R ⋅ K₀, N
end

# ── Identity tensor matching the algebra of the property tensor ─────────────

"""
    _identity_like(P) -> AbstractTens

Identity tensor of the same order/dimension as `P`. Used as the ``\\mathbb{A}^{\\mathrm{dil}}``
weight of the matrix in average-strain schemes (Mori-Tanaka,
self-consistent, …).
"""
_identity_like(C::TensND.AbstractTens{4, 3}) =
    TensND.tens_Id4(Val(3), Val(eltype(C)))
_identity_like(K::TensND.AbstractTens{2, 3}) =
    TensND.tens_Id2(Val(3), Val(eltype(K)))
