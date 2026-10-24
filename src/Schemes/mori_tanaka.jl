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
#  For cracks (`CrackDensity`) the volume fraction vanishes while the strain
#  the openings carry does not: a family adds κ ε H : C₀ to the denominator
#  (the limit of f A) and −κ ε C₀ : H : C₀ to the numerator (the limit of
#  f (C − C₀) : A, a traction-free crack carrying no stress), κ being 4π/3
#  or π. With cracks alone this gives S_MT = S₀ + Σ κ ε H, the
#  non-interaction approximation.
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
    return _mt_kernel(rve, P₀, m, Val(p); kw...)
end

# The phase properties come out of heterogeneous containers, so each phase's
# terms are added to the sums inside one call compiled for their concrete
# types (`_mt_accumulate`), and the closure is another: a few dynamic
# dispatches per phase instead of one per tensor operation. The 4th-order
# (elasticity) and 2nd-order (conductivity) kernels differ only by the
# contraction, which `_compose` carries.
function _mt_kernel(rve, P₀::_PropertyTens, m::Symbol, ::Val{p}; kw...) where {p}
    A_avg = volume_fraction(rve, m) * one(P₀)   # ⟨A_dil⟩, matrix carries A_dil = I
    Nsum = zero(P₀)
    for name in inclusion_phase_names(rve, m)
        A_avg, Nsum = _mt_step(rve.amounts[name], A_avg, Nsum, rve, name, p, P₀; kw...)
    end
    return _mt_close(P₀, Nsum, A_avg)
end

_mt_close(P₀, Nsum, A_avg) = P₀ + _compose(Nsum, inv(A_avg))

# The terms a phase adds to ⟨A⟩ and to the contribution sum, in one loop over
# the phases so that the sums keep their order.
#
# A phase with a volume: its dilute concentration weighted by its fraction. The
# bundled helper shares one concentration solve between `A_dil` and `N`, sums
# the layers of an internally heterogeneous inclusion (LayeredSphere) instead
# of reading a phase property that does not represent it, and symmetrizes the
# product (C_i − C₀):A, not A alone.
function _mt_step(::AbstractAmount, A_avg, Nsum, rve, name, p, P₀; kw...)
    A_dil, N = _phase_dilute_and_contribution(rve, name, p, P₀; kw...)
    return _mt_accumulate(A_avg, Nsum, volume_fraction(rve, name), A_dil, N)
end

# A crack family, in the ECHOES `B · A⁻¹` form: strain-stress term
# ε sym(H_c) : C₀, no stress-stress term (traction-free), and the stiffness
# contribution −ε C₀ : sym(H_c) : C₀ as in the additive form. The crack term in
# the denominator ⟨A⟩ is what prevents the spurious percolation of the
# additive form at moderate density.
function _mt_step(::CrackDensity, A_avg, Nsum, rve, name, p, P₀; kw...)
    H, N = _phase_compliance_and_contribution(rve, name, p, P₀; kw...)
    return _mt_accumulate_crack(A_avg, Nsum, H, P₀, N)
end

_mt_accumulate(A_avg, Nsum, φ, A_dil, N) = (A_avg + φ * A_dil, Nsum + N)
_mt_accumulate_crack(A_avg, Nsum, H, P₀, N) = (A_avg + _compose(H, P₀), Nsum + N)

# The identity of the property's algebra is `one(P)` (TensND): same order and
# dimension, same element type. It weighs the matrix in average-strain schemes
# (Mori-Tanaka, self-consistent, …).
