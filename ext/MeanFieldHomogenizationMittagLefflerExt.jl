# =============================================================================
#  MeanFieldHomogenizationMittagLefflerExt.jl
#
#  Supplies closed-form time-domain values for the fractional models by wiring
#  `MittagLeffler.mittleff` into the `Viscoelasticity._mittag_leffler` hook.
#
#  Without this extension those models still work: their Laplace-Carson
#  transforms are elementary and the numerical inversion covers the time
#  domain to about 1e-12.  The extension is a sharpening and an independent
#  cross-check, never an enabler — the same contract as the DECUHR extension.
# =============================================================================

module MeanFieldHomogenizationMittagLefflerExt

using MittagLeffler
import MeanFieldHomogenization.Viscoelasticity: _mittag_leffler

# ── Why the domain is restricted ────────────────────────────────────────────
#
#  `MittagLeffler.jl` v1.0.0 returns `1.0` for `1 < α < 2` and small `|z|`,
#  silently and with no warning:
#
#      julia> mittleff(1.3, 1.0, -0.5)
#      1.0                       # the Taylor series gives 0.6330079...
#      julia> mittleff(1.3, 1.0, -2.0)
#      0.05434750...             # correct
#
#  That band is not an academic corner: the Rabotnov kernel is built on
#  `E_{α+1,1}`, so for the usual `0 < α < 1` its parameter lands *exactly*
#  there.  Trusting the package uncritically would corrupt precisely the model
#  the extension was added for.
#
#  So the hook only answers on `0 < α ≤ 1`, where the values were checked
#  against the defining series across the range (see
#  `test/Viscoelasticity/test_rheology.jl`).  Everything else declines, and the
#  caller falls back on inverting the closed-form Carson transform — which is
#  correct, and which the test suite pins against the extension in the band
#  where both are available.
const _ML_ALPHA_MAX = 1.0

# The signature is deliberately `AbstractFloat`, not `Real`: a
# `ForwardDiff.Dual` is `<: Real` but not `<: AbstractFloat`, and `mittleff`
# has no derivative rule.  Declining Duals here sends the caller to the
# inversion, which *is* differentiable — so loading this extension never costs
# you autodiff.
const _MLFloat = Union{AbstractFloat, Complex{<:AbstractFloat}}

function _mittag_leffler(a::AbstractFloat, b::AbstractFloat, z::_MLFloat)
    (0 < a ≤ _ML_ALPHA_MAX && b == 1) || return nothing
    return mittleff(a, b, z)
end

end # module
