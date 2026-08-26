# =============================================================================
#  mittag_leffler.jl — optional access to the Mittag-Leffler function.
#
#  Only two models in the catalog have a closed-form *time-domain* expression
#  that needs it: `FractionalZener` and `Rabotnov`.  Both have an elementary
#  Laplace-Carson transform, so both are fully usable without it — the
#  numerical inversion simply takes over.  The extension therefore sharpens a
#  reference curve; it never unlocks a capability.  Same contract as the DECUHR
#  and Lux extensions.
# =============================================================================

"""
    _mittag_leffler(a, b, z) -> value or `nothing`

The two-parameter Mittag-Leffler function ``E_{a,b}(z)``.

Returns `nothing` when `MittagLeffler.jl` is not loaded, which is the signal
callers use to fall back on inverting the closed-form Carson transform:

```julia
ml = _mittag_leffler(α, one(α), -(t / τ)^α)
ml === nothing && return inverse_carson(p -> carson_relaxation(m, p), t, method)
```

Load the package to switch the closed forms on:

```julia
using MittagLeffler        # activates MeanFieldHomogenizationMittagLefflerExt
```

!!! warning "The extension declines more than it accepts"
    `MittagLeffler.jl` v1.0.0 silently returns `1.0` for `1 < a < 2` and small
    `|z|` — `mittleff(1.3, 1.0, -0.5)` gives `1.0` where the defining series
    gives `0.6330079`.  The extension therefore answers only for `0 < a ≤ 1`
    and `b == 1`, the domain checked against that series.

    [`FractionalZener`](@ref) (`0 < α ≤ 1`) and [`Rabotnov`](@ref) with the
    physical `α ∈ (-1, 0)` — whose Mittag-Leffler order `α + 1` then lies in
    `(0, 1)` — are both inside that domain and get their closed forms.  A
    Rabotnov kernel with `α > 0` is not, and falls back on the inversion, which
    agrees with the closed form to about `1e-10` wherever both can be
    evaluated.

!!! note "This replaces a PyCall detour"
    Before this extension, the Rabotnov benchmark reached `E_{a,b}` through
    `PyCall` and an external Python module living outside the repository
    (`scripts/52_rabotnov_mittag_leffler.jl`, and §11 of the viscoelasticity
    manual).  That is no longer needed: the transform is closed-form and the
    time-domain reference is a registered Julia package away.
"""
_mittag_leffler(a, b, z) = nothing
