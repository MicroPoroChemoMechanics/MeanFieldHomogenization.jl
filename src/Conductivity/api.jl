# =============================================================================
#  api.jl — add `_kernel` methods for 2nd-order Hill tensor (conductivity).
#
#  `hill_tensor` itself is defined in the `Elasticity` sub-module — the
#  dispatch on the 2nd-order case goes through the same generic entry
#  point, so we only need to register new `_kernel` methods here.
# =============================================================================

# ── 2nd-order, 3D (conductivity) ─────────────────────────────────────────────

function Elasticity._kernel(
        ell::Elasticity.Ellipsoid{3},
        K₀::TensND.TensISO{2, 3},
        ::MFH_Core.Analytical; kw...
    )
    return _hill_order2_3d_iso(ell, K₀)
end

function Elasticity._kernel(
        ell::Elasticity.Ellipsoid{3},
        K₀::TensND.AbstractTens{2, 3},
        ::MFH_Core.Analytical; kw...
    )
    return _hill_order2_3d_aniso(ell, K₀)
end

# ── 2nd-order, 3D (conductivity) — infinite cylinder ────────────────────────

function Elasticity._kernel(
        cyl::Elasticity.Cylinder,
        K₀::TensND.TensISO{2, 3},
        ::MFH_Core.Analytical; kw...
    )
    return _hill_order2_3d_iso(cyl, K₀)
end

function Elasticity._kernel(
        cyl::Elasticity.Cylinder,
        K₀::TensND.AbstractTens{2, 3},
        ::MFH_Core.Analytical; kw...
    )
    return _hill_order2_3d_aniso(cyl, K₀)
end

# ── 2nd-order, 2D (conductivity) ─────────────────────────────────────────────

function Elasticity._kernel(
        ell::Elasticity.Ellipsoid{2},
        K₀::TensND.TensISO{2, 2},
        ::MFH_Core.Analytical; kw...
    )
    return _hill_order2_2d_iso(ell, K₀)
end

function Elasticity._kernel(
        ell::Elasticity.Ellipsoid{2},
        K₀::TensND.AbstractTens{2, 2},
        ::MFH_Core.Analytical; kw...
    )
    return _hill_order2_2d(ell, K₀)
end

# ── Eshelby tensor (2nd order) — s = P · K₀ ──────────────────────────────────

"""
    eshelby_tensor(incl::AbstractEllipsoidalInclusion, K₀::TensND.AbstractTens{2}; kw...)

2nd-order Eshelby tensor of an ellipsoidal inclusion in a matrix of
conductivity ``\\boldsymbol{K}_0``:

```math
\\boldsymbol{S}^{\\mathrm{E}} = \\boldsymbol{P}\\cdot\\boldsymbol{K}_0 .
```

For the sphere in an isotropic conductor ``\\boldsymbol{S}^{\\mathrm{E}} = \\tfrac{1}{3}
\\boldsymbol{1}`` (independent of ``k_0``). Thin wrapper around
[`hill_tensor`](@ref).
"""
MFH_Core.eshelby_tensor(
    incl::MFH_Core.AbstractEllipsoidalInclusion, K₀::TensND.AbstractTens{2, 3};
    kw...
) = Elasticity.hill_tensor(incl, K₀; kw...) ⋅ K₀

MFH_Core.eshelby_tensor(
    incl::MFH_Core.AbstractEllipsoidalInclusion, K₀::TensND.AbstractTens{2, 2};
    kw...
) = Elasticity.hill_tensor(incl, K₀; kw...) ⋅ K₀

MFH_Core.eshelby_tensor(
    incl::MFH_Core.AbstractCustomInclusion, K₀::TensND.AbstractTens{2, 3};
    kw...
) = Elasticity.hill_tensor(incl, K₀; kw...) ⋅ K₀
