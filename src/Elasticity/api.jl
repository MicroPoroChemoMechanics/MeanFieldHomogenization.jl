# =============================================================================
#  api.jl
#
#  Public entry point `hill_tensor(ell, C₀; method, kw...)` and the
#  `_kernel` method table for elasticity and 2nd-order Hill tensors
#  (4th-order case handled in this module; 2nd-order case handled in
#  `Conductivity`).  Dispatch is delegated to `Core._resolve_algo`.
# =============================================================================

"""
    hill_tensor(ell, C₀; method=:auto, abstol=1e-8, reltol=1e-6, maxiters=1_000_000)
        → AbstractTens

Hill polarization tensor **P** for an ellipsoidal inclusion `ell`
embedded in a reference medium `C₀`.  `C₀` can be a 4th-order stiffness
(elasticity) or a 2nd-order conductivity tensor — dispatch selects
the appropriate formulation automatically.

The general expression of the elastic polarization tensor is
([willis1977](@cite), [mura1987](@cite)):

```
P(A, C) = (det A)/(4π) ∫_{|ξ|=1} ξ ⊗ˢ (ξ·C·ξ)⁻¹ ⊗ˢ ξ / ‖A·ξ‖³ dS_ξ
```

The isotropic case (`C₀::TensISO`) is evaluated analytically; the
anisotropic case uses the Cauchy-residue reduction of
[masson2008](@cite) (trait `Residue`) or the DECUHR
adaptive cubature of [espelid1994](@cite)
(trait `DECUHR`). See the `Hill polarization tensors` theory page
for the full dispatch table and return types.
"""
function hill_tensor(
        ell::AbstractEllipsoidalInclusion,
        C₀::TensND.AbstractTens;
        method::Symbol = :auto,
        abstol::Float64 = 1.0e-8,
        reltol::Float64 = 1.0e-6,
        maxiters::Int = 1_000_000
    )
    return _hill_tensor_entry(ell, C₀, method, abstol, reltol, maxiters)
end

# User-defined morphologies (`Core.AbstractCustomInclusion`) reach the same
# `_resolve_algo` / `_kernel` table as the built-in families.  This branch of
# the hierarchy is disjoint from `AbstractEllipsoidalInclusion`, so the two
# entry points can never be ambiguous.  A custom inclusion either registers
# `Elasticity._kernel(::MyIncl, C₀, ::Analytical; kw...)` methods, or bypasses
# the table altogether by defining `hill_tensor` on its own concrete type.
function hill_tensor(
        incl::MFH_Core.AbstractCustomInclusion,
        C₀::TensND.AbstractTens;
        method::Symbol = :auto,
        abstol::Float64 = 1.0e-8,
        reltol::Float64 = 1.0e-6,
        maxiters::Int = 1_000_000
    )
    return _hill_tensor_entry(incl, C₀, method, abstol, reltol, maxiters)
end

function _hill_tensor_entry(incl, C₀, method::Symbol, abstol, reltol, maxiters)
    MFH_Core._bump!(MFH_Core.HILL_CALLS)
    algo = MFH_Core._resolve_algo(Val(method), incl, C₀)
    return _kernel(incl, C₀, algo; abstol = abstol, reltol = reltol, maxiters = maxiters)
end

# ── 4th-order, 3D ────────────────────────────────────────────────────────────

_kernel(ell::Ellipsoid{3}, C₀::TensND.TensISO{4, 3}, ::MFH_Core.Analytical; kw...) =
    _hill_3d_iso(ell, C₀)

# Analytical TI-coaxial branch — Barthélémy 2020
_kernel(ell::Ellipsoid{3}, C₀::TensND.TensTI{4}, ::MFH_Core.Analytical; kw...) =
    _hill_3d_ti_coaxial(ell, C₀)

_kernel(ell::Ellipsoid{3}, C₀::TensND.AbstractTens{4, 3}, ::MFH_Core.Residue; kw...) =
    _hill_3d_aniso_residue(
    ell, C₀;
    abstol = get(kw, :abstol, 1.0e-8),
    reltol = get(kw, :reltol, 1.0e-6),
    maxiters = get(kw, :maxiters, 1_000_000)
)

_kernel(ell::Ellipsoid{3}, C₀::TensND.AbstractTens{4, 3}, ::MFH_Core.DECUHR; kw...) =
    _hill_3d_aniso_decuhr(
    ell, C₀;
    abstol = get(kw, :abstol, 1.0e-8),
    reltol = get(kw, :reltol, 1.0e-6),
    maxiters = get(kw, :maxiters, 1_000_000)
)

_kernel(ell::Ellipsoid{3}, C₀::TensND.AbstractTens{4, 3}, ::MFH_Core.NestedQuadGK; kw...) =
    _hill_3d_aniso_nestedquadgk(
    ell, C₀;
    abstol = get(kw, :abstol, 1.0e-8),
    reltol = get(kw, :reltol, 1.0e-6),
    maxiters = get(kw, :maxiters, 1_000_000)
)

# ── 4th-order, 3D — infinite cylinder ────────────────────────────────────────

_kernel(cyl::Cylinder, C₀::TensND.TensISO{4, 3}, ::MFH_Core.Analytical; kw...) =
    _hill_3d_iso(cyl, C₀)

_kernel(cyl::Cylinder, C₀::TensND.AbstractTens{4, 3}, ::MFH_Core.CylinderQuadrature; kw...) =
    _hill_3d_cylinder_aniso(
    cyl, C₀;
    abstol = get(kw, :abstol, 1.0e-8),
    reltol = get(kw, :reltol, 1.0e-6),
    maxiters = get(kw, :maxiters, 1_000_000)
)

# ── 4th-order, 2D ────────────────────────────────────────────────────────────

_kernel(ell::Ellipsoid{2}, C₀::TensND.TensISO{4, 2}, ::MFH_Core.Analytical; kw...) =
    _hill_2d_iso(ell, C₀)

_kernel(ell::Ellipsoid{2}, C₀::TensND.AbstractTens{4, 2}, ::MFH_Core.Analytical; kw...) =
    _hill_2d_aniso(
    ell, C₀;
    abstol = get(kw, :abstol, 1.0e-8),
    reltol = get(kw, :reltol, 1.0e-6),
    maxiters = get(kw, :maxiters, 1_000_000)
)

# ── Eshelby tensor (4th order) — S = P : C₀ ──────────────────────────────────

"""
    eshelby_tensor(incl::AbstractEllipsoidalInclusion, C₀::TensND.AbstractTens{4}; kw...)

4th-order Eshelby tensor ``\\mathbb S = \\mathbb P : \\mathbb C_0``
of an ellipsoidal inclusion `incl` embedded in a matrix of stiffness
`C₀`. Thin wrapper around [`hill_tensor`](@ref) followed by the double
contraction with `C₀`.
"""
MFH_Core.eshelby_tensor(
    incl::AbstractEllipsoidalInclusion, C₀::TensND.AbstractTens{4, 3};
    kw...
) = hill_tensor(incl, C₀; kw...) ⊡ C₀

MFH_Core.eshelby_tensor(
    incl::AbstractEllipsoidalInclusion, C₀::TensND.AbstractTens{4, 2};
    kw...
) = hill_tensor(incl, C₀; kw...) ⊡ C₀

MFH_Core.eshelby_tensor(
    incl::MFH_Core.AbstractCustomInclusion, C₀::TensND.AbstractTens{4, 3};
    kw...
) = hill_tensor(incl, C₀; kw...) ⊡ C₀
