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

Hill polarization tensor ``\\mathbb{P}`` (``\\boldsymbol{P}`` in conduction) for an ellipsoidal inclusion `ell`
embedded in a reference medium `C₀`.  `C₀` can be a 4th-order stiffness
(elasticity) or a 2nd-order conductivity tensor — dispatch selects
the appropriate formulation automatically.

The general expression of the elastic polarization tensor is
[willis1977, mura1987](@cite):

```math
\\mathbb{P}(\\boldsymbol{A},\\mathbb{C})
= \\frac{\\det\\boldsymbol{A}}{4\\pi}
\\int_{\\|\\underline{\\xi}\\|=1}
\\frac{\\underline{\\xi}\\stackrel{s}{\\otimes}
      \\bigl(\\underline{\\xi}\\cdot\\mathbb{C}\\cdot\\underline{\\xi}\\bigr)^{-1}
      \\stackrel{s}{\\otimes}\\underline{\\xi}}
     {\\|\\boldsymbol{A}\\cdot\\underline{\\xi}\\|^{3}}\\,\\mathrm{d}S_{\\xi}
```

The isotropic case (`C₀::TensISO`) is evaluated analytically; the
anisotropic case uses the Cauchy-residue reduction of
[masson2008](@cite) (trait `Residue`) or the DECUHR
adaptive cubature of [espelid1994](@citet)
(trait `DECUHR`). See the `Hill polarization tensors` theory page
for the full dispatch table and return types.
"""
Base.@constprop :aggressive @inline function hill_tensor(
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
Base.@constprop :aggressive @inline function hill_tensor(
        incl::MFH_Core.AbstractCustomInclusion,
        C₀::TensND.AbstractTens;
        method::Symbol = :auto,
        abstol::Float64 = 1.0e-8,
        reltol::Float64 = 1.0e-6,
        maxiters::Int = 1_000_000
    )
    return _hill_tensor_entry(incl, C₀, method, abstol, reltol, maxiters)
end

# Inlined, with the keyword methods above, down to `_with_method`: Julia 1.12
# propagates the constant default `method = :auto` for inference but compiles
# the entry for a run-time `Symbol`, whose other branch returns an abstract type
# and boxes the result. Inlined, the literal reaches `_with_method` and selects
# the branch at compile time on every version.
Base.@constprop :aggressive @inline function _hill_tensor_entry(incl, C₀, method::Symbol, abstol, reltol, maxiters)
    MFH_Core._bump!(MFH_Core.HILL_CALLS)
    return MFH_Core._with_method(_hill_tensor_at, method, incl, C₀, abstol, reltol, maxiters)
end

_hill_tensor_at(m::Val, incl, C₀, abstol, reltol, maxiters) =
    _hill_kernel(incl, C₀, MFH_Core._resolve_algo(m, incl, C₀), abstol, reltol, maxiters)

# A closed form is cheaper than a lookup: it never consults a `HillCache`. A
# numerical one is read from the cache in scope (`with_hill_cache`) when its
# inputs are `Float64` and its shape keys it.
_hill_kernel(incl, C₀, algo::MFH_Core.Analytical, abstol, reltol, maxiters) =
    _kernel(incl, C₀, algo; abstol, reltol, maxiters)
_hill_kernel(incl, C₀, algo, abstol, reltol, maxiters) = MFH_Core._hill_cached(
    () -> _kernel(incl, C₀, algo; abstol, reltol, maxiters),
    () -> _hill_cache_key(incl, C₀, algo, abstol, reltol, maxiters),
)

# The exact values a numerical Hill tensor depends on, as plain tuples (a frame
# or a tensor object would not hash by its values): `nothing` for an input that
# is not `Float64` or a shape not keyed here, which bypasses the cache.
_hill_cache_key(incl::Union{Ellipsoid{3}, Cylinder}, C₀, algo, abstol, reltol, maxiters) =
    _hill_cache_key(Val(_all_float64(incl, C₀)), incl, C₀, algo, abstol, reltol, maxiters)
_hill_cache_key(incl, C₀, algo, abstol, reltol, maxiters) = nothing
_hill_cache_key(::Val{false}, incl, C₀, algo, abstol, reltol, maxiters) = nothing
_hill_cache_key(::Val{true}, incl, C₀, algo, abstol, reltol, maxiters) = (
    nameof(typeof(incl)), Tuple(incl.semi_axes), Tuple(TensND.vecbasis(incl.basis, :cov)),
    Tuple(TensND.canonical_components(C₀)), typeof(algo), abstol, reltol, maxiters,
)
_all_float64(incl, C₀) = eltype(incl.semi_axes) === Float64 && eltype(incl.basis) === Float64 && eltype(C₀) === Float64

# ── 4th-order, 3D ────────────────────────────────────────────────────────────

_kernel(ell::Ellipsoid{3}, C₀::TensND.TensISO{4, 3}, ::MFH_Core.Analytical; kw...) =
    _hill_3d_iso(ell, C₀)

# Analytical TI-coaxial branch — Barthélémy 2020
_kernel(ell::Ellipsoid{3}, C₀::TensND.TensTI{4}, ::MFH_Core.Analytical; kw...) =
    _hill_3d_ti_coaxial(ell, C₀)

function _kernel(ell::Ellipsoid{3}, C₀::TensND.AbstractTens{4, 3}, ::MFH_Core.Residue; kw...)
    MFH_Core._residues_in_floats(ell, C₀)
    return _hill_3d_aniso_residue(
        ell, C₀;
        abstol = get(kw, :abstol, 1.0e-8),
        reltol = get(kw, :reltol, 1.0e-6),
        maxiters = get(kw, :maxiters, 1_000_000)
    )
end

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

4th-order Eshelby tensor ``\\mathbb{S}^{\\mathrm{E}} = \\mathbb{P}:\\mathbb{C}_0``
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

# ── Second Hill tensor (4th order) — Q = C₀ − C₀ : P : C₀ ─────────────────────

MFH_Core.second_hill_tensor(
    incl::AbstractEllipsoidalInclusion, C₀::TensND.AbstractTens{4, 3};
    kw...
) = MFH_Core._second_hill_tensor(hill_tensor(incl, C₀; kw...), C₀)

MFH_Core.second_hill_tensor(
    incl::AbstractEllipsoidalInclusion, C₀::TensND.AbstractTens{4, 2};
    kw...
) = MFH_Core._second_hill_tensor(hill_tensor(incl, C₀; kw...), C₀)

MFH_Core.second_hill_tensor(
    incl::MFH_Core.AbstractCustomInclusion, C₀::TensND.AbstractTens{4, 3};
    kw...
) = MFH_Core._second_hill_tensor(hill_tensor(incl, C₀; kw...), C₀)
