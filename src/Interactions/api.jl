# =============================================================================
#  api.jl
#
#  Public entry points `interaction_tensor` / `self_interaction_tensor` and
#  the `_pair_kernel` method table.  Dispatch is delegated to
#  `Core._resolve_pair_algo`, the two-inclusion counterpart of
#  `Core._resolve_algo`.
#
#  SIGN CONVENTION — read this before transcribing any formula.
#
#  Three sign conventions coexist in the literature and mixing them is the
#  single most likely way to get a wrong answer here:
#
#    * Molinari & El Mouden (1996) and Berveiller et al. (1987) write the
#      Lippmann-Schwinger equation as ε^I = ε⁰ + Σ_J Γ^{IJ}:δℂ^J:ε^J, so
#      Γ^{IJ}:τ is the strain *induced* by the polarization τ, and the
#      self term is Γ^{II} = -ℙ.
#    * Brisard et al. (2014, 2023) define Γ₀ as mapping the polarization onto
#      *minus* the induced field, so their influence tensors are the opposite
#      of the above and their self term is +ℙ.
#
#  This package uses the FIRST convention throughout, because it is the one
#  that keeps `self_interaction_tensor` a direct function of `hill_tensor`,
#  which is the object the rest of MFH is built on.  Every scheme kernel that
#  transcribes a formula from Brisard flips the sign explicitly and says so.
# =============================================================================

"""
    interaction_tensor(incl_a, incl_b, r, P₀; method=:auto, kw...) -> AbstractTens

Interaction tensor ``\\mathbb{T}^{ab}`` between two non-overlapping inclusions
embedded in an infinite reference medium `P₀`, with `r` the vector joining the
center of `incl_a` to the center of `incl_b`.

It is the double average of the real-space Green operator over the two
regions,

```math
\\mathbb{T}^{ab} = \\frac{1}{|\\Omega_a|}
   \\int_{\\Omega_a}\\int_{\\Omega_b} \\mathbb{G}^0(x - y)\\, dV_y\\, dV_x ,
```

so that contracting it with a *uniform polarization* carried by `incl_b`
returns the **average field induced in `incl_a`** — strain in elasticity,
minus the gradient in conduction. Its self counterpart is
[`self_interaction_tensor`](@ref), equal to ``-\\mathbb{P}``.

`P₀` may be a 4th-order stiffness (elasticity) or a 2nd-order conductivity
tensor; dispatch selects the corresponding formulation, in 2D or 3D. Two
balls (3D) or two disks (2D) in an isotropic reference are evaluated by the
closed form of [Molinari & El Mouden 1996](@cite molinari1996) and
[Berveiller et al. 1987](@cite berveiller1987), which is **exact at any
separation**; other geometries use the truncated multipole expansion of
[Brisard et al. 2014](@cite brisard2014), §4.2.

`method` selects the back-end explicitly: `:analytical`, `:multipole`,
`:quadrature`, or `:auto` (default).

This is the shared numerical ingredient of both N-body models in the package,
[`EquivalentInclusion`](@ref MeanFieldHomogenization.Schemes.EquivalentInclusion) and [`ClusterModel`](@ref MeanFieldHomogenization.Schemes.ClusterModel) — Brisard et al.
(2014), §3.1, note that their influence pseudotensors of order `k = l = 0`
coincide with the interaction tensors of Molinari & El Mouden.

!!! note "Isotropic reference only"
    The real-space Green operator is implemented for isotropic references
    only. An anisotropic `P₀` raises an `ArgumentError` rather than silently
    falling back to an isotropic kernel.
"""
function interaction_tensor(
        incl_a::MFH_Core.AbstractInclusion,
        incl_b::MFH_Core.AbstractInclusion,
        r::AbstractVector,
        P₀::TensND.AbstractTens;
        method::Symbol = :auto,
        kw...
    )
    algo = MFH_Core._resolve_pair_algo(Val(method), incl_a, incl_b, P₀)
    return _pair_kernel(incl_a, incl_b, r, P₀, algo; kw...)
end

"""
    self_interaction_tensor(incl, P₀; kw...) -> AbstractTens

Self term of the interaction family, ``\\mathbb{T}^{aa} = -\\mathbb{P}_a``: the
average field induced in an inclusion by its *own* uniform polarization.

It is minus the Hill polarization tensor, so it inherits every back-end of
[`hill_tensor`](@ref) — closed forms for isotropic and transversely isotropic
references, the residue algorithm, the DECUHR and nested-QuadGK cubatures, in
2D and 3D, for elasticity and for conduction. Keyword arguments are forwarded
to `hill_tensor`.

The identity ``\\mathbb{T}^{aa} = -\\mathbb{P}_a`` is what ties the two N-body
schemes of this package to the one-site schemes: it is the reason the cluster
model collapses onto Mori-Tanaka when the cluster is reduced to a single
inclusion ([Molinari & El Mouden 1996](@cite molinari1996), App. C).
"""
self_interaction_tensor(incl::MFH_Core.AbstractInclusion, P₀::TensND.AbstractTens; kw...) =
    -Elasticity.hill_tensor(incl, P₀; kw...)

# ─── Kernel table ────────────────────────────────────────────────────────────

_pair_kernel(
    incl_a, incl_b, r, P₀, ::MFH_Core.Analytical; kw...
) = _pair_ball_iso(_ball_radius(incl_a), _ball_radius(incl_b), r, P₀)

_pair_kernel(
    incl_a, incl_b, r, P₀, ::MFH_Core.Multipole; kw...
) = _pair_multipole(incl_a, incl_b, r, P₀; kw...)

_pair_kernel(
    incl_a, incl_b, r, P₀, ::MFH_Core.NestedQuadGK; kw...
) = _pair_quadrature(incl_a, incl_b, r, P₀; kw...)

# ─── Ball / disk recognition ─────────────────────────────────────────────────
#
# The closed form applies to a region that is a ball (3D) or a disk (2D).
# Recognition goes through the shape trait rather than the concrete type, so
# that any inclusion advertising a spherical or circular shape reaches it.

MFH_Core._is_ball_like(ell::Ellipsoid{3, Elasticity.Spherical}) = true
MFH_Core._is_ball_like(ell::Ellipsoid{2, Elasticity.Circular}) = true

"""
    _ball_radius(incl) -> Real

Radius of a ball- or disk-shaped inclusion. Errors on any other shape — the
closed-form kernel must never be reached with a non-spherical region.
"""
_ball_radius(ell::Ellipsoid{3, Elasticity.Spherical}) = ell.semi_axes[1]
_ball_radius(ell::Ellipsoid{2, Elasticity.Circular}) = ell.semi_axes[1]

_ball_radius(incl::MFH_Core.AbstractInclusion) = throw(
    ArgumentError(
        "interaction_tensor: the closed-form (`:analytical`) kernel applies to " *
            "balls and disks only; got a $(nameof(typeof(incl))) with shape trait " *
            "$(nameof(typeof(MFH_Core.shape_trait(incl)))). Use `method = :multipole` " *
            "or `method = :quadrature` for a general ellipsoid."
    )
)
