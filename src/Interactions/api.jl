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
#  Two sign conventions coexist in the literature and mixing them is the single
#  most likely way to get a wrong answer here:
#
#    * Brisard, Bertin & Legoll (2023), Eq. (9) — THE ONE THIS PACKAGE USES —
#      define the Green operator as mapping a polarization onto MINUS the
#      induced field,
#
#          ε = E - 𝔾⁰ * τ ,       𝔾⁰_ijkl = -[∂²G_ik/∂x_j∂x_l]_{(ij)(kl)} ,
#
#      so its Fourier symbol is positive semi-definite (their Eq. (11),
#      Γ̂₀(k) = σ₀⁻¹k⊗k/‖k‖² in conduction) and its self term is
#
#          𝕋^{aa} = +ℙ ,
#
#      positive definite like the Hill tensor itself.  The one-inclusion case
#      is then ε = -ℙ:τ, which is what `theory/eshelby_problem.md` and
#      `theory/viscoelasticity.md` already write.
#
#    * Molinari & El Mouden (1996) and Berveiller et al. (1987) write
#      ε^I = ε⁰ + Σ_J Γ^{IJ}:δℂ^J:ε^J, so their Γ^{IJ} is the OPPOSITE of the
#      above and their self term is Γ^{II} = -ℙ.
#
#  Consequence: any formula transcribed from Molinari or Berveiller — their
#  App. A component table in particular — must be flipped on the way in.  The
#  closed forms of `pair_ball_iso.jl` are flipped at their κ, once, with a
#  comment saying so.  Nothing else in the package carries a compensating sign.
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
returns **minus the average field induced in `incl_a`** — strain in
elasticity, gradient of the temperature in conduction. Its self counterpart is
[`self_interaction_tensor`](@ref), equal to ``+\\mathbb{P}``.

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

!!! warning "Molinari's convention is the opposite one"
    The package follows [Brisard et al. 2023](@cite brisard2023), Eq. (9), for
    which ``\\mathbb{T}^{aa} = +\\mathbb{P}``. Molinari & El Mouden (1996) and
    Berveiller et al. (1987) use ``\\Gamma^{II} = -\\mathbb{P}``, so a formula
    taken from them — their Appendix A table in particular — must be flipped
    before it is compared with anything here.

An anisotropic reference is supported in 3D elasticity and in conduction; only
plane-strain elasticity with an anisotropic `P₀` is missing, and it raises an
`ArgumentError` naming the limitation rather than falling back to an isotropic
kernel.
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

Self term of the interaction family, ``\\mathbb{T}^{aa} = +\\mathbb{P}_a``:
minus the average field induced in an inclusion by its *own* uniform
polarization, which is the Eshelby result ``\\varepsilon = -\\mathbb{P}:\\tau``.

It **is** the Hill polarization tensor, so it inherits every back-end of
[`hill_tensor`](@ref MeanFieldHomogenization.Elasticity.hill_tensor) — closed
forms for isotropic and transversely isotropic references, the residue
algorithm, the DECUHR and nested-QuadGK cubatures, in 2D and 3D, for elasticity
and for conduction. Keyword arguments are forwarded to `hill_tensor`.

That the self term is *plus* the Hill tensor is the whole reason the package
follows [Brisard et al. 2023](@cite brisard2023) rather than Molinari's
opposite sign: it makes the N-body kernels and the one-site schemes share one
object, and it is why the cluster model collapses onto Mori-Tanaka when the
cluster is reduced to a single inclusion
([Molinari & El Mouden 1996](@cite molinari1996), App. C).
"""
self_interaction_tensor(incl::MFH_Core.AbstractInclusion, P₀::TensND.AbstractTens; kw...) =
    Elasticity.hill_tensor(incl, P₀; kw...)

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
