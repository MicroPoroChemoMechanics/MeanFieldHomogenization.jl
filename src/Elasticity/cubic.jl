# =============================================================================
#  cubic.jl — the cubic symmetry class, as this package needs it.
#
#  The algebra itself lives in TensND: `TensCubic` stores the three constants
#  and a cube frame, its products and inverses are componentwise, and
#  `proj_tens(Val(:CUBIC), …)` projects onto the class and reports the distance.
#  Nothing of that is reimplemented here.
#
#  What this file adds is the two things a homogenization package wants on top:
#  the symmetry **trait**, so that `material_symmetry` answers honestly for a
#  cubic tensor, and a name for the projection **residual**, which is the
#  quantity a solver actually reads.
# =============================================================================

"""
    cubic_residual(t, frame = CanonicalBasis{3,Float64}()) -> Real

Relative Frobenius distance from `t` to its projection onto the cubic class
with cube axes `frame` — the third value returned by
`proj_tens(Val(:CUBIC), t, frame)`, named because it is the one that gets read.

**This is a free error bar.** Where the morphology and the medium leave the
octahedral group invariant, the answer belongs to the class **by group theory**,
so this distance is bounded by the discretization error and by nothing else: no
reference solution appears in it. What lives *inside* the class, and is
therefore not measured here, is
`TensND.cubic_anisotropy`.
Read together they separate a real morphological anisotropy from a numerical
artifact — an artifact breaks the symmetry, a real anisotropy does not.

The default frame is the canonical one. Pass the inclusion's own basis when the
cube is not aligned with it: a rotated cubic tensor is not cubic about the
canonical axes, and the residual would say so, correctly but uselessly.
"""
cubic_residual(
    t::TensND.AbstractTens{4, 3},
    frame::TensND.AbstractBasis = TensND.CanonicalBasis{3, Float64}(),
) = TensND.proj_tens(Val(:CUBIC), t, frame)[3]

# ─── The symmetry trait ──────────────────────────────────────────────────────
#
#  Now that TensND has a storage type for the class, the trait can dispatch on
#  it and mean something.  Before that it could not: `material_symmetry`
#  answers from the *container*, and inventing a trait with no type behind it
#  would have let dispatch claim a structure the object did not carry.

"Cubic material (TensND `TensCubic`) — three independent constants."
struct CubicSym <: Core.MaterialSymmetry end

Core.material_symmetry(::TensND.TensCubic) = CubicSym()
