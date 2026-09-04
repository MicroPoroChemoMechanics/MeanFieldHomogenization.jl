# =============================================================================
#  27_user_inclusion_sensitivity.jl
#
#  Demonstrates that the sensitivity API extends to *user-defined* inclusion
#  types with no changes to `parameters.jl`: the `_replace_geom_field`
#  helper (based on `@generated`) rebuilds any struct whose differentiable
#  fields are `<:Number` (or tuples of such), provided the parametric
#  constructor follows the standard Julia auto-generated pattern.
#
#  Case study: a minimal `MyBlob{T,B}` that delegates the Hill/Eshelby
#  kernels to an equivalent `Ellipsoid`. We then differentiate with respect
#  to its `radius` and `eccentricity` fields, and finally to a composite
#  parameter combined via the `sensitivity` closure entry point — no
#  library code change required.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using ForwardDiff
using TensND
using Printf

# ── User-defined inclusion type ────────────────────────────────────────────

"""
    MyBlob(radius, eccentricity, basis)

Demonstration user inclusion. The Hill / Eshelby kernels are delegated to
an `Ellipsoid(radius, radius·(1-e), radius·(1-e²))`, which makes the
sensitivity to either `radius` or `eccentricity` non-trivial (the aspect
ratio depends explicitly on `e`).
"""
struct MyBlob{T <: Number, B <: TensND.AbstractBasis} <:
    MeanFieldHomogenization.AbstractEllipsoidalInclusion{3, T}
    radius::T
    eccentricity::T
    basis::B
end

# Equivalent geometry used for the Hill / Eshelby kernels.
_blob_as_ellipsoid(b::MyBlob) =
    Ellipsoid(b.radius, b.radius * (1 - b.eccentricity), b.radius * (1 - b.eccentricity)^2)

# Delegate accessors and kernels to the equivalent `Ellipsoid`. Signatures
# are made as specific as the generic `Elasticity` ones to avoid dispatch
# ambiguities.
MeanFieldHomogenization.hill_tensor(b::MyBlob, C₀::TensND.AbstractTens; kw...) =
    MeanFieldHomogenization.hill_tensor(_blob_as_ellipsoid(b), C₀; kw...)
MeanFieldHomogenization.eshelby_tensor(b::MyBlob, C₀::TensND.AbstractTens; kw...) =
    MeanFieldHomogenization.eshelby_tensor(_blob_as_ellipsoid(b), C₀; kw...)
MeanFieldHomogenization.material_symmetry(b::MyBlob) =
    MeanFieldHomogenization.material_symmetry(_blob_as_ellipsoid(b))
MeanFieldHomogenization.dimension(::MyBlob) = 3
MeanFieldHomogenization.shape_trait(b::MyBlob) = MeanFieldHomogenization.shape_trait(_blob_as_ellipsoid(b))
MeanFieldHomogenization.shape_tensor(b::MyBlob) = MeanFieldHomogenization.shape_tensor(_blob_as_ellipsoid(b))
MeanFieldHomogenization.inclusion_basis(b::MyBlob) = b.basis

# ── RVE with a `MyBlob` inclusion ───────────────────────────────────────────

basis = TensND.CanonicalBasis{3, Float64}()

println("="^78)
println("MeanFieldHomogenization — sensitivity on a user-defined inclusion type (MyBlob)")
println("="^78)

rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
add_phase!(
    rve, :B, MyBlob(1.0, 0.2, basis),
    Dict(:C => TensISO{3}(60.0, 20.0));
    fraction = 0.2
)

const idxC = C -> get_array(C)[1, 1, 1, 1]

# ── (1) Sensitivity to the radius ───────────────────────────────────────────
∂_r = derivative(rve, Dilute(), geometry(:B, :radius); indexer = idxC)
@printf "\n[1] ∂C[1111]/∂radius (MyBlob)        = %.6e\n" ∂_r

# ── (2) Sensitivity to the eccentricity — drives the Hill tensor change ────
∂_e = derivative(rve, Dilute(), geometry(:B, :eccentricity); indexer = idxC)
@printf "[2] ∂C[1111]/∂eccentricity           = %.6e\n" ∂_e

# ── (3) Cross-check via finite differences ─────────────────────────────────
function f_eval_e(de)
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        r, :B, MyBlob(1.0, 0.2 + de, basis),
        Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.2
    )
    return idxC(homogenize(r, Dilute()))
end
h = 1.0e-5
∂_e_fd = (f_eval_e(h) - f_eval_e(-h)) / (2h)
@printf "[3] FD reference for ∂e             = %.6e   (rel. err = %.2e)\n" ∂_e_fd abs(∂_e - ∂_e_fd) / abs(∂_e_fd)

# ── (4) Gradient w.r.t. three parameters: f, radius, eccentricity ──────────
∇ = gradient(
    rve, Dilute(),
    [amount(:B), geometry(:B, :radius), geometry(:B, :eccentricity)];
    indexer = idxC
)
println("\n[4] gradient on [f_B, radius, eccentricity] :")
@printf "    [%.4f, %.6e, %.6e]\n" ∇[1] ∇[2] ∇[3]

# ── (5) sensitivity() with a fully user-defined closure ───────────────────
println("\n[5] sensitivity(closure) on a composite parameter (radius and eccentricity):")
f_composed = α -> begin
    r2 = RVE()
    add_phase!(r2, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(30.0, 10.0)); fraction = :rest)
    add_phase!(
        r2, :B, MyBlob(α, α / 5, basis),
        Dict(:C => TensISO{3}(60.0, 20.0));
        fraction = 0.2
    )
    return idxC(homogenize(r2, Dilute()))
end
∂_α = sensitivity(f_composed, 1.0)
@printf "    ∂C[1111]/∂α (where r=α, e=α/5) = %.6e\n" ∂_α

println("\nDone.")
