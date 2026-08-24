# # Periodic multilayer: the exact laminate solution
#
# A laminate is a periodic unit cell of parallel layers: no matrix, no
# auxiliary Eshelby problem, no reference medium — and an **exact** effective
# behavior rather than an estimate. It is the deterministic counterpart of
# the random morphologies the mean-field schemes describe.
#
# This script builds one, checks it against the closed form of
# Backus (1962), and shows the two bound saturations that make a laminate a
# useful calibration case: it is *exactly* Voigt in the plane of the layers
# and *exactly* Reuss across them, simultaneously.
#
# Theory: the [laminate page](@ref th-laminate).

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using Printf
using LinearAlgebra

# ## A two-layer cell
#
# A stiff layer and a compliant one, with a 30/70 split. Layers are given in
# stacking order; the normal defaults to `e₃`.

k₁, μ₁ = 2.0, 0.8
k₂, μ₂ = 0.5, 0.2
λ(k, μ) = k - 2μ / 3

C₁ = TensISO{3}(3k₁, 2μ₁)
C₂ = TensISO{3}(3k₂, 2μ₂)
f₁, f₂ = 0.3, 0.7

lam = Laminate(; normal = (0, 0, 1))
add_layer!(lam, :A, Dict(:C => C₁, :K => TensISO{3}(2.0)); fraction = f₁)
add_layer!(lam, :B, Dict(:C => C₂, :K => TensISO{3}(0.3)); fraction = f₂)

# The effective stiffness comes out as an **exact** `TensTI{4}` about the
# layer normal: isotropic layers stack into a transversely isotropic medium,
# and the symmetry is decided from the declared classes of the inputs, not
# from a numerical fit of the output.

Cᵉᶠᶠ = homogenize(lam, Laminated(), :C)
println("effective stiffness : ", typeof(Cᵉᶠᶠ))
println(Cᵉᶠᶠ)

# ## Against the closed form of Backus (1962)
#
# For isotropic layers the general formula collapses to averages of simple
# functions of the Lamé coefficients — a harmonic mean across the layers, an
# arithmetic mean within them.

avg(g) = f₁ * g(λ(k₁, μ₁), μ₁) + f₂ * g(λ(k₂, μ₂), μ₂)
r₃₃ = 1 / avg((l, m) -> 1 / (l + 2m))
rλ = avg((l, m) -> l / (l + 2m))

M = Matrix(KM(Cᵉᶠᶠ))
backus = (
    ("C₁₁₁₁", M[1, 1], avg((l, m) -> 4m * (l + m) / (l + 2m)) + r₃₃ * rλ^2),
    ("C₁₁₂₂", M[1, 2], avg((l, m) -> 2m * l / (l + 2m)) + r₃₃ * rλ^2),
    ("C₁₁₃₃", M[1, 3], r₃₃ * rλ),
    ("C₃₃₃₃", M[3, 3], r₃₃),
    ("C₂₃₂₃", M[4, 4] / 2, 1 / avg((l, m) -> 1 / m)),
    ("C₁₂₁₂", M[6, 6] / 2, avg((l, m) -> m)),
)

println("\ncomponent      computed        Backus 1962        |Δ|")
println("─"^58)
for (name, got, want) in backus
    @printf "  %-8s %12.8f    %12.8f    %.2e\n" name got want abs(got - want)
end

# ## The two exact bound saturations
#
# `Voigt` and `Reuss` need no matrix phase, so they apply to a laminate. They
# bracket the exact answer — and two of the bracketings are equalities.

Cᵥ = homogenize(lam, Voigt(), :C)
Cᵣ = homogenize(lam, Reuss(), :C)
Mv, Mr = Matrix(KM(Cᵥ)), Matrix(KM(Cᵣ))

@printf "\nout-of-plane  C₃₃₃₃ : Reuss %.6f ≤ exact %.6f ≤ Voigt %.6f\n" Mr[3, 3] M[3, 3] Mv[3, 3]
@printf "  exact − Reuss = %.2e   (the laminate SATURATES the Reuss bound)\n" abs(M[3, 3] - Mr[3, 3])
@printf "in-plane      C₁₂₁₂ : Reuss %.6f ≤ exact %.6f ≤ Voigt %.6f\n" Mr[6, 6] / 2 M[6, 6] / 2 Mv[6, 6] / 2
@printf "  Voigt − exact = %.2e   (and SATURATES the Voigt bound)\n" abs(M[6, 6] - Mv[6, 6])

# These two statements hold for arbitrary anisotropy, not just here: the
# out-of-plane response is the harmonic average of the acoustic tensors, the
# in-plane one the arithmetic average of the Schur complements.

# ## Transport, from the same cell
#
# The physics is chosen by the *order* of the stored property. The layers
# already carry a `:K`, so nothing else is needed: series across the layers,
# parallel within them.

Kᵉᶠᶠ = homogenize(lam, Laminated(), :K)
Karr = Matrix(components(Kᵉᶠᶠ))
@printf "\nconduction  k_∥ = %.6f  (want %.6f, parallel)\n" Karr[1, 1] (f₁ * 2.0 + f₂ * 0.3)
@printf "            k_⊥ = %.6f  (want %.6f, series)\n" Karr[3, 3] 1 / (f₁ / 2.0 + f₂ / 0.3)

# ## Localization
#
# The layer strains follow from the same tensors. Their fraction-weighted average is the
# identity, and the in-plane block of every ``\mathbb A_i`` is the identity: the
# macroscopic in-plane strain reaches each layer unchanged, which is the compatibility
# condition read backwards.

𝔸 = Dict(nm => layer_strain_localization(lam, nm) for nm in layer_names(lam))
𝔹 = Dict(nm => layer_stress_localization(lam, nm) for nm in layer_names(lam))
fs = Dict(nm => layer_volume_fraction(lam, nm) for nm in layer_names(lam))

sumA = sum(fs[nm] * 𝔸[nm] for nm in layer_names(lam))
sumB = sum(fs[nm] * 𝔹[nm] for nm in layer_names(lam))
@printf "\n‖Σ fᵢ 𝔸ᵢ − 𝕀‖∞ = %.2e\n" maximum(abs, Matrix(KM(sumA)) - I)
@printf "‖Σ fᵢ 𝔹ᵢ − 𝕀‖∞ = %.2e\n" maximum(abs, Matrix(KM(sumB)) - I)

MA = Matrix(KM(𝔸[:A]))
@printf "in-plane block of 𝔸_A is the identity : %.2e\n" maximum(
    abs, MA[[1, 2, 6], [1, 2, 6]] - I
)

# The two Hill tensors of a layer are available as well. ``\mathbb P`` is the flat limit
# of the Hill polarization tensor and operates only within out-of-plane tensors;
# ``\mathbb Q`` operates only within in-plane ones.

ℙ, ℚ = laminate_hill(lam, :A)
@printf "\n‖ℙ:ℂ:ℙ − ℙ‖∞ = %.2e     ‖ℚ:ℙ‖∞ = %.2e\n" maximum(
    abs, Matrix(KM(ℙ ⊡ C₁ ⊡ ℙ)) - Matrix(KM(ℙ))
) maximum(abs, Matrix(KM(ℚ ⊡ ℙ)))

# ## Anisotropic layers, and an arbitrary normal
#
# Nothing above is special to isotropy or to `n = e₃`. With a tilted normal
# the result is a generic `Tens` in the laminate frame, and the exact
# out-of-plane law still holds.

lam_tilt = Laminate(; normal = (1, 1, 1))
add_layer!(lam_tilt, :A, Dict(:C => C₁); fraction = f₁)
add_layer!(lam_tilt, :B, Dict(:C => C₂); fraction = f₂)
C_tilt = homogenize(lam_tilt, Laminated(), :C)
println("\ntilted normal (1,1,1) → ", typeof(C_tilt))
println("  axis = ", TensND.axis(C_tilt))

# The Walpole coefficients are unchanged — only the axis moved, as they must
# be for an isotropic-layer stack.
@printf "  same Walpole coefficients as n = e₃ : %.2e\n" maximum(
    abs, collect(TensND.get_data(C_tilt)) .- collect(TensND.get_data(Cᵉᶠᶠ))
)
