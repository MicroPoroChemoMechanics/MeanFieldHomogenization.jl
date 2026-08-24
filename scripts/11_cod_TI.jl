# =============================================================================
#  scripts/11_cod_TI.jl
#
#  Analytical COD tensor of an elliptical / ribbon crack whose plane is
#  orthogonal to the isotropy axis of a transversely isotropic matrix.
#  Demonstrates MeanFieldHomogenization.jl's high-level API.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Plots

# TI moduli  (E₁=E, E₃=H·E₁, ν₁₂=ν₁, ν₃₁=H·ν₂, G₃₁=Γ·E₁/(2(1+ν₁₂)))
E_, H_, ν₁, ν₂, Γ_ = 1.0, 2.0, 0.4, 0.3, 3.0

n̂ = tens_basis(CanonicalBasis{3, Float64}(), 3)
E₁ = E_
E₃ = H_ * E₁
ν₁₂ = ν₁
ν₃₁ = H_ * ν₂
G₃₁ = Γ_ * E₁ / (2 * (1 + ν₁₂))
S = tens_TI(inv(E₁), -ν₁₂ / E₁, -ν₃₁ / E₃, inv(E₃), inv(4G₃₁), n̂)
C₀ = inv(S)

# Penny in TI isotropy plane
pc = PennyCrack(1.0)
Bpen = cod_tensor(pc, C₀)
println("Penny  B_ll = $(Bpen[1, 1])")
println("Penny  B_mm = $(Bpen[2, 2])")
println("Penny  B_nn = $(Bpen[3, 3])")

# Ribbon in TI isotropy plane
r = RibbonCrack(1.0)
Brib = cod_tensor(r, C₀)
println("Ribbon B_ll = $(Brib[1, 1])")
println("Ribbon B_mm = $(Brib[2, 2])")
println("Ribbon B_nn = $(Brib[3, 3])")

# Sweep aspect ratio for an elliptical crack with normal along TI axis
ηrange = 0.05:0.05:1.0
Bnn = [cod_tensor(EllipticCrack(1.0, η), C₀)[3, 3] for η in ηrange]
Bmm = [cod_tensor(EllipticCrack(1.0, η), C₀)[2, 2] for η in ηrange]
Bll = [cod_tensor(EllipticCrack(1.0, η), C₀)[1, 1] for η in ηrange]

plt = plot(
    ηrange, Bll, label = "B_ll", xlabel = "η", lw = 2,
    title = "TI matrix, crack normal ≡ TI axis"
)
plot!(plt, ηrange, Bmm, label = "B_mm", lw = 2)
plot!(plt, ηrange, Bnn, label = "B_nn", lw = 2)

figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
figpath = joinpath(figdir, "11_cod_TI.png")
savefig(plt, figpath)
display(plt)
println("Saved : ", figpath)
