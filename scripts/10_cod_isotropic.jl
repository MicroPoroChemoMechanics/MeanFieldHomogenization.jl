# =============================================================================
#  scripts/10_cod_isotropic.jl
#
#  Compare the analytical isotropic COD tensor against both numerical
#  backends, sweep the aspect ratio η ∈ (0, 1] and plot the three
#  diagonal components of 𝐁 in the crack-local frame.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Plots

E, ν = 210.0, 0.3
k, μ = E / (3(1 - 2ν)), E / (2(1 + ν))
C₀ = TensISO{3}(3k, 2μ)

ηrange = 0.05:0.02:1.0
Bll = zeros(length(ηrange))
Bmm = similar(Bll)
Bnn = similar(Bll)

for (i, η) in enumerate(ηrange)
    c = EllipticCrack(1.0, η)
    B = cod_tensor(c, C₀)
    Bll[i] = B[1, 1]
    Bmm[i] = B[2, 2]
    Bnn[i] = B[3, 3]
end

plt = plot(
    ηrange, Bll, label = "B_ll", xlabel = "η = b/a",
    ylabel = "component of 𝐁", title = "COD tensor — isotropic matrix",
    lw = 2
)
plot!(plt, ηrange, Bmm, label = "B_mm", lw = 2)
plot!(plt, ηrange, Bnn, label = "B_nn", lw = 2)

# Penny and ribbon reference lines
pc = PennyCrack(1.0)
B_p = cod_tensor(pc, C₀)
hline!(plt, [B_p[3, 3]], label = "penny B_nn", linestyle = :dash)

r = RibbonCrack(1.0)
B_r = cod_tensor(r, C₀)
hline!(plt, [B_r[3, 3]], label = "ribbon B_nn (B²ᵈ = (3π/8)·lim_{η→0} B³ᵈ)", linestyle = :dot)

figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
figpath = joinpath(figdir, "10_cod_isotropic.png")
savefig(plt, figpath)
display(plt)
println("Saved : ", figpath)
