# =============================================================================
#  scripts/14_sif_computation.jl
#
#  Plot the three SIF modes along the elliptic crack front versus the
#  tip polar angle θˣ, for an elliptic crack with aspect ratio 0.1 in a
#  TI matrix.  Exercises the elliptic branch of `sif(...)`.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Plots

# TI matrix parameters
E_, H_, ν₁, ν₂, Γ_ = 1.0, 2.0, 0.4, 0.3, 3.0
n̂ = tens_basis(CanonicalBasis{3, Float64}(), 3)
E₁ = E_; E₃ = H_ * E₁; ν₁₂ = ν₁; ν₃₁ = H_ * ν₂
G₃₁ = Γ_ * E₁ / (2 * (1 + ν₁₂))
S = tens_TI(inv(E₁), -ν₁₂ / E₁, -ν₃₁ / E₃, inv(E₃), inv(4G₃₁), n̂)
C₀ = inv(S)

a, b = 1.0, 0.1
crack = EllipticCrack(a, b)
e1, e2, e3 = tens_basis(CanonicalBasis{3, Float64}())
Σ = e3 ⊗ˢ e3

# Sweep θˣ = atan(b sin θʸ, a cos θʸ) along the crack front
lθʸ = range(0, π, length = 101)
lKᴵ = Float64[]
lKᴵᴵ = Float64[]
lKᴵᴵᴵ = Float64[]
lθˣ = similar(lθʸ, Float64)
for (i, θʸ) in enumerate(lθʸ)
    y₀ = cos(θʸ) * e1 + sin(θʸ) * e2
    𝐊, (Kᴵ, Kᴵᴵ, Kᴵᴵᴵ) = sif(crack, C₀, Σ; y₀ = y₀)
    push!(lKᴵ, abs(Kᴵ) / √b)
    push!(lKᴵᴵ, abs(Kᴵᴵ) / √b)
    push!(lKᴵᴵᴵ, abs(Kᴵᴵᴵ) / √b)
    lθˣ[i] = atan(b * sin(θʸ), a * cos(θʸ))
end

plt = plot(lθˣ, lKᴵ, label = "Kᴵ", xlabel = "θₓ", lw = 2)
plot!(plt, lθˣ, lKᴵᴵ, label = "Kᴵᴵ", lw = 2)
plot!(plt, lθˣ, lKᴵᴵᴵ, label = "Kᴵᴵᴵ", lw = 2)
title!(plt, "Elliptic crack in TI matrix, η = $(b / a)")

figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
figpath = joinpath(figdir, "14_sif_computation.png")
savefig(plt, figpath)
display(plt)
println("Saved : ", figpath)
