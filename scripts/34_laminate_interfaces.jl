# # Imperfect interfaces in a laminate, and the size effect
#
# A planar interface is the curvature-free case of the spherical one, so the
# four interface models of `LayeredSpheres` carry over unchanged — and the
# algebra collapses to **two additive terms** whose effects are exactly
# complementary:
#
# - **primal** (`SpringInterface`, `KapitzaInterface`) — a jump of the field.
#   Moves the *out-of-plane* response and leaves the in-plane one untouched.
# - **dual** (`MembraneInterface`, `SurfaceConductiveInterface`) — a surface
#   stiffness. The interfaces being planar,
#   ``\mathrm{div}_s\,\boldsymbol\sigma^s = 0``: there is *no traction jump at
#   all*, and only the *in-plane* response moves.
#
# Both enter with the weight `1/L`, an interface **density**: at fixed volume
# fractions, doubling the period halves the correction. That is why a laminate
# stores thicknesses and not just fractions.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using Printf
using LinearAlgebra
using Plots

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

k₁, μ₁, k₂, μ₂ = 2.0, 0.8, 0.5, 0.2
C₁ = TensISO{3}(3k₁, 2μ₁)
C₂ = TensISO{3}(3k₂, 2μ₂)
K₁, K₂ = TensISO{3}(2.0), TensISO{3}(0.3)

# A bilayer parameterized by its interfaces and by an overall length scale.
function bilayer(; itf = PerfectInterface(), itf2 = PerfectInterface(), L = 1.0)
    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(lam, :A, Dict(:C => C₁, :K => K₁); thickness = 0.3L, interface = itf)
    add_layer!(lam, :B, Dict(:C => C₂, :K => K₂); thickness = 0.7L, interface = itf2)
    return lam
end

C₃₃(lam) = Matrix(KM(homogenize(lam, Laminated(), :C)))[3, 3]
C₁₂(lam) = Matrix(KM(homogenize(lam, Laminated(), :C)))[6, 6] / 2
k_perp(lam) = Matrix(components(homogenize(lam, Laminated(), :K)))[3, 3]
k_par(lam) = Matrix(components(homogenize(lam, Laminated(), :K)))[1, 1]

ref = bilayer()
@printf "perfect bonding : C₃₃₃₃ = %.6f   C₁₂₁₂ = %.6f\n" C₃₃(ref) C₁₂(ref)

# ## The primal interface acts out of plane only
#
# `SpringInterface(kn, kt)` takes the interface **stiffnesses**; the keyword
# form `SpringInterface(; sn, st)` takes the matching compliances, which is
# what the type stores. `sn = st = 0` is perfect bonding, and growing the
# compliance decouples the layers.

println("\nspring interface (kn = kt)")
println("       kn        C₃₃₃₃        C₁₂₁₂")
println("─"^42)
for kn in (0.0, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e0, 1.0e2)
    cell = bilayer(; itf = SpringInterface(; sn = kn, st = kn))
    @printf "  %8.1e  %10.6f  %10.6f\n" kn C₃₃(cell) C₁₂(cell)
end
println("  → C₁₂₁₂ never moves: a spring is invisible in the plane.")

# The out-of-plane law stays exact, with the compliance simply added to the
# series:
#
# ```math
# (\underline{n}\cdot\mathbb{C}^{hom}\cdot\underline{n})^{-1}
#  = \sum_i f_i (\underline{n}\cdot\mathbb{C}_i\cdot\underline{n})^{-1}
#  + \frac{1}{L}\sum_j \boldsymbol{\mathcal{K}}_j
# ```

kn = 5.0e-2
lam = bilayer(; itf = SpringInterface(; sn = kn, st = kn))
λ(k, μ) = k - 2μ / 3
series = 0.3 / (λ(k₁, μ₁) + 2μ₁) + 0.7 / (λ(k₂, μ₂) + 2μ₂)
@printf "\nexact check : 1/C₃₃₃₃ = %.10f   series + kn/L = %.10f\n" 1 / C₃₃(lam) (series + kn)

# ## The dual interface acts in plane only
#
# A Gurtin-Murdoch membrane of surface moduli `(κs, μs)` adds `μs/L` to
# `C₁₂₁₂` — exactly, additively — and leaves `C₃₃₃₃` alone.

println("\nmembrane interface (κs, μs = κs/2)")
println("       κs        C₃₃₃₃        C₁₂₁₂      ΔC₁₂₁₂")
println("─"^54)
for κs in (0.0, 0.02, 0.05, 0.10, 0.20)
    cell = bilayer(; itf = MembraneInterface(κs, κs / 2))
    @printf "  %8.3f  %10.6f  %10.6f  %10.6f\n" κs C₃₃(cell) C₁₂(cell) (C₁₂(cell) - C₁₂(ref))
end
println("  → ΔC₁₂₁₂ = μs/L exactly, and C₃₃₃₃ never moves.")

# ## The size effect
#
# Keep the volume fractions fixed and scale the whole cell. With perfect
# bonding nothing happens — the classical result depends on fractions alone.
# With an interface, the correction decays like `1/L`.

Ls = 10 .^ range(-1.5, 2.5; length = 60)
kn_fixed = 5.0e-2
c33_spring = [C₃₃(bilayer(; itf = SpringInterface(; sn = kn_fixed, st = kn_fixed), L = L)) for L in Ls]
c33_perf = [C₃₃(bilayer(; L = L)) for L in Ls]

p1 = plot(
    Ls, c33_spring; xscale = :log10, lw = 2, label = "spring, kn = $(kn_fixed)",
    xlabel = "period L", ylabel = "C₃₃₃₃", legend = :bottomright
)
plot!(p1, Ls, c33_perf; lw = 2, ls = :dash, label = "perfect bonding")

κs_fixed = 0.05
c12_memb = [C₁₂(bilayer(; itf = MembraneInterface(κs_fixed, κs_fixed / 2), L = L)) for L in Ls]
c12_perf = [C₁₂(bilayer(; L = L)) for L in Ls]
p2 = plot(
    Ls, c12_memb; xscale = :log10, lw = 2, label = "membrane, κs = $(κs_fixed)",
    xlabel = "period L", ylabel = "C₁₂₁₂", legend = :topright
)
plot!(p2, Ls, c12_perf; lw = 2, ls = :dash, label = "perfect bonding")

# A thin cell is dominated by its interfaces; a thick one forgets them. The
# spring softens (compliance added), the membrane stiffens (stiffness added).

plot(p1, p2; layout = (1, 2), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (900, 350))

# ## Transport: Kapitza and the conductive surface layer
#
# The same two roles, in closed form. An interfacial resistance adds to the
# series law; a conductive surface layer adds to the parallel one.

println("\ntransport")
println("        ρ (Kapitza)   1/k⊥          k∥")
println("─"^48)
for ρ in (0.0, 0.05, 0.2, 1.0)
    cell = bilayer(; itf = KapitzaInterface(ρ))
    @printf "  %10.3f   %10.6f   %10.6f\n" ρ 1 / k_perp(cell) k_par(cell)
end
@printf "  (series alone = %.6f)\n" (0.3 / 2.0 + 0.7 / 0.3)

println("\n        ks (surface)  1/k⊥          k∥")
println("─"^48)
for ks in (0.0, 0.05, 0.2, 1.0)
    cell = bilayer(; itf = SurfaceConductiveInterface(ks))
    @printf "  %10.3f   %10.6f   %10.6f\n" ks 1 / k_perp(cell) k_par(cell)
end
@printf "  (parallel alone = %.6f)\n" (0.3 * 2.0 + 0.7 * 0.3)

# ## The displacement jump
#
# What distinguishes a spring interface from a softer layer: the compliance
# shows up as a *discontinuity*, not as a strain.

E = Tens([0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0e-3])     # pure normal strain
lam = bilayer(; itf = SpringInterface(; sn = 1.0e-2, st = 0.0))
jump = interface_jump(lam, 1, E)
Σ33 = Matrix(components(homogenize(lam, Laminated(), :C) ⊡ E))[3, 3]
@printf "\nunder E₃₃ = 1e-3 : Σ₃₃ = %.6e,  [u] = (%.2e, %.2e, %.3e)\n" Σ33 jump[1] jump[2] jump[3]
@printf "  [u]₃ / (kn Σ₃₃) = %.12f\n" jump[3] / (1.0e-2 * Σ33)
