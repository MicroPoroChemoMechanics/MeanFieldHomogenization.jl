# =============================================================================
#  20_voigt_reuss_bounds.jl
#
#  Plots the Voigt and Reuss bulk-modulus bounds of an iso 2-phase composite
#  versus the inclusion volume fraction, on the same axes as the dilute,
#  Mori-Tanaka, self-consistent and differential schemes.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using Printf
using Plots

const k_m, μ_m = 30.0, 10.0    # 3κ, 2μ of matrix
const k_i, μ_i = 60.0, 20.0    # 3κ, 2μ of inclusion (stiffer)

function effective_bulk(rve, scheme)
    C = homogenize(rve, scheme)
    arr = get_array(C)
    return arr[1, 1, 1, 1]    # representative component for iso C
end

fs = collect(range(0.0, 0.6; length = 25))
schemes = [
    (Voigt(), :red, :solid),
    (Reuss(), :red, :dash),
    (Dilute(), :blue, :solid),
    (MoriTanaka(), :green, :solid),
    (SelfConsistent(; abstol = 1.0e-10, maxiters = 200), :purple, :solid),
    (DifferentialScheme(; nsteps = 200), :orange, :solid),
]

p = plot(;
    xlabel = "inclusion volume fraction f", ylabel = "C[1111]",
    title = "Bounds and schemes — iso 2-phase", legend = :topleft
)

for (sch, col, ls) in schemes
    ys = Float64[]
    for f in fs
        rve = RVE()
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(k_m, μ_m)); fraction = :rest)
        if f > 0
            add_phase!(
                rve, :I, Ellipsoid(1.0),
                Dict(:C => TensISO{3}(k_i, μ_i)); fraction = f
            )
        end
        push!(ys, effective_bulk(rve, sch))
    end
    plot!(
        p, fs, ys; label = string(typeof(sch).name.name), color = col, linestyle = ls,
        lw = 2
    )
end

figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
figpath = joinpath(figdir, "voigt_reuss_bounds.png")
savefig(p, figpath)
display(p)
println("Saved : ", figpath)

@printf("\nf = 0.30 :\n")
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(k_m, μ_m)); fraction = :rest)
add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(k_i, μ_i)); fraction = 0.3)
for (sch, _, _) in schemes
    @printf("  %-30s : %.4f\n", string(typeof(sch).name.name), effective_bulk(rve, sch))
end
