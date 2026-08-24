# =============================================================================
#  21_dilute_vs_mori_tanaka.jl
#
#  Demonstrates the convergence of Dilute and Mori-Tanaka schemes in the
#  dilute limit (small f) and their divergence at finite f.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using Printf
using Plots

const k_m, μ_m = 30.0, 10.0
const k_i, μ_i = 60.0, 20.0

function build(f)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => TensISO{3}(k_m, μ_m)))
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(k_i, μ_i)); fraction = f)
    return rve
end

fs = exp10.(range(-4, log10(0.6); length = 30))
y_dil = [get_array(homogenize(build(f), Dilute()))[1, 1, 1, 1] for f in fs]
y_mt = [get_array(homogenize(build(f), MoriTanaka()))[1, 1, 1, 1] for f in fs]

p = plot(
    fs, y_dil; xscale = :log10, label = "Dilute", lw = 2, color = :blue,
    xlabel = "f", ylabel = "C[1111]",
    title = "Dilute vs Mori-Tanaka — iso 2-phase"
)
plot!(p, fs, y_mt; label = "MoriTanaka", lw = 2, color = :green)
plot!(p, fs, fill(y_dil[1], length(fs)); label = "matrix", lw = 1, color = :gray, ls = :dash)

figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
figpath = joinpath(figdir, "dilute_vs_mt.png")
savefig(p, figpath)
display(p)
println("Saved : ", figpath)

@printf("\n  f       Dilute      MT          Δ(MT-Dil)\n")
for (i, f) in enumerate(fs)
    if i % 5 == 1
        @printf(
            "  %.2e   %.4e   %.4e   %.2e\n",
            f, y_dil[i], y_mt[i], y_mt[i] - y_dil[i]
        )
    end
end
