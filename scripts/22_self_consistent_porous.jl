# =============================================================================
#  22_self_consistent_porous.jl
#
#  Self-consistent scheme on a porous iso material : voids (zero stiffness)
#  are too soft for the standard SC iteration, so we use the
#  AsymmetricSelfConsistent variant which switches automatically to the
#  compliance-form iteration. Compares against Voigt / Reuss bounds.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using Printf
using Plots

const k_solid, μ_solid = 90.0, 30.0
# Soft (not strictly zero) inclusion to keep all bounds well-defined
const k_void, μ_void = 0.01, 0.005

fs = collect(range(0.0, 0.5; length = 20))

function bulk_at(f, scheme)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => TensISO{3}(k_solid, μ_solid)))
    f > 0 && add_phase!(
        rve, :V, Ellipsoid(1.0),
        Dict(:C => TensISO{3}(k_void, μ_void)); fraction = f
    )
    return get_array(homogenize(rve, scheme))[1, 1, 1, 1]
end

p = plot(;
    xlabel = "porosity f", ylabel = "C_eff[1111]",
    title = "Porous iso composite — schemes vs bounds", legend = :topright
)
plot!(p, fs, [bulk_at(f, Voigt())                       for f in fs]; label = "Voigt", color = :red, lw = 2, ls = :dash)
plot!(p, fs, [bulk_at(f, Reuss())                       for f in fs]; label = "Reuss", color = :gray, lw = 2, ls = :dash)
plot!(p, fs, [bulk_at(f, MoriTanaka())                  for f in fs]; label = "MoriTanaka", color = :green, lw = 2)
plot!(p, fs, [bulk_at(f, DiluteDual())                  for f in fs]; label = "DiluteDual", color = :blue, lw = 2)
plot!(
    p, fs, [
        bulk_at(f, AsymmetricSelfConsistent(; abstol = 1.0e-10, maxiters = 200))
            for f in fs
    ]; label = "ASC", color = :purple, lw = 2
)
plot!(
    p, fs, [
        bulk_at(f, DifferentialScheme(; nsteps = 200))
            for f in fs
    ]; label = "Differential", color = :orange, lw = 2
)

figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
figpath = joinpath(figdir, "self_consistent_porous.png")
savefig(p, figpath)
display(p)
println("Saved : ", figpath)
