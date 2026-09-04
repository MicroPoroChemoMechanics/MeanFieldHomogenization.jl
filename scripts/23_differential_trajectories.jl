# =============================================================================
#  23_differential_trajectories.jl
#
#  Compares the Proportional, Sequential and CustomPath trajectories of the
#  differential scheme on a 2-phase RVE (one stiff, one soft inclusion).
#
#  In the dilute limit all trajectories agree; at finite fractions they
#  disagree by an amount proportional to f, illustrating the path-dependence
#  of the differential scheme for multi-phase RVEs.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using Printf
using Plots

const k_m, μ_m = 30.0, 10.0
const k_s, μ_s = 90.0, 30.0     # stiff inclusion
const k_w, μ_w = 5.0, 2.0       # soft inclusion

function build(f1, f2)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(k_m, μ_m)); fraction = :rest)
    f1 > 0 && add_phase!(
        rve, :STIFF, Ellipsoid(1.0),
        Dict(:C => TensISO{3}(k_s, μ_s)); fraction = f1
    )
    f2 > 0 && add_phase!(
        rve, :SOFT, Ellipsoid(1.0),
        Dict(:C => TensISO{3}(k_w, μ_w)); fraction = f2
    )
    return rve
end

fs = collect(range(0.005, 0.3; length = 25))   # skip f=0 (Sequential needs the named phases)

# Same total inclusion fraction in two phases : f_stiff = f_soft = f_total / 2
y_prop = Float64[]
y_stiff_first = Float64[]
y_soft_first = Float64[]

for ftot in fs
    f1 = f2 = ftot / 2
    rve = build(f1, f2)
    push!(y_prop, get_array(homogenize(rve, DifferentialScheme(; nsteps = 200)))[1, 1, 1, 1])
    push!(
        y_stiff_first,
        get_array(
            homogenize(
                rve, DifferentialScheme(;
                    trajectory = Sequential([:STIFF, :SOFT]), nsteps = 200
                )
            )
        )[1, 1, 1, 1]
    )
    push!(
        y_soft_first,
        get_array(
            homogenize(
                rve, DifferentialScheme(;
                    trajectory = Sequential([:SOFT, :STIFF]), nsteps = 200
                )
            )
        )[1, 1, 1, 1]
    )
end

p = plot(;
    xlabel = "total inclusion fraction f₁ + f₂",
    ylabel = "C_eff[1111]",
    title = "Differential scheme — trajectory dependence",
    legend = :topleft
)
plot!(p, fs, y_prop; label = "Proportional", lw = 2, color = :blue)
plot!(p, fs, y_stiff_first; label = "Sequential [stiff,soft]", lw = 2, color = :green)
plot!(p, fs, y_soft_first; label = "Sequential [soft,stiff]", lw = 2, color = :red)

figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
figpath = joinpath(figdir, "differential_trajectories.png")
savefig(p, figpath)
display(p)
println("Saved : ", figpath)

@printf("\nf_total      Prop          Stiff→Soft     Soft→Stiff\n")
for (i, ftot) in enumerate(fs)
    if i % 4 == 1
        @printf(
            "  %.3f       %.4f       %.4f       %.4f\n",
            ftot, y_prop[i], y_stiff_first[i], y_soft_first[i]
        )
    end
end
