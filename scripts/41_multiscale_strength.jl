# =============================================================================
#  41_multiscale_strength.jl  (was 28_*)
#
#  Three-scale upscaling of cement-paste / mortar elasticity and quasi-brittle
#  strength — Pichler & Hellmich (2011), Cement and Concrete Research 41,
#  467-476, https://doi.org/10.1016/j.cemconres.2011.01.010.
#
#  This is the DEMO / PLOT front-end.  The model itself lives in
#  `scripts/common/quasibrittle_strength.jl` and is built entirely on the public
#  MeanFieldHomogenization API : a multi-bin Self-Consistent hydrate foam whose several
#  non-coaxial `TISymmetrize` needle families are averaged EXACTLY about the
#  global axis (`TensTI{4,T,8}`), then two Mori-Tanaka stages (CP, MO).  The
#  strength sensitivity is one ForwardDiff pass through the whole chain —
#  reproducing Fig. 4 of the paper.  Cross-validated against echoes in
#  `scripts/bench_echoes/benchmark_strength.jl` (moduli 1 %, fc 2 %).
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using Printf
using Plots

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

include(joinpath(@__DIR__, "common", "quasibrittle_strength.jl"))

println("="^78)
println("Multi-scale upscaling of cement-paste / mortar (Pichler-Hellmich 2011)")
println("(NTHETA = $NTHETA, ω = $ω_aspect)")
println("="^78)

const wcs = [0.157, 0.25, 0.35, 0.5, 0.65, 0.8]
const sc_default = 0.0
const N_α = 20
const α_min = 0.005

p1 = plot(; xlabel = "α", ylabel = "k_mortar (GPa)", xlims = (0, 1), ylims = (0, 35), legend = :topleft)
p2 = plot(; xlabel = "α", ylabel = "μ_mortar (GPa)", xlims = (0, 1), ylims = (0, 20), legend = false)
p3 = plot(; xlabel = "α", ylabel = "f_c / σ_ult", xlims = (0, 1), ylims = (0, 2), legend = false)
p4 = plot(; xlabel = "f_c / σ_ult", ylabel = "E_mortar (GPa)", xlims = (0, 2), ylims = (0, 50), legend = false)

for wc in wcs
    αs = collect(range(α_min, αmax(wc) * (1 - 1.0e-12); length = N_α))
    K_arr = Float64[]; μ_arr = Float64[]; E_arr = Float64[]; fc_arr = Float64[]
    α_kept = Float64[]
    for α_p in αs
        try
            r = compute_point(wc, α_p; sc = sc_default)
            push!(K_arr, r.K_mo); push!(μ_arr, r.μ_mo); push!(E_arr, r.E_mo)
            push!(fc_arr, r.fc); push!(α_kept, α_p)
        catch e
            @warn "compute_point failed at wc=$wc α=$α_p" exception = (e, catch_backtrace())
        end
    end
    if !isempty(K_arr)
        plot!(p1, α_kept, K_arr, lw = 2, label = "wc = $wc")
        plot!(p2, α_kept, μ_arr, lw = 2)
        plot!(p3, α_kept, fc_arr, lw = 2)
        plot!(p4, fc_arr, E_arr, lw = 2)
    end
end

p_full = plot(
    p1, p2, p3, p4; layout = (2, 2), size = (1000, 800),
    plot_title = "Multi-scale strength upscaling (Pichler-Hellmich 2011)"
)

figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
figpath = joinpath(figdir, "41_multiscale_strength.png")
savefig(p_full, figpath)
display(p_full)
@printf "\nSaved : %s\n" figpath

println("\n[Tabular] wc = 0.50")
@printf "  %5s   %10s   %10s   %10s   %10s\n" "α" "k_mo" "μ_mo" "fc" "E_mo"
println("  " * "─"^60)
let wc = 0.5
    αs = collect(filter(α -> α > 0.05, range(0.05, αmax(wc); length = 6)))
    for α_p in αs
        try
            r = compute_point(wc, α_p)
            @printf "  %.3f   %10.4f   %10.4f   %10.4f   %10.4f\n" α_p r.K_mo r.μ_mo r.fc r.E_mo
        catch e
            @printf "  %.3f   (failed : %s)\n" α_p typeof(e)
        end
    end
end

println("\nDone.")
