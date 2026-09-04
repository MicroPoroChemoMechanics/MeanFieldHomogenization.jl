# =============================================================================
#  24_differential_loading_paths.jl
#
#  Demonstrates the **path-dependence of the differential homogenization
#  scheme** (DEM) by computing the effective stiffness of a 3-phase
#  composite (matrix + 2 solid inclusions) along several incorporation
#  trajectories `τ -> f_α(τ)` :
#
#    1. `Proportional()`           — both phases grow linearly together.
#    2. `Sequential([:I1, :I2])`   — phase 1 first, then phase 2.
#    3. `Sequential([:I2, :I1])`   — phase 2 first, then phase 1.
#    4. `Path(:I1 => τ -> τ², :I2 => τ -> 2τ - τ²)`
#                                  — phase 2 frontloaded (concave), phase
#                                    1 backloaded (convex), both reach 1
#                                    at τ = 1.
#
#  All trajectories satisfy `f_α(0) = 0`, `f_α(1) = 1`, so they reach
#  the same **target** volume fractions at τ = 1.  But the resulting
#  effective stiffness `C^hom(τ = 1)` **differs** : DEM is genuinely
#  path-dependent because each infinitesimal volume increment is added
#  as a dilute inclusion in the *current* effective medium, which
#  itself depends on the prior incorporation history.
#
#  Output : `scripts/figures/24_differential_loading_paths.png` plotting
#  the bulk and shear moduli of `C^hom(τ)` along each path.
#
#  Usage  : julia --project scripts/24_differential_loading_paths.jl
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using Plots

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
using Printf

# ─── RVE : matrix + 2 solid inclusions ─────────────────────────────────────

const C_M = TensISO{3}(3 * 5.0, 2 * 2.0)        # matrix : k = 5,    μ = 2
const C_I1 = TensISO{3}(3 * 30.0, 2 * 12.0)      # phase 1 : stiff   (5× stiffer)
const C_I2 = TensISO{3}(3 * 0.5, 2 * 0.2)       # phase 2 : compliant (0.1× softer)

const F1, F2 = 0.2, 0.2    # target volume fractions

function build_rve()
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_M); fraction = :rest)
    add_phase!(rve, :I1, Ellipsoid(1.0), Dict(:C => C_I1); fraction = F1)
    add_phase!(rve, :I2, Ellipsoid(1.0), Dict(:C => C_I2); fraction = F2)
    return rve
end

# ─── Compute C^hom along τ for each trajectory ─────────────────────────────
#
# `differential_path` returns the states saved along τ (`nsteps + 1` of
# them), instead of the τ = 1 value that `homogenize` returns.

const NSTEPS = 200

function eval_path(traj)
    scheme = DifferentialScheme(;
        trajectory = traj, nsteps = NSTEPS,
        abstol = 1.0e-9, reltol = 1.0e-7
    )
    τ, Cs = differential_path(build_rve(), scheme, :C)
    kμ = k_mu.(Cs)
    return τ, first.(kμ), last.(kμ)
end

println("Computing four loading-path scenarios on the same target (f₁=$F1, f₂=$F2)…")

paths_to_run = (
    ("Proportional", Proportional()),
    ("Sequential :I1 → :I2", Sequential(:I1, :I2)),
    ("Sequential :I2 → :I1", Sequential(:I2, :I1)),
    (
        "Path (I1 ∝ τ²,  I2 ∝ 2τ−τ²)",
        Path(:I1 => τ -> τ^2, :I2 => τ -> 2τ - τ^2),
    ),
)

results = Dict{String, NTuple{3, Vector{Float64}}}()
for (name, traj) in paths_to_run
    println("  $name…")
    τ, k, μ = eval_path(traj)
    results[name] = (τ, k, μ)
    @printf "    k_eff(τ=1) = %.5f   μ_eff(τ=1) = %.5f\n" k[end] μ[end]
end

# ─── Plot ──────────────────────────────────────────────────────────────────

p_k = plot(
    xlabel = "τ  (fictitious incorporation time)",
    ylabel = "k_eff(τ)",
    title = "Bulk modulus along the trajectory",
    legend = :right
)
p_μ = plot(
    xlabel = "τ",
    ylabel = "μ_eff(τ)",
    title = "Shear modulus along the trajectory",
    legend = :right
)

colors = (:black, :red, :blue, :green)
for (i, (name, _)) in enumerate(paths_to_run)
    τ, k, μ = results[name]
    plot!(p_k, τ, k; label = name, color = colors[i], linewidth = 2)
    plot!(p_μ, τ, μ; label = name, color = colors[i], linewidth = 2)
end

fig = plot(p_k, p_μ; layout = (1, 2), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (1400, 600))
mkpath(joinpath(@__DIR__, "figures"))
out = joinpath(@__DIR__, "figures", "24_differential_loading_paths.png")
savefig(fig, out)
display(fig)
println("\nSaved : $out")

# ─── Numeric report ────────────────────────────────────────────────────────

println()
println("═══════════════════════════════════════════════════════════════════")
println(" Effective moduli at τ = 1 (same target volume fractions, different paths)")
println("═══════════════════════════════════════════════════════════════════")
@printf "  %-32s  %-12s  %-12s\n" "trajectory" "k_eff" "μ_eff"
for (name, _) in paths_to_run
    _, k, μ = results[name]
    @printf "  %-32s  %-12.5f  %-12.5f\n" name k[end] μ[end]
end
println()
println("The DEM scheme is genuinely **path-dependent** : different")
println("incorporation sequences `τ → f_α(τ)` reaching the same target")
println("volume fractions `f_α^∞` at τ=1 give different `C^hom(τ=1)`.")
