# =============================================================================
#  44_lavergne_hydration_micromechanics.jl
#
#  Elastic properties of a hydrating blended cement paste, with the volume
#  fractions COMPUTED BY THE CHEMISTRY rather than correlated.
#
#    Lavergne, F., Ben Fraj, A., Bayane, I. and Barthélémy, J.-F. (2018),
#    "Estimating the mechanical properties of hydrating blended cementitious
#    materials: an investigation based on micromechanics",
#    Cement and Concrete Research 104, 37-60.
#    https://doi.org/10.1016/j.cemconres.2017.10.018
#
#  The chain, end to end:
#
#    binder composition
#      → Parrot-Killoh (clinker) + Waller (addition)        [ChemistryLab]
#      → moles of every phase, by stoichiometry             [ChemistryLab]
#      → volume fractions + chemical-shrinkage void         [ChemistryLab]
#      → four-scale SC/SC/MT/MT homogenization              [MeanFieldHomogenization]
#      → E(t)
#
#  Every other cement model in this repository starts from a Powers-type
#  correlation `volume_fractions(w/c, α)`. This one does not: the fractions come
#  out of a kinetics ODE and a molar-volume balance, so the composition of the
#  binder — not just its water content — reaches the effective modulus.
#
#  The model itself lives in `scripts/common/lavergne_model.jl` (micromechanics)
#  and `scripts/common/lavergne_hydration.jl` (chemistry).
#
#  ENVIRONMENT — unlike every other script here, this one activates `docs/`
#  rather than the repository root: it needs ChemistryLab.jl and OrdinaryDiffEq,
#  which are not dependencies of MeanFieldHomogenization. `docs/Project.toml`
#  carries them (with ChemistryLab as a path source), so
#      julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#  is all the setup required.
#
#  Usage:
#    julia scripts/44_lavergne_hydration_micromechanics.jl
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using ForwardDiff
using Printf
using Plots
gr()

include(joinpath(@__DIR__, "common", "lavergne_model.jl"))
include(joinpath(@__DIR__, "common", "lavergne_hydration.jl"))

# ── §1  Formulations ────────────────────────────────────────────────────────
#
# CEM I 52.5 N of the paper (Table 9): clinker 96 % with a Bogue composition
# C3S 65 / C2S 11 / C3A 11 / C4AF 8, calcite 3.5 %, gypsum 4.6 %, Blaine
# 380 m²/kg. The blends follow the paper's own naming: C100 is the plain paste,
# C85L15 substitutes 15 % limestone filler, C95SF05 5 % silica fume.

const CLINKER = (C3S = 0.65, C2S = 0.11, C3A = 0.11, C4AF = 0.08)
const BLAINE = 380.0u"m^2/kg"
const NTHETA = 20

const MIXES = [
    ("C100   w/b=0.50", (wb = 0.5, filler = 0.035, silica = 0.0)),
    ("C85L15 w/b=0.50", (wb = 0.5, filler = 0.185, silica = 0.0)),
    ("C95SF05 w/b=0.50", (wb = 0.5, filler = 0.035, silica = 0.05)),
    ("C100   w/b=0.32", (wb = 0.32, filler = 0.035, silica = 0.0)),
]

const TIMES = 10 .^ range(log10(0.05 * 86400), log10(90 * 86400); length = 60)

hydrate(mix) = run_hydration(;
    wb = mix.wb, clinker = CLINKER, gypsum = 0.046,
    filler = mix.filler, silica = mix.silica,
    blaine = BLAINE, tend = 90 * 86400.0,
)

# ── §2  Hydration of the reference paste ────────────────────────────────────

@info "Integrating the hydration of the reference paste (C100, w/b = 0.50)…"
run0 = hydrate(MIXES[1][2])
α = degrees_of_hydration(run0.sol, run0.kp; times = TIMES)
ᾱ = mean_degree_of_hydration(run0.sol, run0.kp; times = TIMES)

p_alpha = plot(
    TIMES ./ 86400, [α[k] for k in ("C3S", "C2S", "C3A", "C4AF")];
    xscale = :log10, xlabel = "age [days]", ylabel = "degree of hydration α",
    label = ["C₃S" "C₂S" "C₃A" "C₄AF"], lw = 2, legend = :topleft,
    title = "Parrot & Killoh kinetics",
)
plot!(p_alpha, TIMES ./ 86400, ᾱ; lw = 3, color = :black, ls = :dash, label = "ᾱ (mass)")

# ── §3  Volume fractions ────────────────────────────────────────────────────

_, fs = fraction_history(run0, TIMES)

# Explicit palette: the default cycle repeats after nine series, which put FH3
# and water in the same teal on a chart whose whole point is telling phases apart.
const PHASE_COLOR = [
    "anhydrous" => :grey35, "C-S-H" => :steelblue, "CH" => :indianred,
    "AFt" => :mediumpurple, "AFm" => :orchid, "hydrogarnet" => :darkorange,
    "FH3" => :sienna, "gypsum" => :gold, "calcite" => :darkkhaki,
    "silica" => :olive, "water" => :lightskyblue, "void" => :white,
]
present = [k for (k, _) in PHASE_COLOR if maximum(get.(fs, k, 0.0)) > 1.0e-3]
cols = [Dict(PHASE_COLOR)[k] for k in present]
Y = cumsum(reduce(hcat, [get.(fs, k, 0.0) for k in present]); dims = 2)

p_frac = plot(
    TIMES ./ 86400, Y;
    xscale = :log10, xlabel = "age [days]", ylabel = "cumulative volume fraction",
    label = reshape(present, 1, :), lw = 0.5, linecolor = :grey30,
    fillrange = hcat(zeros(length(TIMES)), Y[:, 1:(end - 1)]),
    fillcolor = reshape(cols, 1, :), fillalpha = 0.9,
    legend = :outerright, ylims = (0, 1),
    title = "Phase assemblage of the sealed paste",
)

# ── §4  Effective Young's modulus of every mix ──────────────────────────────

p_E = plot(;
    xscale = :log10, xlabel = "age [days]", ylabel = "E [GPa]",
    legend = :topleft, title = "Effective Young's modulus",
)
results = Dict{String, Vector{Float64}}()
for (name, mix) in MIXES
    @info "Homogenizing $name…"
    r = hydrate(mix)
    _, f = fraction_history(r, TIMES)
    E = [lavergne_paste_moduli(fi; N = NTHETA).E for fi in f]
    results[name] = E
    plot!(p_E, TIMES ./ 86400, E; lw = 2, label = name)
end

# ── §5  Sensitivity through the chemistry, by ForwardDiff ───────────────────
#
# The homogenization chain is AD-clean, so ∂E/∂(anything the RVE depends on)
# costs one Dual evaluation. Here: the sensitivity of E to the volume fraction
# of each phase family at 28 days — which phase would it pay to grow?

f28 = fraction_history(run0, [28 * 86400.0])[2][1]
@info "Sensitivities of E(28 d) to each phase fraction…"
sens = Dict{String, Float64}()
for k in keys(f28)
    get(f28, k, 0.0) > 1.0e-4 || continue
    g(x) = begin
        f = Dict{String, Any}(kk => vv for (kk, vv) in f28)
        f[k] = x
        lavergne_paste_moduli(f; N = 8).E
    end
    sens[k] = ForwardDiff.derivative(g, f28[k])
end

# ── §6  Summary ─────────────────────────────────────────────────────────────

i28 = findmin(abs.(TIMES .- 28 * 86400))[2]
println()
println("┌──────────────────────────────────────────────────────────┐")
println("│  Lavergne et al. (2018) — chemistry-driven E(t)           │")
println("├──────────────────────┬──────────┬──────────┬─────────────┤")
@printf "│ %-20s │ %8s │ %8s │ %11s │\n" "mix" "E(1 d)" "E(28 d)" "E(90 d)"
println("├──────────────────────┼──────────┼──────────┼─────────────┤")
for (name, _) in MIXES
    E = results[name]
    i1 = findmin(abs.(TIMES .- 86400))[2]
    @printf "│ %-20s │ %6.2f   │ %6.2f   │ %9.2f   │\n" name E[i1] E[i28] E[end]
end
println("└──────────────────────┴──────────┴──────────┴─────────────┘")
println()
println("Volume fractions at 28 days (C100, w/b = 0.50):")
for k in present
    v = get(f28, k, 0.0)
    v > 1.0e-4 && @printf "   %-14s %6.3f   ∂E/∂f = %+8.1f GPa\n" k v get(sens, k, NaN)
end
println()
@printf "ᾱ(28 d) = %.3f   chemical shrinkage = %.3f of the paste volume\n" ᾱ[i28] get(f28, "void", 0.0)

# ── §7  Figures ─────────────────────────────────────────────────────────────

const figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
savefig(p_alpha, joinpath(figdir, "44_hydration_alpha.png"))
savefig(p_frac, joinpath(figdir, "44_volume_fractions.png"))
savefig(p_E, joinpath(figdir, "44_young_modulus.png"))
display(plot(p_alpha, p_frac, p_E; layout = (1, 3), size = (1650, 460)))
@printf "\nSaved: %s\n" joinpath(figdir, "44_*.png")
