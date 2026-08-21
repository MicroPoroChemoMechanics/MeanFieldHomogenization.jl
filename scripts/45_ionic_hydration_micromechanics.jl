# =============================================================================
#  45_ionic_hydration_micromechanics.jl
#
#  Elastic properties of a hydrating ordinary Portland cement paste, with the
#  phase assemblage DECIDED BY THE THERMODYNAMICS rather than written down.
#
#  This is the counterpart to `44_stoichiometric_hydration_micromechanics.jl`. The
#  micromechanics is identical — the same four-scale SC/SC/MT/MT model, the same
#  `paste_micromechanics.jl` — and only the chemistry differs:
#
#    44   clinker → aggregated solid→solid reactions with stated products,
#         and a hand-written priority cascade deciding which forms when.
#
#    45   clinker → dissolution into IONS (Ca²⁺, SiO₂, AlO₂⁻, FeO₂⁻, SO₄²⁻,
#         CO₃²⁻, H⁺) → a Gibbs minimization at every accepted step decides
#         which hydrates are stable and in what amounts. No sequencing rule
#         is written anywhere.
#
#  What the second model gives that the first cannot:
#
#    - the pore-solution pH and composition over time;
#    - the aluminate sequence as a RESULT, where model 44 has to encode it by
#      hand as a priority cascade. And the result depends on the mix, which is
#      the whole point: WITHOUT limestone the ettringite peaks early and
#      converts to monosulphate once the sulfate is spent, the AFm settling at
#      exactly the sulfate budget; WITH the 3.5 % calcite of this formulation
#      the carbonate stabilizes the AFt instead, forming monocarboaluminate, and
#      the ettringite persists to 28 days. Nothing in the code distinguishes the
#      two cases — the free energy does.
#    - a setting threshold: below percolation the self-consistent fixed point
#      collapses and the modulus is zero, which is physical, not a failure.
#
#  The chain, end to end:
#
#    binder composition
#      → Parrot-Killoh dissolution rates into the ions      [ChemistryLab]
#      → Gibbs minimization φ(bₑ) at every step             [ChemistryLab]
#      → volume fractions + chemical-shrinkage void         [ChemistryLab]
#      → four-scale SC/SC/MT/MT homogenization              [MeanFieldHomogenization]
#      → E(t)
#
#  Requires ChemistryLab ≥ 0.7.1: before it, a re-speciation could start outside
#  the feasible set of its own equality constraint, and the full OPC returned an
#  assemblage demanding 174 % of the sulfate present.
#
#  ENVIRONMENT — like script 44, this activates `docs/` rather than the
#  repository root: it needs ChemistryLab and OrdinaryDiffEq, which are not
#  dependencies of MeanFieldHomogenization.
#      julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#
#  Usage:
#    julia scripts/45_ionic_hydration_micromechanics.jl
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using Printf
using Plots
gr()

include(joinpath(@__DIR__, "common", "paste_micromechanics.jl"))
include(joinpath(@__DIR__, "common", "stoichiometric_hydration.jl"))
include(joinpath(@__DIR__, "common", "ionic_hydration.jl"))

# ── §1  Formulation ─────────────────────────────────────────────────────────
#
# The CEM I 52.5 N of Lavergne et al. (2018), Table 9, so the two chapters are
# comparable term by term: clinker 96 % with a Bogue composition C3S 65 / C2S 11
# / C3A 11 / C4AF 8, calcite 3.5 %, gypsum 4.6 %, Blaine 380 m²/kg, w/b = 0.50.

const CLINKER45 = (C3S = 0.65, C2S = 0.11, C3A = 0.11, C4AF = 0.08)
const WB45 = 0.5
const TEND45 = 28 * 86400.0

# Log-spaced, starting after the first hour: the assemblage moves fast early and
# the AFt peak is around 6 hours, so a linear grid would miss it.
const TIMES45 = 10 .^ range(log10(0.05 * 86400), log10(TEND45); length = 40)

# Two mixes, and the difference between them is the point of §5. Only the
# limestone content changes; the clinker, the gypsum and the water are the same.
const MIXES45 = [
    ("with 3.5 % calcite", 0.035),
    ("no limestone", 0.0),
]

results45 = map(MIXES45) do (label, filler)
    @printf "Running the ionic OPC model (%s)…\n" label
    r = run_ionic_hydration(;
        wb = WB45, clinker = CLINKER45, gypsum = 0.046, filler = filler,
        tend = TEND45, system = :opc,
    )
    @printf "  %d accepted steps, retcode = %s\n" length(r.sol.t) r.sol.retcode
    t, f, ph, po = ionic_fraction_history(r, TIMES45)
    E = [paste_moduli(fi; N = NTHETA_PASTE).E for fi in f]
    return (; label, filler, run = r, times = t, fracs = f, pH = ph, poro = po, E)
end
println()

run45 = results45[1].run

# ── §2  Pore solution and phase assemblage ──────────────────────────────────
#
# `ionic_fraction_history` re-solves φ(bₑ) at each instant with a sequential
# warm start, clipping and feasibility-restoring the guess first. That is not
# optional: the warm start is the equilibrium of the PREVIOUS bₑ, so once the
# sulfate is spent it demands more of it than exists.

times45 = results45[1].times
fracs45 = results45[1].fracs
pH45 = results45[1].pH
poro45 = results45[1].poro

# ── §3  Micromechanics — identical to script 44 ─────────────────────────────

E45 = results45[1].E

# The setting threshold: below percolation the SC fixed point collapses and the
# model reports a zero modulus rather than noise.
i_set = findfirst(>(0.0), E45)
t_set = i_set === nothing ? NaN : times45[i_set] / 3600

# ── §4  The same paste through the stoichiometric model, for comparison ─────

println("Running the stoichiometric model (script 44's chemistry) for comparison…")
run44 = run_hydration(;
    wb = WB45, clinker = CLINKER45, gypsum = 0.046, filler = 0.035,
    silica = 0.0, tend = TEND45,
)
_, fr44 = fraction_history(run44, TIMES45)
E44 = [paste_moduli(f; N = NTHETA_PASTE).E for f in fr44]

# ── §5  Report ──────────────────────────────────────────────────────────────

println()
println("┌────────────┬───────┬────────┬────────┬──────────┬──────────┐")
println("│  t         │  pH   │   φ    │   S    │ E ionic  │ E stoich │")
println("├────────────┼───────┼────────┼────────┼──────────┼──────────┤")
for t_target in (0.25, 1.0, 3.0, 7.0, 28.0)
    i = argmin(abs.(times45 ./ 86400 .- t_target))
    @printf "│ %6.2f d   │ %5.2f │ %6.4f │ %6.4f │ %6.2f   │ %6.2f   │\n" (
        times45[i] / 86400
    ) pH45[i] poro45[i].total (poro45[i].liquid / poro45[i].total) E45[i] E44[i]
end
println("└────────────┴───────┴────────┴────────┴──────────┴──────────┘")
println()
@printf "Setting threshold (first non-zero modulus): %.1f h\n" t_set

# The aluminate sequence, which is the point of this script: it is a RESULT.
i28 = argmin(abs.(times45 ./ 86400 .- 28.0))
println()
println("Aluminate sequence — decided by the free-energy minimization, not stated.")
println("Only the limestone content differs between these two mixes:")
println()
println("┌──────────────────────┬──────────┬──────────┬──────────┬──────────┐")
println("│  mix                 │ AFt peak │  at      │ AFt 28 d │ AFm 28 d │")
println("├──────────────────────┼──────────┼──────────┼──────────┼──────────┤")
for r in results45
    aft = [get(f, "AFt", 0.0) for f in r.fracs]
    afm = [get(f, "AFm", 0.0) for f in r.fracs]
    j = argmax(aft)
    @printf "│ %-20s │  %6.4f  │ %5.2f d  │  %6.4f  │  %6.4f  │\n" r.label aft[j] (
        r.times[j] / 86400
    ) aft[end] afm[end]
end
println("└──────────────────────┴──────────┴──────────┴──────────┴──────────┘")
println()

# The sulfate budget is the tell. With no carbonate to stabilize it, the AFt
# must give its sulfate up to the AFm; with calcite, monocarboaluminate forms
# instead and the ettringite survives.
let r = results45[2]
    aft = [get(f, "AFt", 0.0) for f in r.fracs]
    j = argmax(aft)
    depleted = findfirst(<(0.1 * aft[j]), aft[j:end])
    @printf "Without limestone the ettringite is depleted: peak %.4f at %.2f d" aft[j] (r.times[j] / 86400)
    depleted === nothing ? println(", still falling at 28 d.") :
        @printf ", down to a tenth of it by %.2f d.\n" (r.times[j + depleted - 1] / 86400)
end

println()
println("Volume fractions at 28 days:")
for (k, v) in sort(collect(fracs45[i28]); by = last, rev = true)
    v > 1.0e-4 && @printf "   %-14s %6.4f\n" k v
end
@printf "\nChemical shrinkage = %.3f of the fresh paste volume\n" get(fracs45[i28], "void", 0.0)

# ── §6  Figures ─────────────────────────────────────────────────────────────

td = times45 ./ 86400

p_pH = plot(
    td, pH45; xscale = :log10, lw = 2, legend = false,
    xlabel = "time [days]", ylabel = "pore-solution pH",
    title = "pH — not available from model 44", ylims = (11.5, 13.5),
)

p_seq = plot(;
    xscale = :log10, xlabel = "time [days]", ylabel = "volume fraction",
    title = "aluminates, as a result (calcite stabilises AFt)", legend = :topleft
)
for (i, r) in enumerate(results45)
    ls = i == 1 ? :solid : :dash
    for (lbl, key, c) in (("AFt", "AFt", 1), ("AFm", "AFm", 2))
        plot!(
            p_seq, td, [get(f, key, 0.0) for f in r.fracs];
            lw = 2, ls = ls, color = c, label = "$lbl — $(r.label)",
        )
    end
end

p_E45 = plot(
    td, E45; xscale = :log10, lw = 2, label = "ionic (45)",
    xlabel = "time [days]", ylabel = "E [GPa]", title = "Young's modulus", legend = :topleft,
)
plot!(p_E45, td, E44; lw = 2, ls = :dash, label = "stoichiometric (44)")

const figdir45 = joinpath(@__DIR__, "figures")
isdir(figdir45) || mkdir(figdir45)
savefig(p_pH, joinpath(figdir45, "45_pore_solution_pH.png"))
savefig(p_seq, joinpath(figdir45, "45_aluminate_sequence.png"))
savefig(p_E45, joinpath(figdir45, "45_young_modulus.png"))
display(plot(p_pH, p_seq, p_E45; layout = (1, 3), size = (1650, 460)))
@printf "\nSaved: %s\n" joinpath(figdir45, "45_*.png")
