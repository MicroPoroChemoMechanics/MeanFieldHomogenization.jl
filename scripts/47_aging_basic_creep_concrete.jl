# =============================================================================
#  47_aging_basic_creep_concrete.jl
#
#  Aging basic creep of two hydrating concretes, from their composition, against
#  the creep tests of Le Roy (1995) at several loading ages.
#
#    Lavergne, F. and Barthélémy, J.-F. (2020), "Confronting a refined multiscale
#    estimate for the aging basic creep of concrete with a comprehensive
#    experimental database", Cement and Concrete Research 136, 106163.
#    https://doi.org/10.1016/j.cemconres.2020.106163
#
#  The chain:
#
#    cement composition, w/c, silica fume                       [lavergne2020.json]
#      → hydration, volume fractions at each instant             [ChemistryLab]
#      → aging viscoelastic phases, one per step of hydration    [precipitated, dissolved]
#      → four scales SC / SC / MT / MT, each handing its operator on   [homogenize_alv]
#        the shared grid to the next
#      → J(t, t') of the concrete for every loading age
#      → against the measured compliances                       [leroy1995.json]
#
#  The model lives in `scripts/common/aging_creep_micromechanics.jl` (functions
#  only), on top of the elastic model of `paste_micromechanics.jl` and the
#  chemistry of `stoichiometric_hydration.jl`; this script runs it (`main()`, at
#  the end). The results are written to
#  `docs/src/assets/applications/aging_basic_creep/results.json`, which the
#  Applications chapter reads: the documentation build does not rerun this.
#
#  ENVIRONMENT: as script 44, the `docs/` project (ChemistryLab, OrdinaryDiffEq).
#      julia --project=docs -e 'using Pkg; Pkg.update()'
#  (`update` rather than `instantiate`: the docs manifest is not versioned, and
#  `instantiate` refuses a local one older than `docs/Project.toml`.)
#      julia scripts/47_aging_basic_creep_concrete.jl
#  From the REPL, `include("scripts/47_aging_basic_creep_concrete.jl")` runs it too.
#  About five minutes: four aging scales on a grid of some fifty instants, the
#  self-consistent foams dominating.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using JSON
using Printf

include(joinpath(@__DIR__, "common", "paste_micromechanics.jl"))
include(joinpath(@__DIR__, "common", "stoichiometric_hydration.jl"))
include(joinpath(@__DIR__, "common", "aging_creep_micromechanics.jl"))

const LITERATURE = joinpath(@__DIR__, "..", "data", "literature")
const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets", "applications", "aging_basic_creep")

const PAPER = JSON.parsefile(joinpath(LITERATURE, "lavergne2020.json"))["quantities"]
const TESTS = JSON.parsefile(joinpath(LITERATURE, "leroy1995.json"))["tables"]["basic_creep"]
q(key) = Float64(PAPER[key]["value"])

# ── §1  The Saint-Vigor cement (Table 4 of the paper) ───────────────────────
#
# The clinker phases are fractions of the clinker; the sulfate, given as CaSO4,
# enters as gypsum of the same calcium, through the molar masses of the library;
# the calcite as filler. The few percent the table leaves out (free lime, MgO,
# alkalis) are not modeled: the binder is renormalized over what is.

molar_mass(formula) = ustrip(us"g/mol", Species(formula).M)
const CLINKER_PCT = (C3S = q("saint_vigor_C3S"), C2S = q("saint_vigor_C2S"), C3A = q("saint_vigor_C3A"), C4AF = q("saint_vigor_C4AF"))
const GYPSUM_PCT = q("saint_vigor_CaSO4") * molar_mass("CaSO4(H2O)2") / molar_mass("CaSO4")
const CALCITE_PCT = q("saint_vigor_CaCO3")
const CEMENT_PCT = sum(CLINKER_PCT) + GYPSUM_PCT + CALCITE_PCT
const CLINKER = map(x -> x / sum(CLINKER_PCT), CLINKER_PCT)

# The two concretes, with the effective w/c of the paper (Fig. 11 g and h). The
# silica fume is a fraction of the cement; the binder is cement plus silica fume.
# The aggregate density is not reported: 2.6 t/m³, an ordinary value, is assumed.
const AGGREGATE_DENSITY = 2600.0     # kg/m³, an assumption of this script
concrete(name, prefix, files) = (
    name = name, files = files,
    w_c = q(prefix * "_w_c"), a_c = q(prefix * "_a_c"), cement = q(prefix * "_cement"),
    silica = haskey(PAPER, prefix * "_silica_fume") ? q(prefix * "_silica_fume") / 100 : 0.0,
)
const CONCRETES = [
    concrete("ordinary, w/c = 0.463", "leroy_ordinary", ["c_123_0$i" for i in 1:5]),
    concrete("high-performance, w/c = 0.298, 10 % silica fume", "leroy_hpc", ["c_123_0$i" for i in 6:9] ∪ ["c_123_10"]),
]

function hydrate(c, tend)
    binder = 1 + c.silica                                   # per kg of cement
    return run_hydration(;
        wb = c.w_c / binder, clinker = CLINKER,
        gypsum = GYPSUM_PCT / CEMENT_PCT / binder, filler = CALCITE_PCT / CEMENT_PCT / binder,
        silica = c.silica / binder, blaine = q("saint_vigor_blaine") * u"m^2/kg", tend = tend * 86400.0,
    )
end
aggregate_fraction(c) = c.a_c * c.cement / AGGREGATE_DENSITY

# The volume fractions at each instant of the grid; an instant given twice (a loading age)
# has one state of hydration, computed once.
function fractions_on_grid(run, times)
    instants = unique(times)
    _, fs = fraction_history(run, instants .* 86400)
    return [fs[searchsortedfirst(instants, t)] for t in times]
end

# ── §2  The model, one concrete at a time ───────────────────────────────────
#
# The grid holds every loading age twice: the stress jumps there. The compliance of the
# test loaded at t′ is the creep kernel J(t, t′) of the concrete, its relaxation operator
# inverted, read at any instant: on the grid for the curves, at the measured instants for
# the comparison.

function model_concrete(c)
    mine = filter(t -> t["nu_file"] in c.files, TESTS)
    ages = sort(unique(Float64[t["loading_age_days"] for t in mine]))
    t_end = maximum(t["loading_age_days"] + maximum(first.(t["creep_compliance"]["rows"])) for t in mine)
    times = loading_grid(0.3, 1.2 * t_end, ages; per_decade = 6)
    t0 = time()
    run = hydrate(c, 1.3 * t_end)
    R = concrete_relaxation(fractions_on_grid(run, times), times; f_aggregate = aggregate_fraction(c))
    J = uniaxial_compliance(R)
    @printf "%-48s %3d instants, %.0f s\n" c.name length(times) time() - t0
    curves = map(ages) do a
        t = times[findlast(==(a), times):end]                 # from the second node of a on
        (; age = a, t = t, J = [1000 * J(tᵢ, a) for tᵢ in t])  # 1e-6/MPa
    end
    for test in mine
        a = Float64(test["loading_age_days"])
        rows = test["creep_compliance"]["rows"]
        dt, measured = rows[end][1], rows[end][2]
        @printf "  t' = %5.2f d   J(t', t') %5.1f model %5.1f   last point (dt %5.0f d) %5.1f model %5.1f\n" a rows[1][2] 1000 * J(a, a) dt measured 1000 * J(a + dt, a)
    end
    return Dict("name" => c.name, "files" => c.files, "curves" => curves)
end

function main()
    results = map(model_concrete, CONCRETES)
    mkpath(OUT)
    open(joinpath(OUT, "results.json"), "w") do io
        JSON.print(io, Dict("source" => "scripts/47_aging_basic_creep_concrete.jl", "unit" => "1e-6/MPa", "concretes" => results), 1)
    end
    println("wrote ", relpath(joinpath(OUT, "results.json"), joinpath(@__DIR__, "..")))
    return results
end

main()
