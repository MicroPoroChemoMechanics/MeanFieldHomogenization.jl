# =============================================================================
#  ionic_hydration.jl — hydration by dissolution into IONS and precipitation
#  driven by thermodynamics, as the counterpart to `lavergne_hydration.jl`.
#
#  Shared by `scripts/45_ionic_hydration_micromechanics.jl` and the Applications
#  chapter `docs/src/applications/ionic_hydrating_paste.md`.
#
#  The difference with `lavergne_hydration.jl` is where the products come from.
#  There, aggregated solid → solid reactions state the products in advance, and a
#  hand-written priority cascade decides which of them forms when. Here the
#  clinker only dissolves — into Ca²⁺, SiO₂, AlO₂⁻, FeO₂⁻, SO₄²⁻, CO₃²⁻, H⁺ —
#  and a Gibbs minimization decides, at every accepted step, which hydrates are
#  stable and in what amounts. No sequencing rule is written anywhere.
#
#  The micromechanics (`lavergne_model.jl`) is reused unchanged: only the
#  chemistry differs.
#
#  Requires ChemistryLab ≥ 0.7.1 (before it, a re-speciation could start outside
#  the feasible set of its own equality constraint and the full OPC returned an
#  assemblage demanding 174 % of the sulfate present) and an equilibrium
#  back-end, here OptimaSolver.
# =============================================================================

using ChemistryLab
using DynamicQuantities
using OrdinaryDiffEq
using OptimaSolver
using OrderedCollections
using SciMLBase

const IONIC_CEMDATA = joinpath(pkgdir(ChemistryLab), "data", "cemdata18-thermofun.json")

# Four systems. `:opc` is the model; the three smaller ones are the ladder that
# was used to isolate what made it hard, and they are kept because reproducing an
# intermediate is the fastest way to localize a regression.
#
# All four run with ChemistryLab >= 0.7.1. `:opc` over 28 days: 202 accepted
# steps, retcode Success, pore solution at pH 12.58, sulfate and aluminum
# budgets closing on the last digit.
const IONIC_SYSTEMS = Dict(
    :silicates => (
        anhydrous = ["C3S", "C2S"],
        hydrates = ["Portlandite", "Jennite"],
    ),
    # Intermediate systems, used to isolate what makes the full OPC hard.
    :aluminate => (
        anhydrous = ["C3S", "C2S", "C3A"],
        hydrates = ["Portlandite", "Jennite", "C3AH6"],
    ),
    :sulfate => (
        anhydrous = ["C3S", "C2S", "C3A", "Gp"],
        hydrates = ["Portlandite", "Jennite", "ettringite", "monosulphate12", "C3AH6"],
    ),
    :opc => (
        anhydrous = ["C3S", "C2S", "C3A", "C4AF", "Gp", "Cal"],
        hydrates = [
            "Portlandite", "Jennite", "ettringite", "monosulphate12",
            "monocarbonate", "C3AH6", "FeOOHmic",
        ],
    ),
)

"""
    IONIC_DEFAULT_SYSTEM

`:opc` — the full ordinary Portland cement: alite, belite, aluminate, ferrite,
gypsum and calcite dissolving to ions, with the assemblage left to the
thermodynamics.

Measured over 28 days: 202 accepted steps, `retcode = Success`, pore solution
holding at pH 12.58. The aluminate sequence comes out of the free-energy
minimization with no sequencing rule written anywhere — ettringite forms early,
peaks around 6 hours and converts to monosulphate once the sulfate is spent, the
AFm settling at exactly the sulfate budget. That is the whole point of this
second model, and what `lavergne_hydration.jl` has to encode by hand.

This needs ChemistryLab ≥ 0.7.1. Before it, a re-speciation could start outside
the feasible set of its own equality constraint, and the full OPC returned an
assemblage demanding 174 % of the sulfate present while reporting a residual of
1.4e-2 — the residual being normalized by the 34 mol water budget, which hid a
0.465 mol violation on the 0.27 mol of sulfate.

!!! danger "Judge these runs on the pH, not on the retcode"
    `EXACT_HESSIAN` looks like it helps and is wrong: it tightens the reported
    element balance while collapsing the pore solution from pH 12.06 to 5.54. A
    pure phase has `∂μ/∂n = 0` exactly, so flooring the Hessian diagonal does
    move the minerals, but to a point of *higher* Gibbs energy with katoite
    replacing the AFm. The pH and the element budgets are the physical tells.
"""
const IONIC_DEFAULT_SYSTEM = :opc

const IONIC_ANHYDROUS = IONIC_SYSTEMS[IONIC_DEFAULT_SYSTEM].anhydrous
const IONIC_HYDRATES = IONIC_SYSTEMS[IONIC_DEFAULT_SYSTEM].hydrates

# Same phase families as the stoichiometric model, so the two are comparable
# term by term through `lavergne_model.jl`.
const IONIC_GROUPS = [
    "anhydrous" => ["C3S", "C2S", "C3A", "C4AF"],
    "gypsum" => "Gp",
    "calcite" => "Cal",
    "C-S-H" => "Jennite",
    "CH" => "Portlandite",
    "AFt" => "ettringite",
    "AFm" => ["monosulphate12", "monocarbonate"],
    "hydrogarnet" => "C3AH6",
    "FH3" => "FeOOHmic",
    # "water" is filled in per system by `ionic_water_group`: it must collect
    # EVERY aqueous species, not just H2O@. The RVE's water phase is the pore
    # solution, and at cement ionic strength the solutes are not negligible —
    # their negative partial molar volumes sum to -3e-4 of the paste volume,
    # which left as an unassigned remainder trips the RVE's own sanity check.
]

"""
    ionic_water_group(cs) -> Pair

The `"water"` phase of the RVE: every aqueous species of `cs`, i.e. the whole
pore solution rather than the solvent alone.
"""
ionic_water_group(cs) =
    "water" => [symbol(s) for s in cs.species if aggregate_state(s) == AS_AQUEOUS]

"""
    build_ionic_system() -> ChemicalSystem

The CEMDATA18 subset for the ionic model: the six dissolving phases, the
candidate hydrates, and every aqueous species `speciation` pulls in from
`CEMDATA_PRIMARIES` — 49 of them, which is what carries the pore chemistry.
"""
function build_ionic_system(system::Symbol = IONIC_DEFAULT_SYSTEM)
    haskey(IONIC_SYSTEMS, system) ||
        throw(ArgumentError("system must be :silicates or :opc, got :$system"))
    spec = IONIC_SYSTEMS[system]
    subs = build_species(IONIC_CEMDATA)
    sp = speciation(subs, vcat(spec.anhydrous, spec.hydrates); aggregate_state = [AS_AQUEOUS])
    return ChemicalSystem(sp, CEMDATA_PRIMARIES)
end

"""
    ionic_reactions(cs; wb, blaine, calibration = IONIC_CALIBRATION) -> Vector{KineticReaction}

Congruent dissolution of each anhydrous phase into the primary aqueous species,
with a Parrot–Killoh rate scaled by a per-phase calibration factor.

The reactions are **balanced by ChemistryLab** from the phase and the primaries,
and come out in the expected acid-driven form, e.g.

    C₃S = 3 H₂O + 3 Ca²⁺ + SiO₂ − 6 H⁺

Rates reuse [`parrot_killoh_avrami`](@ref) rather than a transition-state law:
no Palandri–Kharaka parameter set is shipped for clinker phases, and inventing
one would be a fabrication. What the calibration factors buy is that the degrees
of hydration follow those of the stoichiometric model, so the two chapters differ
only in *what forms*, not in *how fast* — which is the comparison worth making.
"""
function ionic_reactions(cs; wb, blaine, system::Symbol = IONIC_DEFAULT_SYSTEM, calibration = IONIC_CALIBRATION)
    nmv = symbol.(cs.species)
    prim = [p for p in ("Ca+2", "SiO2@", "AlO2-", "FeO2-", "SO4-2", "CO3-2", "H2O@", "H+") if p in nmv]
    α_max = powers_alpha_max(wb)

    pk_of = Dict(
        "C3S" => PK84_PARAMS_C3S, "C2S" => PK84_PARAMS_C2S,
        "C3A" => PK84_PARAMS_C3A, "C4AF" => PK84_PARAMS_C4AF,
    )

    out = KineticReaction[]
    for a in IONIC_SYSTEMS[system].anhydrous
        rxn = Reaction([cs[a]], [cs[p] for p in prim]; symbol = "$a dissolution")

        base = if haskey(pk_of, a)
            parrot_killoh_avrami(pk_of[a], a; α_max, blaine)
        else
            # Gypsum and calcite are not clinker: give them a fast first-order
            # release so sulfate and carbonate are available to the minimization
            # from the start, rather than rate-limiting it.
            k = get(calibration, a, 1.0) * 1.0e-4
            KineticFunc(
                (T, P, t, n, lna, n0) -> k * max(n[a], zero(eltype(n.data))),
                (T = 293.15u"K", P = 1.0e5u"Pa"), u"mol/s",
            )
        end

        f = get(calibration, a, 1.0)
        rate = haskey(pk_of, a) ?
            KineticFunc(
                (T, P, t, n, lna, n0) -> f * base(T, P, t, n, lna, n0),
                (T = 293.15u"K", P = 1.0e5u"Pa"), u"mol/s",
            ) : base

        rxn[:rate] = rate
        push!(out, KineticReaction(cs, rxn))
    end
    return out
end

"""
    IONIC_CALIBRATION

Per-phase multipliers on the dissolution rates, adjusted so that the degrees of
hydration of the ionic model follow those of the stoichiometric one.

They are close to 1 by construction: the same Parrot–Killoh laws drive both, and
the factors only absorb the fact that here a phase dissolves congruently into
ions instead of being consumed by a specific reaction.
"""
const IONIC_CALIBRATION = Dict(
    "C3S" => 1.0, "C2S" => 1.0, "C3A" => 1.0, "C4AF" => 1.0,
    "Gp" => 1.0, "Cal" => 1.0,
)

"""
    run_ionic_hydration(; wb, clinker, gypsum, filler, blaine, tend) -> NamedTuple

Integrate the coupled problem for 1 kg of binder: dissolution kinetics on the six
anhydrous phases, Gibbs minimization on everything else, once per accepted step.

Returns `(; cs, state0, kp, sol)`. The activity model is `HKFActivityModel` on
both halves — a cement pore solution sits at I ≈ 0.1–0.7 mol/kg, where the dilute
model every shipped example uses is not defensible.
"""
function run_ionic_hydration(;
        wb = 0.5,
        clinker = (C3S = 0.65, C2S = 0.11, C3A = 0.11, C4AF = 0.08),
        gypsum = 0.046, filler = 0.035,
        blaine = 380.0u"m^2/kg", tend = 90 * 86400.0,
        reltol = 1.0e-7, abstol = 1.0e-10,
        system::Symbol = IONIC_DEFAULT_SYSTEM,
    )
    cs = build_ionic_system(system)
    nmv = symbol.(cs.species)
    f_clinker = 1.0 - gypsum - filler

    state0 = ChemicalState(cs)
    for (nm, w) in pairs(clinker)
        string(nm) in nmv && set_quantity!(state0, string(nm), (f_clinker * w)u"kg")
    end
    gypsum > 0 && "Gp" in nmv && set_quantity!(state0, "Gp", gypsum * u"kg")
    filler > 0 && "Cal" in nmv && set_quantity!(state0, "Cal", filler * u"kg")
    set_quantity!(state0, "H2O@", wb * u"kg")

    model = HKFActivityModel()
    kp = KineticsProblem(
        cs, ionic_reactions(cs; wb, blaine, system), state0, (0.0, tend);
        activity_model = model,
        equilibrium_solver = EquilibriumSolver(cs, model, OptimaOptimizer()),
    )
    ks = KineticsSolver(; ode_solver = Rodas5P(), reltol = reltol, abstol = abstol)
    return (; cs, state0, kp, system, sol = integrate(kp, ks))
end

"""
    speciated_states(run, times) -> Vector{ChemicalState}

The **speciated** compositions at `times`, computed sequentially with a warm
start.

[`state_at`](@ref) deliberately does not replay the re-speciation — the
redistribution performed by the equilibrium solve is not recoverable from the
stoichiometry — so it returns the purely kinetic reconstruction, which for this
model is meaningless: all the calcium in solution and not one hydrate. The
equilibrium partition is recovered here the way the solver computed it, by
solving `φ(bₑ)` on the sub-system with the element totals carried by the ODE
state.

The sequence matters. Solved from a cold guess, `φ(bₑ)` does not converge for this
system: the H⁺ component of `bₑ` reaches −14 mol, and an interior-point method
started from pure water lands nowhere near it — the first attempt at this function
reported no hydrates at all and a pore solution at pH 6, while the run itself had
computed 2.2 mol of C-S-H and 2.8 mol of portlandite. Walking the instants in
order and carrying the previous speciation as the guess fixes it, which is exactly
what `respeciate!` does inside the run.
"""
function speciated_states(run, times)
    kp, sol = run.kp, run.sol
    p = sol.prob.p
    sub = ChemistryLab._equilibrium_subsystem(kp.system, kp.idx_equilibrium)
    subnm = symbol.(sub.species)
    es = EquilibriumSolver(sub, kp.activity_model, OptimaOptimizer())

    issorted(times) || throw(ArgumentError("`times` must be ascending: the solves warm-start"))

    # Cold start for the first instant: the initial composition of the partition.
    guess = [max(p.n_eq_init[j], 1.0e-10) for j in eachindex(p.n_eq_init)]

    Ae = Float64.(sub.SM.A)

    out = ChemicalState[]
    for t in times
        u = sol(t)
        be = collect(u[1:(p.n_be)])
        # Carry the previous speciation onto this instant's element totals before
        # solving. The warm start is the equilibrium of the PREVIOUS `bₑ`, so once
        # an element has been spent — the sulfate of an OPC, after the gypsum is
        # gone — it demands more of it than now exists and the solve starts
        # outside its own feasible set. Without these two the full OPC returned an
        # assemblage claiming 174 % of the sulfate present; with them the AFm
        # settles at exactly the sulfate budget. Both are inert on a guess that is
        # already feasible, which is the ordinary case.
        ChemistryLab._budget_clip!(guess, Ae, be)
        ChemistryLab._restore_feasibility!(guess, Ae, be)
        eq = SciMLBase.solve(
            es, ChemicalState(sub, [g * u"mol" for g in guess]; T = p.T_q[], P = p.P_q[]);
            b = be
        )
        guess = [max(ustrip(us"mol", x), 1.0e-10) for x in eq.n]

        n = zeros(Float64, length(kp.system.species))
        for (j, idx) in enumerate(kp.idx_equilibrium)
            n[idx] = ustrip(us"mol", eq.n[j])
        end
        for (j, idx) in enumerate(kp.idx_kinetic)
            n[idx] = max(u[p.n_be + j], 0.0)
        end
        push!(
            out, ChemicalState(
                kp.system; T = p.T * u"K", P = p.P * u"Pa",
                n = [x * u"mol" for x in n]
            )
        )
    end
    return out
end

"""
    ionic_fraction_history(run, times; gel_water = GEL_WATER_PER_CSH)
        -> (times, Vector{Dict{String,Float64}}, Vector{Float64}, Vector{NamedTuple})

Volume fractions of the phase families of `IONIC_GROUPS` at each instant, plus
the pore-solution pH — which the stoichiometric model cannot produce at all.

The sealed-curing convention and the gel-water transfer are those of
`lavergne_hydration.jl`, so the two models are directly comparable.
"""
function ionic_fraction_history(run, times; gel_water = GEL_WATER_PER_CSH)
    V_ref = volume(run.state0).total
    fs = Vector{Dict{String, Float64}}(undef, length(times))
    phs = Vector{Float64}(undef, length(times))
    pors = Vector{NamedTuple}(undef, length(times))
    nmv = symbol.(run.cs.species)
    groups = vcat(
        [
            k => v for (k, v) in IONIC_GROUPS
                if any(x -> x in nmv, v isa AbstractString ? [v] : v)
        ],
        [ionic_water_group(run.cs)],
    )
    states = speciated_states(run, times)
    for (i, st) in enumerate(states)
        f = volume_fractions(st, groups; reference = run.state0)
        d = Dict{String, Float64}(k => v for (k, v) in f)

        if gel_water > 0
            n_csh = ustrip(us"mol", moles(st, "Jennite"))
            V_H2O = ustrip(run.cs["H2O@"][:V⁰](T = temperature(st), P = pressure(st); unit = true) / V_ref)
            f_gel = min(n_csh * gel_water * V_H2O, get(d, "water", 0.0))
            d["C-S-H"] = get(d, "C-S-H", 0.0) + f_gel
            d["water"] = get(d, "water", 0.0) - f_gel
        end
        fs[i] = d
        phs[i] = something(pH(st), NaN)
        # Sealed-curing porosity: referred to the fresh specimen volume, with the
        # chemical shrinkage counted as empty porosity. `porosity(st)` alone would
        # divide by the shrunken current volume and miss the void entirely —
        # understating it by some five points on a w/c = 0.5 paste at 28 days.
        pors[i] = porosity(st, run.state0)
    end
    return times, fs, phs, pors
end
