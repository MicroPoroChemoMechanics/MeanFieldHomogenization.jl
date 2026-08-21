# =============================================================================
#  ionic_hydration.jl — hydration by dissolution into IONS and precipitation
#  driven by thermodynamics, as the counterpart to `stoichiometric_hydration.jl`.
#
#  Shared by `scripts/45_ionic_hydration_micromechanics.jl` and the Applications
#  chapter `docs/src/applications/ionic_hydrating_paste.md`.
#
#  The difference with `stoichiometric_hydration.jl` is where the products come from.
#  There, aggregated solid → solid reactions state the products in advance, and a
#  hand-written priority cascade decides which of them forms when. Here the
#  clinker only dissolves — into Ca²⁺, SiO₂, AlO₂⁻, FeO₂⁻, SO₄²⁻, CO₃²⁻, H⁺ —
#  and a Gibbs minimization decides, at every accepted step, which hydrates are
#  stable and in what amounts. No sequencing rule is written anywhere.
#
#  The micromechanics (`paste_micromechanics.jl`) is reused unchanged: only the
#  chemistry differs.
#
#  Requires ChemistryLab ≥ 0.9, whose `speciated_states` certifies every instant
#  against the KKT conditions, and an equilibrium back-end, here OptimaSolver.
# =============================================================================

using ChemistryLab
using DynamicQuantities
using OrdinaryDiffEq
using OptimaSolver
using OrderedCollections

const IONIC_CEMDATA = joinpath(pkgdir(ChemistryLab), "data", "cemdata18-thermofun.json")

# Four systems. `:opc` is the model; the three smaller ones are the ladder that
# was used to isolate what made it hard, and they are kept because reproducing an
# intermediate is the fastest way to localize a regression.
#
# All four run with ChemistryLab >= 0.9. `:opc` over 28 days: 202 accepted steps,
# retcode Success, pore solution at pH 12.52-12.58, every replayed instant
# CERTIFIED optimal with an element balance between 1e-11 and 1e-13 mol.
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
second model, and what `stoichiometric_hydration.jl` has to encode by hand.

This needs ChemistryLab ≥ 0.9, which certifies each replayed speciation. Before
0.7.1 a re-speciation could start outside the feasible set of its own equality
constraint, and the full OPC returned an assemblage demanding 174 % of the
sulfate present.

!!! note "The reported compositions are certified"
    From ChemistryLab 0.9, `speciated_states` passes each instant to
    `DualEquilibriumSolver`, which solves the KKT system and returns a proof: the
    Gibbs problem is convex, so stationarity of the interior species, the
    component balance, and undersaturation of every absent phase are together
    sufficient for **global** optimality. On this OPC every replayed instant is
    certified, with an element balance between 1e-11 and 1e-13 mol.

    That matters because the interior-point solver alone is not reliable here. On
    the package's own calcite reference it returns pH 6.96 where the certified
    answer is 9.90, and it misses a trace species by 147 %; it also rarely reports
    convergence at all, so its return code cannot be used to tell the two cases
    apart.
"""
const IONIC_DEFAULT_SYSTEM = :opc

const IONIC_ANHYDROUS = IONIC_SYSTEMS[IONIC_DEFAULT_SYSTEM].anhydrous
const IONIC_HYDRATES = IONIC_SYSTEMS[IONIC_DEFAULT_SYSTEM].hydrates

# Same phase families as the stoichiometric model, so the two are comparable
# term by term through `paste_micromechanics.jl`.
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
    ionic_reactions(cs; wb, blaine, calibration, induction, induction_phases)
        -> Vector{KineticReaction}

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
function ionic_reactions(
        cs; wb, blaine, system::Symbol = IONIC_DEFAULT_SYSTEM,
        calibration = IONIC_CALIBRATION,
        induction = ionic_induction(),
        induction_phases = IONIC_INDUCTION_PHASES,
    )
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
        β = (induction !== nothing && a in induction_phases) ? induction : (_t -> 1.0)
        rate = haskey(pk_of, a) ?
            KineticFunc(
                (T, P, t, n, lna, n0) -> f * β(t) * base(T, P, t, n, lna, n0),
                (T = 293.15u"K", P = 1.0e5u"Pa"), u"mol/s",
            ) : base

        rxn[:rate] = rate
        push!(out, KineticReaction(cs, rxn))
    end
    return out
end

"""
    IONIC_INDUCTION_TAU, IONIC_INDUCTION_M, IONIC_INDUCTION_PHASES

The dormant period of the clinker silicates, as a factor
`β(t) = 1 - exp(-(t/τ)^m)` on their dissolution rate. **On by default.**

Parrot–Killoh has no induction period: `parrot_killoh_avrami` floors its Avrami
argument at `PK_AVRAMI_SEED` so the ODE can leave `ξ = 0` at all, which means the
clinker starts hydrating the instant the water does. Measured against a real CEM I
isothermal-calorimetry record in ChemistryLab's `scripts/hydration_calibration.jl`,
that released 23.8 J/g by 2.7 h where the calorimeter saw 4.7, and the model then
ran some 60 J/g behind from one day onwards.

# Why this chapter cares, and it is not only about heat

Volume fractions drive the micromechanics here, and a model that hydrates from
`t = 0` crosses the percolation threshold too early. `paste_moduli`
reports setting as a genuine zero of the self-consistent fixed point, so the
**setting time** this chapter prints moves when the dormant period is added. It is
a reported result, not an input.

`τ = 5 h` and `m = 2.5` are round on purpose: three independent routes in that
calibration put the dormancy at 5 to 5.6 h, while its own identifiability analysis
shows `τ` correlated with the alite rate constant at 0.994 — so the timescale is
well determined and its precision is not. Applied to the **silicates only**: the
aluminate reacts within minutes of wetting.
"""
const IONIC_INDUCTION_TAU = 5.0 * 3600.0

@doc (@doc IONIC_INDUCTION_TAU)
const IONIC_INDUCTION_M = 2.5

@doc (@doc IONIC_INDUCTION_TAU)
const IONIC_INDUCTION_PHASES = ("C3S", "C2S")

"""
    ionic_induction(τ = IONIC_INDUCTION_TAU, m = IONIC_INDUCTION_M) -> Function

The default dormant-period factor. Pass `induction = nothing` to
[`run_ionic_hydration`](@ref) for the published Parrot–Killoh behavior.
"""
ionic_induction(τ = IONIC_INDUCTION_TAU, m = IONIC_INDUCTION_M) =
    t -> -expm1(-(max(t, zero(t)) / τ)^m)

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
    run_ionic_hydration(; wb, clinker, gypsum, filler, blaine, tend,
                        binder_mass, calorimeter) -> NamedTuple

Integrate the coupled problem: dissolution kinetics on the six anhydrous phases,
Gibbs minimization on everything else, once per accepted step.

`binder_mass` is the mass of binder simulated (1 kg by default, so every extensive
result is per kilogram of binder). `calorimeter`, when given, couples the thermal
balance into the same ODE: `IsothermalCalorimeter` integrates the heat at fixed
temperature, `SemiAdiabaticCalorimeter` lets the temperature follow the balance
and feeds it back into the Arrhenius factor of every dissolution law.

Returns `(; cs, state0, kp, sol, calorimeter)`. The activity model is
`HKFActivityModel` on both halves — a cement pore solution sits at
I ≈ 0.1–0.7 mol/kg, where the dilute model every shipped example uses is not
defensible.
"""
function run_ionic_hydration(;
        wb = 0.5,
        clinker = (C3S = 0.65, C2S = 0.11, C3A = 0.11, C4AF = 0.08),
        gypsum = 0.046, filler = 0.035,
        blaine = 380.0u"m^2/kg", tend = 90 * 86400.0,
        reltol = 1.0e-7, abstol = 1.0e-10,
        system::Symbol = IONIC_DEFAULT_SYSTEM,
        binder_mass = 1.0u"kg",
        calorimeter = nothing,
        induction = ionic_induction(),
        induction_phases = IONIC_INDUCTION_PHASES,
    )
    cs = build_ionic_system(system)
    nmv = symbol.(cs.species)
    f_clinker = 1.0 - gypsum - filler

    mb = ustrip(us"kg", binder_mass)
    T0 = calorimeter === nothing ? 293.15u"K" : _calorimeter_T0(calorimeter)

    state0 = ChemicalState(cs; T = T0)
    for (nm, w) in pairs(clinker)
        string(nm) in nmv && set_quantity!(state0, string(nm), (mb * f_clinker * w)u"kg")
    end
    gypsum > 0 && "Gp" in nmv && set_quantity!(state0, "Gp", (mb * gypsum)u"kg")
    filler > 0 && "Cal" in nmv && set_quantity!(state0, "Cal", (mb * filler)u"kg")
    set_quantity!(state0, "H2O@", (mb * wb)u"kg")

    model = HKFActivityModel()
    kp = if calorimeter === nothing
        KineticsProblem(
            cs, ionic_reactions(cs; wb, blaine, system, induction, induction_phases),
            state0, (0.0, tend);
            activity_model = model,
            equilibrium_solver = EquilibriumSolver(cs, model, OptimaOptimizer()),
        )
    else
        KineticsProblem(
            cs, ionic_reactions(cs; wb, blaine, system, induction, induction_phases),
            state0, (0.0, tend);
            activity_model = model,
            equilibrium_solver = EquilibriumSolver(cs, model, OptimaOptimizer()),
            calorimeter = calorimeter,
        )
    end
    ks = KineticsSolver(; ode_solver = Rodas5P(), reltol = reltol, abstol = abstol)
    return (; cs, state0, kp, system, calorimeter, sol = integrate(kp, ks))
end

_calorimeter_T0(cal::IsothermalCalorimeter) = cal.T
_calorimeter_T0(cal::SemiAdiabaticCalorimeter) = cal.T0

# ── calorimetry, after Lavergne et al. (2018) §4.1 ───────────────────────────

"""
    CALORIMETRY_MIX_C100

Mix proportions of the plain-cement semi-adiabatic test of Lavergne et al.
(2018), Table 11, at w/b = 0.5: 371 g of binder, 1113 g of dry sand, 196 g of
water. The sand is there to keep the temperature rise moderate, as NF EN 196-9
prescribes; it takes no part in the chemistry and enters only through its heat
capacity.
"""
const CALORIMETRY_MIX_C100 = (binder = 0.371u"kg", sand = 1.113u"kg", water = 0.196u"kg")

"""
    CALORIMETRY_LOSS_A, CALORIMETRY_LOSS_B

Calibration of the calorimeter's heat loss, Lavergne et al. (2018) Eq. (23):
`φ(ΔT) = a ΔT + b ΔT²`, with `a = 75 J/(h·K)` and `b = 0.260 J/(h·K²)` from the
NF EN 196-9 calibration. Converted here to watts.
"""
const CALORIMETRY_LOSS_A = 75.0 / 3600            # W/K
const CALORIMETRY_LOSS_B = 0.26 / 3600           # W/K²

"""
    CALORIMETRY_VESSEL_CP

Heat capacity of the calorimeter vessel, **380 J/K**.

The paper prints "about 380 kJ/K", and that cannot be the figure its own results
correspond to. Its Table 11 mix holds 371 g of binder, which releases of order
350 J/g, so about 130 kJ in total; against 380 kJ/K the temperature would rise by
0.3 K, where the test reports tens of kelvin. The rest of the setup is consistent
with joules: sand and water alone contribute roughly 1.7 kJ/K, so a vessel of
380 J/K puts the total near 2 kJ/K and the adiabatic rise near 60 K, which is the
order the measurements show. It is read as 380 J/K here, and this note is
deliberate: the alternative is to change a published number in silence.
"""
const CALORIMETRY_VESSEL_CP = 380.0               # J/K

const _SAND_CP_PER_KG = Ref{Float64}(NaN)

"""
    sand_heat_capacity(mass) -> Float64

Heat capacity [J/K] of `mass` of quartz sand, from the `Qtz` entry of CEMDATA18
rather than from a remembered figure.
"""
function sand_heat_capacity(mass)
    if isnan(_SAND_CP_PER_KG[])
        q = first(s for s in build_species(IONIC_CEMDATA) if symbol(s) == "Qtz")
        cp = ustrip(us"J/K/mol", q[:Cp⁰](T = 293.15, P = 1.0e5, unit = true))
        _SAND_CP_PER_KG[] = cp / ustrip(us"kg/mol", q[:M])
    end
    return _SAND_CP_PER_KG[] * ustrip(us"kg", mass)
end

"""
    semiadiabatic_cell(; mix = CALORIMETRY_MIX_C100, T0 = 293.15u"K", T_env = T0)

The NF EN 196-9 device of Lavergne et al. (2018), as a `SemiAdiabaticCalorimeter`.

`Cp` here is what the ODE does NOT compute for itself: the vessel and the inert
sand. The paste's own heat capacity is `Σᵢ nᵢ Cp⁰ᵢ(T)`, which `ChemistryLab` adds
at every step from the database, so it must not be counted twice.
"""
function semiadiabatic_cell(;
        mix = CALORIMETRY_MIX_C100, T0 = 293.15u"K", T_env = T0,
    )
    Cp_fixed = CALORIMETRY_VESSEL_CP + sand_heat_capacity(mix.sand)
    return SemiAdiabaticCalorimeter(;
        Cp = Cp_fixed * u"J/K",
        heat_loss = ΔT -> CALORIMETRY_LOSS_A * ΔT + CALORIMETRY_LOSS_B * ΔT^2,
        T_env = T_env,
        T0 = T0,
    )
end

"""
    speciated_states(run, times) -> Vector{ChemicalState}

The **speciated** compositions at `times`, delegated to ChemistryLab.

This used to be reimplemented here against three private internals of that
package. It is now [`ChemistryLab.speciated_states`](@ref), public since 0.8.0,
which is where it belongs: the guards it needs — a warm start capped at the
element budget and projected back into `{Aₑn = bₑ, n ≥ 0}`, and a back-end
instance that has not been used by the integration — are properties of the
coupling, not of this chapter.
"""
speciated_states(run, times) = ChemistryLab.speciated_states(run.sol, run.kp; times = times)

"""
    ionic_fraction_history(run, times; gel_water = GEL_WATER_PER_CSH)
        -> (times, Vector{Dict{String,Float64}}, Vector{Float64}, Vector{NamedTuple})

Volume fractions of the phase families of `IONIC_GROUPS` at each instant, plus
the pore-solution pH — which the stoichiometric model cannot produce at all.

The sealed-curing convention and the gel-water transfer are those of
`stoichiometric_hydration.jl`, so the two models are directly comparable.
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
