# =============================================================================
#  stoichiometric_hydration.jl — the ChemistryLab half of the Lavergne et al. (2018)
#  chain: binder composition → α(t) → moles → volume fractions.
#
#  Shared by `scripts/44_stoichiometric_hydration_micromechanics.jl` and by the
#  Applications chapter `docs/src/applications/hydrating_blended_paste.md`.
#
#  Requires ChemistryLab.jl and OrdinaryDiffEq, neither of which is a dependency
#  of MeanFieldHomogenization: they resolve from the stacked default environment
#  (as `Plots` already does for every script here) and are listed in
#  `docs/Project.toml` for the documentation build.
# =============================================================================

using ChemistryLab
using DynamicQuantities
using OrdinaryDiffEq
using OrderedCollections

const CEMDATA = joinpath(pkgdir(ChemistryLab), "data", "cemdata18-thermofun.json")

# Phase families the micromechanical model consumes. Keys must match
# `PASTE_HYDRATES`, `PASTE_INCLUSIONS` and `PASTE_PORES` of
# `paste_micromechanics.jl`; `volume_fractions` refuses to put a species in two.
# The AFm and hydrogarnet families each gather two phases. That is faithful
# rather than lazy: Table 5 of Lavergne et al. gives monosulfo- and
# monocarboaluminate the same (42.3 GPa, 0.324), and C₃AH₆ and the siliceous
# hydrogarnet the same (22.4 GPa, 0.25), so the micromechanics cannot tell them
# apart anyway.
const PASTE_PHASE_GROUPS = [
    "anhydrous" => ["C3S", "C2S", "C3A", "C4AF"],
    "gypsum" => "Gp",
    "calcite" => "Cal",
    "silica" => "Amor-Sl",
    "C-S-H" => "Jennite",
    "CH" => "Portlandite",
    "AFt" => "ettringite",
    "AFm" => ["monosulphate12", "monocarbonate"],
    "hydrogarnet" => ["C3AH6", "C3AFS0.84H4.32"],
    "FH3" => "FeOOHmic",
    "water" => "H2O@",
]

"""
    build_cement_system() -> ChemicalSystem

The CEMDATA18 subset used throughout: four clinker phases, gypsum, calcite
filler, amorphous silica standing for the pozzolanic addition, the hydrates they
produce, and water.
"""
function build_cement_system()
    substances = build_species(CEMDATA)
    syms = split(
        "C3S C2S C3A C4AF Gp Cal Amor-Sl " *
            "Jennite Portlandite ettringite monosulphate12 monocarbonate " *
            "C3AH6 C3AFS0.84H4.32 FeOOHmic " *
            "H2O@"
    )
    sp = speciation(substances, syms; aggregate_state = [AS_AQUEOUS])
    return ChemicalSystem(sp, CEMDATA_PRIMARIES)
end

"""
    STOICH_INDUCTION_PHASES, stoich_induction

The dormant period of the clinker silicates, `β(t) = 1 - exp(-(t/τ)^m)` with
`τ = 5 h`, `m = 2.5`, **on by default** — the same factor and the same values the
ionic route uses, so the two chapters remain comparable.

That comparability is the point. `IONIC_CALIBRATION` exists so the two routes
differ only in *what forms*, not in *how fast*; adding a dormant period to one and
not the other would break exactly that. See `common/ionic_hydration.jl` for why the
factor is needed and why the values are round.

`waller` is deliberately excluded: its sigmoid already vanishes as `t → 0`, so an
SCM entered through it carries its own onset delay.
"""
const STOICH_INDUCTION_PHASES = ("C3S", "C2S")

@doc (@doc STOICH_INDUCTION_PHASES)
stoich_induction(τ = 5.0 * 3600.0, m = 2.5) =
    t -> -expm1(-(max(t, zero(t)) / τ)^m)

"""
    hydration_reactions(cs; wb, blaine, humidity, c4af_priority, induction,
                        induction_phases) -> Vector{Reaction}

The reaction set, with a Parrot–Killoh rate on each clinker phase and a Waller
rate on the silica.

Every reaction is **balanced by ChemistryLab** from its reactants and products
rather than by hand: CEMDATA18 stores Jennite with a rounded Ca:Si of 1.666667,
so hand-written 4/3 and 103/30 coefficients leave a residual.

## The aluminate cascade

C₃A and C₄AF each drive **several** competing reactions, taken in a fixed order
of priority. The scheme this follows calls every reaction in turn, at each time
step, with the *remainder* of the phase's kinetic increment: each reaction takes
what its scarcest co-reactant allows and passes the rest on. Depletion is never
tested for explicitly; it is absorbed by that clipping.

The continuous analog used here is a **partition of unity** over the routes.
With `gᵢ = xᵢ/(xᵢ+ε)` a smooth availability gate on the co-reactant of route `i`,
expressed in moles of anhydrous phase that co-reactant can support,

    w₁ = g₁,   w₂ = (1-w₁)g₂,   w₃ = (1-w₁-w₂)g₃,   …,   w_last = 1 - Σ wᵢ

so `Σ wᵢ = 1` and the phase's total Parrot–Killoh rate is conserved exactly,
whichever routes are open. The last route needs no co-reactant and takes the
remainder, exactly as in the reference.

Priority orders, from `iter()` of the reference:

  - **C₃A** — gypsum → calcite → ettringite → water only.
  - **C₄AF** — C₃S → C₂S → gypsum → calcite → ettringite → water only.

### The C₄AF ordering is genuinely ambiguous in the sources

`c4af_priority` selects between two readings, because the sources do not agree:

  - `:silicate_first` (default) — what `iter()` of the reference actually does:
    `RfC4AF` (with C₃S, l. 1163) and `ReC4AF` (with C₂S, l. 1164) are called
    **before** `RaC4AF` (gypsum, l. 1167), each receiving only the remainder. The
    paper's §1.1.1 supports it: the Brouwers reactions apply "*until all sources of
    silica are depleted*", which is a priority rule.
  - `:sulfate_first` — the order in which Table 2 of the paper **lists** the
    reactions, and the order of the `d_reac_C4AF` dictionary itself (l. 225, where
    `RaC4AF` with gypsum is the first entry).

A listing order is not an execution order, and the dictionary is not iterated in
`iter()` — so `:silicate_first` is the defensible default. Figure 1 of the paper
cannot settle it either way: it groups the aluminate hydrates into a single
"AFt, AFm" band with no separate hydrogarnet. The keyword exists so the difference
can be measured rather than argued about.

Two consequences worth knowing. Monocarboaluminate comes **before**
monosulfoaluminate (Lothenbach 2008), and the *dominant* route for C₄AF is the
siliceous hydrogarnet consuming C₃S then C₂S — not the sulfate route. The
deduction of the C₃S/C₂S eaten there, which the reference performs by hand on the
silicate targets, is automatic here: C₃S is a kinetic species, so that
consumption enters its own degree of reaction and hence its own rate.

The gates read amounts of species that are not themselves kinetic (`Gp`, `Cal`,
`ettringite`). That is correct because the ODE state carries the reaction
extents, from which the residual reconstructs every species before evaluating the
rates (ChemistryLab ≥ 0.5).

### Why smooth gates and not callbacks

The `x/(x+ε)` form makes each consumer's rate proportional to `x` as the
co-reactant runs out, so the stock decays **exponentially towards zero and cannot
cross it**. That is a structural property of the gate, not luck: measured over a
90-day run refined around depletion, the most negative value reached by any
species is −1.7e-16 mol, i.e. rounding.

A `ContinuousCallback` on "gypsum exhausted" would therefore be rooted on a
quantity that never actually reaches zero, and would switch the rate
discontinuously — which is what a Rosenbrock method handles worst, since it needs
both a Jacobian and a time gradient of the residual. Nothing here needs an event.

### The one thing the smooth form does not encode

The reference consumes gypsum from a **shared, sequentially mutated** dictionary,
so C₃A has absolute priority over C₄AF. Here both aluminates read the same
`n["Gp"]` and draw simultaneously. That turns out not to matter: C₃A's
Parrot–Killoh rate is far faster early on (k₁ = 1.0 against 0.37 d⁻¹) and C₄AF's
two silicate routes are open ahead of its sulfate route, so C₃A takes essentially
all the sulfate anyway — 3 × 0.0891 = 0.267 mol against 0.2672 mol available, with
the C₄AF sulfate route left at zero. The priority *emerges from the kinetics*
rather than being imposed. Worth re-checking if the clinker composition is changed
to something where C₄AF outpaces C₃A.
"""
function hydration_reactions(
        cs; wb, blaine, humidity = nothing, c4af_priority::Symbol = :silicate_first,
        induction = stoich_induction(), induction_phases = STOICH_INDUCTION_PHASES,
    )
    s(n) = cs[n]
    α_max = powers_alpha_max(wb)
    wrap(f) = KineticFunc(f, (T = 293.15u"K", P = 1.0e5u"Pa"), u"mol/s")

    # The dormant period, on the silicates only. `waller` below is left untouched:
    # its sigmoid α(t) = 1/(1 + (τ/t)^n) already vanishes as t → 0, so the silica
    # fume carries its own onset delay in its own τ and damping it again would count
    # the delay twice.
    β(n) = (induction !== nothing && n in induction_phases) ? induction : (_t -> 1.0)
    pk(p, n) = begin
        base = parrot_killoh_avrami(p, n; α_max, blaine, humidity)
        g = β(n)
        induction === nothing || !(n in induction_phases) ? base :
            wrap((T, P, t, nn, lna, n0) -> g(t) * base(T, P, t, nn, lna, n0))
    end

    # Mole scale of the availability gates, not a tolerance: it sets how sharply
    # a route closes as its co-reactant runs out.
    ε_av = 1.0e-3

    # Availability of `name` expressed in moles of anhydrous phase it can supply,
    # given that one mole of that phase consumes `need` moles of it.
    gate(n, name, need) = begin
        x = max(n[name], zero(eltype(n.data))) / need
        x / (x + ε_av)
    end

    # Monocarboaluminate is disabled above 48 °C (Lothenbach 2008). Smooth over
    # ~1 K so the residual stays differentiable; inert at 20 °C.
    θ48(T) = one(T) / (one(T) + exp((T - oftype(T, 321.15)) / oftype(T, 1.0)))

    # Build the rate of route `k` of a cascade: base rate × its weight in the
    # partition of unity. `gates` returns the availability gate of each route
    # except the last, which takes the remainder.
    function cascade_rate(base, gates, k, nroutes)
        return wrap(
            (T, P, t, n, lna, n0) -> begin
                r = base(T, P, t, n, lna, n0)
                rest = one(r)
                w = zero(r)
                for i in 1:(nroutes - 1)
                    gᵢ = gates(i, T, n)
                    w = rest * gᵢ
                    i == k && return r * w
                    rest = rest - w
                end
                return r * rest          # last route: the remainder
            end
        )
    end

    # ── Silicates ────────────────────────────────────────────────────────────
    r_c3s = Reaction([s("C3S"), s("H2O@")], [s("Jennite"), s("Portlandite")]; symbol = "C3S hydration")
    r_c3s[:rate] = pk(PK84_PARAMS_C3S, "C3S")

    r_c2s = Reaction([s("C2S"), s("H2O@")], [s("Jennite"), s("Portlandite")]; symbol = "C2S hydration")
    r_c2s[:rate] = pk(PK84_PARAMS_C2S, "C2S")

    # ── C₃A: gypsum → calcite → ettringite → water only ─────────────────────
    pk_c3a = pk(PK84_PARAMS_C3A, "C3A")
    c3a_gates(i, T, n) =
        i == 1 ? gate(n, "Gp", 3.0) :
        i == 2 ? gate(n, "Cal", 1.0) * θ48(T) :
        gate(n, "ettringite", 0.5)

    c3a_routes = [
        ("C3A + gypsum -> AFt", [s("C3A"), s("Gp"), s("H2O@")], [s("ettringite")]),
        ("C3A + calcite -> Mc", [s("C3A"), s("Cal"), s("H2O@")], [s("monocarbonate")]),
        ("C3A + AFt -> AFm", [s("C3A"), s("ettringite"), s("H2O@")], [s("monosulphate12")]),
        ("C3A -> hydrogarnet", [s("C3A"), s("H2O@")], [s("C3AH6")]),
    ]
    r_c3a = map(enumerate(c3a_routes)) do (k, (name, reac, prod))
        r = Reaction(reac, prod; symbol = name)
        r[:rate] = cascade_rate(pk_c3a, c3a_gates, k, length(c3a_routes))
        r
    end

    # ── C₄AF: C₃S → C₂S → gypsum → calcite → ettringite → water only ────────
    pk_c4af = pk(PK84_PARAMS_C4AF, "C4AF")
    hg = s("C3AFS0.84H4.32")

    # (gate specification, reaction) pairs, listed in the order of priority.
    r_si3 = (
        "C4AF + C3S -> Fe hydrogarnet", ("C3S", 1.68),
        [s("C4AF"), s("C3S"), s("H2O@")], [hg, s("Portlandite")],
    )
    r_si2 = (
        "C4AF + C2S -> Fe hydrogarnet", ("C2S", 1.68),
        [s("C4AF"), s("C2S"), s("H2O@")], [hg, s("Portlandite")],
    )
    r_gp = (
        "C4AF + gypsum -> AFt", ("Gp", 3.0),
        [s("C4AF"), s("Gp"), s("H2O@")],
        [s("ettringite"), s("Portlandite"), s("FeOOHmic")],
    )
    r_cal = (
        "C4AF + calcite -> Mc", ("Cal", 1.0),
        [s("C4AF"), s("Cal"), s("H2O@")],
        [s("monocarbonate"), s("Portlandite"), s("FeOOHmic")],
    )
    r_aft = (
        "C4AF + AFt -> AFm", ("ettringite", 0.5),
        [s("C4AF"), s("ettringite"), s("H2O@")],
        [s("monosulphate12"), s("Portlandite"), s("FeOOHmic")],
    )
    r_last = (
        "C4AF -> hydrogarnet", nothing,
        [s("C4AF"), s("H2O@")],
        [s("C3AH6"), s("Portlandite"), s("FeOOHmic")],
    )

    ordered = if c4af_priority === :silicate_first
        [r_si3, r_si2, r_gp, r_cal, r_aft, r_last]
    elseif c4af_priority === :sulfate_first
        [r_gp, r_cal, r_aft, r_si3, r_si2, r_last]
    else
        throw(
            ArgumentError(
                "c4af_priority must be :silicate_first or :sulfate_first, " *
                    "got :$c4af_priority"
            )
        )
    end

    # Calcite is the one route carrying the 48 C switch.
    c4af_gates(i, T, n) = begin
        (name, need) = ordered[i][2]
        g = gate(n, name, need)
        name == "Cal" ? g * θ48(T) : g
    end

    c4af_routes = [(nm, reac, prod) for (nm, _, reac, prod) in ordered]
    r_c4af = map(enumerate(c4af_routes)) do (k, (name, reac, prod))
        r = Reaction(reac, prod; symbol = name)
        r[:rate] = cascade_rate(pk_c4af, c4af_gates, k, length(c4af_routes))
        r
    end

    # ── Pozzolanic reaction: amorphous silica consumes portlandite ──────────
    r_sil = Reaction(
        [s("Amor-Sl"), s("Portlandite"), s("H2O@")], [s("Jennite")];
        symbol = "pozzolanic reaction"
    )
    r_sil[:rate] = waller(
        WALLER_PARAMS_SILICA_FUME, "Amor-Sl";
        α_max = 0.95, blaine = 2000.0u"m^2/kg", humidity
    )

    return vcat([r_c3s, r_c2s], r_c3a, r_c4af, [r_sil])
end

"""
    run_hydration(; wb, clinker, gypsum, filler, silica, blaine, tend, humidity)
        -> (; cs, state0, kp, sol)

Integrate the hydration of 1 kg of binder at `wb` water-to-binder ratio.

`clinker` is a `NamedTuple` of clinker mass fractions **within the clinker**;
`gypsum`, `filler` and `silica` are mass fractions of the binder, the clinker
making up the rest. Amounts are in moles per kilogram of binder, so all volume
fractions below are per unit volume of fresh paste.
"""
function run_hydration(;
        wb = 0.5,
        clinker = (C3S = 0.65, C2S = 0.11, C3A = 0.11, C4AF = 0.08),
        gypsum = 0.046, filler = 0.035, silica = 0.0,
        blaine = 380.0u"m^2/kg", tend = 90 * 86400.0, humidity = nothing,
        c4af_priority::Symbol = :silicate_first,
        induction = stoich_induction(), induction_phases = STOICH_INDUCTION_PHASES,
    )
    cs = build_cement_system()
    f_clinker = 1.0 - gypsum - filler - silica
    f_clinker > 0 || error("the additions exceed the whole binder")

    state0 = ChemicalState(cs)
    for (name, w) in pairs(clinker)
        set_quantity!(state0, string(name), (f_clinker * w)u"kg")
    end
    gypsum > 0 && set_quantity!(state0, "Gp", gypsum * u"kg")
    filler > 0 && set_quantity!(state0, "Cal", filler * u"kg")
    silica > 0 && set_quantity!(state0, "Amor-Sl", silica * u"kg")
    set_quantity!(state0, "H2O@", wb * u"kg")

    rxns = hydration_reactions(
        cs; wb, blaine, humidity, c4af_priority, induction, induction_phases
    )
    kp = KineticsProblem(cs, rxns, state0, (0.0, tend); equilibrium_solver = nothing)
    ks = KineticsSolver(; ode_solver = Rodas5P(), reltol = 1.0e-7, abstol = 1.0e-10)
    return (; cs, state0, kp, sol = integrate(kp, ks))
end

# ── Gel water: reconciling two conventions ──────────────────────────────────
#
# CEMDATA18 and Lavergne et al. do not draw the boundary of "C-S-H" in the same
# place, and the difference is large enough to change the answer by a factor of
# two if it is ignored.
#
#   * CEMDATA18 counts only STRUCTURAL water inside the C-S-H. Its Jennite
#     end-member is (SiO2)(CaO)1.667(H2O)2.1, molar volume 78.4 cm³/mol of Si.
#     The gel water then belongs to the aqueous phase, which is thermodynamically
#     the cleaner convention.
#   * Lavergne et al. use C1.7SH4, molar volume 108.3 cm³/mol of Si: the same
#     solid plus about 1.9 mol of gel water per silicon, held inside the hydrate.
#
# Both are self-consistent. But the micromechanical parameters of the paper —
# E_CSH = 25 GPa, aspect ratio 7, and the percolation threshold that follows —
# were calibrated on the SECOND object. Feeding the first into them would count
# the gel water as capillary porosity, overstate the porosity by some ten points,
# and understate E by roughly a factor of two.
#
# So the gel water is moved from the aqueous phase into the C-S-H here, in the
# bridge, where the two conventions meet. The chemistry is untouched: this is a
# re-partition of a computed volume, not a change to the thermodynamics.
const GEL_WATER_PER_CSH = 4.0 - 2.1     # mol H2O per mol of Si, from C1.7SH4

"""
    fraction_history(run, times; gel_water = GEL_WATER_PER_CSH)
        -> (times, Vector{Dict{String,Float64}})

Volume fractions of the phase families of `PASTE_PHASE_GROUPS` at each instant, in
the sealed-curing convention: referred to the initial paste volume, with the
chemical shrinkage appearing as a `"void"` phase.

`gel_water` moles of water per mole of C-S-H are transferred from `"water"` to
`"C-S-H"`, reconciling the CEMDATA18 and Lavergne conventions — see the
discussion above. Pass `gel_water = 0` to keep the thermodynamic partition.
"""
function fraction_history(run, times; gel_water = GEL_WATER_PER_CSH)
    ξ = reaction_extents(run.sol, run.kp; times = times)
    V_ref = volume(run.state0).total
    fs = Vector{Dict{String, Float64}}(undef, length(times))
    for (i, t) in enumerate(times)
        st = state_at(run.sol, run.kp, t; ξ = ξ[1:i, :])
        f = volume_fractions(st, PASTE_PHASE_GROUPS; reference = run.state0)
        d = Dict{String, Float64}(k => v for (k, v) in f)

        if gel_water > 0
            n_csh = ustrip(us"mol", moles(st, "Jennite"))
            V_H2O = ustrip(
                _molar_volume_of(run.cs, "H2O@")(
                    T = temperature(st), P = pressure(st); unit = true
                ) / V_ref
            )
            # Self-desiccation caps the transfer: the gel cannot hold water that
            # is no longer there.
            f_gel = min(n_csh * gel_water * V_H2O, d["water"])
            d["C-S-H"] += f_gel
            d["water"] -= f_gel
        end
        fs[i] = d
    end
    return times, fs
end

_molar_volume_of(cs, sym) = cs[sym][:V⁰]
