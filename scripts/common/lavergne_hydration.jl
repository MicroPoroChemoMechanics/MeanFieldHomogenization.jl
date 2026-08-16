# =============================================================================
#  lavergne_hydration.jl — the ChemistryLab half of the Lavergne et al. (2018)
#  chain: binder composition → α(t) → moles → volume fractions.
#
#  Shared by `scripts/44_lavergne_hydration_micromechanics.jl` and by the
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
# `LAVERGNE_HYDRATES`, `LAVERGNE_INCLUSIONS` and `LAVERGNE_PORES` of
# `lavergne_model.jl`; `volume_fractions` refuses to put a species in two.
const LAVERGNE_GROUPS = [
    "anhydrous" => ["C3S", "C2S", "C3A", "C4AF"],
    "gypsum" => "Gp",
    "calcite" => "Cal",
    "silica" => "Amor-Sl",
    "C-S-H" => "Jennite",
    "CH" => "Portlandite",
    "AFt" => "ettringite",
    "AFm" => "monosulphate12",
    "hydrogarnet" => "C3AH6",
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
            "Jennite Portlandite ettringite monosulphate12 C3AH6 FeOOHmic " *
            "H2O@"
    )
    sp = speciation(substances, syms; aggregate_state = [AS_AQUEOUS])
    return ChemicalSystem(sp, CEMDATA_PRIMARIES)
end

"""
    hydration_reactions(cs; wb, blaine, humidity) -> Vector{Reaction}

The reaction set, with a Parrot–Killoh rate on each clinker phase and a Waller
rate on the silica.

Every reaction is **balanced by ChemistryLab** from its reactants and products
rather than by hand: CEMDATA18 stores Jennite with a rounded Ca:Si of 1.666667,
so hand-written 4/3 and 103/30 coefficients leave a residual.

C₃A and C₄AF each drive two reactions, one consuming gypsum to form ettringite
and one forming hydrogarnet without it. The Parrot–Killoh rate of the phase is
split between them by a smooth switch on the remaining gypsum, which reproduces
the sequencing rule of Lavergne et al. — ettringite while sulfate lasts, then
the sulfate-free product — while staying differentiable.
"""
function hydration_reactions(cs; wb, blaine, humidity = nothing)
    s(n) = cs[n]
    α_max = powers_alpha_max(wb)
    pk(p, n) = parrot_killoh_avrami(p, n; α_max, blaine, humidity)

    # Smooth partition of an aluminate rate between the sulfated and the
    # sulfate-free route, gated on the gypsum still present. `ε` is a mole scale,
    # not a tolerance: it sets how sharply the switch happens as sulfate runs out.
    #
    # Gypsum is consumed but is not a kinetic species. Reading it here is correct
    # because the ODE state carries the reaction extents, from which the residual
    # reconstructs every species before evaluating the rates (ChemistryLab ≥ 0.5).
    # The gate is kept smooth rather than a hard `> 0` test, so the residual stays
    # continuous and differentiable for the stiff solver.
    ε_gp = 1.0e-3
    sulfated(rate) = (T, P, t, n, lna, n0) -> begin
        g = max(n["Gp"], zero(eltype(n.data)))
        rate(T, P, t, n, lna, n0) * g / (g + ε_gp)
    end
    unsulfated(rate) = (T, P, t, n, lna, n0) -> begin
        g = max(n["Gp"], zero(eltype(n.data)))
        rate(T, P, t, n, lna, n0) * ε_gp / (g + ε_gp)
    end
    wrap(f) = KineticFunc(f, (T = 293.15u"K", P = 1.0e5u"Pa"), u"mol/s")

    r_c3s = Reaction([s("C3S"), s("H2O@")], [s("Jennite"), s("Portlandite")]; symbol = "C3S hydration")
    r_c3s[:rate] = pk(PK84_PARAMS_C3S, "C3S")

    r_c2s = Reaction([s("C2S"), s("H2O@")], [s("Jennite"), s("Portlandite")]; symbol = "C2S hydration")
    r_c2s[:rate] = pk(PK84_PARAMS_C2S, "C2S")

    pk_c3a = pk(PK84_PARAMS_C3A, "C3A")
    r_c3a_aft = Reaction(
        [s("C3A"), s("Gp"), s("H2O@")], [s("ettringite")]; symbol = "C3A + gypsum -> AFt"
    )
    r_c3a_aft[:rate] = wrap(sulfated(pk_c3a))
    r_c3a_hg = Reaction([s("C3A"), s("H2O@")], [s("C3AH6")]; symbol = "C3A -> hydrogarnet")
    r_c3a_hg[:rate] = wrap(unsulfated(pk_c3a))

    pk_c4af = pk(PK84_PARAMS_C4AF, "C4AF")
    r_c4af_aft = Reaction(
        [s("C4AF"), s("Gp"), s("H2O@")],
        [s("ettringite"), s("Portlandite"), s("FeOOHmic")];
        symbol = "C4AF + gypsum -> AFt"
    )
    r_c4af_aft[:rate] = wrap(sulfated(pk_c4af))
    r_c4af_hg = Reaction(
        [s("C4AF"), s("H2O@")], [s("C3AH6"), s("Portlandite"), s("FeOOHmic")];
        symbol = "C4AF -> hydrogarnet"
    )
    r_c4af_hg[:rate] = wrap(unsulfated(pk_c4af))

    # Pozzolanic reaction: amorphous silica consumes portlandite.
    r_sil = Reaction(
        [s("Amor-Sl"), s("Portlandite"), s("H2O@")], [s("Jennite")];
        symbol = "pozzolanic reaction"
    )
    r_sil[:rate] = waller(
        WALLER_PARAMS_SILICA_FUME, "Amor-Sl";
        α_max = 0.95, blaine = 2000.0u"m^2/kg", humidity
    )

    return [r_c3s, r_c2s, r_c3a_aft, r_c3a_hg, r_c4af_aft, r_c4af_hg, r_sil]
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

    rxns = hydration_reactions(cs; wb, blaine, humidity)
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

Volume fractions of the phase families of `LAVERGNE_GROUPS` at each instant, in
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
        f = volume_fractions(st, LAVERGNE_GROUPS; reference = run.state0)
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
