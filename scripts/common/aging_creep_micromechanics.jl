# =============================================================================
#  aging_creep_micromechanics.jl — aging basic creep of a hydrating concrete.
#
#    Lavergne, F. and Barthélémy, J.-F. (2020), "Confronting a refined
#    multiscale estimate for the aging basic creep of concrete with a
#    comprehensive experimental database", Cement and Concrete Research 136,
#    106163. https://doi.org/10.1016/j.cemconres.2020.106163
#
#  Included by `scripts/47_aging_basic_creep_concrete.jl` and by the Applications
#  chapter `docs/src/applications/aging_basic_creep.md`, after
#  `paste_micromechanics.jl` (phase stiffnesses, families, the 2018 scales).
#
#  The four scales of the elastic model, in aging linear viscoelasticity. A phase
#  keeps a fixed volume fraction over the whole history; what changes with time
#  is when it carries stress. Each step of the hydration therefore adds phases:
#
#    1. Hydrates      — the hydrates formed during step j, self-consistent,
#                       spherical: one law per step.
#    2. Hydrate foam  — the foam born at step i, self-consistent: needles of
#                       aspect ratio 7, randomly oriented, of the hydrates
#                       precipitated at each step j ≥ i, plus its porosity.
#    3. Cement paste  — Mori-Tanaka on the foam born first: each grain consumed
#                       at step i is a phase that dissolves at tᵢ while the foam
#                       born at tᵢ precipitates in its place.
#    4. Concrete      — Mori-Tanaka on the paste, with elastic aggregates.
#
#  Times are in days, moduli in GPa.
# =============================================================================

using MeanFieldHomogenization
using TensND
using LinearAlgebra

# ── The creep of the hydrates, LB2020 Eqs. (10)-(11) ────────────────────────
#
# A logarithmic creep compliance with a time-independent Poisson ratio,
#     J(t, t') = [1/E + ln(1 + (t - t')/τ)/E_D] S₁,   S₁ = C⁻¹(E = 1, ν),
# whose ratio E_D/E for C-S-H is the paper's correlation with the Ca/Si ratio
# and the relative humidity. The coefficients below DEFINE that model.

const CREEP_TAU = 0.5                          # days, §2.3.1
csh_log_ratio(ca_si, rh) = 1.2 * (14 * ca_si^2 - 33 * ca_si + 24) * 0.11 / rh   # Eq. (11)

# The C-S-H of the hydration run: the Jennite end-member of CEMDATA18.
const CSH_CA_SI = 1.667
# AFm takes the ratio of a C-S-H of Ca/Si 1.7 (§2.3.1).
const AFM_CA_SI = 1.7

"""
    log_creep_law(E, ν, E_D; τ = CREEP_TAU) -> ViscoLaw

The logarithmic creep law of Eq. (10) as a `:creep` kernel: elastic compliance `1/E`, a
logarithmic term of modulus `E_D`, the same Poisson ratio `ν` for both.
"""
function log_creep_law(E, ν, E_D; τ = CREEP_TAU)
    S₁ = inv(iso_stiffness_E_nu(one(E), ν))
    return ViscoLaw((t, tp) -> t < tp ? zero(S₁) : (inv(E) + log1p((t - tp) / τ) / E_D) * S₁, :creep)
end

"""
    hydrate_law(name; rh = 1.0) -> ViscoLaw or elastic tensor

The law of one hydrate family: logarithmic creep for C-S-H and AFm, elastic otherwise (the
stiffness itself, whose kernel is C H(t - t′)), with the stiffnesses of `PASTE_E_NU`.
"""
function hydrate_law(name; rh = 1.0)
    E, ν = PASTE_E_NU[name]
    name == "C-S-H" && return log_creep_law(E, ν, csh_log_ratio(CSH_CA_SI, rh) * E)
    name == "AFm" && return log_creep_law(E, ν, csh_log_ratio(AFM_CA_SI, rh) * E)
    return _stiff(name)   # an elastic tensor: its kernel is C H(t - t′)
end

# ── Kernels on the grid ─────────────────────────────────────────────────────

# Every scale hands the next its relaxation operator on the grid (`VolterraOperator`),
# isotropic here: two n×n blocks. Two coexisting histories in the same volume (a grain
# that dissolves while the foam precipitates in its place) are the sum of operators.

const TINY_VOID = TensISO{3}(3 * TINY_LAV, 2 * TINY_LAV)

# ── Scale 1: the hydrates formed during one step ────────────────────────────

"""
    hydrate_mix(composition, times; rh = 1.0) -> VolterraOperator

The self-consistent estimate of the hydrates of `composition` (volumes per family), all
spherical: its relaxation operator on `times`.
"""
function hydrate_mix(composition, times; rh = 1.0)
    total = sum(values(composition))
    rve = RVE()
    add_phase!(rve, :SEED, Ellipsoid(1.0), Dict(:C => hydrate_law("C-S-H"; rh)); fraction = :rest)
    for (name, v) in composition
        add_phase!(
            rve, Symbol(replace(name, "-" => "")), Ellipsoid(1.0), Dict(:C => hydrate_law(name; rh));
            fraction = v / total * _SC_SLACK
        )
    end
    return homogenize_alv(rve, SelfConsistent(), :C; times)
end

# ── Scale 2: the foam born at step i ────────────────────────────────────────

"""
    foam_law(i, porosity, mixes, times) -> VolterraOperator

The foam born at `times[i]`: its solid fraction `1 - porosity[i]` at birth is the mix of
step `i`, then each later step `j` fills `porosity[j-1] - porosity[j]` of it with the mix of
step `j`, precipitated stress-free at `times[j]`; what remains is pore. Self-consistent,
hydrates as needles of aspect ratio `ω_hyd` randomly oriented.
"""
function foam_law(i, porosity, mixes, times)
    n = length(times)
    rve = RVE()
    add_phase!(rve, :SEED, Ellipsoid(1.0), Dict(:C => TINY_VOID); fraction = :rest)
    for j in i:n
        f = j == i ? 1 - porosity[i] : porosity[j - 1] - porosity[j]
        (f > 0 && mixes[j] !== nothing) || continue
        add_phase!(
            rve, Symbol(:H, j), Spheroid(ω_hyd), Dict(:C => precipitated(mixes[j], times[j]));
            fraction = f * _SC_SLACK, symmetrize = :iso
        )
    end
    add_phase!(rve, :PORE, Ellipsoid(1.0), Dict(:C => TINY_VOID); fraction = porosity[n] * _SC_SLACK)
    return homogenize_alv(rve, SelfConsistent(), :C; times)
end

# ── Scales 3 and 4 ──────────────────────────────────────────────────────────

"""
    concrete_relaxation(fs, times; f_aggregate, rh = 1.0) -> VolterraOperator

The relaxation operator of the concrete on `times`, from the volume fractions `fs[k]` of the
paste at `times[k]` (families of `PASTE_PHASE_GROUPS`, fractions of the paste) and the
aggregate fraction of the concrete.
"""
function concrete_relaxation(fs, times; f_aggregate, rh = 1.0)
    n = length(times)
    hydrates(k) = Dict(h => get(fs[k], h, 0.0) for h in PASTE_HYDRATES)
    # The hydrates formed at each step: the increase of each family over the step.
    mixes = map(1:n) do k
        inc = Dict(h => max(v - (k == 1 ? 0.0 : hydrates(k - 1)[h]), 0.0) for (h, v) in hydrates(k))
        filter!(p -> p.second > 0, inc)
        isempty(inc) ? nothing : hydrate_mix(inc, times; rh)
    end
    # Capillary porosity of the foam: pores over pores plus hydrates.
    porosity = map(1:n) do k
        pores = get(fs[k], "water", 0.0) + get(fs[k], "void", 0.0)
        pores / (pores + sum(values(hydrates(k))))
    end
    foams = [foam_law(i, porosity, mixes, times) for i in 1:n]

    paste = RVE()
    reactant(k, name) = get(fs[k], name, 0.0)
    add_phase!(paste, :FOAM1, Ellipsoid(1.0), Dict(:C => foams[1]); fraction = :rest)
    for name in PASTE_INCLUSIONS
        grain = VolterraOperator(_stiff(name), times)
        reactant(n, name) > 0 && add_phase!(
            paste, Symbol(name), Ellipsoid(1.0), Dict(:C => grain); fraction = reactant(n, name)
        )
        for i in 2:n
            dv = reactant(i - 1, name) - reactant(i, name)
            dv > 0 || continue
            R = dissolved(grain, times[i]) + precipitated(foams[i], times[i])
            add_phase!(paste, Symbol(name, :_, i), Ellipsoid(1.0), Dict(:C => R); fraction = dv)
        end
    end
    # The paste is homogenized where the concrete needs it, on the same grid.
    concrete = RVE()
    add_phase!(concrete, :PASTE, Ellipsoid(1.0), Dict(:C => Homogenized(paste, MoriTanaka())); fraction = :rest)
    add_phase!(concrete, :AGG, Ellipsoid(1.0), Dict(:C => _stiff("aggregate")); fraction = f_aggregate)
    return homogenize_alv(concrete, MoriTanaka(), :C; times)
end

"""
    uniaxial_compliance(R, times) -> Matrix

`J[i, k]`, the axial strain at `times[i]` for a unit uniaxial stress held from `times[k]`
on, from the relaxation operator `R` of an isotropic material: the response of the inverse
operator to that stress history. On the grid, a load "held from `times[k]`" rises over
`[times[k-1], times[k]]`; `loading_grid` places an instant just before each loading age so
that this rise is a step.
"""
function uniaxial_compliance(R, times)
    Jop = volterra_inverse(R; block_size = 6)
    n = length(times)
    axial(i, j) = Jop[6 * (i - 1) + 1, 6 * (j - 1) + 1]
    return [k ≤ i ? sum(axial(i, j) for j in k:i) : NaN for i in 1:n, k in 1:n]
end

"""
    loading_grid(t_start, t_end, ages; per_decade = 6, lead = 1.0e-3, after = (0.02, 0.05, 0.1, 0.2, 0.5)) -> Vector

A logarithmic time grid from `t_start` to `t_end` that holds every loading age of `ages`,
just before each the instant `age * (1 - lead)`, and after each the instants
`age * (1 + a)` for `a` in `after`, where the compliance changes fastest.
"""
function loading_grid(t_start, t_end, ages; per_decade = 6, lead = 1.0e-3, after = (0.02, 0.05, 0.1, 0.2, 0.5))
    m = round(Int, per_decade * log10(t_end / t_start)) + 1
    base = exp10.(range(log10(t_start), log10(t_end); length = m))
    near = [a * (1 + d) for a in ages for d in after]
    return sort(unique(vcat(base, ages, ages .* (1 - lead), near)))
end
