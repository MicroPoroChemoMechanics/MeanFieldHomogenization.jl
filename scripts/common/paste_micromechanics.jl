# =============================================================================
#  paste_micromechanics.jl — shared four-scale model of a hydrating blended paste.
#
#    Lavergne, F., Ben Fraj, A., Bayane, I. and Barthélémy, J.-F. (2018),
#    "Estimating the mechanical properties of hydrating blended cementitious
#    materials: an investigation based on micromechanics",
#    Cement and Concrete Research 104, 37-60.
#    https://doi.org/10.1016/j.cemconres.2017.10.018
#
#  Included by `scripts/44_stoichiometric_hydration_micromechanics.jl` and by the
#  Applications chapter `docs/src/applications/hydrating_blended_paste.md`.
#
#  The point of this file is that it takes VOLUME FRACTIONS, never (w/c, α):
#  the fractions come from a ChemistryLab hydration run, not from a Powers-type
#  correlation. Nothing here knows about cement chemistry — it knows about
#  phases, their stiffnesses and their shapes.
#
#  Four scales (Fig. 2 of the paper):
#
#    1. Hydrates      — Self-Consistent, all phases SPHERICAL.
#    2. Hydrate foam  — Self-Consistent, hydrates as spheroids of aspect ratio
#                       ω = 7 over NTHETA orientations, plus capillary porosity.
#    3. Cement paste  — Mori-Tanaka, foam matrix + spherical unreacted grains
#                       and inert fillers.
#    4. Concrete      — Mori-Tanaka, paste matrix + aggregates + entrapped air.
#
#  Scale 1 is what distinguishes this model from `quasibrittle_strength.jl`, which
#  collapses every hydrate into a single stiffness. Here the hydrate species are
#  merged first, and only the RESULT is given the fibrillar morphology.
# =============================================================================

using MeanFieldHomogenization
using TensND
using ForwardDiff
using LinearAlgebra

# ── Phase stiffnesses, Table 5 of the paper (E in GPa, ν) ────────────────────
#
# The C-S-H entry is the median of the HD (29.4 GPa) and LD (21.7 GPa) values
# identified by Constantinides & Ulm on C-S-H foams: the model carries a single
# C-S-H of uniform density, which the paper's own §5 lists as a limitation.

const PASTE_E_NU = Dict{String, Tuple{Float64, Float64}}(
    "C-S-H" => (25.0, 0.24),      # C₁.₇SH₄ and pozzolanic C₁.₁SH₃.₉ alike
    "CH" => (42.3, 0.324),        # portlandite
    "AFt" => (22.4, 0.25),        # ettringite
    "AFm" => (42.3, 0.324),       # monosulfo- and monocarboaluminate
    "hydrogarnet" => (22.4, 0.25),   # C₃AH₆, C₆AFS₂.₁₈H₁₉
    "FH3" => (22.4, 0.25),        # iron oxide-hydroxide
    "anhydrous" => (130.0, 0.3),  # C₃S, C₂S, C₃A, C₄AF
    "gypsum" => (45.7, 0.33),
    "calcite" => (83.8, 0.31),    # limestone filler
    "silica" => (72.8, 0.167),    # unreacted silica fume / fly ash
    "aggregate" => (60.0, 0.25),  # siliceous sand, a representative value
)

# Which families are hydrates (scale 1), which are inclusions in the paste
# (scale 3). Anything not listed in either is an error rather than a silent drop.
const PASTE_HYDRATES = ["C-S-H", "CH", "AFt", "AFm", "hydrogarnet", "FH3"]
const PASTE_INCLUSIONS = ["anhydrous", "gypsum", "calcite", "silica"]
const PASTE_PORES = ["water", "void"]

# ── Model constants ─────────────────────────────────────────────────────────

# Aspect ratio of the fibrillar hydrates at the foam scale. The paper settles on
# 7 — large enough to set the percolation threshold near the observed hydration
# degree at setting (2.7-4.3 % for w/c in 0.25-0.4), small enough to keep the
# early-age strength estimate reasonable. Aspect ratios of 21 or 35 would place
# the threshold at the measured gel-space ratios of 13 % and 7 % respectively.
const ω_hyd = 7.0

# Orientation bins for the foam. The paper uses 20; the chapter runs fewer to
# keep the docs build short, which moves the percolation threshold by well under
# a percent of hydration degree.
const NTHETA_PASTE = 20

# Bulk modulus of the capillary water, GPa. Zero for a STATIC Young's modulus —
# the pore fluid does not carry load in a drained measurement — and 2.2 GPa for
# a dynamic (ultrasonic) one, where it does. The paper is explicit that this
# choice dominates the estimated Poisson's ratio.
const K_WATER_STATIC = 0.0
const K_WATER_DYNAMIC = 2.2

# Empty porosity and (for a static measurement) water are given a small positive
# stiffness rather than an exact zero: it selects the percolating branch of the
# self-consistent fixed point while keeping the iteration smooth. Same device,
# and same justification, as `quasibrittle_strength.jl`.
const TINY_LAV = 1.0e-6

_stiff(name) = iso_stiffness_E_nu(PASTE_E_NU[name]...)
_stiff(name, ::Type{Float64}) = _stiff(name)
function _stiff(name, ::Type{T}) where {T}
    E, ν = PASTE_E_NU[name]
    return iso_stiffness_E_nu(convert(T, E), convert(T, ν))
end

# Both self-consistent stages use a ZERO-volume matrix phase as the fixed-point
# seed, so their inclusion fractions must sum to one. Summing to 1 + 2e-16 makes
# `RVE` warn about a negative matrix fraction, so leave a hair of headroom.
const _SC_SLACK = 1.0 - 1.0e-12

"""
    normalize_fractions(f) -> Dict{String, Float64}

Clamp tiny negative fractions to zero and renormalize to a unit sum.

`volume_fractions` keeps the negative partial molar volumes of aqueous solutes,
which is right for a volume balance and meaningless for an RVE. Anything more
negative than `-1e-8` is an error, not a rounding artifact.
"""
function normalize_fractions(f)
    # Generic in the element type: the fractions may be `ForwardDiff.Dual` when
    # a sensitivity ∂E/∂fᵢ is taken through the whole chain.
    T = mapreduce(typeof, promote_type, values(f); init = Float64)
    out = Dict{String, T}()
    for (k, v) in f
        _val(v) < -1.0e-8 && error("phase \"$k\" has a strongly negative fraction $v")
        out[k] = max(convert(T, v), zero(T))
    end
    s = sum(values(out))
    _val(s) > 0 || error("all volume fractions are zero")
    for k in keys(out)
        out[k] /= s
    end
    return out
end

# Primal value of a possibly-dual number, for comparisons and guards.
_val(x::Real) = x
_val(x::ForwardDiff.Dual) = ForwardDiff.value(x)

# Element type carried by a fraction dictionary.
_fT(f) = mapreduce(typeof, promote_type, values(f); init = Float64)

# ── Scale 1: the hydrates, self-consistent, all spherical ───────────────────
#
# A polycrystal with no matrix phase: the SC seed carries zero volume and every
# hydrate family enters as an inclusion, which is why the fractions are
# renormalized over the hydrates alone.
function build_hydrates(f; T = _fT(f))
    f_h = Dict(k => convert(T, get(f, k, zero(T))) for k in PASTE_HYDRATES)
    total = sum(values(f_h))
    _val(total) > 0 || return nothing         # nothing has precipitated yet

    rve = RVE(:SEED; T = T)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => _stiff("C-S-H", T)); symmetrize = :iso)
    for name in PASTE_HYDRATES
        frac = f_h[name] / total
        _val(frac) > 0 || continue
        add_phase!(
            rve, Symbol(replace(name, "-" => "")), Ellipsoid(1.0),
            Dict(:C => _stiff(name, T)); fraction = frac * _SC_SLACK, symmetrize = :iso
        )
    end
    return homogenize(rve, SelfConsistent(; abstol = 1.0e-10, maxiters = 500), :C)
end

# ── Scale 2: the hydrate foam, self-consistent, ω = 7 needles + porosity ────
#
# The hydrates are spread over NTHETA orientation bins, each exactly averaged
# about the global axis by `TISymmetrize`. This is the stage that percolates:
# below a critical hydrate fraction the fixed point returns a vanishing
# stiffness, which is the setting transition.
function build_foam(f, C_hyd; N::Int = NTHETA_PASTE, ω::Real = ω_hyd, K_water = K_WATER_STATIC)
    T = promote_type(_fT(f), eltype(C_hyd))
    f_hyd = sum(convert(T, get(f, k, 0.0)) for k in PASTE_HYDRATES)
    f_water = convert(T, get(f, "water", 0.0))
    f_void = convert(T, get(f, "void", 0.0))
    total = f_hyd + f_water + f_void
    _val(total) > 0 || error("the hydrate foam is empty")

    ez = (0.0, 0.0, 1.0)
    rve = RVE(:SEED; T = T)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_hyd); symmetrize = :iso)

    for bin in polar_orientation_bins(N)
        add_phase!(
            rve, Symbol(:HYD, round(Int, 1.0e4 * bin.θ)),
            Spheroid(ω; euler_angles = (bin.θ, 0.0, 0.0)),
            Dict(:C => C_hyd);
            fraction = (f_hyd / total) * bin.weight * _SC_SLACK,
            symmetrize = TISymmetrize(ez)
        )
    end

    # Water-filled capillary porosity: bulk K_water, no shear.
    C_w = TensISO{3}(convert(T, 3 * max(K_water, TINY_LAV)), convert(T, 2 * TINY_LAV))
    add_phase!(
        rve, :WATER, Ellipsoid(1.0), Dict(:C => C_w);
        fraction = (f_water / total) * _SC_SLACK, symmetrize = :iso
    )
    # Empty porosity left by the chemical shrinkage.
    C_v = TensISO{3}(convert(T, 3 * TINY_LAV), convert(T, 2 * TINY_LAV))
    add_phase!(
        rve, :VOID, Ellipsoid(1.0), Dict(:C => C_v);
        fraction = (f_void / total) * _SC_SLACK, symmetrize = :iso
    )

    return homogenize(
        rve, SelfConsistent(; abstol = 1.0e-8, maxiters = 1000, damping = 0.5), :C;
        select_best = true
    )
end

# ── Scale 3: the cement paste, Mori-Tanaka on the foam ──────────────────────
function build_paste(f, C_foam)
    T = promote_type(_fT(f), eltype(C_foam))
    rve = RVE(:FOAM; T = T)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_foam))
    for name in PASTE_INCLUSIONS
        frac = convert(T, get(f, name, 0.0))
        _val(frac) > 0 || continue
        add_phase!(
            rve, Symbol(name), Ellipsoid(1.0), Dict(:C => _stiff(name, T)); fraction = frac
        )
    end
    return homogenize(rve, MoriTanaka(), :C)
end

# ── Scale 4: concrete, Mori-Tanaka on the paste ─────────────────────────────
function build_concrete(C_paste, f_aggregate, f_air)
    T = eltype(C_paste)
    rve = RVE(:PASTE; T = T)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_paste))
    f_aggregate > 0 && add_phase!(
        rve, :AGG, Ellipsoid(1.0), Dict(:C => _stiff("aggregate")); fraction = f_aggregate
    )
    f_air > 0 && add_phase!(
        rve, :AIR, Ellipsoid(1.0),
        Dict(:C => TensISO{3}(convert(T, 3 * TINY_LAV), convert(T, 2 * TINY_LAV)));
        fraction = f_air
    )
    return homogenize(rve, MoriTanaka(), :C)
end

"""
    paste_moduli(f; N, ω, K_water) -> (; K, μ, E, ν)

Effective elastic moduli of a cement paste whose phase volume fractions are `f`,
a `Dict`-like mapping of the family names of `PASTE_HYDRATES`,
`PASTE_INCLUSIONS` and `PASTE_PORES` to fractions summing to one.

Returns `(K = 0, μ = 0, E = 0, ν = NaN)` before the hydrate foam percolates —
the setting transition — rather than a spurious small stiffness.
"""
function paste_moduli(
        f; N::Int = NTHETA_PASTE, ω::Real = ω_hyd, K_water = K_WATER_STATIC
    )
    fn = normalize_fractions(f)

    Tf = _fT(fn)
    C_hyd = build_hydrates(fn)
    C_hyd === nothing && return (; K = zero(Tf), μ = zero(Tf), E = zero(Tf), ν = Tf(NaN))

    # The paste-scale fractions are those of the inclusions; the foam occupies
    # the rest, so its internal fractions are renormalized inside `build_foam`.
    f_inc = sum(convert(Tf, get(fn, k, 0.0)) for k in PASTE_INCLUSIONS)
    _val(f_inc) < 1.0 || return (; K = zero(Tf), μ = zero(Tf), E = zero(Tf), ν = Tf(NaN))

    C_foam = build_foam(fn, C_hyd; N = N, ω = ω, K_water = K_water)
    K_f, μ_f = k_mu(TensND.proj_tens(Val(:ISO), get_array(C_foam))[1])
    # Below percolation the SC fixed point collapses; report setting, not noise.
    _val(μ_f) <= 1.0e-4 && return (; K = zero(Tf), μ = zero(Tf), E = zero(Tf), ν = Tf(NaN))

    C_foam_iso = iso_stiffness(K_f, μ_f)
    C_paste = build_paste(fn, C_foam_iso)
    K_p, μ_p = k_mu(TensND.proj_tens(Val(:ISO), get_array(C_paste))[1])
    E_p, ν_p = E_nu(iso_stiffness(K_p, μ_p))
    return (; K = K_p, μ = μ_p, E = E_p, ν = ν_p)
end

"""
    concrete_moduli(f, f_aggregate; f_air, kwargs...) -> (; K, μ, E, ν)

Effective moduli of a concrete: the paste of [`paste_moduli`](@ref)
with `f_aggregate` of aggregates and `f_air` of entrapped air added at the
upper scale. `f_aggregate` and `f_air` are fractions of the concrete, while `f`
describes the paste.
"""
function concrete_moduli(f, f_aggregate; f_air = 0.0, kwargs...)
    m = paste_moduli(f; kwargs...)
    _val(m.E) > 0 || return m
    C_paste = iso_stiffness(m.K, m.μ)
    C_conc = build_concrete(C_paste, f_aggregate, f_air)
    K_c, μ_c = k_mu(TensND.proj_tens(Val(:ISO), get_array(C_conc))[1])
    E_c, ν_c = E_nu(iso_stiffness(K_c, μ_c))
    return (; K = K_c, μ = μ_c, E = E_c, ν = ν_c)
end
