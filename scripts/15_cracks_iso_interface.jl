# =============================================================================
#  15_cracks_iso_interface.jl
#
#  Cross-check **elastic + conductivity homogenization with cracks** vs
#  a reference implementation.
#
#  Setup :
#    * Iso solid matrix : `C_s = stiff_Enu(E = 1, ν = 0.25)`,
#                          `K_s = k_o · 𝟏` (small background conductivity)
#    * Crack : penny / very-thin spheroid with `aspect_ratio = 1e-3`,
#              random orientation distribution (`symmetrize=[ISO]`).
#    * **Elastic side**  : *traction-free* crack (`prop["C"] = 0` in
#      ECHOES) — no interface stiffness.
#    * **Conduction side** : the crack is modeled as a thin highly-
#      conductive inclusion `K_crack = (kt, kt, kn) · 𝟏 = K_t · 𝟏`,
#      ECHOES uses `prop={"K": tensor(kt,kt,kn)}` on the `crack()`
#      factory.  In MeanFieldHomogenization we use a `Spheroid(1e-3)` with a
#      `VolumeFraction = density · 4π/3 · aspect_ratio` (translation of
#      Budiansky `density = N a b² → volume fraction (4π/3) ε ω` for a
#      thin spheroid of aspect ratio ω).
#
#  Both implementations cover schemes : Mori-Tanaka, Asymmetric
#  Self-Consistent (cracks supported in MFH only via ASC, not SC),
#  Differential, PCW, Maxwell.  Density sweep `d ∈ [0, 1.2]`.
#
#  Output : two-panel plot with Julia × markers / ECHOES — solid lines.
#
#  Usage  : julia --project scripts/15_cracks_iso_interface.jl
#  Output : scripts/figures/15_cracks_iso_interface.png
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using PyCall
using Plots

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

# ─── Common parameters ────────────────────────────────────────────────────

const ω = 1.0e-3
const k_o = 1.0e-9
const K_t = 1.0
const E_s, ν_s = 1.0, 0.25
const k_s = E_s / (3 * (1 - 2 * ν_s))
const μ_s = E_s / (2 * (1 + ν_s))
const C_s = TensISO{3}(3 * k_s, 2 * μ_s)
const K_s = TensISO{3}(k_o)

const DENS = collect(range(0.0, 1.0; length = 30))
const SCHEME_NAMES = ("MT", "SC", "ASC", "DIFF", "PCW", "MAX")
const SCHEME_COLORS = Dict(
    "MT" => :black, "SC" => :red, "ASC" => :orange,
    "DIFF" => :blue, "PCW" => :green, "MAX" => :purple
)

# ─── ECHOES side via PyCall ────────────────────────────────────────────────

const echoes = pyimport("echoes")
println("ECHOES imported")

py"""
from numpy import *
from echoes import *

_SCHEMES = {"MT": MT, "SC": SC, "ASC": ASC, "DIFF": DIFF, "PCW": PCW, "MAX": MAX}

def run_echoes(dens, scheme_name, omega, ko, kt, kn, E_s, nu_s):
    sch = _SCHEMES[scheme_name]
    Cs = stiff_Enu(E_s, nu_s)
    ver = rve(matrix="SOLID")
    ver["SOLID"] = ellipsoid(shape=spheroidal(1),
                              prop={"C":Cs, "K":ko*tId2})
    ver["CRACK"] = crack(shape=spheroidal(omega), symmetrize=[ISO],
                          prop={"C":tZ4, "K":tensor(kt,kt,kn)})
    ver["CRACK"].density = float(dens)
    try:
        C = homogenize(prop="C", rve=ver, scheme=sch, maxnb=200,
                       epsrel=1.e-5, select_best=True, verbose=False)
        k_eff = float(C.k); mu_eff = float(C.mu)
    except Exception:
        k_eff = float('nan'); mu_eff = float('nan')
    try:
        K = homogenize(prop="K", rve=ver, scheme=sch, maxnb=200,
                       epsrel=1.e-5, select_best=True, verbose=False)
        K_eff = float(K.param[0])
    except Exception:
        K_eff = float('nan')
    return k_eff, mu_eff, K_eff
"""

println("Running ECHOES sweep over density × {MT, ASC, DIFF, PCW, MAX} …")
echoes_results = Dict{String, NamedTuple}()
for name in SCHEME_NAMES
    @printf "  ECHOES scheme=%s\n" name
    k_arr = Float64[]; μ_arr = Float64[]; K_arr = Float64[]
    for d in DENS
        k_e, μ_e, K_e = py"run_echoes"(d, name, ω, k_o, K_t, K_t, E_s, ν_s)
        push!(k_arr, max(k_e / k_s, 0.0))
        push!(μ_arr, max(μ_e / μ_s, 0.0))
        push!(K_arr, K_e / (ω * K_t))
    end
    echoes_results[name] = (; k = k_arr, μ = μ_arr, K = K_arr)
end

# ─── Julia side ────────────────────────────────────────────────────────────
#
# Elastic : traction-free penny crack, density = d.
# Conduction : the reference model treats the crack as a high-K thin
#              spheroid inclusion (NOT a free crack with interface
#              conductance).  In MFH the equivalent is a
#              `Spheroid(ω)` with `K = TensISO{3}(K_t)` and a volume
#              fraction `f = (4π/3) · ε · ω` (Budiansky-O'Connell
#              translation for thin oblate spheroids).

const JULIA_SCHEME_OBJ = Dict(
    "MT" => MoriTanaka(),
    "SC" => SelfConsistent(),
    "ASC" => AsymmetricSelfConsistent(),
    "DIFF" => DifferentialScheme(; nsteps = 100),
    "PCW" => PonteCastanedaWillis(),
    "MAX" => Maxwell(),
)

function _build_rve_elastic(d::Real)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_s))
    add_phase!(
        rve, :CRACK, PennyCrack(1.0), Dict(:C => C_s);
        density = d, symmetrize = :iso
    )
    return rve
end

function _build_rve_conduction(d::Real)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:K => K_s))
    # Thin spheroid (aspect ratio ω) with high conductivity ; volume
    # fraction ↔ density translation : f ≈ (4π/3) · ε · ω.
    f = (4π / 3) * d * ω
    add_phase!(
        rve, :CRACK_K, Spheroid(ω),
        Dict(:K => TensISO{3}(K_t));
        fraction = f, symmetrize = :iso
    )
    return rve
end

function _kμK_julia(scheme, d::Real)
    extra_kw = scheme isa Union{SelfConsistent, AsymmetricSelfConsistent} ?
        (;
            abstol = 1.0e-9, reltol = 1.0e-7, maxiters = 500,
            select_best = true, damping = 0.3,
        ) : (;)
    k_r, μ_r, K_r = NaN, NaN, NaN
    rve_C = _build_rve_elastic(d)
    try
        C = homogenize(rve_C, scheme, :C; extra_kw...)
        α, β = TensND.get_data(C)
        k_r = max(α / 3 / k_s, 0.0)
        μ_r = max(β / 2 / μ_s, 0.0)
    catch
    end
    rve_K = _build_rve_conduction(d)
    try
        K = homogenize(rve_K, scheme, :K; extra_kw...)
        K_r = first(TensND.get_data(K)) / (ω * K_t)
    catch
    end
    return (k = k_r, μ = μ_r, K = K_r)
end

println("Running MeanFieldHomogenization.jl sweep …")
julia_results = Dict{String, NamedTuple}()
for name in SCHEME_NAMES
    @printf "  Julia scheme=%s\n" name
    k_arr = Float64[]; μ_arr = Float64[]; K_arr = Float64[]
    for d in DENS
        r = _kμK_julia(JULIA_SCHEME_OBJ[name], d)
        push!(k_arr, r.k); push!(μ_arr, r.μ); push!(K_arr, r.K)
    end
    julia_results[name] = (; k = k_arr, μ = μ_arr, K = K_arr)
end

# ─── Plot ──────────────────────────────────────────────────────────────────

p_elastic = plot(
    xlabel = "crack density",
    ylabel = "k_eff/k_s,  μ_eff/μ_s",
    title = "Elasticity (traction-free cracks)",
    legend = :bottomleft
)
p_perm = plot(
    xlabel = "crack density",
    ylabel = "K_eff / (ω · K_t)",
    title = "Conductivity (high-K thin spheroid)",
    legend = :topleft
)

for name in SCHEME_NAMES
    col = SCHEME_COLORS[name]
    e = echoes_results[name]
    j = julia_results[name]
    plot!(
        p_elastic, DENS, e.k;
        color = col, linestyle = :solid, linewidth = 1.6,
        label = "ECHOES k $name"
    )
    plot!(
        p_elastic, DENS, e.μ;
        color = col, linestyle = :dash, linewidth = 1.6, alpha = 0.6,
        label = "ECHOES μ $name"
    )
    scatter!(
        p_elastic, DENS, j.k;
        color = col, marker = :circle, markersize = 3, label = ""
    )
    scatter!(
        p_elastic, DENS, j.μ;
        color = col, marker = :diamond, markersize = 3, alpha = 0.6,
        label = ""
    )
    plot!(
        p_perm, DENS, e.K;
        color = col, linestyle = :solid, linewidth = 1.6, label = "ECHOES $name"
    )
    scatter!(
        p_perm, DENS, j.K;
        color = col, marker = :circle, markersize = 3, label = "Julia $name"
    )
end

fig = plot(p_elastic, p_perm; layout = (1, 2), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (1500, 700))

mkpath(joinpath(@__DIR__, "figures"))
out = joinpath(@__DIR__, "figures", "15_cracks_iso_interface.png")
savefig(fig, out)
display(fig)
println("Saved : $out")

# ─── Numeric report ────────────────────────────────────────────────────────

println()
println("═══════════════════════════════════════════════════════════════════")
println(" Comparison Julia ↔ ECHOES at density = 0.5")
println("═══════════════════════════════════════════════════════════════════")
i_05 = argmin(abs.(DENS .- 0.5))
@printf "  %-4s  %-7s  %-12s %-12s %-12s\n" "scheme" "qty" "Julia" "ECHOES" "rel.err"
for name in SCHEME_NAMES
    j = julia_results[name]
    e = echoes_results[name]
    for (qty, jq, eq) in (
            ("k", j.k[i_05], e.k[i_05]),
            ("μ", j.μ[i_05], e.μ[i_05]),
            ("K", j.K[i_05], e.K[i_05]),
        )
        rel = isnan(jq) || isnan(eq) || iszero(eq) ? NaN :
            abs(jq - eq) / abs(eq)
        @printf "  %-4s  %-7s  %-12.4e %-12.4e %-12.2e\n" name qty jq eq rel
    end
end
