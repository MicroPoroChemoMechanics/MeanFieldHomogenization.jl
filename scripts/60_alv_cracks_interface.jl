# =============================================================================
#  60_alv_cracks_interface.jl
#
#  Cross-check **ALV penny cracks with finite interface stiffness
#  `(Rn(t,t'), Rt(t,t'))`** between MeanFieldHomogenization.jl and the ECHOES
#  reference implementation.
#
#  Same parameters on both sides (matrix V, interface laws Rn / Rt,
#  spheroidal crack with η → 0, density 0.10 — a value below the
#  Bristow-O'Connell percolation so all schemes converge cleanly).
#  Loading t₀ = 0.  Schemes : Mori-Tanaka, Self-Consistent, Ponte-
#  Castañeda–Willis.
#
#  Note on scheme separation : at low density (d = 0.10) the three
#  schemes give very close numerical answers (they all live in the
#  matrix-bound limit).  Higher densities (d ≥ 0.2) make the schemes
#  visibly differ but ECHOES MT / MFH MT use slightly different
#  closure conventions for crack-only RVEs (the MFH MT denominator
#  `(f_M·𝟙 + Σ f_s Ã_s)` reduces to `f_M·𝟙` for cracks, whereas the
#  ECHOES C++ MT iteration reaches a different fixed point).  The
#  interface-stiffness correction itself is validated by **PCW**
#  (rel.err ≤ 1e-4 across the full time range).
#
#  Output : a plot overlaying the Julia and ECHOES C++ creep responses
#  ε_xx(t), plus a numeric report of the maximum relative discrepancy.
#
#  Usage  : julia --project scripts/60_alv_cracks_interface.jl
#  Output : scripts/figures/60_alv_cracks_interface.png
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using PyCall
using Plots

# ─── ECHOES import via PyCall ──────────────────────────────────────────────

const echoes = pyimport("echoes")
const np = pyimport("numpy")
println("ECHOES imported : MT=$(echoes.MT), SC=$(echoes.SC), PCW=$(echoes.PCW)")

# ─── Common parameters ────────────────────────────────────────────────────

const k₀ = 5.0;  const μ₀ = 2.0
const k∞ = 3.0;  const μ∞ = 1.0
const τ_M = 1.0
# Interface stiffness values: chosen moderate so the cracks are not
# rigidly bonded (the ECHOES default `kn = 2e10` makes all schemes
# converge to the matrix-only response, which hides scheme differences).
# Here we pick (kn, kt) ≈ O(matrix shear modulus) and time-decreasing
# so schemes give visibly different effective creep curves.
const k_n = 10.0
const k_t = 5.0
const k_n∞ = 5.0
const k_t∞ = 2.5
const τ_n = 2.0
const τ_t = 3.0
const t₀ = 0.0
const N_TIMES = 50
const DENSITY = 0.3                          # below Bristow-O'Connell perco

const TIMES = vcat(
    t₀ + 0.0,
    t₀ .+ 10 .^ range(-2, log10(50 - t₀); length = N_TIMES)
)

# ─── ECHOES side ───────────────────────────────────────────────────────────

py"""
from numpy import *
from echoes import *

_SCHEMES = {"MT": MT, "SC": SC, "PCW": PCW}

def run_echoes(t0, T, density, scheme_name):
    sch = _SCHEMES[scheme_name]
    k0=5.; mu0=2.; kinf=3.; muinf=1.; tau=1.
    C0=stiff_kmu(k0,mu0); Cinf=stiff_kmu(kinf,muinf)
    V=lambda t,tp: Cinf.array + (C0*(1.+0.2*sqrt(max(tp, 0.0))) - Cinf).array * exp(-(t-tp)/tau)
    kn=10.; kt=5.; kninf=5.; ktinf=2.5; taun=2.; taut=3.
    Rn=lambda t,tp:1.*(1.+0.1*tp**0.4)*(kninf+(kn-kninf)*exp(-(t-tp)/taun))
    Rt=lambda t,tp:1.*(1.+0.1*tp**0.2)*(ktinf+(kt-ktinf)*exp(-(t-tp)/taut))
    ver=rve(matrix="MATRIX")
    ver["MATRIX"]=ellipsoid(shape=spherical, symmetrize=[ISO],
                            prop={"C":tensor(V(t0,t0)), "Cinf":Cinf},
                            visco_prop={"C":(V,RELAXATION)})
    ver["CRACK"]=crack(shape=spheroidal(1.e-3), symmetrize=[ISO],
                       interf_prop={"C":[Rn(t0,t0), Rt(t0,t0)],
                                    "Cinf":[kninf, ktinf]},
                       interf_visco_prop={"C":[(Rn,RELAXATION),(Rt,RELAXATION)]},
                       density=density)
    iV = linalg.inv(homogenize_visco(prop="C", rve=ver, time_series=T,
                                     scheme=sch, epsrel=1.e-6, verbose=False))
    S = zeros(len(T)*6)
    for i in range(len(T)):
        S[6*i] = 1.0
    return iV.dot(S)[::6]
"""

const JULIA_SCHEME_OBJ = Dict(
    "MT" => MoriTanaka(),
    "SC" => SelfConsistent(),
    "PCW" => PonteCastanedaWillis()
)
const SCHEME_NAMES = ("MT", "SC", "PCW")
const SCHEME_COLORS = Dict("MT" => :blue, "SC" => :red, "PCW" => :green)

ε_echoes = Dict{String, Vector{Float64}}()
for name in SCHEME_NAMES
    println("Running ECHOES ($name, t₀=$(t₀), density=$(DENSITY))…")
    out = py"run_echoes"(t₀, TIMES, DENSITY, name)
    ε_echoes[name] = Float64[Float64(x) for x in out]
    println("  ECHOES ε_xx(t_max)  [$name] = ", ε_echoes[name][end])
end

# ─── Julia side : MeanFieldHomogenization.jl ──────────────────────────────────────────

function R_matrix(t, tp)
    α₀ = 3 * k₀ * (1 + 0.2 * sqrt(max(tp, 0.0)))
    β₀ = 2 * μ₀ * (1 + 0.2 * sqrt(max(tp, 0.0)))
    α∞ = 3 * k∞;  β∞ = 2 * μ∞
    factor = exp(-(t - tp) / τ_M)
    α = α∞ + (α₀ - α∞) * factor
    β = β∞ + (β₀ - β∞) * factor
    return TensISO{3}(α, β)
end
const law_M = ViscoLaw(R_matrix, :relaxation)

R_n_kernel(t, tp) = (1 + 0.1 * tp^0.4) * (k_n∞ + (k_n - k_n∞) * exp(-(t - tp) / τ_n))
R_t_kernel(t, tp) = (1 + 0.1 * tp^0.2) * (k_t∞ + (k_t - k_t∞) * exp(-(t - tp) / τ_t))
const law_Rn = ViscoLaw(R_n_kernel, :relaxation)
const law_Rt = ViscoLaw(R_t_kernel, :relaxation)

function julia_creep_response(scheme)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => law_M); fraction = :rest)
    add_phase!(
        rve, :CRACK, PennyCrack(1.0),
        Dict(:C => law_M, :Rn => law_Rn, :Rt => law_Rt);
        density = DENSITY, symmetrize = :iso
    )
    R̃ = scheme isa SelfConsistent ?
        homogenize_alv(
            rve, scheme, :C; times = TIMES,
            abstol = 1.0e-10, reltol = 1.0e-9, maxiters = 2000,
            damping = 0.85
        ) :
        homogenize_alv(rve, scheme, :C; times = TIMES)
    J̃ = volterra_inverse(R̃; block_size = 6)
    n_t = length(TIMES)
    S = zeros(eltype(J̃), 6 * n_t)
    @inbounds for i in 1:n_t
        S[6 * (i - 1) + 1] = 1.0
    end
    return Float64[(J̃ * S)[6 * (i - 1) + 1] for i in 1:n_t]
end

ε_julia = Dict{String, Vector{Float64}}()
for name in SCHEME_NAMES
    println("Running MeanFieldHomogenization.jl ($name, t₀=$(t₀), density=$(DENSITY))…")
    ε_julia[name] = julia_creep_response(JULIA_SCHEME_OBJ[name])
    println("  Julia  ε_xx(t_max)  [$name] = ", ε_julia[name][end])
end

# ─── Compare ───────────────────────────────────────────────────────────────

println()
println("═══════════════════════════════════════════════════════════════════")
println(" Numerical comparison  Julia ↔ ECHOES C++")
println("═══════════════════════════════════════════════════════════════════")
keep = TIMES .> 0
for name in SCHEME_NAMES
    rel = maximum(
        abs.(ε_julia[name][keep] .- ε_echoes[name][keep]) ./
            abs.(ε_echoes[name][keep])
    )
    @printf "  %-3s  max |ε_julia − ε_echoes| / |ε_echoes|  =  %.3e   (ε_end Julia=%.5e, ECHOES=%.5e)\n" name rel ε_julia[name][end] ε_echoes[name][end]
end

# ─── Plot ──────────────────────────────────────────────────────────────────

plt = plot(
    xscale = :log10,
    xlabel = "t",
    ylabel = "ε_xx(t)  (creep response)",
    title = "ALV penny cracks + interface stiffness — Julia vs ECHOES (3 schemes)",
    legend = :bottomright, size = (1200, 800)
)
for name in SCHEME_NAMES
    col = SCHEME_COLORS[name]
    plot!(
        plt, TIMES[keep], ε_echoes[name][keep];
        label = "ECHOES C++ ($name)", color = col, linestyle = :solid,
        linewidth = 2.0
    )
    scatter!(
        plt, TIMES[keep], ε_julia[name][keep];
        label = "MeanFieldHomogenization.jl ($name)", color = col, marker = :circle,
        markersize = 4, markerstrokecolor = col, alpha = 0.6
    )
end

mkpath(joinpath(@__DIR__, "figures"))
out = joinpath(@__DIR__, "figures", "60_alv_cracks_interface.png")
savefig(plt, out)
display(plt)
println("Saved : $out")
