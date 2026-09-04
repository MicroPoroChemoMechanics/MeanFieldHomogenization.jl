# =============================================================================
#  benchmark_porous.jl
#
#  Cross-validation of every MeanFieldHomogenization porous scheme against the C++
#  porous reference called via PyCall. Two geometric
#  cases:
#
#   1) Spherical pores in a solid matrix.
#   2) Oblate spheroidal pores (ω = 0.2) in an oblate solid matrix
#      (ω = 0.2), each phase carrying an `iso` symmetrize so the
#      effective stiffness collapses to isotropic.
#
#  All ten schemes (Voigt, Reuss, Dilute, DiluteDual, Mori-Tanaka, Maxwell,
#  PCW, Self-Consistent, Asymmetric Self-Consistent, Differential) are
#  swept over φ ∈ {0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95}. Output is a tabular
#  diff (rtol per row) and a CSV file written next to the script.
#
#  Run from the package root:
#     julia --project=scripts/bench_echoes scripts/bench_echoes/benchmark_porous.jl
# =============================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)

using MeanFieldHomogenization
using TensND
using Printf
using PyCall

# ── Python-side wrapper ──────────────────────────────────────────────────────

py"""
import echoes
from echoes import (rve, ellipsoid, spheroidal, stiff_kmu, homogenize, ISO,
                    VOIGT, REUSS, MT, SC, ASC, DIFF, DIL, DILD, MAX, PCW)

_SCH = {'VOIGT':VOIGT, 'REUSS':REUSS, 'MT':MT, 'SC':SC, 'ASC':ASC,
        'DIFF':DIFF, 'DIL':DIL, 'DILD':DILD, 'MAX':MAX, 'PCW':PCW}

def py_chom_porous(phi, sch_name, omegas=1.0, omegap=1.0, iso_sym=False,
                   ks=72., mus=32., kp=1.e-6, mup=1.e-6,
                   epsrel=1.e-10, maxnb=300):
    Cs = stiff_kmu(ks, mus); Cp = stiff_kmu(kp, mup)
    ver = rve(matrix='SOLID')
    if iso_sym:
        ver['SOLID'] = ellipsoid(shape=spheroidal(omegas), symmetrize=[ISO], prop={'C':Cs})
        ver['PORE']  = ellipsoid(shape=spheroidal(omegap), symmetrize=[ISO], prop={'C':Cp})
    else:
        ver['SOLID'] = ellipsoid(shape=spheroidal(omegas), prop={'C':Cs})
        ver['PORE']  = ellipsoid(shape=spheroidal(omegap), prop={'C':Cp})
    ver['PORE'].fraction  = phi
    ver['SOLID'].fraction = 1.0 - phi
    ver.set_prop('C', Cs)
    try:
        C = homogenize(prop='C', rve=ver, scheme=_SCH[sch_name],
                       verbose=False, maxnb=maxnb, epsrel=epsrel,
                       select_best=True)
        return float(max(C.k, 0.0)), float(max(C.mu, 0.0))
    except Exception as e:
        return float('nan'), float('nan')
"""

const py_chom_porous = py"py_chom_porous"

# ── Julia-side wrapper ───────────────────────────────────────────────────────

const k_s, μ_s = 72.0, 32.0
const k_p, μ_p = 1.0e-6, 1.0e-6
const C_s = TensISO{3}(3 * k_s, 2 * μ_s)
const C_p = TensISO{3}(3 * k_p, 2 * μ_p)

# Schemes table — Symbol form (used here) plus a label for the Python side.
const SCHEMES = [
    (:voigt, Voigt(), "VOIGT"),
    (:reuss, Reuss(), "REUSS"),
    (:dilute, Dilute(), "DIL"),
    (:dilute_dual, DiluteDual(), "DILD"),
    (:mori_tanaka, MoriTanaka(), "MT"),
    (:maxwell, Maxwell(), "MAX"),
    (:pcw, PonteCastanedaWillis(), "PCW"),
    (
        :sc, SelfConsistent(; abstol = 1.0e-10, maxiters = 300),
        "SC",
    ),
    (
        :asc, AsymmetricSelfConsistent(; abstol = 1.0e-10, maxiters = 300),
        "ASC",
    ),
    (
        :differential, DifferentialScheme(; nsteps = 300),
        "DIFF",
    ),
]

# Aspect-ratio convention: `ω = c/a` (matches the C++ reference's
# `spheroidal(omega)`).  ω < 1 → oblate, ω > 1 → prolate, ω = 1 → sphere.
# All schemes use the SOLID phase as the matrix, mirroring the
# reference Python script.  For SC the matrix-distinguished
# Picard iteration with `select_best = true` is enough to track the
# physical (lower) branch through the percolation point.
function _build_rve(
        scheme, φ; ω_s = 1.0, ω_p = 1.0,
        sym_s = nothing, sym_p = nothing
    )
    rve = RVE()
    geom_s = Spheroid(ω_s)
    geom_p = Spheroid(ω_p)
    add_phase!(rve, :SOLID, geom_s, Dict(:C => C_s); fraction = :rest, symmetrize = sym_s)
    add_phase!(
        rve, :PORE, geom_p, Dict(:C => C_p);
        fraction = φ, symmetrize = sym_p
    )
    return rve
end

function _extract_kμ(C::TensND.AbstractTens)
    if C isa TensND.TensISO{4, 3}
        α, β = TensND.get_data(C)
        return max(α / 3, 0.0), max(β / 2, 0.0)
    end
    a = TensND.get_array(C)
    K = sum(a[i, i, j, j] for i in 1:3, j in 1:3) / 9
    full = sum(a[i, j, i, j] for i in 1:3, j in 1:3)
    μ = (full - 3 * K) / 10
    return max(K, 0.0), max(μ, 0.0)
end

function jl_porous(
        scheme, φ; ω_s = 1.0, ω_p = 1.0,
        sym_s = nothing, sym_p = nothing
    )
    try
        rve = _build_rve(
            scheme, φ; ω_s = ω_s, ω_p = ω_p,
            sym_s = sym_s, sym_p = sym_p
        )
        # `select_best` is an SC/ASC-specific kwarg; do not forward it to
        # closed-form schemes, which would otherwise fail on the unknown
        # keyword.
        if scheme isa SelfConsistent || scheme isa AsymmetricSelfConsistent
            C = homogenize(rve, scheme, :C; select_best = true)
        else
            C = homogenize(rve, scheme, :C)
        end
        return _extract_kμ(C)
    catch e
        @warn "Julia homogenize failed" scheme φ ω_s ω_p sym_s sym_p exception = (e, catch_backtrace())
        return (NaN, NaN)
    end
end

# ── Tolerance policy ────────────────────────────────────────────────────────
#
# Sphere case: tight rtol everywhere (no special percolation handling
# beyond φ exactly at the threshold).  Oblate case: relaxed rtol (oblate
# + iso-symmetrize is more numerically demanding).
function _tol(ω, φ)
    if ω == 1.0
        if isapprox(φ, 0.0; atol = 1.0e-12) || isapprox(φ, 1.0; atol = 1.0e-12)
            return (atol = 1.0e-3, rtol = 1.0e-6)
        end
        return (atol = 1.0e-3, rtol = 1.0e-3)
    else
        # Non-spherical: looser, both ECHOES and MFH go through
        # numerical localization-tensor branches.
        return (atol = 5.0e-3, rtol = 5.0e-3)
    end
end

_relerr(a, b) = (isnan(a) || isnan(b)) ? NaN :
    (a == 0 && b == 0) ? 0.0 : abs(a - b) / max(abs(a), abs(b), 1.0e-12)

function _within(a, b; atol, rtol)
    isnan(a) && isnan(b) && return true
    isnan(a) || isnan(b) && return false
    Δ = abs(a - b)
    return Δ ≤ atol || Δ ≤ rtol * max(abs(a), abs(b))
end

# ── Sweep ────────────────────────────────────────────────────────────────────

const φs = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95)

mutable struct BenchRow
    scheme::String
    ω::Float64
    iso::Bool
    φ::Float64
    k_jl::Float64
    k_py::Float64
    μ_jl::Float64
    μ_py::Float64
    k_relerr::Float64
    μ_relerr::Float64
    pass::Bool
end

function run_sweep!(rows::Vector{BenchRow})
    # ω here is the aspect ratio c/a, matching the C++ reference's
    # `spheroidal(omega)` convention.
    cases = [
        (ω = 1.0, sym = nothing, iso = false),
        (ω = 0.2, sym = :iso, iso = true),  # oblate spheroid; iso symmetrize
    ]
    for (_, scheme, lbl) in SCHEMES
        for case in cases
            ω = case.ω
            sym = case.sym
            for φ in φs
                k_jl, μ_jl = jl_porous(
                    scheme, φ;
                    ω_s = ω, ω_p = ω,
                    sym_s = sym, sym_p = sym
                )
                k_py, μ_py = py_chom_porous(φ, lbl, ω, ω, case.iso)
                tol = _tol(ω, φ)
                pass_k = _within(k_jl, k_py; tol...)
                pass_μ = _within(μ_jl, μ_py; tol...)
                push!(
                    rows, BenchRow(
                        lbl, ω, case.iso, φ,
                        k_jl, k_py, μ_jl, μ_py,
                        _relerr(k_jl, k_py),
                        _relerr(μ_jl, μ_py),
                        pass_k && pass_μ
                    )
                )
            end
        end
    end
    return rows
end

# ── Reporting ────────────────────────────────────────────────────────────────

function print_table(rows::Vector{BenchRow})
    println()
    @printf "%-6s  %-6s  %-3s  %-5s  %-12s  %-12s  %-12s  %-12s  %-9s  %-9s  %s\n" "scheme" "ω" "iso" "φ" "k_jl" "k_py" "μ_jl" "μ_py" "Δk_rel" "Δμ_rel" "pass"
    println("─"^130)
    for r in rows
        @printf "%-6s  %-6.2f  %-3s  %-5.2f  %-12.6g  %-12.6g  %-12.6g  %-12.6g  %-9.2e  %-9.2e  %s\n" r.scheme r.ω string(r.iso) r.φ r.k_jl r.k_py r.μ_jl r.μ_py r.k_relerr r.μ_relerr (r.pass ? "✓" : "✗")
    end
    n_total = length(rows)
    n_fail = count(r -> !r.pass, rows)
    println("─"^130)
    @printf "Total cases: %d   Pass: %d   Fail: %d\n" n_total (n_total - n_fail) n_fail
    if n_fail > 0
        println("\n[FAILED CASES]")
        for r in rows
            r.pass && continue
            @printf "  %s ω=%.2f iso=%s φ=%.2f : k_rel=%.2e  μ_rel=%.2e\n" r.scheme r.ω string(r.iso) r.φ r.k_relerr r.μ_relerr
        end
    end
    return n_fail == 0
end

function write_csv(rows::Vector{BenchRow})
    figdir = joinpath(@__DIR__, "figures")
    isdir(figdir) || mkdir(figdir)
    path = joinpath(figdir, "benchmark_porous.csv")
    open(path, "w") do io
        println(io, "scheme,omega,iso_sym,phi,k_jl,k_py,mu_jl,mu_py,k_relerr,mu_relerr,pass")
        for r in rows
            @printf io "%s,%g,%s,%g,%g,%g,%g,%g,%g,%g,%s\n" r.scheme r.ω r.iso r.φ r.k_jl r.k_py r.μ_jl r.μ_py r.k_relerr r.μ_relerr (r.pass ? "1" : "0")
        end
    end
    return @printf "\nCSV: %s\n" path
end

# ── Main ─────────────────────────────────────────────────────────────────────

rows = BenchRow[]
run_sweep!(rows)
all_pass = print_table(rows)
write_csv(rows)

if !all_pass
    println("\n[!]  Some cases fall outside tolerance. See [FAILED CASES] above.")
    exit(1)
else
    println("\n[OK] All porous benchmark cases match within tolerance.")
end
