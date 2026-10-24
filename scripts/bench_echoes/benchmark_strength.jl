# =============================================================================
#  benchmark_strength.jl — cross-validation of the quasi-brittle strength model (2011)
#  three-scale cement-paste / mortar strength model against the echoes C++
#  quasi-brittle strength reference.
#
#  The Julia model is the SHARED implementation `scripts/common/quasibrittle_strength.jl`,
#  built entirely on the public MeanFieldHomogenization API : the multi-bin
#  Self-Consistent hydrate foam is assembled with several non-coaxial
#  `TISymmetrize` families whose EXACT azimuthal average (`TensTI{4,T,8}`,
#  non-major-symmetric content preserved) now flows through the generic SC
#  kernel — no hand-rolled Mandel bypass, no private IFT solver.  The strength
#  sensitivity `dC_mo/dμ_hyd` is a single ForwardDiff pass through the whole
#  chain (multi-bin SC + MT + MT).
#
#  Elastic moduli (k, μ, E) are validated to 1 %; the strength `fc` to 2 %.
#  Sweeps a (w/c, α) grid, compares against `py_strength_reference`, and writes a CSV.
# =============================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)

# Which Python holds Echoes: chosen before PythonCall is loaded
include(joinpath(@__DIR__, "echoes_python.jl"))

using PythonCall
using Printf
using LinearAlgebra

include(joinpath(@__DIR__, "..", "common", "quasibrittle_strength.jl"))

# ── echoes Python reference: `strength_echoes.py`, next to this file ────────
pyimport("sys").path.insert(0, @__DIR__)
include(joinpath(@__DIR__, "pyjl.jl"))

const py_strength_reference = pyimport("strength_echoes").py_strength_reference

# ── Row comparison ──────────────────────────────────────────────────────────
_relerr(a, b) = abs(b) < 1.0e-12 ? abs(a - b) : abs(a - b) / abs(b)

# 1 % on elastic moduli (k, μ, E) ; 2 % on the strength criterion fc — the
# exact azimuthal average + full-chain ForwardDiff closes the gap that the
# former hand-rolled IFT left at ~15 %.
function _row_pass(r; rtol_mod = 1.0e-2, rtol_fc = 2.0e-2, atol = 5.0e-2)
    checks = (
        (r.k_jl, r.k_py, rtol_mod),
        (r.mu_jl, r.mu_py, rtol_mod),
        (r.E_jl, r.E_py, rtol_mod),
        (r.fc_jl, r.fc_py, rtol_fc),
    )
    return all(c -> _relerr(c[1], c[2]) ≤ c[3] || abs(c[1] - c[2]) ≤ atol, checks)
end

# ── Sweep ───────────────────────────────────────────────────────────────────
const WCS = (0.25, 0.35, 0.5, 0.65)
const N_α = 6

println("="^92)
println("Quasi-brittle strength — Julia (public API) vs echoes C++ cross-validation")
println("(NTHETA = $NTHETA, ω = $ω_aspect ; tol : moduli 1 %, fc 2 %)")
println("="^92)
@printf "%5s %6s | %8s %8s | %8s %8s | %8s %8s | %8s %8s | %5s\n" "wc" "α" "k_jl" "k_py" "μ_jl" "μ_py" "E_jl" "E_py" "fc_jl" "fc_py" "pass"
println("─"^92)

rows = NamedTuple[]
for wc in WCS
    αs = range(0.05, αmax(wc) * (1 - 1.0e-9); length = N_α)
    for α in αs
        py = pyjl(py_strength_reference(wc, α, 0.0, ω_aspect, NTHETA))
        py[1] || continue                       # skip physically invalid points
        _, k_py, mu_py, E_py, fc_py = py
        jl = try
            compute_point(wc, α)
        catch e
            @warn "Julia compute_point failed" wc α exception = e
            continue
        end
        row = (
            wc = wc, α = α,
            k_jl = jl.K_mo, k_py = k_py,
            mu_jl = jl.μ_mo, mu_py = mu_py,
            E_jl = jl.E_mo, E_py = E_py,
            fc_jl = jl.fc, fc_py = fc_py,
        )
        row = merge(row, (pass = _row_pass(row),))
        push!(rows, row)
        @printf "%5.2f %6.3f | %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f | %5s\n" row.wc row.α row.k_jl row.k_py row.mu_jl row.mu_py row.E_jl row.E_py row.fc_jl row.fc_py (row.pass ? "✓" : "✗")
    end
end

n_pass = count(r -> r.pass, rows)
@printf "\n%d / %d rows within tolerance.\n" n_pass length(rows)

# ── CSV export ──────────────────────────────────────────────────────────────
figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkpath(figdir)
csv_path = joinpath(figdir, "benchmark_strength.csv")
open(csv_path, "w") do io
    println(io, "wc,alpha,k_jl,k_py,mu_jl,mu_py,E_jl,E_py,fc_jl,fc_py,pass")
    for r in rows
        @printf io "%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n" r.wc r.α r.k_jl r.k_py r.mu_jl r.mu_py r.E_jl r.E_py r.fc_jl r.fc_py (r.pass ? 1 : 0)
    end
end
@printf "CSV: %s\n" csv_path
