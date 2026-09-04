# =============================================================================
#  bench_sc_solvers.jl — SC / ASC solver comparison: Picard vs built-in
#  Newton vs NonlinearSolve.jl (NewtonRaphson / TrustRegion).
#
#  This script explicitly `using NonlinearSolve` (a weak dependency of
#  the main package), so it needs it as a *direct* dependency somewhere
#  resolvable — provided by the self-contained `scripts/bench/Project.toml`
#  (same pattern as `scripts/bench_echoes/`), rather than the main
#  package environment (where NonlinearSolve is only a weakdep).
#
#  Run from the MeanFieldHomogenization.jl package root (instantiate once first):
#    julia --project=scripts/bench -e 'using Pkg; Pkg.instantiate()'
#    julia --project=scripts/bench scripts/bench/bench_sc_solvers.jl
#
#  Output: stdout (one solver per line, time + allocations), plus a
#          derivative comparison and a strength-criterion cross-check.
#
#  Uses `@elapsed`/`@allocated` (3 warm-ups + 5 samples, reports the
#  minimum) — same style as `bench_alv.jl`; no BenchmarkTools dependency
#  is added.
# =============================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using NonlinearSolve
using Printf

# ─── helpers ───────────────────────────────────────────────────────────────

"""
    bench(f, label; warmups = 3, samples = 5) -> (t_min_s, alloc_bytes)

Run `f()` `warmups + samples` times, return the minimum recorded time and
the allocation of that same fastest sample. See `bench_alv.jl` for the
identical convention used across the package's benchmark scripts.
"""
function bench(f::Function, label::String; warmups::Int = 3, samples::Int = 5)
    for _ in 1:warmups
        f()
    end
    t_min = Inf
    bytes = 0
    GC.gc()
    for _ in 1:samples
        s_alloc = @allocated f()
        t = @elapsed f()
        if t < t_min
            t_min = t
            bytes = s_alloc
        end
    end
    @printf "  %-38s  %8.3f ms   %10.3f MiB\n" label (t_min * 1.0e3) (bytes / 2^20)
    return (t_min, bytes)
end

println("="^78)
println("MeanFieldHomogenization — SC / ASC solver benchmark (Picard vs Newton vs NonlinearSolve)")
println("="^78)

# ─── Setup: a high-contrast 2-phase iso RVE (away from percolation) ────────

const k_m, μ_m = 30.0, 10.0
const k_i, μ_i = 300.0, 100.0    # 10x contrast — enough to make Picard work

function build_rve(; f = 0.3)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(k_m, μ_m)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(k_i, μ_i)); fraction = f)
    return rve
end

const SOLVERS = [
    ("Picard (AndersonDefault)", SelfConsistent()),
    ("Newton (built-in, dep-free)", SelfConsistent(; algorithm = NewtonDefault())),
    ("NewtonRaphson (NonlinearSolve)", SelfConsistent(; algorithm = NewtonRaphson())),
    ("TrustRegion (NonlinearSolve)", SelfConsistent(; algorithm = TrustRegion())),
    ("AutoNonlinear", SelfConsistent(; algorithm = AutoNonlinear())),
]

println("\n[1] Forward solve — homogenize(rve, SelfConsistent(algorithm=…))")
println("-"^78)
rve = build_rve()
results = Dict{String, TensND.AbstractTens}()
for (label, scheme) in SOLVERS
    C = homogenize(rve, scheme)   # warm up dispatch / precompile before timing
    results[label] = C
    bench(() -> homogenize(rve, scheme), label)
end

println("\n  Cross-check (all solvers agree on the same fixed point):")
C_ref = results[first(SOLVERS)[1]]
for (label, _) in SOLVERS
    rel = maximum(abs.(get_array(results[label]) .- get_array(C_ref))) / maximum(abs.(get_array(C_ref)))
    @printf "    %-38s  max rel. diff. vs Picard = %.3e\n" label rel
end

# ─── ASC too (stiffness branch here; compliance branch mirrors it) ─────────

println("\n[2] Forward solve — AsymmetricSelfConsistent")
println("-"^78)
const ASC_SOLVERS = [
    ("Picard (AndersonDefault)", AsymmetricSelfConsistent()),
    ("TrustRegion (NonlinearSolve)", AsymmetricSelfConsistent(; algorithm = TrustRegion())),
]
for (label, scheme) in ASC_SOLVERS
    homogenize(rve, scheme)
    bench(() -> homogenize(rve, scheme), label)
end

# ─── Derivative w.r.t. a phase modulus (the strength-criterion use case) ───

println("\n[3] ForwardDiff derivative ∂C[1111]/∂K_I — Picard vs Newton vs NonlinearSolve")
println("-"^78)

idxC = C -> get_array(C)[1, 1, 1, 1]

const DERIV_SOLVERS = [
    ("Picard (AndersonDefault)", SelfConsistent()),
    ("Newton (built-in, dep-free)", SelfConsistent(; algorithm = NewtonDefault())),
    ("NewtonRaphson (NonlinearSolve, IFT lift)", SelfConsistent(; algorithm = NewtonRaphson())),
    ("TrustRegion (NonlinearSolve, IFT lift)", SelfConsistent(; algorithm = TrustRegion())),
]

deriv_results = Dict{String, Float64}()
for (label, scheme) in DERIV_SOLVERS
    d = derivative(rve, scheme, property(:I, :C, :bulk); indexer = idxC)
    deriv_results[label] = d
    bench(() -> derivative(rve, scheme, property(:I, :C, :bulk); indexer = idxC), label)
end

println("\n  Cross-check (central finite difference, solver-independent ground truth):")
h = 1.0e-5
function f_modulus(K_I)
    r = RVE()
    add_phase!(r, :SOLID, Ellipsoid(1.0), Dict(:C => TensISO{3}(k_m, μ_m)); fraction = :rest)
    add_phase!(r, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(K_I, μ_i)); fraction = 0.3)
    return idxC(homogenize(r, SelfConsistent()))
end
d_fd = (f_modulus(k_i + h) - f_modulus(k_i - h)) / (2h)
@printf "    %-42s  %14.6f\n" "central finite difference" d_fd
for (label, _) in DERIV_SOLVERS
    @printf "    %-42s  %14.6f   rel. err. vs FD = %.3e\n" label deriv_results[label] abs(deriv_results[label] - d_fd) / abs(d_fd)
end

# ─── Strength criterion cross-check (echoes of scripts/40_…) ──────────────
#
# Ports the ellipse formula of `scripts/40_porous_strength_criterion.jl`,
# but drives the underlying SC solve through both a Picard and a
# NonlinearSolve algorithm, to show the strength-criterion application
# (mentioned in the task) is unaffected by which solver computes the
# homogenized moduli and their sensitivity.

println("\n[4] Strength-criterion ellipse (A, B) — Picard vs NonlinearSolve")
println("-"^78)

const k_s, μs_value = 1.0e6, 1.0
const TINY = 1.0e-12
const ω_aspect = 0.1
const φ_value = 0.15

function _C_hom_iso_2vec(μs::Real, scheme)
    T = typeof(μs)
    r = RVE(; T = T)
    add_phase!(r, :SOLID, Spheroid(ω_aspect), Dict(:C => TensISO{3}(convert(T, 3 * k_s), 2 * μs)); fraction = :rest, symmetrize = :iso)
    add_phase!(
        r, :PORE, Spheroid(ω_aspect),
        Dict(:C => TensISO{3}(convert(T, 3 * TINY), convert(T, 2 * TINY)));
        fraction = convert(T, φ_value), symmetrize = :iso
    )
    C = homogenize(r, scheme, :C)
    C_iso = MeanFieldHomogenization.Schemes._apply_symmetrize(C, MeanFieldHomogenization.Schemes.IsoSymmetrize())
    α, β = TensND.get_data(C_iso)
    return [α, β]
end

function ellipse_AB(scheme)
    Cp0 = _C_hom_iso_2vec(μs_value, scheme)
    K_hom, μ_hom = Cp0[1] / 3, Cp0[2] / 2
    dCp = ForwardDiff.derivative(μ -> _C_hom_iso_2vec(μ, scheme), μs_value)
    dK_dμs = dCp[1] / 3
    dμ_dμs = dCp[2] / 2
    A = (μs_value / K_hom)^2 * dK_dμs
    B = (μs_value / μ_hom)^2 * dμ_dμs
    return A, B
end

const STRENGTH_SOLVERS = [
    ("Picard (AndersonDefault)", SelfConsistent(; abstol = 1.0e-10, maxiters = 300, select_best = true)),
    ("TrustRegion (NonlinearSolve)", SelfConsistent(; algorithm = TrustRegion())),
]
for (label, scheme) in STRENGTH_SOLVERS
    A, B = ellipse_AB(scheme)
    a = sqrt((1 - φ_value) / (2A))
    b = sqrt((1 - φ_value) / B)
    @printf "    %-30s  A=%.6g  B=%.6g  a=%.6g  b=%.6g\n" label A B a b
end

println("\nDone.")
