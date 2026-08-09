# =============================================================================
#  bench_alv.jl — micro-benchmarks for the Viscoelasticity hot paths.
#
#  Usage : julia --project bench/bench_alv.jl
#  Output: stdout (one ALV operation per line, time + allocations)
#
#  Designed to be re-run after each optimization step in the audit plan
#  to track regressions.  Uses `@time` (3 warm-ups + 5 measurements,
#  reports the minimum) — good enough for relative comparisons; switch
#  to `BenchmarkTools.@btime` for finer numbers.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf

# ─── helpers ───────────────────────────────────────────────────────────────

"""
    bench(f, label; warmups = 3, samples = 5) -> (t_min_s, alloc_bytes)

Run `f()` `warmups + samples` times, return the minimum recorded time
and total allocated bytes of the best sample.  Argument order is
`(f, label)` so the `do` syntax works : `bench("name") do ... end`.
"""
function bench(f::Function, label::String; warmups::Int = 3, samples::Int = 5)
    for _ in 1:warmups
        f()
    end
    t_min   = Inf
    bytes   = 0
    GC.gc()
    for _ in 1:samples
        s_alloc = @allocated f()
        t = @elapsed f()
        if t < t_min
            t_min = t
            bytes = s_alloc
        end
    end
    @printf "  %-50s  %8.3f ms   %10.3f MiB\n" label (t_min * 1e3) (bytes / 2^20)
    return (t_min, bytes)
end

# ─── ALV setup (Maxwell iso, 50 time points, MT scheme) ────────────────────

function _build_setup(n::Int; transversely_isotropic::Bool = false)
    times = collect(range(0.0, 5.0; length = n))

    function R_iso(t, tp)
        α = 3 * (1.0 + 4.0 * exp(-(t - tp) / 1.0))
        β = 2 * (0.5 + 1.5 * exp(-(t - tp) / 0.5))
        return TensISO{3}(α, β)
    end
    law_M = ViscoLaw(R_iso, :relaxation)

    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => law_M))

    C_I_t = TensISO{3}(60.0, 20.0)
    add_phase!(rve, :I, Ellipsoid(1.0, 1.0, 0.5),
               Dict(:C => heaviside_law(C_I_t));
               fraction = 0.2)

    if transversely_isotropic
        # Add another inclusion to push out of pure iso
        ℓ_I = TensTI{4}(20.0, 30.0, 4.0, 5.0, 8.0, (0.0, 0.0, 1.0))
        add_phase!(rve, :J, Spheroid(1.5),
                   Dict(:C => heaviside_law(ℓ_I));
                   fraction = 0.05)
    end
    return (; rve, times, law_M)
end

# ─── benchmarks ────────────────────────────────────────────────────────────

function run_all_benches()
    println("=" ^ 80)
    println("  bench_alv.jl — ALV micro-benchmarks (MeanFieldHomogenization v$(VERSION))")
    println("=" ^ 80)

    for n in (50, 100, 200)
        ctx = _build_setup(n)
        rve, times = ctx.rve, ctx.times
        law_M = ctx.law_M
        println()
        println("---  n = $n time points  --- (matrix size = $(6n)×$(6n))")

        bench("trapezoidal_matrix(law_M, times)") do
            trapezoidal_matrix(law_M, times)
        end

        M = trapezoidal_matrix(law_M, times)
        bench("volterra_inverse(M, block_size = 6)") do
            volterra_inverse(M; block_size = 6)
        end

        bench("homogenize_alv(rve, Voigt(), :C; times)") do
            homogenize_alv(rve, Voigt(), :C; times = times)
        end

        bench("homogenize_alv(rve, Reuss(), :C; times)") do
            homogenize_alv(rve, Reuss(), :C; times = times)
        end

        bench("homogenize_alv(rve, Dilute(), :C; times)") do
            homogenize_alv(rve, Dilute(), :C; times = times)
        end

        bench("homogenize_alv(rve, MoriTanaka(), :C; times)") do
            homogenize_alv(rve, MoriTanaka(), :C; times = times)
        end

        bench("homogenize_alv(rve, Maxwell(), :C; times)") do
            homogenize_alv(rve, Maxwell(), :C; times = times)
        end

        if n <= 100
            bench("homogenize_alv(rve, SelfConsistent(), :C; times)") do
                homogenize_alv(rve, SelfConsistent(), :C; times = times,
                               abstol = 1e-10, reltol = 1e-10, maxiters = 200)
            end
        end

        # Iso/TI/ortho fast path overhead — heuristic detection cost
        bench("_is_iso_block(M)") do
            MeanFieldHomogenization.Viscoelasticity._is_iso_block(M)
        end
        bench("_is_ti_block(M)") do
            MeanFieldHomogenization.Viscoelasticity._is_ti_block(M)
        end
        bench("_is_ortho_block(M)") do
            MeanFieldHomogenization.Viscoelasticity._is_ortho_block(M)
        end

        # Conversions
        α, β = iso_params_from_blocks(M)
        bench("iso_params_from_blocks(M)") do
            iso_params_from_blocks(M)
        end
        bench("iso_blocks_from_params(α, β)") do
            iso_blocks_from_params(α, β)
        end

        ℓ = ti_params_from_blocks(M)
        bench("ti_params_from_blocks(M)") do
            ti_params_from_blocks(M)
        end
        bench("ti_blocks_from_params(ℓ)") do
            ti_blocks_from_params(ℓ)
        end

        o = ortho_params_from_blocks(M)
        bench("ortho_params_from_blocks(M)") do
            ortho_params_from_blocks(M)
        end
        bench("ortho_blocks_from_params(o)") do
            ortho_blocks_from_params(o)
        end
    end

    println()
    println("=" ^ 80)
    println("  done")
    println("=" ^ 80)
end

run_all_benches()
