using Test
using MeanFieldHomogenization
using TensND

# =============================================================================
#  test_counters.jl — the work counters of the benchmark harness, and the
#  shared caches, under threads.
#
#  The counters are bumped at seams that threaded assemblies reach; a plain
#  `Ref{Int}` increment loses counts when two threads meet there, and a `Dict`
#  cache filled from two threads at once is corrupted. With one thread the
#  checks below hold either way; with several they are what fails.
# =============================================================================

const MC = MeanFieldHomogenization.Core

@testset "Work counters" begin
    MC.reset_counters!()
    @test all(iszero, values(MC.read_counters()))
    n = 10_000
    Threads.@threads for _ in 1:n
        MC._bump!(MC.SC_ITERATIONS)
    end
    @test MC.read_counters().sc_iterations == n

    # The per-node counter counts only when asked, and counts every node.
    KM = [
        210.0 80.0 75.0 5.0 4.0 3.0; 80.0 195.0 90.0 -2.0 3.0 -1.0; 75.0 90.0 220.0 1.0 -2.0 2.0;
        5.0 -2.0 1.0 60.0 2.5 1.5; 4.0 3.0 -2.0 2.5 65.0 -1.0; 3.0 -1.0 2.0 1.5 -1.0 55.0
    ]
    C = TensND.inv_KM(KM, TensND.CanonicalBasis{3, Float64}())
    ell = Ellipsoid(1.0, 0.7, 0.3)
    MC.reset_counters!()
    P = hill_tensor(ell, C; method = :nestedquadgk)
    @test MC.read_counters().integrand_evals == 0
    @test MC.read_counters().hill_calls == 1
    P_counted = MC.with_integrand_counting(() -> hill_tensor(ell, C; method = :nestedquadgk))
    @test MC.read_counters().integrand_evals > 0
    @test get_array(P_counted) == get_array(P)
    MC.reset_counters!()
end

@testset "Gauss-Legendre rules from several threads" begin
    ns = repeat(collect(70:90), 8)
    rules = Vector{Any}(undef, length(ns))
    Threads.@threads for i in eachindex(ns)
        rules[i] = MC.gauss_legendre_nodes(ns[i])
    end
    for (n, (x, w)) in zip(ns, rules)
        @test length(x) == n && sum(w) ≈ 2 && sum(x .^ 2 .* w) ≈ 2 / 3
    end
end
