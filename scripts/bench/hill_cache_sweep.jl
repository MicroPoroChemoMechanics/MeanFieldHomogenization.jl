# =============================================================================
#  hill_cache_sweep.jl — a sweep of the fractions in an anisotropic matrix,
#  with and without a HillCache: the time of the sweep, the share of the Hill
#  tensors in it, and the check that the cache gives the same tensors.
#
#    julia --project=scripts/bench scripts/bench/hill_cache_sweep.jl
# =============================================================================

using MeanFieldHomogenization, TensND, BenchmarkTools
const MC = MeanFieldHomogenization.Core

KM = [
    210.0 80 75 5 4 3; 80 195 90 -2 3 -1; 75 90 220 1 -2 2;
    5 -2 1 60 2.5 1.5; 4 3 -2 2.5 65 -1; 3 -1 2 1.5 -1 55
]
C_tri = TensND.from_mandel(KM, CanonicalBasis{3, Float64}())
function cell(f)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_tri); fraction = :rest)
    add_phase!(rve, :A, Ellipsoid(2.0, 1.0, 0.5), Dict(:C => iso_stiffness(300.0, 150.0)); fraction = f)
    add_phase!(rve, :B, Ellipsoid(1.0, 0.7, 0.3; euler_angles = (0.4, 1.0, 0.2)), Dict(:C => iso_stiffness(5.0, 2.0)); fraction = f / 2)
    return rve
end
sweep() = [homogenize(cell(f), MoriTanaka(), :C) for f in range(0.01, 0.3; length = 100)]
fresh() = with_hill_cache(sweep, HillCache())

sweep(); fresh()
t = minimum(@benchmark sweep() samples = 5 evals = 1).time / 1.0e6
tc = minimum(@benchmark fresh() samples = 5 evals = 1).time / 1.0e6
same = all(TensND.canonical_components(a) == TensND.canonical_components(b) for (a, b) in zip(sweep(), fresh()))
println("100 Mori–Tanaka cells, triclinic matrix: ", round(t; digits = 1), " ms without a cache, ",
    round(tc; digits = 1), " ms with a new HillCache; the same tensors: ", same)
