# `HillCache` and `with_hill_cache`: a series of calls that keeps the shapes and
# the anisotropic matrix computes each numerical Hill tensor once. The results
# are the bits of the calls without the cache; a closed form never consults it;
# a dual input bypasses it, its derivative unchanged; the key tells apart what
# changes the tensor and recognizes what does not; the store stays bounded and
# safe across threads.

using Test
using MeanFieldHomogenization
using TensND
using ForwardDiff
using LinearAlgebra

const _HC_KM = [
    210.0 80 75 5 4 3; 80 195 90 -2 3 -1; 75 90 220 1 -2 2;
    5 -2 1 60 2.5 1.5; 4 3 -2 2.5 65 -1; 3 -1 2 1.5 -1 55
]
_hc_matrix(KM = _HC_KM) = TensND.from_mandel(KM, CanonicalBasis{3, Float64}())
_hc_vals(t) = vec(collect(TensND.canonical_components(t)))

function _hc_cell(C, f)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C); fraction = :rest)
    add_phase!(rve, :A, Ellipsoid(2.0, 1.0, 0.5), Dict(:C => iso_stiffness(300.0, 150.0)); fraction = f)
    add_phase!(rve, :B, Ellipsoid(1.0, 0.7, 0.3; euler_angles = (0.4, 1.0, 0.2)), Dict(:C => iso_stiffness(5.0, 2.0)); fraction = f / 2)
    return rve
end

@testset "HillCache: a sweep computes each Hill tensor once, same bits" begin
    C = _hc_matrix()
    fractions = range(0.02, 0.3; length = 15)
    plain = [homogenize(_hc_cell(C, f), MoriTanaka(), :C) for f in fractions]
    cache = HillCache()
    cached = with_hill_cache(cache) do
        [homogenize(_hc_cell(C, f), MoriTanaka(), :C) for f in fractions]
    end
    @test all(_hc_vals(a) == _hc_vals(b) for (a, b) in zip(plain, cached))
    @test cache.misses[] == 2 && cache.hits[] == 2 * length(fractions) - 2
    @test length(cache) == 2
    # Outside the scope, nothing is read or stored.
    homogenize(_hc_cell(C, 0.1), MoriTanaka(), :C)
    @test cache.misses[] == 2
    @test length(empty!(cache)) == 0
end

@testset "HillCache: what the key tells apart" begin
    C = _hc_matrix()
    ell = Ellipsoid(2.0, 1.0, 0.5; euler_angles = (0.3, 0.8, 1.1))
    cache = HillCache()
    with_hill_cache(cache) do
        P = hill_tensor(ell, C)
        # The same shape rebuilt in the same frame: read back.
        @test _hc_vals(hill_tensor(Ellipsoid(2.0, 1.0, 0.5; euler_angles = (0.3, 0.8, 1.1)), C)) == _hc_vals(P)
        @test cache.hits[] == 1
        # Another matrix, another method, other tolerances: computed.
        KM = copy(_HC_KM)
        KM[1, 1] += 1.0
        hill_tensor(ell, _hc_matrix(KM))
        hill_tensor(ell, C; method = :residues)
        hill_tensor(ell, C; reltol = 1.0e-8)
        @test cache.misses[] == 4
        # An infinite cylinder in the same matrix is keyed too.
        cyl = Cylinder(1.0, 0.6)
        Pc = hill_tensor(cyl, C)
        @test _hc_vals(hill_tensor(cyl, C)) == _hc_vals(Pc)
        @test cache.misses[] == 5 && cache.hits[] == 2
        # A closed form never consults the cache.
        hill_tensor(ell, iso_stiffness(30.0, 10.0))
        hill_tensor(Ellipsoid(1.0, 0.5), TensND.Tens(Array(TensISO{2}(15.0, 4.0))))
        @test cache.misses[] == 5 && cache.hits[] == 2
    end
end

@testset "HillCache: derivatives bypass it, bounded store, threads, inference" begin
    C = _hc_matrix()
    ell = Ellipsoid(2.0, 1.0, 0.5)
    canon(P) = Array(TensND.canonical_components(P))
    cache = HillCache()
    P, dP = with_hill_cache(cache) do
        canon(hill_tensor(ell, C)), ForwardDiff.derivative(λ -> canon(hill_tensor(ell, λ * C)), 1.0)
    end
    # ℙ(λℂ) = ℙ(ℂ)/λ: the derivative is −ℙ; the dual call was not stored.
    @test dP ≈ -P rtol = 1.0e-6
    @test cache.misses[] == 1 && length(cache) == 1

    small = HillCache(; maxentries = 1)
    shapes = (Ellipsoid(2.0, 1.0, 0.5), Ellipsoid(1.0, 0.7, 0.3))
    reference = [_hc_vals(hill_tensor(s, C)) for s in shapes]
    with_hill_cache(small) do
        for _ in 1:2, (s, ref) in zip(shapes, reference)
            @test _hc_vals(hill_tensor(s, C)) == ref
        end
    end
    @test length(small) == 1 && small.misses[] == 4

    shared = HillCache()
    results = Vector{Vector{Float64}}(undef, 8)
    with_hill_cache(shared) do
        Threads.@threads for i in 1:8
            results[i] = _hc_vals(hill_tensor(shapes[mod1(i, 2)], C))
        end
    end
    @test all(results[i] == reference[mod1(i, 2)] for i in 1:8)
    @test shared.hits[] + shared.misses[] == 8 && length(shared) == 2

    inferred(f) = only(Base.return_types(f, ()))
    @test inferred(() -> hill_tensor(ell, C)) == inferred(() -> with_hill_cache(() -> hill_tensor(ell, C), shared))
    @test_throws ArgumentError HillCache(; maxentries = 0)
end
