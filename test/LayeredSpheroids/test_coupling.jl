using Test
using LinearAlgebra
using MeanFieldHomogenization.LayeredSpheroids: coupling_matrices, _gamma_table

# =============================================================================
#  test_coupling.jl — interface coupling matrices I, J, K, L.
#
#  The default `:quadrature` path (stable in Float64) is cross-checked
#  against the `:series` path (faithful BigFloat monomial-coefficient
#  port of `spheroid_nlayers.py`, an independent derivation) — the two
#  should agree to machine precision for moderate 𝒩, exactly the point
#  of the precision argument in the paper's numerical-convergence
#  appendix (`eq:maxlogcij`).
# =============================================================================

@testset "coupling_matrices — γ table against known Legendre products" begin
    # Coefficients of Pᵢ(x)·Pⱼ(x), i,j = 0,…,5 (rational, hand-derived).
    refs = Dict(
        (0, 0) => [1.0],
        (1, 1) => [0.0, 0.0, 1.0],
        (1, 2) => [0.0, -0.5, 0.0, 1.5],
        (2, 2) => [0.25, 0.0, -1.5, 0.0, 2.25],
        (3, 3) => [0.0, 0.0, 2.25, 0.0, -7.5, 0.0, 6.25],
    )
    γ = _gamma_table(6, BigFloat)
    for ((i, j), ref) in refs
        @test length(γ[i + 1, j + 1]) == length(ref)
        @test all(isapprox.(Float64.(γ[i + 1, j + 1]), ref; atol = 1.0e-12))
    end
end

@testset "coupling_matrices — an unknown method is refused" begin
    @test_throws ArgumentError coupling_matrices(1.3, 2; method = :spline)
end

@testset "coupling_matrices — quadrature ≡ series (prolate)" begin
    for q in (1.3, 1.7, 3.3, 10.0), Nseries in (1, 2, 3, 5)
        I, J, K, L = coupling_matrices(q, Nseries; method = :quadrature)
        Is, Js, Ks, Ls = coupling_matrices(q, Nseries; method = :series)
        @test norm(I - Is) / max(norm(Is), 1) < 1.0e-10
        @test norm(J - Js) / max(norm(Js), 1) < 1.0e-10
        @test norm(K - Ks) / max(norm(Ks), 1) < 1.0e-10
        @test norm(L - Ls) / max(norm(Ls), 1) < 1.0e-10
    end
end

@testset "coupling_matrices — quadrature ≡ series (oblate, q = iτ)" begin
    for τ in (1.5, 4.0), Nseries in (2, 4, 6)
        q = im * τ
        I, J, K, L = coupling_matrices(q, Nseries; method = :quadrature)
        Is, Js, Ks, Ls = coupling_matrices(q, Nseries; method = :series)
        @test norm(I - Is) / max(norm(Is), 1) < 1.0e-10
        @test norm(J - Js) / max(norm(Js), 1) < 1.0e-10
        @test norm(K - Ks) / max(norm(Ks), 1) < 1.0e-10
        @test norm(L - Ls) / max(norm(Ls), 1) < 1.0e-10
    end
end

@testset "coupling_matrices — symmetry" begin
    I, J, K, L = coupling_matrices(2.1, 5; method = :quadrature)
    @test I ≈ I'
    @test J ≈ J'
    @test K ≈ K'
    @test L ≈ L'
end
