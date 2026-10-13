using Test
using MeanFieldHomogenization
using TensND

@testset "eshelby_tensor — sphere in isotropic matrix" begin
    E, ν = 210.0e3, 0.3
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2 * (1 + ν))
    C₀ = TensISO{3}(3 * (λ + 2μ / 3), 2μ)

    sphere = Ellipsoid(1.0)
    S = eshelby_tensor(sphere, C₀)

    # Classical Eshelby components for a sphere in an isotropic matrix
    s1 = (7 - 5ν) / (15 * (1 - ν))     # S_1111
    s2 = (5ν - 1) / (15 * (1 - ν))     # S_1122
    s4 = (4 - 5ν) / (15 * (1 - ν))     # S_1212

    @test S[1, 1, 1, 1] ≈ s1  rtol = 1.0e-12
    @test S[2, 2, 2, 2] ≈ s1  rtol = 1.0e-12
    @test S[3, 3, 3, 3] ≈ s1  rtol = 1.0e-12
    @test S[1, 1, 2, 2] ≈ s2  rtol = 1.0e-12
    @test S[1, 1, 3, 3] ≈ s2  rtol = 1.0e-12
    @test S[2, 2, 3, 3] ≈ s2  rtol = 1.0e-12
    @test S[1, 2, 1, 2] ≈ s4  rtol = 1.0e-12
    @test S[1, 3, 1, 3] ≈ s4  rtol = 1.0e-12
    @test S[2, 3, 2, 3] ≈ s4  rtol = 1.0e-12
end

@testset "eshelby_tensor — consistency S = P : C₀ (triaxial ellipsoid, iso matrix)" begin
    E, ν = 210.0e3, 0.3
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2 * (1 + ν))
    C₀ = TensISO{3}(3 * (λ + 2μ / 3), 2μ)

    ell = Ellipsoid(3.0, 2.0, 1.0; euler_angles = (π / 5, π / 4, π / 7))
    P = hill_tensor(ell, C₀)
    S_ref = P ⊡ C₀
    S = eshelby_tensor(ell, C₀)

    @test S ≈ S_ref rtol = 1.0e-12
end

@testset "eshelby_tensor — method keyword forwarded to hill_tensor" begin
    E, ν = 210.0e3, 0.3
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2 * (1 + ν))
    C₀ = TensISO{3}(3 * (λ + 2μ / 3), 2μ)

    ell = Ellipsoid(3.0, 2.0, 1.0)
    S_auto = eshelby_tensor(ell, C₀)                         # analytical (iso)
    S_decuhr = eshelby_tensor(ell, C₀; method = :decuhr)      # forced cubature
    @test S_auto ≈ S_decuhr rtol = 1.0e-4
end
