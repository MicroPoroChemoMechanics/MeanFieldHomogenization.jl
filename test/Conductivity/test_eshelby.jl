using Test
using MeanFieldHomogenization
using TensND

@testset "eshelby_tensor — sphere in isotropic conductivity" begin
    K₀ = TensISO{3}(2.5)
    sphere = Ellipsoid(1.0)
    s = eshelby_tensor(sphere, K₀)

    # s = P · K₀ and for a sphere in isotropic K₀ = K·I, P = I/(3K)
    # ⇒ s = (1/3) I
    @test s[1, 1] ≈ 1 / 3  rtol = 1.0e-12
    @test s[2, 2] ≈ 1 / 3  rtol = 1.0e-12
    @test s[3, 3] ≈ 1 / 3  rtol = 1.0e-12
    @test isapprox(s[1, 2], 0.0; atol = 1.0e-12)
    @test isapprox(s[1, 3], 0.0; atol = 1.0e-12)
    @test isapprox(s[2, 3], 0.0; atol = 1.0e-12)
end

@testset "eshelby_tensor — consistency s = P · K₀ (triaxial ellipsoid, iso conductivity)" begin
    K₀ = TensISO{3}(1.3)
    ell = Ellipsoid(3.0, 2.0, 1.0; euler_angles = (π / 5, π / 4, π / 7))
    P = hill_tensor(ell, K₀)
    s_ref = P ⋅ K₀
    s = eshelby_tensor(ell, K₀)
    @test s ≈ s_ref rtol = 1.0e-12
end

@testset "eshelby_tensor — 2D sphere (circle) in isotropic 2-D conductivity" begin
    K₀ = TensISO{2}(1.0)
    circle = Ellipsoid(1.0; dim = 2)
    s = eshelby_tensor(circle, K₀)
    @test s[1, 1] ≈ 1 / 2  rtol = 1.0e-12
    @test s[2, 2] ≈ 1 / 2  rtol = 1.0e-12
    @test isapprox(s[1, 2], 0.0; atol = 1.0e-12)
end
