using Test
using MeanFieldHomogenization

@testset "Core — newton potentials" begin
    # Sphere: ΣIᵢ = 4π
    Iv, IIv = MeanFieldHomogenization.Core.newton_potential_3d(1.0, 1.0, 1.0)
    @test sum(Iv) ≈ 4π atol = 1.0e-12

    # Near the sphere the spheroid formulas divided the difference of two nearly
    # equal potentials by c² − a², and the second-order potentials came out far
    # from their sphere values (a Hill tensor 72 % off at |ω − 1| = 2e-6). They
    # now move from the sphere values by no more than the shape does.
    for (a, b, c) in ((1.0, 1.0, 1 - 2.0e-6), (1 + 2.0e-6, 1.0, 1.0), (1.0, 1.0, 0.9999), (1.0001, 1.0, 1.0))
        Iv, IIv = MeanFieldHomogenization.Core.newton_potential_3d(a, b, c)
        δ = a - c
        @test sum(Iv) ≈ 4π rtol = 1.0e-14
        @test all(abs(I - 4π / 3) < 50δ for I in Iv)
        @test all(abs(I - 4π / 5) < 50δ for I in IIv)
    end

    # 2D circle
    Ia, Ib = MeanFieldHomogenization.Core.newton_potential_2d(1.0, 1.0)
    @test Ia ≈ π atol = 1.0e-12
    @test Ib ≈ π atol = 1.0e-12

    # Moduli extractors
    C_iso = TensND.TensISO{3}(3.0, 2.0)
    E, ν = MeanFieldHomogenization.Core.extract_iso_moduli(C_iso)
    @test E > 0
    @test -1 < ν < 0.5
end
