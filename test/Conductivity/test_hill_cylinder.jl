using Test
using MeanFieldHomogenization
using TensND

@testset "Conductivity — Hill tensor for infinite cylinders" begin

    @testset "Isotropic conductivity" begin
        k = 3.5
        K = TensISO{3}(k)
        cyl = Cylinder(2.0, 1.0)
        H = hill_tensor(cyl, K)
        b, c = 2.0, 1.0
        @test H[1, 1] ≈ 0.0 atol = 1.0e-14
        @test H[2, 2] ≈ c / (k * (b + c)) atol = 1.0e-12
        @test H[3, 3] ≈ b / (k * (b + c)) atol = 1.0e-12
        # Sum of transverse capacities = 1/k
        @test H[2, 2] + H[3, 3] ≈ 1 / k atol = 1.0e-12

        # Circular case
        cyl2 = Cylinder(1.5)
        H2 = hill_tensor(cyl2, K)
        @test H2[1, 1] ≈ 0.0 atol = 1.0e-14
        @test H2[2, 2] ≈ 1 / (2k) atol = 1.0e-12
        @test H2[3, 3] ≈ 1 / (2k) atol = 1.0e-12
    end

    @testset "Anisotropic conductivity" begin
        K_aniso_arr = [2.0 0.0 0.0; 0.0 3.5 0.5; 0.0 0.5 1.8]
        K_aniso = TensND.Tens(K_aniso_arr)
        cyl = Cylinder(2.0, 1.0)
        H = hill_tensor(cyl, K_aniso)
        # Axial components identically zero
        @test H[1, 1] ≈ 0.0 atol = 1.0e-14
        @test H[1, 2] ≈ 0.0 atol = 1.0e-14
        @test H[1, 3] ≈ 0.0 atol = 1.0e-14
        # Consistency with the limit of a very elongated ellipsoid
        ell_big = Ellipsoid(1.0e8, 2.0, 1.0)
        Href = hill_tensor(ell_big, K_aniso)
        @test H[2, 2] ≈ Href[2, 2] rtol = 1.0e-10
        @test H[3, 3] ≈ Href[3, 3] rtol = 1.0e-10
        @test H[2, 3] ≈ Href[2, 3] rtol = 1.0e-10
    end
end
