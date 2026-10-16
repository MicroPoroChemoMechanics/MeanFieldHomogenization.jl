using Test
using MeanFieldHomogenization
using TensND

@testset "Cracks — cod_tensor smoke tests" begin
    E, ν = 210.0, 0.3
    k = E / (3 * (1 - 2ν))
    μ = E / (2 * (1 + ν))
    C₀ = TensISO{3}(3k, 2μ)

    # Penny / elliptic / ribbon — all return finite positive B[3,3]
    for c in (PennyCrack(1.0), EllipticCrack(1.0, 0.3), RibbonCrack(0.5))
        B = cod_tensor(c, C₀)
        @test B[3, 3] > 0
    end

    # Compliance contribution — size-independent H (4th-order tensor)
    pc = PennyCrack(1.0)
    H = inverse_property_contribution(pc, C₀)
    @test H[1, 1, 1, 1] == H[1, 1, 1, 1]   # no NaN
    # Budiansky helper reintroduces the density ε
    ΔS = delta_inverse_property(pc, H, 0.1)
    @test ΔS[1, 1, 1, 1] == ΔS[1, 1, 1, 1]   # no NaN
end
