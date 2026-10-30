using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import ForwardDiff
import Random

# =============================================================================
#  test_second_hill.jl — the second Hill tensor Q = C − C:P:C (order 4) and
#  Q = K − K·P·K (order 2), held to closed forms that do not go through
#  `hill_tensor`, and to the Kelvin-Mandel matrices of the operands in random
#  bases.
# =============================================================================

# Q of a sphere in an isotropic matrix: 12kμ/(3k+4μ) 𝕁 + 2μ(9k+8μ)/(5(3k+4μ)) 𝕂.
_sh_sphere_Q(k, μ) = (12k * μ / (3k + 4μ), 2μ * (9k + 8μ) / (5 * (3k + 4μ)))

@testset "second_hill_tensor — sphere in an isotropic matrix" begin
    k, μ = 10.0, 4.0
    Q = @inferred second_hill_tensor(Ellipsoid(1.0), TensISO{3}(3k, 2μ))
    @test Q isa TensISO
    @test collect(TensND.get_data(Q)) ≈ collect(_sh_sphere_Q(k, μ)) rtol = 1.0e-13

    # The derivative with respect to the shear modulus, against the derivative
    # of the closed form.
    dQ = ForwardDiff.derivative(μ -> collect(TensND.get_data(second_hill_tensor(Ellipsoid(1.0), TensISO{3}(3k, 2μ)))), μ)
    dQ_ref = [36k^2 / (3k + 4μ)^2, 2 * (27k^2 + 48k * μ + 32μ^2) / (5 * (3k + 4μ)^2)]
    @test dQ ≈ dQ_ref rtol = 1.0e-12
end

@testset "second_hill_tensor — circle in plane strain" begin
    # Mura's Eshelby tensor of a circular cylinder, written in ν alone:
    # Q = C:(𝕀 − 𝕊), with no Hill tensor involved.
    α, β = 24.0, 6.0
    C = TensISO{2}(α, β)
    ν = (α - β) / (2α)          # plane strain, as in test_hill_2d.jl
    s1111 = (5 - 4ν) / (8 * (1 - ν))
    s1122 = (4ν - 1) / (8 * (1 - ν))
    s1212 = (3 - 4ν) / (8 * (1 - ν))
    S_M = [s1111 s1122 0; s1122 s1111 0; 0 0 2s1212]
    C_M = TensND.mandel(C, CanonicalBasis{2, Float64}())
    Q = second_hill_tensor(Ellipsoid(1.0; dim = 2), C)
    @test TensND.mandel(Q, CanonicalBasis{2, Float64}()) ≈ C_M * (I - S_M) rtol = 1.0e-12
end

@testset "second_hill_tensor — rotated ellipsoid, anisotropic matrix, random bases" begin
    rng = Random.Xoshiro(20261025)
    for _ in 1:3
        C = rand(rng, Tens{4, 3}; definite = true) + 10 * tens_Id4(Val(3), Val(Float64))
        R = TensND.random_basis(rng)
        C_rot = change_tens(C, R)
        ell = Ellipsoid(2.0, 1.0, 0.5; euler_angles = Tuple(π * rand(rng, 3)))
        Q = second_hill_tensor(ell, C_rot; method = :nestedquadgk)
        # The same product on the canonical Kelvin-Mandel matrices.
        canon(t) = TensND.mandel(t, CanonicalBasis{3, Float64}())
        C_M = canon(C)
        P_M = canon(hill_tensor(ell, C; method = :nestedquadgk))
        @test canon(Q) ≈ C_M - C_M * P_M * C_M rtol = 1.0e-9
    end
end

@testset "second_hill_tensor — order 2" begin
    k = 5.0
    Q = @inferred second_hill_tensor(Ellipsoid(1.0), TensISO{3}(k))
    @test Q isa TensISO
    @test Array(components_canon(Q)) ≈ (2k / 3) * I(3) rtol = 1.0e-13
    Q2 = second_hill_tensor(Ellipsoid(1.0; dim = 2), TensISO{2}(k))
    @test Array(components_canon(Q2)) ≈ (k / 2) * I(2) rtol = 1.0e-13

    # An anisotropic conductor in a random basis: the product on the canonical
    # components.
    rng = Random.Xoshiro(20261026)
    A = randn(rng, 3, 3)
    K = TensND.Tens(A * transpose(A) + 3I)
    K_rot = change_tens(K, TensND.random_basis(rng))
    ell = Ellipsoid(3.0, 1.0, 0.4; euler_angles = (0.3, 1.1, -0.7))
    Kc = Array(components_canon(K))
    Pc = Array(components_canon(hill_tensor(ell, K)))
    @test Array(components_canon(second_hill_tensor(ell, K_rot))) ≈ Kc - Kc * Pc * Kc rtol = 1.0e-11
end
