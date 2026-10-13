using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  Contribution tensors N (stiffness) and H (compliance) for ellipsoids.
# =============================================================================

@testset "stiffness_contribution — identity limit and sign" begin
    k₀ = 100.0
    μ₀ = 70.0
    C₀ = TensISO{3}(3k₀, 2μ₀)
    ell = Ellipsoid(1.0)

    # C₁ = C₀ → N = 0
    N0 = stiffness_contribution(ell, C₀, C₀)
    @test maximum(abs, N0) < 1.0e-12

    # Stiffer inclusion → N is positive-definite contribution
    C₁ = TensISO{3}(3k₀ * 1.5, 2μ₀ * 1.5)
    N = stiffness_contribution(ell, C₁, C₀)
    @test N[1, 1, 1, 1] > 0
end

@testset "Dilute coherence: C_eff ≈ C₀ + f·N to first order in f" begin
    k₀ = 1.0
    μ₀ = 0.5
    C₀ = TensISO{3}(3k₀, 2μ₀)
    C₁ = TensISO{3}(3k₀ * 3.0, 2μ₀ * 3.0)   # stiff inclusion
    ell = Ellipsoid(1.0)

    N = stiffness_contribution(ell, C₁, C₀)
    H = compliance_contribution(ell, C₁, C₀)

    f = 0.02   # small volume fraction
    ΔC = delta_stiffness(N, f)
    ΔS = delta_compliance(H, f)
    C_eff = C₀ + ΔC
    S_eff = inv(C₀) + ΔS

    # S_eff ≈ inv(C_eff) to first order in f
    S_from_C = inv(C_eff)
    # First-order agreement: residual ~ f² × scale
    @test maximum(abs, components_canon(S_eff - S_from_C)) / maximum(abs, components_canon(S_from_C)) < 5 * f^2
end

@testset "compliance_contribution — crack case (2-arg) still works" begin
    # Regression: the top-level extension must not break Cracks methods.
    E, ν = 1.0, 0.25
    k = E / (3 * (1 - 2ν))
    μ = E / (2 * (1 + ν))
    C₀ = TensISO{3}(3k, 2μ)

    pc = PennyCrack(1.0)
    H_crack = compliance_contribution(pc, C₀)     # 2-arg method from Cracks
    @test H_crack[3, 3, 3, 3] > 0
end

@testset "delta_compliance / delta_stiffness — 2-arg generic (ellipsoid)" begin
    k₀ = 100.0
    μ₀ = 70.0
    C₀ = TensISO{3}(3k₀, 2μ₀)
    C₁ = TensISO{3}(3k₀ * 2, 2μ₀ * 2)
    ell = Ellipsoid(1.0)
    N = stiffness_contribution(ell, C₁, C₀)
    H = compliance_contribution(ell, C₁, C₀)

    f = 0.1
    ΔC = delta_stiffness(N, f)
    ΔS = delta_compliance(H, f)
    @test ΔC[1, 1, 1, 1] ≈ f * N[1, 1, 1, 1] rtol = 1.0e-12
    @test ΔS[1, 1, 1, 1] ≈ f * H[1, 1, 1, 1] rtol = 1.0e-12
end
