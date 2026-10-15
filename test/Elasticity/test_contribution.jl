using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
# `import`, not `using`: SymPy and Symbolics both export `@variables`.
import Symbolics

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

# A void and an insulator have finite compliance / resistivity contributions,
# although their stiffness has no inverse: the contribution is read as
# ℍ = −𝕊₀ : ℕ : 𝕊₀. Before 0.16.3 the literal (𝕊₁ − 𝕊₀) : 𝔸_σσ multiplied an
# infinite compliance by a zero concentration and returned NaN, and with it
# DiluteDual and the compliance form of the differential scheme on pores.
# Reference: a spherical void in an isotropic matrix,
# ℍ = (1/(3k₀) + 1/(4μ₀)) 𝕁 + 5(3k₀ + 4μ₀) / (2μ₀(9k₀ + 8μ₀)) 𝕂.
@testset "compliance_contribution — void, closed form, AD and symbolic" begin
    hJ(k, μ) = 1 / (3k) + 1 / (4μ)
    hK(k, μ) = 5 * (3k + 4μ) / (2μ * (9k + 8μ))
    k₀, μ₀ = 10.0, 5.0
    C₀ = TensISO{3}(3k₀, 2μ₀)
    sphere = Ellipsoid(1.0)

    H = @inferred compliance_contribution(sphere, zero(C₀), C₀)
    @test H ≈ TensISO{3}(hJ(k₀, μ₀), hK(k₀, μ₀)) rtol = 1.0e-13

    # For a solid inclusion the identity reproduces the definition.
    spheroid = Ellipsoid(1.0, 1.0, 0.3)
    C₁ = TensISO{3}(90.0, 40.0)
    literal = (inv(C₁) - inv(C₀)) ⊡ (stress_strain_concentration(spheroid, C₁, C₀) ⊡ inv(C₀))
    @test @inferred(compliance_contribution(spheroid, C₁, C₀)) ≈ literal rtol = 1.0e-12

    # ForwardDiff through the matrix moduli, against the closed form.
    jcoef(T) = sum(T[i, i, j, j] for i in 1:3, j in 1:3) / 3
    void_H(k, μ) = compliance_contribution(sphere, zero(TensISO{3}(3k, 2μ)), TensISO{3}(3k, 2μ))
    @test ForwardDiff.derivative(k -> jcoef(void_H(k, μ₀)), k₀) ≈
        ForwardDiff.derivative(k -> hJ(k, μ₀), k₀) rtol = 1.0e-12
    @test ForwardDiff.derivative(μ -> void_H(k₀, μ)[1, 2, 1, 2], μ₀) ≈
        ForwardDiff.derivative(μ -> hK(k₀, μ) / 2, μ₀) rtol = 1.0e-12

    # Symbolic moduli: the closed form itself.
    Symbolics.@variables k μ
    Hs = void_H(k, μ)
    @test isequal(Symbolics.simplify(jcoef(Hs) - hJ(k, μ)), 0)
    @test isequal(Symbolics.simplify(2 * Hs[1, 2, 1, 2] - hK(k, μ)), 0)
end
